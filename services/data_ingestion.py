"""
Este módulo maneja la ingesta y transformación de datos financieros
desde múltiples fuentes (CSV, Excel, APIs, bases de datos externas).
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime, date
import json
import requests
from dataclasses import dataclass, field
from sqlalchemy.orm import Session
import io
import zipfile
import tempfile

from ..preprocessing.data_processor import DataPreprocessor, PreprocessingConfig
from ...database.connection import get_db_session
from ...database.models import (
    Empresa, DatoFinanciero, DatoMacroeconomico, LogETL,
    TamañoEmpresa, TipoDato, EmpresaStatus
)

# Configurar logging
logger = logging.getLogger(__name__)

@dataclass
class IngestionConfig:
    """Configuración para el proceso de ingesta"""
    
    # Validación de datos
    validate_data: bool = True
    skip_duplicates: bool = True
    update_existing: bool = False
    
    # Procesamiento
    auto_calculate_ratios: bool = True
    auto_detect_data_types: bool = True
    fill_missing_values: bool = True
    
    # Límites
    max_records_per_batch: int = 1000
    max_file_size_mb: int = 100
    
    # Configuración de errores
    continue_on_error: bool = True
    max_error_rate: float = 0.1  # 10% máximo de errores
    
    # Fuentes de datos
    allowed_file_types: List[str] = field(default_factory=lambda: ['.csv', '.xlsx', '.xls', '.json'])
    encoding: str = 'utf-8'
    
    # Mapeo de columnas
    column_mapping: Dict[str, str] = field(default_factory=dict)
    required_columns: List[str] = field(default_factory=lambda: ['empresa_id', 'fecha_corte'])

class DataValidator:
    """Validador de datos financieros"""
    
    def __init__(self, config: IngestionConfig):
        self.config = config
    
    def validate_empresa_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida datos de empresa"""
        errors = []
        
        # Validaciones obligatorias
        if not data.get('rut'):
            errors.append("RUT es obligatorio")
        elif not self._validate_rut(data['rut']):
            errors.append("RUT tiene formato inválido")
        
        if not data.get('razon_social'):
            errors.append("Razón social es obligatoria")
        
        if not data.get('sector'):
            errors.append("Sector es obligatorio")
        
        # Validar tamaño de empresa
        if data.get('tamaño') and data['tamaño'] not in [e.value for e in TamañoEmpresa]:
            errors.append(f"Tamaño de empresa inválido: {data['tamaño']}")
        
        # Validar fechas
        if data.get('fecha_constitucion'):
            if not self._validate_date(data['fecha_constitucion']):
                errors.append("Fecha de constitución inválida")
        
        return len(errors) == 0, errors
    
    def validate_financial_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida datos financieros"""
        errors = []
        
        # Validaciones obligatorias
        if not data.get('empresa_id') and not data.get('rut'):
            errors.append("empresa_id o RUT es obligatorio")
        
        if not data.get('fecha_corte'):
            errors.append("fecha_corte es obligatoria")
        elif not self._validate_date(data['fecha_corte']):
            errors.append("fecha_corte tiene formato inválido")
        
        if not data.get('total_activos') or data['total_activos'] <= 0:
            errors.append("total_activos debe ser mayor a 0")
        
        # Validaciones de consistencia
        if data.get('total_activos') and data.get('total_pasivos') and data.get('patrimonio'):
            if abs(data['total_activos'] - (data['total_pasivos'] + data['patrimonio'])) > 0.01:
                errors.append("Ecuación contable no cuadra: Activos ≠ Pasivos + Patrimonio")
        
        # Validar ratios financieros
        if data.get('liquidez_corriente') and data['liquidez_corriente'] < 0:
            errors.append("Liquidez corriente no puede ser negativa")
        
        # Validar rangos razonables
        if data.get('margen_bruto') and (data['margen_bruto'] < -1 or data['margen_bruto'] > 1):
            errors.append("Margen bruto fuera de rango razonable (-100% a 100%)")
        
        return len(errors) == 0, errors
    
    def _validate_rut(self, rut: str) -> bool:
        """Valida formato de RUT chileno"""
        try:
            # Remover puntos y guión
            rut_clean = rut.replace('.', '').replace('-', '')
            
            if len(rut_clean) < 8 or len(rut_clean) > 9:
                return False
            
            # Separar número y dígito verificador
            numero = rut_clean[:-1]
            dv = rut_clean[-1].upper()
            
            # Calcular dígito verificador
            suma = 0
            multiplicador = 2
            
            for digit in reversed(numero):
                suma += int(digit) * multiplicador
                multiplicador = multiplicador + 1 if multiplicador < 7 else 2
            
            resto = suma % 11
            dv_calculado = 'K' if resto == 1 else ('0' if resto == 0 else str(11 - resto))
            
            return dv == dv_calculado
            
        except:
            return False
    
    def _validate_date(self, date_value: Any) -> bool:
        """Valida formato de fecha"""
        try:
            if isinstance(date_value, (date, datetime)):
                return True
            
            if isinstance(date_value, str):
                # Intentar varios formatos
                formats = ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d']
                for fmt in formats:
                    try:
                        datetime.strptime(date_value, fmt)
                        return True
                    except:
                        continue
            
            return False
        except:
            return False

class FileProcessor:
    """Procesador de archivos de datos"""
    
    def __init__(self, config: IngestionConfig):
        self.config = config
    
    def process_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Procesa un archivo y retorna DataFrame"""
        file_path = Path(file_path)
        
        # Validar tamaño de archivo
        if file_path.stat().st_size > self.config.max_file_size_mb * 1024 * 1024:
            raise ValueError(f"Archivo excede el tamaño máximo de {self.config.max_file_size_mb}MB")
        
        # Procesar según extensión
        if file_path.suffix.lower() == '.csv':
            return self._process_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return self._process_excel(file_path)
        elif file_path.suffix.lower() == '.json':
            return self._process_json(file_path)
        elif file_path.suffix.lower() == '.zip':
            return self._process_zip(file_path)
        else:
            raise ValueError(f"Tipo de archivo no soportado: {file_path.suffix}")
    
    def _process_csv(self, file_path: Path) -> pd.DataFrame:
        """Procesa archivo CSV"""
        try:
            # Detectar separador
            with open(file_path, 'r', encoding=self.config.encoding) as f:
                sample = f.read(1024)
            
            separator = ',' if sample.count(',') > sample.count(';') else ';'
            
            # Leer CSV
            df = pd.read_csv(
                file_path,
                sep=separator,
                encoding=self.config.encoding,
                low_memory=False,
                na_values=['', 'NULL', 'null', 'N/A', 'n/a', '#N/A']
            )
            
            logger.info(f"CSV procesado: {len(df)} registros, {len(df.columns)} columnas")
            return df
            
        except Exception as e:
            logger.error(f"Error procesando CSV {file_path}: {str(e)}")
            raise
    
    def _process_excel(self, file_path: Path) -> pd.DataFrame:
        """Procesa archivo Excel"""
        try:
            # Leer todas las hojas y combinar
            excel_file = pd.ExcelFile(file_path)
            dataframes = []
            
            for sheet_name in excel_file.sheet_names:
                df_sheet = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    na_values=['', 'NULL', 'null', 'N/A', 'n/a', '#N/A']
                )
                
                if not df_sheet.empty:
                    df_sheet['_source_sheet'] = sheet_name
                    dataframes.append(df_sheet)
            
            if not dataframes:
                raise ValueError("No se encontraron datos en el archivo Excel")
            
            df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Excel procesado: {len(df)} registros de {len(dataframes)} hojas")
            return df
            
        except Exception as e:
            logger.error(f"Error procesando Excel {file_path}: {str(e)}")
            raise
    
    def _process_json(self, file_path: Path) -> pd.DataFrame:
        """Procesa archivo JSON"""
        try:
            with open(file_path, 'r', encoding=self.config.encoding) as f:
                data = json.load(f)
            
            # Convertir a DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError("Formato JSON no soportado")
            
            logger.info(f"JSON procesado: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"Error procesando JSON {file_path}: {str(e)}")
            raise
    
    def _process_zip(self, file_path: Path) -> pd.DataFrame:
        """Procesa archivo ZIP con múltiples archivos"""
        try:
            dataframes = []
            
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                for file_name in zip_file.namelist():
                    if any(file_name.lower().endswith(ext) for ext in self.config.allowed_file_types):
                        # Extraer archivo temporal
                        with tempfile.NamedTemporaryFile(suffix=Path(file_name).suffix) as temp_file:
                            temp_file.write(zip_file.read(file_name))
                            temp_file.flush()
                            
                            # Procesar archivo extraído
                            df_file = self.process_file(temp_file.name)
                            if not df_file.empty:
                                df_file['_source_file'] = file_name
                                dataframes.append(df_file)
            
            if not dataframes:
                raise ValueError("No se encontraron archivos válidos en el ZIP")
            
            df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"ZIP procesado: {len(df)} registros de {len(dataframes)} archivos")
            return df
            
        except Exception as e:
            logger.error(f"Error procesando ZIP {file_path}: {str(e)}")
            raise

class DataTransformer:
    """Transformador de datos para normalización"""
    
    def __init__(self, config: IngestionConfig):
        self.config = config
    
    def transform_empresa_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforma datos de empresas"""
        df = df.copy()
        
        # Aplicar mapeo de columnas
        if self.config.column_mapping:
            df = df.rename(columns=self.config.column_mapping)
        
        # Normalizar RUT
        if 'rut' in df.columns:
            df['rut'] = df['rut'].astype(str).str.upper().str.strip()
        
        # Normalizar texto
        text_columns = ['razon_social', 'nombre_fantasia', 'sector', 'subsector']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
        
        # Convertir tamaño de empresa
        if 'tamaño' in df.columns:
            df['tamaño'] = df['tamaño'].str.lower().map({
                'micro': TamañoEmpresa.MICRO,
                'pequeña': TamañoEmpresa.PEQUEÑA,
                'pequeña': TamañoEmpresa.PEQUEÑA,
                'mediana': TamañoEmpresa.MEDIANA,
                'grande': TamañoEmpresa.GRANDE
            })
        
        # Convertir fechas
        date_columns = ['fecha_constitucion', 'fecha_inicio_actividades']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def transform_financial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforma datos financieros"""
        df = df.copy()
        
        # Aplicar mapeo de columnas
        if self.config.column_mapping:
            df = df.rename(columns=self.config.column_mapping)
        
        # Convertir fecha de corte
        if 'fecha_corte' in df.columns:
            df['fecha_corte'] = pd.to_datetime(df['fecha_corte'], errors='coerce')
            
            # Extraer año y trimestre
            df['año'] = df['fecha_corte'].dt.year
            df['trimestre'] = df['fecha_corte'].dt.quarter
        
        # Convertir columnas numéricas
        numeric_columns = [
            'total_activos', 'total_pasivos', 'patrimonio',
            'activos_corrientes', 'pasivos_corrientes',
            'ingresos_operacionales', 'utilidad_neta',
            'flujo_efectivo_operacional'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calcular ratios básicos si no existen
        if self.config.auto_calculate_ratios:
            df = self._calculate_basic_ratios(df)
        
        # Detectar tipo de dato
        if self.config.auto_detect_data_types:
            df = self._detect_data_type(df)
        
        return df
    
    def _calculate_basic_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula ratios financieros básicos"""
        
        # Liquidez corriente
        if 'liquidez_corriente' not in df.columns and 'activos_corrientes' in df.columns and 'pasivos_corrientes' in df.columns:
            df['liquidez_corriente'] = df['activos_corrientes'] / df['pasivos_corrientes'].replace(0, np.nan)
        
        # Endeudamiento
        if 'endeudamiento_total' not in df.columns and 'total_pasivos' in df.columns and 'total_activos' in df.columns:
            df['endeudamiento_total'] = df['total_pasivos'] / df['total_activos'].replace(0, np.nan)
        
        # ROA
        if 'roa' not in df.columns and 'utilidad_neta' in df.columns and 'total_activos' in df.columns:
            df['roa'] = df['utilidad_neta'] / df['total_activos'].replace(0, np.nan)
        
        # Margen neto
        if 'margen_neto' not in df.columns and 'utilidad_neta' in df.columns and 'ingresos_operacionales' in df.columns:
            df['margen_neto'] = df['utilidad_neta'] / df['ingresos_operacionales'].replace(0, np.nan)
        
        return df
    
    def _detect_data_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta el tipo de dato (trimestral, anual, mensual)"""
        
        if 'tipo_dato' not in df.columns and 'fecha_corte' in df.columns:
            # Analizar frecuencia de fechas
            dates = pd.to_datetime(df['fecha_corte']).dropna()
            
            if len(dates) > 1:
                # Calcular diferencias entre fechas
                date_diffs = dates.sort_values().diff().dropna()
                avg_diff = date_diffs.mean().days
                
                if 80 <= avg_diff <= 100:  # ~3 meses
                    df['tipo_dato'] = TipoDato.TRIMESTRAL
                elif 350 <= avg_diff <= 380:  # ~1 año
                    df['tipo_dato'] = TipoDato.ANUAL
                elif 25 <= avg_diff <= 35:  # ~1 mes
                    df['tipo_dato'] = TipoDato.MENSUAL
                else:
                    df['tipo_dato'] = TipoDato.TRIMESTRAL  # Default
            else:
                df['tipo_dato'] = TipoDato.TRIMESTRAL  # Default
        
        return df

class DataIngestionService:
    """Servicio principal de ingesta de datos"""
    
    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        self.validator = DataValidator(self.config)
        self.file_processor = FileProcessor(self.config)
        self.transformer = DataTransformer(self.config)
        self.preprocessor = DataPreprocessor(PreprocessingConfig())
    
    def ingest_empresas_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Ingesta empresas desde archivo"""
        
        log_id = self._start_etl_log("ingest_empresas", str(file_path))
        
        try:
            # Procesar archivo
            df = self.file_processor.process_file(file_path)
            
            # Transformar datos
            df = self.transformer.transform_empresa_data(df)
            
            # Ingestar en lotes
            results = self._ingest_empresas_batch(df)
            
            # Finalizar log
            self._finish_etl_log(log_id, "COMPLETADO", results)
            
            return results
            
        except Exception as e:
            self._finish_etl_log(log_id, "ERROR", error_detail=str(e))
            raise
    
    def ingest_financial_data_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Ingesta datos financieros desde archivo"""
        
        log_id = self._start_etl_log("ingest_financial_data", str(file_path))
        
        try:
            # Procesar archivo
            df = self.file_processor.process_file(file_path)
            
            # Transformar datos
            df = self.transformer.transform_financial_data(df)
            
            # Ingestar en lotes
            results = self._ingest_financial_data_batch(df)
            
            # Finalizar log
            self._finish_etl_log(log_id, "COMPLETADO", results)
            
            return results
            
        except Exception as e:
            self._finish_etl_log(log_id, "ERROR", error_detail=str(e))
            raise
    
    def _ingest_empresas_batch(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Ingesta empresas en lotes"""
        
        total_records = len(df)
        successful_records = 0
        error_records = 0
        errors = []
        
        with get_db_session() as session:
            for i in range(0, total_records, self.config.max_records_per_batch):
                batch = df.iloc[i:i + self.config.max_records_per_batch]
                
                for _, row in batch.iterrows():
                    try:
                        # Validar datos
                        is_valid, validation_errors = self.validator.validate_empresa_data(row.to_dict())
                        
                        if not is_valid:
                            if self.config.continue_on_error:
                                errors.extend(validation_errors)
                                error_records += 1
                                continue
                            else:
                                raise ValueError(f"Errores de validación: {validation_errors}")
                        
                        # Verificar si existe
                        existing = session.query(Empresa).filter_by(rut=row['rut']).first()
                        
                        if existing:
                            if self.config.skip_duplicates:
                                continue
                            elif self.config.update_existing:
                                # Actualizar empresa existente
                                for key, value in row.to_dict().items():
                                    if hasattr(existing, key) and pd.notna(value):
                                        setattr(existing, key, value)
                            else:
                                error_records += 1
                                errors.append(f"Empresa con RUT {row['rut']} ya existe")
                                continue
                        else:
                            # Crear nueva empresa
                            empresa_data = row.to_dict()
                            empresa_data = {k: v for k, v in empresa_data.items() if pd.notna(v)}
                            
                            empresa = Empresa(**empresa_data)
                            session.add(empresa)
                        
                        successful_records += 1
                        
                    except Exception as e:
                        error_records += 1
                        errors.append(f"Error procesando fila {i}: {str(e)}")
                        
                        if not self.config.continue_on_error:
                            raise
                
                # Commit por lote
                session.commit()
        
        return {
            'total_records': total_records,
            'successful_records': successful_records,
            'error_records': error_records,
            'errors': errors[:10],  # Primeros 10 errores
            'error_rate': error_records / total_records if total_records > 0 else 0
        }
    
    def _ingest_financial_data_batch(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Ingesta datos financieros en lotes"""
        
        total_records = len(df)
        successful_records = 0
        error_records = 0
        errors = []
        
        with get_db_session() as session:
            for i in range(0, total_records, self.config.max_records_per_batch):
                batch = df.iloc[i:i + self.config.max_records_per_batch]
                
                for _, row in batch.iterrows():
                    try:
                        # Validar datos
                        is_valid, validation_errors = self.validator.validate_financial_data(row.to_dict())
                        
                        if not is_valid:
                            if self.config.continue_on_error:
                                errors.extend(validation_errors)
                                error_records += 1
                                continue
                            else:
                                raise ValueError(f"Errores de validación: {validation_errors}")
                        
                        # Buscar empresa
                        empresa = None
                        if 'empresa_id' in row and pd.notna(row['empresa_id']):
                            empresa = session.query(Empresa).filter_by(id=row['empresa_id']).first()
                        elif 'rut' in row and pd.notna(row['rut']):
                            empresa = session.query(Empresa).filter_by(rut=row['rut']).first()
                        
                        if not empresa:
                            error_records += 1
                            errors.append(f"Empresa no encontrada para fila {i}")
                            continue
                        
                        # Verificar si existe dato financiero
                        existing = session.query(DatoFinanciero).filter_by(
                            empresa_id=empresa.id,
                            fecha_corte=row['fecha_corte'],
                            tipo_dato=row.get('tipo_dato', TipoDato.TRIMESTRAL)
                        ).first()
                        
                        if existing:
                            if self.config.skip_duplicates:
                                continue
                            elif self.config.update_existing:
                                # Actualizar dato existente
                                for key, value in row.to_dict().items():
                                    if hasattr(existing, key) and pd.notna(value):
                                        setattr(existing, key, value)
                            else:
                                error_records += 1
                                errors.append(f"Dato financiero ya existe para empresa {empresa.rut} en {row['fecha_corte']}")
                                continue
                        else:
                            # Crear nuevo dato financiero
                            financial_data = row.to_dict()
                            financial_data['empresa_id'] = empresa.id
                            financial_data = {k: v for k, v in financial_data.items() if pd.notna(v)}
                            
                            dato = DatoFinanciero(**financial_data)
                            session.add(dato)
                        
                        successful_records += 1
                        
                    except Exception as e:
                        error_records += 1
                        errors.append(f"Error procesando fila {i}: {str(e)}")
                        
                        if not self.config.continue_on_error:
                            raise
                
                # Commit por lote
                session.commit()
        
        return {
            'total_records': total_records,
            'successful_records': successful_records,
            'error_records': error_records,
            'errors': errors[:10],  # Primeros 10 errores
            'error_rate': error_records / total_records if total_records > 0 else 0
        }
    
    def _start_etl_log(self, proceso: str, archivo_origen: str) -> int:
        """Inicia log de proceso ETL"""
        with get_db_session() as session:
            log = LogETL(
                proceso=proceso,
                fecha_inicio=datetime.now(),
                estado="INICIADO",
                archivo_origen=archivo_origen
            )
            session.add(log)
            session.commit()
            return log.id
    
    def _finish_etl_log(self, log_id: int, estado: str, results: Optional[Dict] = None, error_detail: Optional[str] = None):
        """Finaliza log de proceso ETL"""
        with get_db_session() as session:
            log = session.query(LogETL).filter_by(id=log_id).first()
            if log:
                log.fecha_fin = datetime.now()
                log.estado = estado
                
                if results:
                    log.registros_procesados = results.get('total_records', 0)
                    log.registros_exitosos = results.get('successful_records', 0)
                    log.registros_error = results.get('error_records', 0)
                    log.mensaje = f"Procesamiento completado. Tasa de error: {results.get('error_rate', 0):.2%}"
                
                if error_detail:
                    log.error_detalle = error_detail
                
                session.commit()

# Funciones de conveniencia
def ingest_empresas_from_file(file_path: Union[str, Path], config: Optional[IngestionConfig] = None) -> Dict[str, Any]:
    """Función de conveniencia para ingestar empresas"""
    service = DataIngestionService(config)
    return service.ingest_empresas_from_file(file_path)

def ingest_financial_data_from_file(file_path: Union[str, Path], config: Optional[IngestionConfig] = None) -> Dict[str, Any]:
    """Función de conveniencia para ingestar datos financieros"""
    service = DataIngestionService(config)
    return service.ingest_financial_data_from_file(file_path)

if __name__ == "__main__":
    # Ejemplo de uso
    import logging
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuración de ejemplo
    config = IngestionConfig(
        validate_data=True,
        skip_duplicates=True,
        auto_calculate_ratios=True,
        max_records_per_batch=500
    )
    
    # Crear servicio
    service = DataIngestionService(config)
    
    print("Servicio de ingesta de datos inicializado")
    print(f"Tipos de archivo soportados: {config.allowed_file_types}")
    print(f"Tamaño máximo de archivo: {config.max_file_size_mb}MB")
    print(f"Registros por lote: {config.max_records_per_batch}")

