"""
Este módulo implementa el servicio de preprocesamiento de datos financieros,
incluyendo limpieza, validación, ingeniería de características y cálculo de ratios.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

# Suprimir warnings de pandas
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configurar logging
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuración para el preprocesamiento de datos"""
    # Estrategias de imputación
    numerical_imputation_strategy: str = "median"  # "mean", "median", "most_frequent", "knn"
    categorical_imputation_strategy: str = "most_frequent"
    knn_neighbors: int = 5
    
    # Estrategias de escalado
    scaling_strategy: str = "robust"  # "standard", "robust", "minmax", "none"
    
    # Detección de outliers
    outlier_detection: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    outlier_threshold: float = 3.0
    outlier_action: str = "cap"  # "remove", "cap", "transform"
    
    # Ingeniería de características
    create_ratios: bool = True
    create_trends: bool = True
    create_volatility: bool = True
    create_sector_features: bool = True
    
    # Validación de datos
    min_data_completeness: float = 0.7  # 70% de datos completos mínimo
    max_missing_ratio: float = 0.3  # 30% máximo de valores faltantes por columna
    
    # Ventanas temporales para características
    trend_window: int = 4  # Trimestres para calcular tendencias
    volatility_window: int = 8  # Trimestres para calcular volatilidad

class DataQualityValidator:
    """Validador de calidad de datos"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.validation_results = {}
    
    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Valida la calidad de los datos de entrada
        
        Args:
            df: DataFrame con los datos a validar
            
        Returns:
            Tuple con (es_válido, diccionario_de_resultados)
        """
        results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'data_types': {},
            'outliers': {},
            'duplicates': 0,
            'completeness': 0.0,
            'issues': []
        }
        
        try:
            # Verificar datos faltantes
            missing_data = df.isnull().sum()
            missing_ratios = missing_data / len(df)
            results['missing_data'] = missing_ratios.to_dict()
            
            # Verificar completitud general
            overall_completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
            results['completeness'] = overall_completeness
            
            # Verificar tipos de datos
            results['data_types'] = df.dtypes.astype(str).to_dict()
            
            # Verificar duplicados
            duplicates = df.duplicated().sum()
            results['duplicates'] = duplicates
            
            # Identificar columnas con demasiados valores faltantes
            problematic_columns = missing_ratios[missing_ratios > self.config.max_missing_ratio].index.tolist()
            if problematic_columns:
                results['issues'].append(f"Columnas con >30% valores faltantes: {problematic_columns}")
            
            # Verificar completitud mínima
            if overall_completeness < self.config.min_data_completeness:
                results['issues'].append(f"Completitud de datos ({overall_completeness:.2%}) menor al mínimo ({self.config.min_data_completeness:.2%})")
            
            # Verificar outliers en columnas numéricas
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                outliers = self._detect_outliers(df[col].dropna())
                if len(outliers) > 0:
                    results['outliers'][col] = len(outliers)
            
            # Determinar si los datos son válidos
            is_valid = (
                overall_completeness >= self.config.min_data_completeness and
                len(problematic_columns) == 0 and
                len(df) > 0
            )
            
            logger.info(f"Validación de calidad completada. Válido: {is_valid}")
            return is_valid, results
            
        except Exception as e:
            logger.error(f"Error en validación de calidad: {str(e)}")
            results['issues'].append(f"Error en validación: {str(e)}")
            return False, results
    
    def _detect_outliers(self, series: pd.Series) -> List[int]:
        """Detecta outliers en una serie numérica"""
        if self.config.outlier_method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        
        elif self.config.outlier_method == "zscore":
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = series[z_scores > self.config.outlier_threshold].index.tolist()
        
        else:  # isolation_forest
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(series.values.reshape(-1, 1))
            outliers = series[outlier_labels == -1].index.tolist()
        
        return outliers

class FinancialRatioCalculator:
    """Calculadora de ratios financieros avanzados"""
    
    @staticmethod
    def calculate_altman_z_score(
        wc_ta: float, re_ta: float, ebit_ta: float, 
        me_tl: float, s_ta: float
    ) -> float:
        """
        Calcula el Altman Z-Score original
        
        Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*ME/TL + 1.0*S/TA
        """
        try:
            z_score = (
                1.2 * (wc_ta or 0) +
                1.4 * (re_ta or 0) +
                3.3 * (ebit_ta or 0) +
                0.6 * (me_tl or 0) +
                1.0 * (s_ta or 0)
            )
            return z_score
        except Exception:
            return np.nan
    
    @staticmethod
    def calculate_altman_z_score_modified(
        wc_ta: float, re_ta: float, ebit_ta: float, 
        bv_tl: float, s_ta: float
    ) -> float:
        """
        Calcula el Altman Z-Score modificado para empresas privadas
        
        Z = 0.717*WC/TA + 0.847*RE/TA + 3.107*EBIT/TA + 0.420*BV/TL + 0.998*S/TA
        """
        try:
            z_score = (
                0.717 * (wc_ta or 0) +
                0.847 * (re_ta or 0) +
                3.107 * (ebit_ta or 0) +
                0.420 * (bv_tl or 0) +
                0.998 * (s_ta or 0)
            )
            return z_score
        except Exception:
            return np.nan
    
    @staticmethod
    def calculate_piotroski_f_score(row: pd.Series) -> int:
        """
        Calcula el Piotroski F-Score (0-9)
        Evalúa la fortaleza financiera en 9 criterios
        """
        score = 0
        
        try:
            # Criterios de rentabilidad
            if row.get('roa', 0) > 0: score += 1  # ROA positivo
            if row.get('ocf_ta', 0) > 0: score += 1  # OCF positivo
            if row.get('roa', 0) > row.get('roa_prev', 0): score += 1  # ROA creciente
            if row.get('ocf_ta', 0) > row.get('roa', 0): score += 1  # OCF > ROA
            
            # Criterios de apalancamiento/liquidez
            if row.get('debt_assets', 1) < row.get('debt_assets_prev', 1): score += 1  # Deuda decreciente
            if row.get('current_ratio', 0) > row.get('current_ratio_prev', 0): score += 1  # Liquidez creciente
            if row.get('shares_outstanding', 1) <= row.get('shares_outstanding_prev', 1): score += 1  # Sin dilución
            
            # Criterios de eficiencia operativa
            if row.get('gross_margin', 0) > row.get('gross_margin_prev', 0): score += 1  # Margen bruto creciente
            if row.get('asset_turnover', 0) > row.get('asset_turnover_prev', 0): score += 1  # Rotación de activos creciente
            
        except Exception:
            pass
        
        return score
    
    @staticmethod
    def calculate_beneish_m_score(row: pd.Series) -> float:
        """
        Calcula el Beneish M-Score para detectar manipulación de estados financieros
        M-Score > -2.22 sugiere posible manipulación
        """
        try:
            # Variables del modelo Beneish
            dsri = row.get('dsri', 1)  # Days Sales in Receivables Index
            gmi = row.get('gmi', 1)    # Gross Margin Index
            aqi = row.get('aqi', 1)    # Asset Quality Index
            sgi = row.get('sgi', 1)    # Sales Growth Index
            depi = row.get('depi', 1)  # Depreciation Index
            sgai = row.get('sgai', 1)  # Sales General and Administrative expenses Index
            lvgi = row.get('lvgi', 1)  # Leverage Index
            tata = row.get('tata', 0)  # Total Accruals to Total Assets
            
            m_score = (
                -4.840 +
                0.920 * dsri +
                0.528 * gmi +
                0.404 * aqi +
                0.892 * sgi +
                0.115 * depi -
                0.172 * sgai +
                4.679 * tata -
                0.327 * lvgi
            )
            
            return m_score
        except Exception:
            return np.nan

class FeatureEngineer:
    """Ingeniero de características financieras"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def create_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea ratios financieros adicionales"""
        df = df.copy()
        
        try:
            # Ratios de liquidez
            df['quick_ratio'] = (df.get('current_assets', 0) - df.get('inventory', 0)) / df.get('current_liabilities', 1)
            df['cash_ratio'] = df.get('cash_equivalents', 0) / df.get('current_liabilities', 1)
            df['operating_cash_ratio'] = df.get('operating_cash_flow', 0) / df.get('current_liabilities', 1)
            
            # Ratios de actividad
            df['inventory_turnover'] = df.get('cogs', 0) / df.get('inventory', 1)
            df['receivables_turnover'] = df.get('revenue', 0) / df.get('accounts_receivable', 1)
            df['payables_turnover'] = df.get('cogs', 0) / df.get('accounts_payable', 1)
            df['working_capital_turnover'] = df.get('revenue', 0) / df.get('working_capital', 1)
            
            # Ratios de rentabilidad
            df['gross_profit_margin'] = df.get('gross_profit', 0) / df.get('revenue', 1)
            df['operating_profit_margin'] = df.get('operating_income', 0) / df.get('revenue', 1)
            df['net_profit_margin'] = df.get('net_income', 0) / df.get('revenue', 1)
            df['return_on_equity'] = df.get('net_income', 0) / df.get('shareholders_equity', 1)
            df['return_on_invested_capital'] = df.get('nopat', 0) / df.get('invested_capital', 1)
            
            # Ratios de apalancamiento
            df['debt_to_equity'] = df.get('total_debt', 0) / df.get('shareholders_equity', 1)
            df['equity_multiplier'] = df.get('total_assets', 1) / df.get('shareholders_equity', 1)
            df['interest_coverage'] = df.get('ebit', 0) / df.get('interest_expense', 1)
            df['debt_service_coverage'] = df.get('operating_cash_flow', 0) / df.get('debt_service', 1)
            
            # Ratios de eficiencia
            df['asset_turnover'] = df.get('revenue', 0) / df.get('total_assets', 1)
            df['fixed_asset_turnover'] = df.get('revenue', 0) / df.get('fixed_assets', 1)
            df['equity_turnover'] = df.get('revenue', 0) / df.get('shareholders_equity', 1)
            
            # Ratios de mercado (si disponibles)
            if 'market_cap' in df.columns:
                df['price_to_book'] = df.get('market_cap', 0) / df.get('book_value', 1)
                df['price_to_earnings'] = df.get('market_cap', 0) / df.get('net_income', 1)
                df['enterprise_value_ebitda'] = df.get('enterprise_value', 0) / df.get('ebitda', 1)
            
            logger.info(f"Creados {len([c for c in df.columns if c not in df.columns])} ratios financieros")
            
        except Exception as e:
            logger.error(f"Error creando ratios financieros: {str(e)}")
        
        return df
    
    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características de tendencia temporal"""
        if not self.config.create_trends or 'fecha_corte' not in df.columns:
            return df
        
        df = df.copy()
        df['fecha_corte'] = pd.to_datetime(df['fecha_corte'])
        
        try:
            # Ordenar por empresa y fecha
            if 'empresa_id' in df.columns:
                df = df.sort_values(['empresa_id', 'fecha_corte'])
                
                # Calcular cambios año a año
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in ['empresa_id', 'año']:
                        # Cambio YoY
                        df[f'{col}_yoy'] = df.groupby('empresa_id')[col].pct_change(periods=4)  # 4 trimestres
                        
                        # Tendencia (pendiente de regresión lineal sobre ventana)
                        df[f'{col}_trend'] = df.groupby('empresa_id')[col].rolling(
                            window=self.config.trend_window, min_periods=2
                        ).apply(self._calculate_trend, raw=False)
                        
                        # Media móvil
                        df[f'{col}_ma'] = df.groupby('empresa_id')[col].rolling(
                            window=self.config.trend_window
                        ).mean()
            
            logger.info("Características de tendencia creadas")
            
        except Exception as e:
            logger.error(f"Error creando características de tendencia: {str(e)}")
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características de volatilidad"""
        if not self.config.create_volatility or 'fecha_corte' not in df.columns:
            return df
        
        df = df.copy()
        
        try:
            if 'empresa_id' in df.columns:
                df = df.sort_values(['empresa_id', 'fecha_corte'])
                
                # Ratios clave para volatilidad
                key_ratios = ['revenue', 'ebitda', 'net_income', 'operating_cash_flow']
                
                for col in key_ratios:
                    if col in df.columns:
                        # Desviación estándar rolling
                        df[f'{col}_volatility'] = df.groupby('empresa_id')[col].rolling(
                            window=self.config.volatility_window, min_periods=3
                        ).std()
                        
                        # Coeficiente de variación
                        mean_val = df.groupby('empresa_id')[col].rolling(
                            window=self.config.volatility_window, min_periods=3
                        ).mean()
                        std_val = df.groupby('empresa_id')[col].rolling(
                            window=self.config.volatility_window, min_periods=3
                        ).std()
                        df[f'{col}_cv'] = std_val / (mean_val + 1e-8)  # Evitar división por cero
            
            logger.info("Características de volatilidad creadas")
            
        except Exception as e:
            logger.error(f"Error creando características de volatilidad: {str(e)}")
        
        return df
    
    def create_sector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características relativas al sector"""
        if not self.config.create_sector_features or 'sector' not in df.columns:
            return df
        
        df = df.copy()
        
        try:
            # Ratios clave para comparación sectorial
            key_ratios = ['roa', 'roe', 'debt_assets', 'current_ratio', 'gross_profit_margin']
            
            for col in key_ratios:
                if col in df.columns:
                    # Media sectorial
                    sector_mean = df.groupby('sector')[col].transform('mean')
                    df[f'{col}_sector_mean'] = sector_mean
                    
                    # Desviación respecto a la media sectorial
                    df[f'{col}_sector_deviation'] = df[col] - sector_mean
                    
                    # Percentil dentro del sector
                    df[f'{col}_sector_percentile'] = df.groupby('sector')[col].rank(pct=True)
                    
                    # Z-score sectorial
                    sector_std = df.groupby('sector')[col].transform('std')
                    df[f'{col}_sector_zscore'] = (df[col] - sector_mean) / (sector_std + 1e-8)
            
            logger.info("Características sectoriales creadas")
            
        except Exception as e:
            logger.error(f"Error creando características sectoriales: {str(e)}")
        
        return df
    
    def create_macro_features(self, df: pd.DataFrame, macro_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Crea características macroeconómicas"""
        if macro_data is None or 'fecha_corte' not in df.columns:
            return df
        
        df = df.copy()
        
        try:
            # Convertir fechas
            df['fecha_corte'] = pd.to_datetime(df['fecha_corte'])
            macro_data['fecha'] = pd.to_datetime(macro_data['fecha'])
            
            # Merge con datos macro (usando la fecha más cercana)
            df_with_macro = pd.merge_asof(
                df.sort_values('fecha_corte'),
                macro_data.sort_values('fecha'),
                left_on='fecha_corte',
                right_on='fecha',
                by='pais' if 'pais' in df.columns else None,
                direction='backward'
            )
            
            # Crear características derivadas de variables macro
            if 'gdp_yoy' in df_with_macro.columns:
                df_with_macro['gdp_recession'] = (df_with_macro['gdp_yoy'] < 0).astype(int)
            
            if 'y10y' in df_with_macro.columns and 'y3m' in df_with_macro.columns:
                df_with_macro['yield_curve_inversion'] = (
                    df_with_macro['y10y'] < df_with_macro['y3m']
                ).astype(int)
            
            if 'pmi' in df_with_macro.columns:
                df_with_macro['pmi_expansion'] = (df_with_macro['pmi'] > 50).astype(int)
            
            logger.info("Características macroeconómicas creadas")
            return df_with_macro
            
        except Exception as e:
            logger.error(f"Error creando características macroeconómicas: {str(e)}")
            return df
    
    @staticmethod
    def _calculate_trend(series: pd.Series) -> float:
        """Calcula la tendencia (pendiente) de una serie temporal"""
        try:
            if len(series) < 2:
                return np.nan
            
            x = np.arange(len(series))
            y = series.values
            
            # Regresión lineal simple
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
            
        except Exception:
            return np.nan

class DataPreprocessor:
    """Preprocesador principal de datos financieros"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.validator = DataQualityValidator(self.config)
        self.ratio_calculator = FinancialRatioCalculator()
        self.feature_engineer = FeatureEngineer(self.config)
        
        # Pipelines de preprocesamiento
        self.numerical_pipeline = None
        self.categorical_pipeline = None
        self.preprocessor = None
        
        # Metadatos
        self.feature_names = []
        self.is_fitted = False
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str] = None,
        macro_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Ajusta el preprocesador y transforma los datos
        
        Args:
            df: DataFrame con datos financieros
            target_column: Nombre de la columna objetivo (opcional)
            macro_data: DataFrame con datos macroeconómicos (opcional)
            
        Returns:
            DataFrame preprocesado
        """
        logger.info("Iniciando preprocesamiento de datos")
        
        # Validar calidad de datos
        is_valid, validation_results = self.validator.validate_data_quality(df)
        if not is_valid:
            logger.warning(f"Problemas de calidad detectados: {validation_results['issues']}")
        
        # Crear copia para no modificar el original
        processed_df = df.copy()
        
        # Ingeniería de características
        processed_df = self._engineer_features(processed_df, macro_data)
        
        # Limpiar y preparar datos
        processed_df = self._clean_data(processed_df)
        
        # Crear y ajustar pipelines de preprocesamiento
        self._create_preprocessing_pipelines(processed_df, target_column)
        
        # Aplicar transformaciones
        processed_df = self._apply_transformations(processed_df, target_column)
        
        self.is_fitted = True
        logger.info("Preprocesamiento completado")
        
        return processed_df
    
    def transform(
        self, 
        df: pd.DataFrame, 
        macro_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Transforma nuevos datos usando el preprocesador ajustado
        
        Args:
            df: DataFrame con datos financieros
            macro_data: DataFrame con datos macroeconómicos (opcional)
            
        Returns:
            DataFrame preprocesado
        """
        if not self.is_fitted:
            raise ValueError("El preprocesador debe ser ajustado antes de transformar")
        
        logger.info("Transformando nuevos datos")
        
        # Crear copia
        processed_df = df.copy()
        
        # Ingeniería de características
        processed_df = self._engineer_features(processed_df, macro_data)
        
        # Limpiar datos
        processed_df = self._clean_data(processed_df)
        
        # Aplicar transformaciones
        processed_df = self._apply_transformations(processed_df)
        
        return processed_df
    
    def _engineer_features(
        self, 
        df: pd.DataFrame, 
        macro_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Aplica ingeniería de características"""
        
        # Calcular ratios financieros básicos si no existen
        if self.config.create_ratios:
            df = self.feature_engineer.create_financial_ratios(df)
        
        # Calcular Altman Z-Score
        if all(col in df.columns for col in ['wc_ta', 're_ta', 'ebit_ta', 'me_tl', 's_ta']):
            df['altman_z_score'] = df.apply(
                lambda row: self.ratio_calculator.calculate_altman_z_score(
                    row['wc_ta'], row['re_ta'], row['ebit_ta'], 
                    row['me_tl'], row['s_ta']
                ), axis=1
            )
        
        # Crear características de tendencia
        if self.config.create_trends:
            df = self.feature_engineer.create_trend_features(df)
        
        # Crear características de volatilidad
        if self.config.create_volatility:
            df = self.feature_engineer.create_volatility_features(df)
        
        # Crear características sectoriales
        if self.config.create_sector_features:
            df = self.feature_engineer.create_sector_features(df)
        
        # Crear características macroeconómicas
        if macro_data is not None:
            df = self.feature_engineer.create_macro_features(df, macro_data)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y prepara los datos"""
        
        # Remover duplicados
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            logger.info(f"Removidos {initial_rows - len(df)} registros duplicados")
        
        # Manejar outliers
        if self.config.outlier_detection:
            df = self._handle_outliers(df)
        
        # Convertir tipos de datos
        df = self._convert_data_types(df)
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maneja outliers en datos numéricos"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['empresa_id', 'año', 'id']:  # Excluir IDs
                continue
            
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            # Detectar outliers
            outliers = self.validator._detect_outliers(series)
            
            if len(outliers) > 0:
                if self.config.outlier_action == "remove":
                    df = df.drop(outliers)
                elif self.config.outlier_action == "cap":
                    # Winsorización
                    lower_percentile = series.quantile(0.01)
                    upper_percentile = series.quantile(0.99)
                    df[col] = df[col].clip(lower=lower_percentile, upper=upper_percentile)
                elif self.config.outlier_action == "transform":
                    # Transformación logarítmica para valores positivos
                    if (series > 0).all():
                        df[col] = np.log1p(df[col])
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convierte tipos de datos apropiados"""
        
        # Convertir fechas
        date_columns = ['fecha_corte', 'fecha_fundacion', 'fecha_registro']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convertir categóricas
        categorical_columns = ['sector', 'subsector', 'pais', 'region', 'tamaño_empresa']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Asegurar que ratios sean float
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            if col not in categorical_columns and col not in date_columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        return df
    
    def _create_preprocessing_pipelines(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str] = None
    ):
        """Crea pipelines de preprocesamiento"""
        
        # Identificar columnas numéricas y categóricas
        if target_column and target_column in df.columns:
            feature_df = df.drop(columns=[target_column])
        else:
            feature_df = df
        
        # Excluir columnas de ID y fecha
        exclude_columns = ['id', 'empresa_id', 'uuid', 'fecha_corte', 'fecha_registro', 'fecha_fundacion']
        feature_df = feature_df.drop(columns=[col for col in exclude_columns if col in feature_df.columns])
        
        numeric_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = feature_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Pipeline numérico
        if self.config.numerical_imputation_strategy == "knn":
            numeric_imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
        else:
            numeric_imputer = SimpleImputer(strategy=self.config.numerical_imputation_strategy)
        
        if self.config.scaling_strategy == "standard":
            scaler = StandardScaler()
        elif self.config.scaling_strategy == "robust":
            scaler = RobustScaler()
        elif self.config.scaling_strategy == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = None
        
        if scaler:
            self.numerical_pipeline = Pipeline([
                ('imputer', numeric_imputer),
                ('scaler', scaler)
            ])
        else:
            self.numerical_pipeline = Pipeline([
                ('imputer', numeric_imputer)
            ])
        
        # Pipeline categórico
        self.categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=self.config.categorical_imputation_strategy))
        ])
        
        # Combinar pipelines
        transformers = []
        if numeric_features:
            transformers.append(('num', self.numerical_pipeline, numeric_features))
        if categorical_features:
            transformers.append(('cat', self.categorical_pipeline, categorical_features))
        
        if transformers:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'
            )
            
            # Guardar nombres de características
            self.feature_names = numeric_features + categorical_features
    
    def _apply_transformations(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """Aplica las transformaciones de preprocesamiento"""
        
        if self.preprocessor is None:
            return df
        
        # Preparar datos para transformación
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            X = df.copy()
            y = None
        
        # Excluir columnas de ID y fecha
        exclude_columns = ['id', 'empresa_id', 'uuid', 'fecha_corte', 'fecha_registro', 'fecha_fundacion']
        id_columns = {col: X[col] for col in exclude_columns if col in X.columns}
        X_features = X.drop(columns=[col for col in exclude_columns if col in X.columns])
        
        # Aplicar transformaciones
        if not self.is_fitted:
            X_transformed = self.preprocessor.fit_transform(X_features)
        else:
            X_transformed = self.preprocessor.transform(X_features)
        
        # Crear DataFrame resultado
        if hasattr(X_transformed, 'toarray'):  # Sparse matrix
            X_transformed = X_transformed.toarray()
        
        # Reconstruir DataFrame
        result_df = pd.DataFrame(
            X_transformed, 
            columns=self.feature_names,
            index=X.index
        )
        
        # Añadir columnas de ID de vuelta
        for col, values in id_columns.items():
            result_df[col] = values.values
        
        # Añadir target si existe
        if y is not None:
            result_df[target_column] = y.values
        
        return result_df
    
    def save_preprocessor(self, filepath: Union[str, Path]):
        """Guarda el preprocesador entrenado"""
        if not self.is_fitted:
            raise ValueError("El preprocesador debe ser ajustado antes de guardarlo")
        
        preprocessor_data = {
            'config': self.config,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocesador guardado en {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath: Union[str, Path]) -> 'DataPreprocessor':
        """Carga un preprocesador entrenado"""
        preprocessor_data = joblib.load(filepath)
        
        instance = cls(preprocessor_data['config'])
        instance.preprocessor = preprocessor_data['preprocessor']
        instance.feature_names = preprocessor_data['feature_names']
        instance.is_fitted = preprocessor_data['is_fitted']
        
        logger.info(f"Preprocesador cargado desde {filepath}")
        return instance
    
    def get_feature_names(self) -> List[str]:
        """Retorna los nombres de las características"""
        return self.feature_names.copy()
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Retorna un resumen del preprocesamiento aplicado"""
        return {
            'config': self.config.__dict__,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'pipelines': {
                'numerical': str(self.numerical_pipeline) if self.numerical_pipeline else None,
                'categorical': str(self.categorical_pipeline) if self.categorical_pipeline else None
            }
        }

# Función de utilidad para uso rápido
def preprocess_financial_data(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    macro_data: Optional[pd.DataFrame] = None,
    config: Optional[PreprocessingConfig] = None
) -> Tuple[pd.DataFrame, DataPreprocessor]:
    """
    Función de utilidad para preprocesar datos financieros
    
    Args:
        df: DataFrame con datos financieros
        target_column: Nombre de la columna objetivo
        macro_data: DataFrame con datos macroeconómicos
        config: Configuración de preprocesamiento
        
    Returns:
        Tuple con (DataFrame preprocesado, instancia del preprocesador)
    """
    preprocessor = DataPreprocessor(config)
    processed_df = preprocessor.fit_transform(df, target_column, macro_data)
    
    return processed_df, preprocessor

if __name__ == "__main__":
    # Ejemplo de uso
    import pandas as pd
    
    # Crear datos de ejemplo
    sample_data = pd.DataFrame({
        'empresa_id': [1, 1, 2, 2, 3, 3],
        'fecha_corte': ['2023-Q1', '2023-Q2', '2023-Q1', '2023-Q2', '2023-Q1', '2023-Q2'],
        'sector': ['Technology', 'Technology', 'Manufacturing', 'Manufacturing', 'Retail', 'Retail'],
        'wc_ta': [0.1, 0.12, 0.05, 0.03, 0.08, 0.09],
        're_ta': [0.15, 0.16, 0.10, 0.08, 0.12, 0.13],
        'ebit_ta': [0.08, 0.09, 0.06, 0.04, 0.07, 0.08],
        'me_tl': [1.5, 1.6, 0.8, 0.7, 1.2, 1.3],
        's_ta': [1.2, 1.3, 1.0, 0.9, 1.1, 1.2],
        'revenue': [1000, 1100, 800, 750, 600, 650],
        'total_assets': [2000, 2100, 1500, 1400, 1200, 1300],
        'y': [0, 0, 1, 1, 0, 0]  # Target variable
    })
    
    # Preprocesar datos
    processed_data, preprocessor = preprocess_financial_data(
        sample_data, 
        target_column='y'
    )
    
    print("Datos preprocesados:")
    print(processed_data.head())
    print(f"\nCaracterísticas creadas: {len(preprocessor.get_feature_names())}")
    print(f"Resumen: {preprocessor.get_preprocessing_summary()}")

