# 📉 Bankruptcy Prediction System – Machine Learning + Streamlit + Power BI

Este proyecto implementa un **sistema avanzado de predicción de quiebras empresariales** usando modelos de *Machine Learning* (XGBoost calibrado + Altman Z-Score), dashboards interactivos en **Streamlit** y visualización ejecutiva en **Power BI**.  

El sistema integra **alertas automáticas**, monitorización de deriva (drift), registro de modelos y un pipeline completo de ETL para datos financieros y sectoriales.

---

## 🚀 Características principales
- 🔮 **Modelos ML supervisados** (XGBoost con calibración, validación temporal).  
- 📊 **Altman Z-Score** integrado como benchmark financiero.  
- 🏢 **Gestión de empresas y datos financieros** vía API y Streamlit.  
- 🚨 **Sistema de alertas** con notificaciones (email, Slack, Teams).  
- 📈 **Dashboards**:  
  - Streamlit → análisis interactivo y alertas.  
  - Power BI → reportes ejecutivos y análisis sectorial.  
- 🧩 **Arquitectura modular** (Backend + Frontend + Servicios).  
- 📦 **Registro y versionado de modelos** + monitor de drift.  
- 🧪 **Tests unitarios e integración** con Pytest.  

---

## 📂 Estructura del proyecto
```
bankruptcy_prediction/
├── app/            # Frontend Streamlit
├── backend/        # API Backend (Flask/FastAPI + ML)
├── services/       # Servicios auxiliares (ETL, Power BI, reportes, etc.)
├── database/       # schema.sql + conexión a BD
├── config/         # Configuración (settings, env, etc.)
├── tests/          # Unit tests e integración
├── docs/           # Documentación técnica y roadmap
├── assets/         # Recursos gráficos (logos, iconos)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Instalación
1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/bankruptcy-prediction.git
   cd bankruptcy-prediction
   ```

2. Crea un entorno virtual e instala dependencias:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate    # Windows

   pip install -r requirements.txt
   ```

3. Configura tus variables de entorno en un archivo `.env` (ejemplo en `config/.env.example`).

4. Inicializa la base de datos:
   ```bash
   sqlite3 bankruptcy.db < database/schema.sql
   ```
   *(o usa Postgres si está configurado en `.env`)*

---

## ▶️ Ejecución

### 1. Levantar Backend (API)
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
```
*(o con Flask: `flask run` si está configurado así)*

### 2. Levantar Frontend (Streamlit)
```bash
streamlit run app/main.py
```

### 3. Cargar datos demo (CSV)
```bash
curl -F "file=@data/empresas_demo.csv" http://localhost:5000/api/etl/upload-empresas
curl -F "file=@data/finanzas_demo.csv" http://localhost:5000/api/etl/upload-finanzas
```

---

## 🧪 Testing
Ejecutar todos los tests:
```bash
pytest -v
```

Tests disponibles:
- `test_data_processor.py` → ratios financieros, Altman Z, features.  
- `test_ml_services.py` → preprocesamiento → entrenamiento → predicción → drift → registro de modelos.  

---

## 📊 Integraciones
- **Power BI**: endpoints en `/api/powerbi/...` para dashboards ejecutivos.  
- **Alertas**: notificaciones por Email, Slack, Teams.  

---

## 📖 Documentación
La documentación técnica completa está en `/docs/`:
- `arquitectura_tecnica.md`  
- `Arquitectura del Sistema.md`  
- `plan_mejoras.md`  
- `todo.md`  

---

## 👩‍💻 Autora
**Valentina Bailon Cano**  
- Estudiante de Business Analytics – ICADE  
- Máster en Data Science & IA – EVOLVE  
- [LinkedIn](https://www.linkedin.com/)  
