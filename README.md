# TransmiSmart v4.0 — Unified

Plataforma unificada de predicción de congestión para estaciones Transmilenio: **Calle 76, Calle 85 y Los Héroes**.

Fusión de `App_Transmilenio_Avance3` (interfaz + opinión ciudadana) y `transmismart_api_v3` (API Flask + predicción ML + PostgreSQL).

## Características

- **Predicción ML** con modelos regressor + classifier por estación (o modo demo si los `.pkl` no están disponibles)
- **Gráfica funcional** de proyección de pasajeros para las próximas 2–12h, con colores por nivel de congestión
- **Estación recomendada** — resalta automáticamente cuál tiene menor congestión
- **Servicios / rutas** por estación con tiempo estimado de llegada
- **Reportes ciudadanos** guardados en PostgreSQL, con estadísticas de precisión IA vs usuario
- **Feedback de satisfacción** guardado en DB
- **Colores dinámicos**: 🟢 BAJO · 🟡 MEDIO · 🔴 ALTO — tarjetas, línea de estaciones y gráfica
- **Persistencia completa** en PostgreSQL: predicciones, reportes, feedback

## Estructura

```
transmismart_unified/
├── app.py               # Flask backend (API + sirve el HTML)
├── requirements.txt
├── setup_db.sql         # Setup inicial PostgreSQL
├── .env.example         # Variables de entorno
├── models/              # ← Coloca aquí los archivos .pkl y model_metadata.json
│   ├── calle76_regressor.pkl
│   ├── calle76_classifier.pkl
│   ├── calle85_regressor.pkl
│   ├── calle85_classifier.pkl
│   ├── heroes_regressor.pkl
│   ├── heroes_classifier.pkl
│   └── model_metadata.json
└── static/
    └── index.html       # Frontend completo (HTML + CSS + JS)
```

## Instalación rápida

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar base de datos
psql -U postgres -f setup_db.sql

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales de PostgreSQL

# 5. Copiar modelos ML
# Copia los archivos .pkl de tu proyecto anterior a ./models/

# 6. Iniciar servidor
python app.py
# o con gunicorn:
# gunicorn app:app --bind 0.0.0.0:5000 --workers 2
```

Abre http://localhost:5000 en tu navegador.

## Endpoints API

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET/POST | `/predict/now` | Predicción actual (guarda en DB) |
| GET/POST | `/predict` | Predicción para fecha/hora específica |
| POST | `/predict/range` | Proyección para las próximas N horas |
| POST | `/report` | Guardar reporte ciudadano |
| GET | `/report/stats` | Estadísticas de reportes |
| POST | `/feedback` | Guardar feedback de satisfacción |
| GET | `/feedback` | Últimos feedbacks |
| GET | `/servicios/<key>` | Rutas por estación |
| GET | `/estaciones` | Info de todas las estaciones |
| GET | `/umbrales` | Umbrales de congestión por estación |
| GET | `/predicciones/historial` | Historial de predicciones en DB |
| GET | `/health` | Estado del sistema |

## Variables de entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `DATABASE_URL` | URL de conexión PostgreSQL | `postgresql://postgres:12345@localhost:5432/transmismart` |
| `PORT` | Puerto del servidor | `5000` |
| `FLASK_DEBUG` | Modo debug | `0` |

## Notas

- Si los modelos `.pkl` no están disponibles, el sistema arranca en **modo demo** con modelos sintéticos que siguen patrones realistas de hora pico.
- El frontend se sirve directamente desde Flask (`GET /`) — no se necesita servidor web adicional.
- En producción, usar **gunicorn** con al menos 2 workers y poner un **nginx** delante.
