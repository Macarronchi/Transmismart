-- TransmiSmart Unified v4.0 — Setup PostgreSQL
-- Ejecutar como superusuario: psql -U postgres -f setup_db.sql

-- Crear base de datos
CREATE DATABASE transmismart
    WITH ENCODING = 'UTF8'
         LC_COLLATE = 'es_CO.UTF-8'
         LC_CTYPE = 'es_CO.UTF-8'
         TEMPLATE = template0;

-- Crear usuario de aplicación
CREATE USER transmismart_user WITH PASSWORD 'transmismart_pass_2026';
GRANT ALL PRIVILEGES ON DATABASE transmismart TO transmismart_user;

-- Conectar a la base de datos
\c transmismart

-- Las tablas se crean automáticamente con SQLAlchemy al iniciar la API:
--   predicciones          → registro de cada predicción generada
--   reportes_ciudadanos   → opiniones de los usuarios sobre congestión
--   feedback_usuarios     → feedback detallado para mejora del modelo
--   cache_predicciones    → caché de predicciones por slot de tiempo

-- Índices recomendados (opcionales, SQLAlchemy los puede crear automáticamente)
-- CREATE INDEX idx_reportes_station   ON reportes_ciudadanos(station_key);
-- CREATE INDEX idx_reportes_timestamp ON reportes_ciudadanos(timestamp DESC);
-- CREATE INDEX idx_pred_station       ON predicciones(station_key);
-- CREATE INDEX idx_pred_timestamp     ON predicciones(timestamp DESC);
-- CREATE INDEX idx_feedback_station   ON feedback_usuarios(station_key);

-- Para verificar después de iniciar la API:
-- \dt                                          → lista tablas
-- SELECT * FROM reportes_ciudadanos LIMIT 10;  → ver reportes
-- SELECT * FROM predicciones LIMIT 10;         → ver predicciones guardadas
-- SELECT * FROM feedback_usuarios LIMIT 10;    → ver feedback
