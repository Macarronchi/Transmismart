"""
TransmiSmart — API Unificada v4.0 — Flask
Fusión de App_Transmilenio_Avance3 + transmismart_api_v3
Predicción de congestión: Calle 76, Calle 85 y Los Héroes (Transmilenio)
- Modelos ML (regressor + classifier) por estación
- Reportes ciudadanos + Feedback en PostgreSQL
- Predicciones guardadas en DB
- Recomendación de estación con menor congestión
- Servicios/rutas por estación
- Cache de predicciones
"""
import os, math, json, logging
from datetime import datetime, timedelta, date, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── App & DB ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app, resources={r"/*": {"origins": "*"}})

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:12345@localhost:5432/Transmilenio?client_encoding=utf8"
)

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ─── Modelos ORM ──────────────────────────────────────────────────────────────
class Prediccion(db.Model):
    """Registro de cada predicción generada."""
    __tablename__ = "predicciones"
    id                      = db.Column(db.Integer, primary_key=True)
    estacion                = db.Column(db.String(50), nullable=False)
    station_key             = db.Column(db.String(20), nullable=False)
    hora                    = db.Column(db.Integer)
    minuto                  = db.Column(db.Integer)
    dia_semana              = db.Column(db.Integer)
    es_festivo              = db.Column(db.Boolean, default=False)
    validaciones_predichas  = db.Column(db.Integer)
    congestion_predicha     = db.Column(db.String(10))
    confianza               = db.Column(db.Float)
    franja                  = db.Column(db.String(30))
    timestamp               = db.Column(db.DateTime, default=lambda: datetime.now(ZoneInfo("America/Bogota")).replace(tzinfo=None))


class ReporteCiudadano(db.Model):
    """Reporte ciudadano sobre el estado de congestión."""
    __tablename__ = "reportes_ciudadanos"
    id            = db.Column(db.Integer, primary_key=True)
    station_key   = db.Column(db.String(20), nullable=False)
    station_name  = db.Column(db.String(50), nullable=False)
    nivel_usuario = db.Column(db.String(10), nullable=False)   # BAJO/MEDIO/ALTO
    nivel_modelo  = db.Column(db.String(10))
    comentario    = db.Column(db.Text)
    coincide      = db.Column(db.Boolean)
    franja        = db.Column(db.String(30))
    dia_semana    = db.Column(db.String(15))
    timestamp     = db.Column(db.DateTime, default=lambda: datetime.now(ZoneInfo("America/Bogota")).replace(tzinfo=None))
    ip_origen     = db.Column(db.String(45))


class FeedbackUsuario(db.Model):
    """Feedback detallado de usuario para mejora del modelo."""
    __tablename__ = "feedback_usuarios"
    id                    = db.Column(db.Integer, primary_key=True)
    estacion              = db.Column(db.String(50))
    station_key           = db.Column(db.String(20))
    congestion_predicha   = db.Column(db.String(10))
    congestion_reportada  = db.Column(db.String(10))
    coincide              = db.Column(db.Boolean)
    comentario            = db.Column(db.Text)
    satisfaccion          = db.Column(db.Integer)   # 1-5
    fecha_hora            = db.Column(db.DateTime, default=lambda: datetime.now(ZoneInfo("America/Bogota")).replace(tzinfo=None))


class CachePrediccion(db.Model):
    __tablename__ = "cache_predicciones"
    id       = db.Column(db.Integer, primary_key=True)
    slot_key = db.Column(db.String(30), unique=True, nullable=False)
    payload  = db.Column(db.Text, nullable=False)
    creado   = db.Column(db.DateTime, default=lambda: datetime.now(ZoneInfo("America/Bogota")).replace(tzinfo=None))


# ─── Constantes ───────────────────────────────────────────────────────────────
FESTIVOS_CO = {
    date(2025, 7,20), date(2025, 8, 7), date(2025, 8,18), date(2025,10,13),
    date(2025,11, 3), date(2025,11,17), date(2025,12, 8), date(2025,12,25),
    date(2026, 1, 1), date(2026, 1,12), date(2026, 3,23), date(2026, 4, 2),
    date(2026, 4, 3), date(2026, 5, 1), date(2026, 5,18), date(2026, 6, 8),
    date(2026, 6,29), date(2026, 7,20), date(2026, 8, 7), date(2026, 8,17),
    date(2026,10,12), date(2026,11, 2), date(2026,11,16), date(2026,12, 8),
    date(2026,12,25),
}
CONGESTION_LABELS = ["BAJO", "MEDIO", "ALTO"]
NOMBRES  = {"calle76": "Calle 76", "calle85": "Calle 85", "heroes": "Los Héroes"}
COLORES  = {"calle76": "#3b82f6",  "calle85": "#f59e0b",   "heroes": "#ef4444"}
DIAS_ES  = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
BOGOTA_TZ = ZoneInfo("America/Bogota")

FEATURES_ALL = [
    "Hora","Minuto","HoraDecimal",
    "DiaSemana","EsFinSemana","EsHoraPico","EsPicoFuerte",
    "EsFestivo","EsPreFestivo","EsDiaLaboral",
    "Mes","DiaMes","SemanaAnio",
    "HoraSin","HoraCos","DiaSin","DiaCos","MesSin","MesCos",
    "MediaHistorica","MedianaHistorica","StdHistorica","P75Historico",
    "Lag_1","Lag_2","Lag_4","Lag_8","Lag_96",
    "RollingMean_4","RollingMean_8","RollingStd_4","RollingMax_4",
    "RollingMean_96","RatioLag1Hist",
]

RUTAS_POR_ESTACION = {
    "heroes": [
        {"ruta": "4", "destino": "Portal Sur"},
        {"ruta": "4", "destino": "Héroes"},
        {"ruta": "8", "destino": "Terminal"},
        {"ruta": "8", "destino": "Guatoque"},
        {"ruta": "B11", "destino": "Portal Norte / Autonorte"},
        {"ruta": "B75", "destino": "Portal Norte"},
        {"ruta": "G11", "destino": "Portal Sur"},
        {"ruta": "H75", "destino": "Portal Usme"},
        {"ruta": "B26", "destino": "Portal Norte"},
        {"ruta": "B27", "destino": "Portal Norte"},
        {"ruta": "B50", "destino": "Portal Norte"},
        {"ruta": "B55", "destino": "Portal Norte"},
        {"ruta": "B74", "destino": "Portal Norte"},
        {"ruta": "C50", "destino": "Portal Suba"},
        {"ruta": "D55", "destino": "Portal 80"},
        {"ruta": "F26", "destino": "Portal Américas"},
        {"ruta": "H27", "destino": "Portal Tunal"},
        {"ruta": "J27", "destino": "Museo del Oro"},
    ],
    "calle76": [
        {"ruta": "6", "destino": "Portal 80"},
        {"ruta": "6", "destino": "Universidades"},
        {"ruta": "8", "destino": "Terminal"},
        {"ruta": "8", "destino": "Guatoque"},
        {"ruta": "A60", "destino": "Calle 26 / Centro Internacional"},
        {"ruta": "B13", "destino": "Portal Norte"},
        {"ruta": "B18", "destino": "Terminal"},
        {"ruta": "B75", "destino": "Portal Norte"},
        {"ruta": "C15", "destino": "Portal Suba"},
        {"ruta": "D24", "destino": "Portal 80"},
        {"ruta": "F60", "destino": "Portal Américas"},
        {"ruta": "H15", "destino": "Portal Tunal"},
        {"ruta": "H75", "destino": "Portal Usme"},
        {"ruta": "J24", "destino": "Universidades"},
        {"ruta": "L18", "destino": "Portal 20 de Julio"},
        {"ruta": "A52", "destino": "Las Flores / Zona Neutra"},
        {"ruta": "G52", "destino": "Portal Sur"},
    ],
    "calle85": [
        {"ruta": "8", "destino": "Terminal"},
        {"ruta": "8", "destino": "Guatoque"},
        {"ruta": "B10", "destino": "Portal Norte"},
        {"ruta": "B11", "destino": "Portal Norte"},
        {"ruta": "B13", "destino": "Portal Norte"},
        {"ruta": "B23", "destino": "Alcalá / Portal Norte"},
        {"ruta": "D10", "destino": "Portal 80"},
        {"ruta": "G11", "destino": "Portal Sur"},
        {"ruta": "H13", "destino": "Portal Tunal"},
        {"ruta": "K23", "destino": "Portal ElDorado"},
        {"ruta": "B26", "destino": "Portal Norte"},
        {"ruta": "B27", "destino": "Portal Norte"},
        {"ruta": "B50", "destino": "Portal Norte"},
        {"ruta": "C50", "destino": "Portal Suba"},
        {"ruta": "F26", "destino": "Portal Américas"},
        {"ruta": "H27", "destino": "Portal Tunal"},
    ],
}

# ─── Carga de modelos ─────────────────────────────────────────────────────────
MODELOS     = {}
UMBRALES    = {}
HIST_LOOKUP = {}
METRICAS    = {}
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")


def _load_models():
    global MODELOS, UMBRALES, HIST_LOOKUP, METRICAS

    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")

    if not os.path.exists(meta_path):
        log.warning("model_metadata.json no encontrado — usando modo demo")
        _load_demo_models()
        return

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        UMBRALES = meta.get("umbrales", {})
        METRICAS = meta.get("metrics", {})

        for key, records in meta.get("hist_tables", {}).items():
            ht_df = pd.DataFrame(records)
            if not ht_df.empty:
                HIST_LOOKUP[key] = ht_df.set_index(["Hora", "Minuto", "DiaSemana"])
            else:
                HIST_LOOKUP[key] = pd.DataFrame()

        for key in ["calle76", "calle85", "heroes"]:
            reg_path = os.path.join(MODEL_DIR, f"{key}_regressor.pkl")
            clf_path = os.path.join(MODEL_DIR, f"{key}_classifier.pkl")

            if os.path.exists(reg_path) and os.path.exists(clf_path):
                try:
                    MODELOS[key] = {
                        "reg": joblib.load(reg_path),
                        "clf": joblib.load(clf_path),
                    }
                    log.info(f"Modelos cargados para {NOMBRES[key]}")
                except Exception as e:
                    log.error(f"No se pudieron cargar los modelos de {key}: {e}")
                    log.warning("Usando modelos DEMO para que la API pueda iniciar")
                    _load_demo_models()
                    return
            else:
                log.warning(f"Modelos faltantes para {key} — usando modo demo")
                _load_demo_models()
                return

    except Exception as e:
        log.error(f"Error cargando metadata/modelos: {e}")
        log.warning("Usando modelos DEMO")
        _load_demo_models()


def _load_demo_models():
    """Genera modelos demo con distribuciones realistas."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    np.random.seed(42)
    base_vals = {"calle76": 80, "calle85": 95, "heroes": 110}
    for key in ["calle76","calle85","heroes"]:
        bv = base_vals[key]
        N  = 5000
        hora = np.random.randint(0, 24, N)
        pico = ((hora >= 7) & (hora <= 9)) | ((hora >= 17) & (hora <= 19))
        val  = np.maximum(0, (bv + pico * 60 + np.random.normal(0, 20, N)).astype(int))
        p33, p66 = np.percentile(val, 33), np.percentile(val, 66)
        UMBRALES[key] = {"p33": float(p33), "p66": float(p66)}
        labels = np.where(val < p33, 0, np.where(val < p66, 1, 2))
        X_demo = np.column_stack([hora, np.zeros((N, len(FEATURES_ALL)-1))])
        reg      = HistGradientBoostingRegressor(max_iter=50, random_state=42)
        clf_base = RandomForestClassifier(n_estimators=30, random_state=42)
        reg.fit(X_demo, val)
        clf_base.fit(X_demo, labels)
        clf = CalibratedClassifierCV(clf_base, cv="prefit")
        clf.fit(X_demo, labels)
        MODELOS[key] = {"reg": reg, "clf": clf}
        rows = []
        for h in range(24):
            for mi in [0, 15, 30, 45]:
                for d in range(7):
                    pico_f = 1 if (6 <= h <= 9 or 16 <= h <= 20) else 0
                    m = bv + pico_f * 60 + (10 if d < 5 else -20)
                    rows.append({"Hora": h, "Minuto": mi, "DiaSemana": d,
                                  "media": m, "mediana": m, "std": 20.0, "p75": m+15})
        HIST_LOOKUP[key] = pd.DataFrame(rows).set_index(["Hora","Minuto","DiaSemana"])
        METRICAS[key] = {
            "nombre": NOMBRES[key], "mae": 0, "rmse": 0,
            "r2": 0, "accuracy": 0, "p33": float(p33), "p66": float(p66),
            "n_train": N, "n_test": 0,
        }
    log.info("Modelos DEMO cargados")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _franja(h: int) -> str:
    for lim, etq in [(6,"Madrugada"),(9,"Mañana pico"),(12,"Mañana media"),
                     (15,"Tarde temprana"),(18,"Tarde pico"),(22,"Noche")]:
        if h < lim:
            return etq
    return "Noche tardía"


def _get_hist(key, hora, minuto, dia):
    try:
        row = HIST_LOOKUP[key].loc[(hora, minuto, dia)]
        return float(row["media"]), float(row["mediana"]), float(row["std"]), float(row["p75"])
    except (KeyError, TypeError):
        for mi_alt in [0, 15, 30, 45]:
            try:
                row = HIST_LOOKUP[key].loc[(hora, mi_alt, dia)]
                return float(row["media"]), float(row["mediana"]), float(row["std"]), float(row["p75"])
            except KeyError:
                continue
        return 30.0, 25.0, 15.0, 40.0


def _extraer_features(dt, key, lag_1=None):
    h, mi = dt.hour, dt.minute
    hd    = h + mi / 60
    dia   = dt.weekday()
    mes   = dt.month
    fest  = int(dt.date() in FESTIVOS_CO)
    pre_f = int((dt + timedelta(1)).date() in FESTIVOS_CO)
    hm, hmed, hstd, hp75 = _get_hist(key, h, mi, dia)
    l1 = lag_1 if lag_1 is not None else hm
    row = {
        "Hora": h, "Minuto": mi, "HoraDecimal": hd,
        "DiaSemana": dia, "EsFinSemana": int(dia >= 5),
        "EsHoraPico": int(h in list(range(6,10)) + list(range(16,20))),
        "EsPicoFuerte": int(h in [7,8,17,18]),
        "EsFestivo": fest, "EsPreFestivo": pre_f,
        "EsDiaLaboral": int(dia < 5 and not fest),
        "Mes": mes, "DiaMes": dt.day,
        "SemanaAnio": int(dt.isocalendar()[1]),
        "HoraSin": math.sin(2*math.pi*hd/24), "HoraCos": math.cos(2*math.pi*hd/24),
        "DiaSin":  math.sin(2*math.pi*dia/7),  "DiaCos":  math.cos(2*math.pi*dia/7),
        "MesSin":  math.sin(2*math.pi*(mes-1)/12), "MesCos": math.cos(2*math.pi*(mes-1)/12),
        "MediaHistorica": hm, "MedianaHistorica": hmed,
        "StdHistorica": hstd, "P75Historico": hp75,
        "Lag_1": l1, "Lag_2": hm, "Lag_4": hm, "Lag_8": hm, "Lag_96": hm,
        "RollingMean_4": hm, "RollingMean_8": hm,
        "RollingStd_4": hstd, "RollingMax_4": hp75,
        "RollingMean_96": hm, "RatioLag1Hist": min(l1/max(hm,1.0), 5.0),
    }
    return pd.DataFrame([row])[FEATURES_ALL]


def _congestion_from_val(key, val):
    u = UMBRALES.get(key, {"p33": 40, "p66": 80})
    if val < u["p33"]: return "BAJO"
    if val < u["p66"]: return "MEDIO"
    return "ALTO"

def _porcentaje_congestion(key, val):
    """
    Convierte las validaciones estimadas en un porcentaje comparable de congestión.
    Usa el umbral alto de cada estación como referencia.
    """
    umbral_alto = UMBRALES.get(key, {}).get("p66", 80)
    referencia = max(float(umbral_alto) * 1.5, 1)

    pct = round((float(val) / referencia) * 100)
    return max(0, min(pct, 100))

def _cache_key(prefix, dt, extra=""):
    """
    Crea una clave única para cachear predicciones por bloque de 15 minutos.
    Ejemplo: now:202605242130
    """
    dt_slot = dt.replace(
        minute=(dt.minute // 15) * 15,
        second=0,
        microsecond=0,
    )
    extra_txt = f":{extra}" if extra else ""
    return f"{prefix}{extra_txt}:{dt_slot.strftime('%Y%m%d%H%M')}"


def _guardar_cache_prediccion(slot_key, payload):
    """
    Guarda o actualiza la respuesta JSON de predicción en cache_predicciones.
    """
    try:
        payload_txt = json.dumps(payload, ensure_ascii=False)

        cache = db.session.query(CachePrediccion).filter_by(slot_key=slot_key).first()

        if cache:
            cache.payload = payload_txt
            cache.creado = datetime.now(BOGOTA_TZ).replace(tzinfo=None)
        else:
            cache = CachePrediccion(
                slot_key=slot_key,
                payload=payload_txt,
            )
            db.session.add(cache)

        db.session.commit()

    except Exception as e:
        log.warning(f"No se pudo guardar cache de predicción: {e}")
        db.session.rollback()


def predecir_todas(dt, lags=None, save_to_db=False):
    lags = lags or {}
    resultados = {}

    for key in ["calle76", "calle85", "heroes"]:
        X = _extraer_features(dt, key, lags.get(key))

        if MODELOS.get(key):
            reg       = MODELOS[key]["reg"]
            clf       = MODELOS[key]["clf"]
            val       = max(0, int(round(float(reg.predict(X)[0]))))
            clf_idx   = int(clf.predict(X)[0])
            clf_proba = clf.predict_proba(X)[0]
            cong      = CONGESTION_LABELS[clf_idx]
        else:
            hm, _, _, _ = _get_hist(key, dt.hour, dt.minute, dt.weekday())
            val   = max(0, int(round(hm + np.random.normal(0, 10))))
            cong  = _congestion_from_val(key, val)
            clf_idx   = CONGESTION_LABELS.index(cong)
            clf_proba = [0.1, 0.1, 0.1]
            clf_proba[clf_idx] = 0.8

        pct_congestion = _porcentaje_congestion(key, val)

        resultados[key] = {
            "nombre": NOMBRES[key],
            "validaciones": val,
            "porcentaje_congestion": pct_congestion,
            "congestion": cong,
            "probabilidades": {
                CONGESTION_LABELS[i]: round(float(p), 3)
                for i, p in enumerate(clf_proba)
            },
            "confianza": round(float(clf_proba[clf_idx]), 3),
            "umbral_bajo": UMBRALES.get(key, {}).get("p33", 40),
            "umbral_alto": UMBRALES.get(key, {}).get("p66", 80),
            "color": COLORES[key],
        }

    orden = {"BAJO": 0, "MEDIO": 1, "ALTO": 2}

    # Recomendación correcta: estación con MENOR porcentaje de congestión
    mejor = min(resultados, key=lambda k: (
        resultados[k]["porcentaje_congestion"],
        orden[resultados[k]["congestion"]],
        resultados[k]["validaciones"],
    ))

    # Guardar predicciones en PostgreSQL
    if save_to_db:
        try:
            for key, est in resultados.items():
                registro = Prediccion(
                    estacion=NOMBRES[key],
                    station_key=key,
                    hora=dt.hour,
                    minuto=dt.minute,
                    dia_semana=dt.weekday(),
                    es_festivo=(dt.date() in FESTIVOS_CO),
                    validaciones_predichas=est["validaciones"],
                    congestion_predicha=est["congestion"],
                    confianza=est["confianza"],
                    franja=_franja(dt.hour),
                )
                db.session.add(registro)

            db.session.commit()

        except Exception as e:
            log.warning(f"No se pudo guardar predicción en DB: {e}")
            db.session.rollback()

    return {
        "timestamp": dt.strftime("%Y-%m-%dT%H:%M"),
        "dia_semana": DIAS_ES[dt.weekday()],
        "franja": _franja(dt.hour),
        "predicciones": resultados,
        "recomendacion": {
            "station_key": mejor,
            "station_name": NOMBRES[mejor],
            "congestion": resultados[mejor]["congestion"],
            "validaciones": resultados[mejor]["validaciones"],
            "porcentaje_congestion": resultados[mejor]["porcentaje_congestion"],
            "motivo": (
                f"Nivel '{resultados[mejor]['congestion']}' con "
                f"{resultados[mejor]['porcentaje_congestion']}% de congestión estimada."
            ),
        },
        "congestion_map": {
            k: v["congestion"]
            for k, v in resultados.items()
        },
    }

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return send_from_directory("static", "index.html")


@app.route("/health", methods=["GET"])
def health():
    modelos_ok = {k: bool(v) for k, v in MODELOS.items()}
    db_ok = False
    try:
        db.session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass
    return jsonify({
        "status":    "ok",
        "version":   "4.0",
        "modelos":   modelos_ok,
        "db":        db_ok,
        "timestamp": datetime.now(BOGOTA_TZ).isoformat(),
    })


@app.route("/predict/now", methods=["GET", "POST"])
def predict_now():
    body = request.get_json(silent=True) or {}
    lags = {
        "calle76": body.get("lag_calle76"),
        "calle85": body.get("lag_calle85"),
        "heroes":  body.get("lag_heroes"),
    }
    dt = datetime.now(BOGOTA_TZ).replace(tzinfo=None)
    resultado = predecir_todas(dt, lags, save_to_db=True)

    _guardar_cache_prediccion(
        _cache_key("now", dt),
        resultado
    )

    return jsonify(resultado)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    body = request.get_json(silent=True) or {}
    fecha_hora = body.get("fecha_hora")
    if fecha_hora:
        try:
            dt = datetime.fromisoformat(fecha_hora.replace("Z",""))
        except ValueError:
            return jsonify({"error": "Formato inválido. Usa ISO 8601."}), 400
    else:
        dt = datetime.now(BOGOTA_TZ).replace(tzinfo=None)
    lags = {
        "calle76": body.get("lag_calle76"),
        "calle85": body.get("lag_calle85"),
        "heroes":  body.get("lag_heroes"),
    }
    return jsonify(predecir_todas(dt, lags))


@app.route("/predict/range", methods=["POST"])
def predict_range():
    body  = request.get_json(silent=True) or {}
    horas = min(int(body.get("horas", 4)), 12)
    dt_base = datetime.now(BOGOTA_TZ).replace(tzinfo=None, second=0, microsecond=0)
    dt_base = dt_base.replace(minute=(dt_base.minute // 15) * 15)
    resultados = []
    for i in range(horas * 4):
        dt_i = dt_base + timedelta(minutes=15*i)
        r = predecir_todas(dt_i)
        resultados.append({
            "timestamp":      r["timestamp"],
            "franja":         r["franja"],
            "calle76":        r["predicciones"]["calle76"]["validaciones"],
            "calle85":        r["predicciones"]["calle85"]["validaciones"],
            "heroes":         r["predicciones"]["heroes"]["validaciones"],
            "cong_calle76":   r["predicciones"]["calle76"]["congestion"],
            "cong_calle85":   r["predicciones"]["calle85"]["congestion"],
            "cong_heroes":    r["predicciones"]["heroes"]["congestion"],
            "recomendacion":  r["recomendacion"]["station_name"],
        })
    payload = {
        "horas": horas,
        "intervalos": resultados,
    }

    _guardar_cache_prediccion(
        _cache_key("range", dt_base, str(horas)),
        payload
    )

    return jsonify(payload)


@app.route("/report", methods=["POST"])
def report():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Body JSON requerido"}), 400
    station_key   = body.get("station_key")
    nivel_usuario = body.get("nivel_usuario", "").upper()
    if station_key not in NOMBRES:
        return jsonify({"error": f"station_key inválido. Usa: {list(NOMBRES.keys())}"}), 400
    if nivel_usuario not in CONGESTION_LABELS:
        return jsonify({"error": f"nivel_usuario inválido. Usa: {CONGESTION_LABELS}"}), 400
    dt_now = datetime.now(BOGOTA_TZ).replace(tzinfo=None)
    pred   = predecir_todas(dt_now)
    nivel_modelo = pred["predicciones"][station_key]["congestion"]
    coincide     = nivel_usuario == nivel_modelo
    reporte = ReporteCiudadano(
        station_key   = station_key,
        station_name  = NOMBRES[station_key],
        nivel_usuario = nivel_usuario,
        nivel_modelo  = nivel_modelo,
        comentario    = body.get("comentario", "")[:500],
        coincide      = coincide,
        franja        = _franja(dt_now.hour),
        dia_semana    = DIAS_ES[dt_now.weekday()],
        ip_origen     = request.remote_addr,
    )
    feedback = FeedbackUsuario(
        estacion=NOMBRES[station_key],
        station_key=station_key,
        congestion_predicha=nivel_modelo,
        congestion_reportada=nivel_usuario,
        coincide=coincide,
        comentario=body.get("comentario", "")[:500],
        satisfaccion=body.get("satisfaccion"),
    )

    db.session.add(reporte)
    db.session.add(feedback)
    db.session.commit()

    return jsonify({
        "ok": True,
        "id": reporte.id,
        "feedback_id": feedback.id,
        "station": NOMBRES[station_key],
        "nivel_usuario": nivel_usuario,
        "nivel_modelo": nivel_modelo,
        "coincide": coincide,
        "mensaje": ("Tu reporte coincide con el modelo." if coincide
                    else f"El modelo predijo {nivel_modelo}, tú reportaste {nivel_usuario}. "
                         "Tu reporte ayuda a mejorar el sistema."),
    })


@app.route("/report/stats", methods=["GET"])
def report_stats():
    try:
        # Escala numérica para convertir BAJO/MEDIO/ALTO en porcentaje
        nivel_score = {
            "BAJO": 25,
            "MEDIO": 60,
            "ALTO": 90,
        }

        peso_ia = 0.60
        peso_ciudadano = 0.40

        total = db.session.query(ReporteCiudadano).count()
        coincide = db.session.query(ReporteCiudadano).filter_by(coincide=True).count()

        # Predicción actual de la IA
        dt_now = datetime.now(BOGOTA_TZ).replace(tzinfo=None)
        pred_actual = predecir_todas(dt_now)

        by_station = {}
        valores_mixtos = []

        for key in ["calle76", "calle85", "heroes"]:
            # Reportes ciudadanos de esta estación
            reportes = (
                db.session.query(ReporteCiudadano)
                .filter_by(station_key=key)
                .order_by(ReporteCiudadano.timestamp.desc())
                .limit(20)
                .all()
            )

            t = len(reportes)
            c = sum(1 for r in reportes if r.coincide)

            dist = {}
            for nivel in CONGESTION_LABELS:
                dist[nivel] = sum(1 for r in reportes if r.nivel_usuario == nivel)

            # Score IA
            nivel_ia = pred_actual["predicciones"][key]["congestion"]
            score_ia = nivel_score.get(nivel_ia, 60)

            # Score ciudadano
            if t > 0:
                score_ciudadano = round(
                    sum(nivel_score.get(r.nivel_usuario, 60) for r in reportes) / t,
                    1
                )
                congestion_mixta = round(
                    (score_ia * peso_ia) + (score_ciudadano * peso_ciudadano),
                    1
                )
            else:
                score_ciudadano = None
                congestion_mixta = score_ia

            valores_mixtos.append(congestion_mixta)

            by_station[key] = {
                "nombre": NOMBRES[key],
                "total": t,
                "coincide": c,
                "pct_acierto": round(c / t * 100, 1) if t > 0 else 0,

                # Nuevos campos importantes
                "nivel_ia": nivel_ia,
                "score_ia": score_ia,
                "score_ciudadano": score_ciudadano,
                "congestion_mixta": congestion_mixta,

                "distribucion": dist,
            }

        congestion_mixta_global = round(
            sum(valores_mixtos) / len(valores_mixtos),
            1
        ) if valores_mixtos else 0

        ultimos = (
            db.session.query(ReporteCiudadano)
            .order_by(ReporteCiudadano.timestamp.desc())
            .limit(10)
            .all()
        )

        return jsonify({
            "total": total,
            "coincide": coincide,

            # Esto queda como precisión, por si luego lo quieres usar
            "pct_global": round(coincide / total * 100, 1) if total > 0 else 0,

            # Este es el nuevo porcentaje correcto para mostrar
            "congestion_mixta_global": congestion_mixta_global,
            "formula": "60% IA + 40% reportes ciudadanos",

            "by_station": by_station,
            "ultimos": [{
                "id": r.id,
                "station": r.station_name,
                "nivel_usuario": r.nivel_usuario,
                "nivel_modelo": r.nivel_modelo,
                "coincide": r.coincide,
                "franja": r.franja,
                "comentario": r.comentario,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            } for r in ultimos],
        })

    except Exception as e:
        log.error(f"Error en /report/stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def guardar_feedback():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Body JSON requerido"}), 400
    station_key = body.get("station_key", "")
    registro = FeedbackUsuario(
        estacion              = NOMBRES.get(station_key, body.get("estacion", "")),
        station_key           = station_key,
        congestion_predicha   = body.get("congestion_predicha", ""),
        congestion_reportada  = body.get("congestion_reportada", ""),
        coincide              = body.get("congestion_predicha") == body.get("congestion_reportada"),
        comentario            = body.get("comentario", "")[:500],
        satisfaccion          = body.get("satisfaccion"),
    )
    db.session.add(registro)
    db.session.commit()
    return jsonify({"ok": True, "id": registro.id, "mensaje": "Feedback guardado. ¡Gracias!"})


@app.route("/feedback", methods=["GET"])
def obtener_feedback():
    registros = (db.session.query(FeedbackUsuario)
                 .order_by(FeedbackUsuario.fecha_hora.desc())
                 .limit(20).all())
    return jsonify([{
        "estacion":             r.estacion,
        "congestion_predicha":  r.congestion_predicha,
        "congestion_reportada": r.congestion_reportada,
        "coincide":             r.coincide,
        "comentario":           r.comentario,
        "satisfaccion":         r.satisfaccion,
        "fecha":                r.fecha_hora.isoformat() if r.fecha_hora else None,
    } for r in registros])


@app.route("/servicios/<station_key>", methods=["GET"])
def servicios(station_key):
    if station_key not in NOMBRES:
        return jsonify({"error": "Estación no encontrada"}), 404
    dt_now = datetime.now(BOGOTA_TZ).replace(tzinfo=None)
    rutas  = list(RUTAS_POR_ESTACION.get(station_key, []))
    franja = _franja(dt_now.hour)
    freq   = 4 if "pico" in franja.lower() else 8
    for i, r in enumerate(rutas):
        r = dict(r)
        r["tiempo_min"]    = (i * freq + np.random.randint(1, freq)) % 20 + 1
        r["frecuencia_min"] = freq
        rutas[i] = r
    return jsonify({"station_key": station_key, "nombre": NOMBRES[station_key], "rutas": rutas})


@app.route("/estaciones", methods=["GET"])
def estaciones():
    return jsonify({
        key: {
            "nombre":       NOMBRES[key],
            "color":        COLORES[key],
            "umbral_bajo":  UMBRALES.get(key, {}).get("p33", 40),
            "umbral_alto":  UMBRALES.get(key, {}).get("p66", 80),
            "metricas":     METRICAS.get(key, {}),
        }
        for key in ["calle76","calle85","heroes"]
    })


@app.route("/umbrales", methods=["GET"])
def umbrales():
    return jsonify(UMBRALES)


@app.route("/predicciones/historial", methods=["GET"])
def historial_predicciones():
    """Últimas predicciones guardadas en DB."""
    limit  = min(int(request.args.get("limit", 50)), 200)
    key    = request.args.get("station_key")
    query  = db.session.query(Prediccion).order_by(Prediccion.timestamp.desc())
    if key:
        query = query.filter_by(station_key=key)
    registros = query.limit(limit).all()
    return jsonify([{
        "id":          r.id,
        "estacion":    r.estacion,
        "hour":        r.hora,
        "congestion":  r.congestion_predicha,
        "validaciones": r.validaciones_predichas,
        "confianza":   r.confianza,
        "timestamp":   r.timestamp.isoformat() if r.timestamp else None,
    } for r in registros])

@app.route("/cache/historial", methods=["GET"])
def historial_cache():
    """Últimos registros guardados en cache_predicciones."""
    limit = min(int(request.args.get("limit", 20)), 100)

    registros = (
        db.session.query(CachePrediccion)
        .order_by(CachePrediccion.creado.desc())
        .limit(limit)
        .all()
    )

    return jsonify([
        {
            "id": r.id,
            "slot_key": r.slot_key,
            "payload": json.loads(r.payload),
            "creado": r.creado.isoformat() if r.creado else None,
        }
        for r in registros
    ])

# ─── Inicialización ───────────────────────────────────────────────────────────
def init_app():
    with app.app_context():
        try:
            db.create_all()
            log.info("Tablas PostgreSQL creadas/verificadas")
        except Exception as e:
            log.error(f"Error con DB: {e} — continuando sin persistencia")
        _load_models()


init_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG","0") == "1")
