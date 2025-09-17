# ---------------------------------------------------------------
# IAQ Dashboard – courbes brutes, colonnes numériques dynamiques
#   • Export des graphiques en PDF/PNG/SVG/JPEG
#   • Sauvegarde / chargement des paramètres (JSON)
# ---------------------------------------------------------------
import webbrowser, threading

def open_browser():
    webbrowser.open_new("http://localhost:8501")

threading.Timer(1, open_browser).start()


import io, json
from datetime import date
import streamlit as st
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dateutil import parser
import pandas as pd
import plotly.io as pio

st.set_page_config(page_title="Ecozimut", layout="wide")

# ─────────────────── Fonctions utilitaires ────────────────────
def fallback_parse(ts: str):
    """Essaie de parser n’importe quel timestamp texte → datetime."""
    try:
        return parser.parse(ts, dayfirst=True)
    except Exception:
        return None

def load_file(up) -> pl.DataFrame | None:
    """Lit un fichier CSV/XLSX, renvoie un DF avec datetime, sensor, + colonnes numériques (toujours en JJ/MM/AAAA)."""
    raw = up.read()
    is_xls = up.name.lower().endswith((".xls", ".xlsx"))
    sep = ";" if (not is_xls and b";" in raw.splitlines()[0]) else ","

    try:
        if is_xls:
            # Lire toutes les colonnes en texte
            pdf = pd.read_excel(io.BytesIO(raw), sheet_name=0, dtype=str, engine="openpyxl")
        else:
            pdf = pd.read_csv(io.BytesIO(raw), sep=sep, encoding="latin1", dtype=str)
        df = pl.from_pandas(pdf)
    except Exception as e:
        st.error(f"Erreur lors de la lecture de {up.name} : {e}")
        return None

    # Normaliser les noms de colonnes
    df = df.rename({col: col.strip().lower().replace(" ", "") for col in df.columns})
    cols = set(df.columns)

    # Si colonnes 'date' et 'h' existent, les combiner
    if {"date", "h"}.issubset(cols):
        df = df.with_columns((pl.col("date").cast(pl.Utf8).str.strip() + " " +
                              pl.col("h").cast(pl.Utf8).str.strip()).alias("datetime"))
    elif "date" in cols:
        df = df.rename({"date": "datetime"})
    else:
        st.error(f"{up.name} → colonnes 'date' (et optionnel 'h') introuvables.")
        return None

    # Parser robustement en JJ/MM/AAAA HH:MM avec dayfirst=True
    def parse_fr(ts: str):
        try:
            return parser.parse(ts, dayfirst=True)
        except Exception:
            return None

    df = df.with_columns(
        pl.col("datetime").map_elements(parse_fr, return_dtype=pl.Datetime)
    ).drop_nulls("datetime")

    # Colonnes numériques
    reserved = {"datetime", "date", "h"}
    numeric_cols = []
    for c in df.columns:
        if c in reserved:
            continue
        try:
            df = df.with_columns(
                pl.col(c).cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False)
            )
            if df[c].null_count() < df.height:
                numeric_cols.append(c)
        except Exception:
            continue

    if not numeric_cols:
        st.error(f"{up.name} → aucune colonne numérique détectée.")
        return None

    return df.select(["datetime"] + numeric_cols).with_columns(
        pl.lit(up.name).alias("sensor")
    )



def detect_outliers_iqr(df: pl.DataFrame, column: str) -> pl.DataFrame:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df.with_columns(
        pl.col(column).apply(lambda x: x < lower_bound or x > upper_bound).alias(f"is_outlier_{column}")
    )

def display_descriptive_statistics(df: pl.DataFrame, columns: list[str]):
    if not columns:
        st.info("Sélectionnez des variables pour afficher les statistiques descriptives.")
        return
    st.subheader("Statistiques descriptives")
    stats_data = []
    for col in columns:
        if col in df.columns and df[col].dtype.is_numeric():
            series = df[col]
            stats = {
                "Variable": col,
                "Count": series.count(),
                "Mean": series.mean(),
                "Std Dev": series.std(),
                "Min": series.min(),
                "25%": series.quantile(0.25),
                "50% (Median)": series.median(),
                "75%": series.quantile(0.75),
                "Max": series.max()
            }
            outlier_col = f"is_outlier_{col}"
            if outlier_col in df.columns:
                non_outliers = df.filter(pl.col(outlier_col) == False)
                stats["Mean (hors outliers)"] = non_outliers[col].mean()
            else:
                stats["Mean (hors outliers)"] = "—"
            stats_data.append(stats)
    if stats_data:
        st.dataframe(pl.DataFrame(stats_data))
    else:
        st.info("Aucune donnée numérique valide pour les statistiques descriptives.")

def display_correlation_matrix(df: pl.DataFrame, columns: list[str]):
    if not columns or len(columns) < 2:
        st.info("Sélectionnez au moins deux variables numériques pour afficher la matrice de corrélation.")
        return
    st.subheader("Matrice de corrélation")
    numeric_df = df.select([
            col for col in columns
            if df[col].dtype.is_numeric() and df[col].null_count() < df.height
        ])
    if numeric_df.width < 2:
        st.info("Aucune paire de variables numériques valide pour la corrélation.")
        return
    correlation_matrix = numeric_df.corr()
    st.dataframe(correlation_matrix)

# ─────────────────── Interface Streamlit ────────────────────
st.title("📈 Ecozimut Dashboard : Mesures des sondes")

uploads = st.file_uploader(
    "Déposez vos CSV / XLSX (date [+ h] + colonnes numériques) :",
    type=["csv", "xlsx", "xls"], accept_multiple_files=True)

if not uploads:
    st.stop()

frames = [d for f in uploads if (d := load_file(f)) is not None]
if not frames:
    st.stop()

all_numeric = set()
for df in frames:
    all_numeric |= set(df.columns) - {"datetime", "sensor"}
all_numeric = sorted(all_numeric)

aligned = []
for df in frames:
    miss = [c for c in all_numeric if c not in df.columns]
    if miss:
        df = df.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in miss])
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in all_numeric])
    aligned.append(df.select(["datetime", "sensor"] + all_numeric))

data = pl.concat(aligned).sort("datetime") 

data = pl.concat(aligned).sort("datetime")

min_date = data["datetime"].min()
max_date = data["datetime"].max()
sensors = data["sensor"].unique().to_list()

# ───── Paramètres (charger / sauvegarder) ─────
_default_date_range = (min_date.date(), max_date.date())
_default_sel_sensors = sensors
_default_vars = []

with st.sidebar.expander("⚙️ Paramètres (charger / sauvegarder)", expanded=False):
    cfg_file = st.file_uploader("📂 Charger paramètres (.json)", type=["json"], key="cfg_upload")
    if cfg_file is not None:
        try:
            cfg = json.load(cfg_file)
            st.session_state.color_map = cfg.get("color_map", {})
            st.session_state.thresholds_default = cfg.get("thresholds", {})
            st.session_state.y_ranges_default = cfg.get("y_ranges", {})
            st.session_state.vars_default = cfg.get("vars_", _default_vars)
            st.session_state.sel_sensors_default = [
                s for s in cfg.get("sel_sensors", _default_sel_sensors) if s in sensors
            ]
            if "date_range" in cfg:
                try:
                    d0 = pd.to_datetime(cfg["date_range"][0]).date()
                    d1 = pd.to_datetime(cfg["date_range"][1]).date()
                    st.session_state.date_range_default = (d0, d1)
                except Exception:
                    st.session_state.date_range_default = _default_date_range
            st.success("Paramètres chargés ✔️")
        except Exception as e:
            st.error(f"Impossible de lire le JSON : {e}")
    st.caption("Après configuration, cliquez pour sauvegarder.")
    save_params_clicked = st.button("💾 Sauvegarder en JSON", key="save_params_btn")

# Widgets
st.sidebar.markdown("### 📅 Filtrer par date")
date_range = st.sidebar.date_input(
    "Plage de dates :",
    value=st.session_state.get("date_range_default", (min_date.date(), max_date.date())),
    min_value=min_date.date(),
    max_value=max_date.date(),
    format="DD/MM/YYYY"
)

# Filtrer les données selon la plage de dates choisie
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    data = data.filter(
        (pl.col("datetime") >= start_date) & (pl.col("datetime") <= end_date)
    )


sel_sensors = st.sidebar.multiselect(
    "Sondes :",
    sensors,
    default=st.session_state.get("sel_sensors_default", sensors)
)

vars_ = st.sidebar.multiselect(
    "Variables :",
    all_numeric,
    default=st.session_state.get("vars_default", [])
)
if len(vars_) > 2:
    st.sidebar.warning("⚠️ Max 2 variables simultanées.")
    vars_ = vars_[:2]

plot_df = data.filter(pl.col("sensor").is_in(sel_sensors))

# Analyses
with st.expander("📊 Analyses des données", expanded=True):
    if st.checkbox("Activer la détection des valeurs aberrantes (IQR)"):
        outliers_df = pl.DataFrame()
        for var in vars_:
            plot_df = detect_outliers_iqr(plot_df, var)
            outliers_for_var = plot_df.filter(pl.col(f"is_outlier_{var}") == True)
            if not outliers_for_var.is_empty():
                outliers_df = (
                    pl.concat([outliers_df, outliers_for_var], how="diagonal")
                    if outliers_df.height > 0 else outliers_for_var
                )
        if not outliers_df.is_empty():
            st.subheader("Valeurs aberrantes détectées")
            st.dataframe(outliers_df)
        else:
            st.info("Aucune valeur aberrante.")
    display_descriptive_statistics(plot_df, vars_)
    display_correlation_matrix(plot_df, vars_)

# Échelles
y_ranges = {}
with st.sidebar.expander("🔧 Ajuster les échelles", expanded=False):
    y_defaults = st.session_state.get("y_ranges_default", {})
    for v in vars_:
        col1, col2 = st.columns(2)
        v_min_auto = float(plot_df[v].min()) if v in plot_df.columns else 0.0
        v_max_auto = float(plot_df[v].max()) if v in plot_df.columns else 1.0
        saved_range = y_defaults.get(v, None)
        with col1:
            min_val = st.number_input(f"{v} min",
                                      value=saved_range[0] if saved_range else v_min_auto,
                                      key=f"{v}_min")
        with col2:
            max_val = st.number_input(f"{v} max",
                                      value=saved_range[1] if saved_range else v_max_auto,
                                      key=f"{v}_max")
        y_ranges[v] = None if min_val >= max_val else [min_val, max_val]

# Couleurs
if "color_map" not in st.session_state:
    st.session_state.color_map = {}
default_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                   "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                   "#17becf", "#bcbd22"]
with st.sidebar.expander("🎨 Couleurs", expanded=False):
    idx = 0
    for v in vars_:
        for s in sel_sensors:
            key = f"{s} – {v}"
            st.session_state.color_map[key] = st.color_picker(
                key,
                value=st.session_state.color_map.get(
                    key, default_palette[idx % len(default_palette)]
                )
            )
            idx += 1

# Seuils
thresholds = {}
with st.sidebar.expander("🚨 Alertes et Seuils", expanded=False):
    thr_defaults = st.session_state.get("thresholds_default", {})
    for v in vars_:
        st.write(f"**{v}**")
        col1, col2 = st.columns(2)
        saved_thr = thr_defaults.get(v, {})
        with col1:
            min_threshold = st.number_input(f"Seuil min {v}",
                                            key=f"threshold_min_{v}",
                                            value=saved_thr.get("min", float(plot_df[v].min())))
        with col2:
            max_threshold = st.number_input(f"Seuil max {v}",
                                            key=f"threshold_max_{v}",
                                            value=saved_thr.get("max", float(plot_df[v].max())))
        thresholds[v] = {"min": min_threshold, "max": max_threshold}

# Graphique
fig = make_subplots(specs=[[{"secondary_y": len(vars_) == 2}]])
for idx, v in enumerate(vars_):
    secondary = (idx == 1 and len(vars_) == 2)
    for s in sel_sensors:
        sub = plot_df.filter(pl.col("sensor") == s)
        if sub[v].null_count() == sub.height:
            continue
        trace_name = f"{s} – {v}"

        # Courbe principale
        fig.add_trace(
            go.Scattergl(
                x=sub["datetime"], y=sub[v], name=trace_name, mode="lines",
                line=dict(color=st.session_state.color_map[trace_name])
            ),
            secondary_y=secondary
        )

        # Points aberrants si dispo
        outlier_col = f"is_outlier_{v}"
        if outlier_col in sub.columns:
            outliers_sub = sub.filter(pl.col(outlier_col) == True)
            if not outliers_sub.is_empty():
                fig.add_trace(
                    go.Scattergl(
                        x=outliers_sub["datetime"],
                        y=outliers_sub[v],
                        name=f"Outliers {trace_name}",
                        mode="markers",
                        marker=dict(color="red", size=8, symbol="x")
                    ),
                    secondary_y=secondary
                )

        # Seuils min/max
        min_thr = thresholds[v]["min"]
        max_thr = thresholds[v]["max"]
        if min_thr is not None:
            fig.add_hline(y=min_thr, line_dash="dot", line_color="orange",
                          annotation_text=f"Seuil min {v}")
        if max_thr is not None:
            fig.add_hline(y=max_thr, line_dash="dot", line_color="purple",
                          annotation_text=f"Seuil max {v}")

# Appliquer les ranges
for idx, v in enumerate(vars_):
    rng = y_ranges.get(v)
    if rng:
        fig.update_yaxes(range=rng, secondary_y=(idx == 1))

# Layout interactif
fig.update_layout(
    title="Courbes brutes",
    height=600,
    legend=dict(
        orientation="v",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)


# Export graphique
with st.expander("⬇️ Exporter le graphique", expanded=False):
    fmt = st.selectbox("Format", ["pdf", "png", "svg", "jpeg"], index=0)
    scale = st.slider("Échelle (résolution)", min_value=1, max_value=4, value=2)
    try:
        # Forcer la légende horizontale en haut pour l’export
        fig.update_layout(
            legend=dict(
                orientation="h",     # horizontale
                yanchor="bottom",
                y=1.1,               # au-dessus du graphe
                xanchor="center",
                x=0.5
            )
        )
        fig_bytes = fig.to_image(format=fmt, scale=scale)
        file_name = f"graph_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.{fmt}"
        mime = {"pdf": "application/pdf", "png": "image/png",
                "svg": "image/svg+xml", "jpeg": "image/jpeg"}[fmt]
        st.download_button("Télécharger le graphique", data=fig_bytes,
                           file_name=file_name, mime=mime)
    except Exception as e:
        st.error(f"Export impossible : {e}")

# Sauvegarde paramètres
if save_params_clicked:
    config = {
        "color_map": st.session_state.get("color_map", {}),
        "thresholds": thresholds,
        "y_ranges": y_ranges,
        "vars_": vars_,
        "sel_sensors": sel_sensors,
        "date_range": [str(date_range[0]), str(date_range[1])]
    }
    cfg_bytes = json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8")
    st.sidebar.download_button("💾 Télécharger paramètres (params.json)",
                               data=cfg_bytes, file_name="params.json",
                               mime="application/json")

# Export CSV brut
st.download_button("Télécharger CSV concaténé",
                   data=data.write_csv(), file_name="mesures_brutes.csv")
