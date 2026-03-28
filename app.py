import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
import io
import re
import copy
from datetime import datetime
from scipy import stats

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DataStudio",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.block-container { padding-top: 3.5rem; padding-bottom: 2rem; }

.stTabs { margin-top: 0.5rem; }
[data-testid="stTabBar"] { padding-top: 0.25rem; }

.metric-card {
    background: #f8f9fb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}
.metric-card h4 { margin: 0 0 4px 0; font-size: 0.78rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card p  { margin: 0; font-size: 1.5rem; font-weight: 700; color: #111827; }

.section-header {
    font-size: 0.92rem; font-weight: 600; color: #111827;
    border-left: 3px solid #2563eb; padding-left: 0.6rem;
    margin: 0.5rem 0 0.3rem 0;
}

.before-after-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
}
.before-after-box h5 {
    margin: 0 0 0.5rem 0;
    font-size: 0.85rem;
    font-weight: 600;
    color: #0369a1;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

[data-testid="stVerticalBlock"] .stSelectbox,
[data-testid="stVerticalBlock"] .stSlider,
[data-testid="stVerticalBlock"] .stTextInput,
[data-testid="stVerticalBlock"] .stMultiSelect {
    margin-bottom: 0 !important;
}
.recipe-step {
    background: #f0f4ff; border-left: 3px solid #2563eb;
    border-radius: 6px; padding: 0.5rem 0.8rem;
    margin-bottom: 0.35rem; font-size: 0.83rem; font-family: 'DM Mono', monospace;
}
.ai-step { background: #fdf4ff; border-left: 3px solid #9333ea; }
.disclaimer {
    background: #fffbeb; border: 1px solid #fbbf24;
    border-radius: 8px; padding: 0.6rem 0.9rem;
    font-size: 0.82rem; color: #92400e; margin-bottom: 0.8rem;
}
.violation-badge {
    background: #fee2e2; color: #991b1b;
    border-radius: 4px; padding: 2px 8px; font-size: 0.78rem; font-weight: 600;
}

[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] [data-testid="glideDataEditor"] .header-cell,
.stDataFrame thead th {
    text-align: left !important;
    justify-content: flex-start !important;
}
[data-testid="stDataFrame"] td,
.stDataFrame tbody td {
    text-align: left !important;
}
[data-testid="stDataFrame"] [class*="dvn-"] .cell-wrap--number,
[data-testid="stDataFrame"] .gdg-cell[data-type="number"] {
    text-align: right !important;
    justify-content: flex-end !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "df_original": None,
        "df_working": None,
        "df_history": [],
        "recipe": [],
        "ai_enabled": False,
        "openai_key": "",
        "upload_key": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def log_step(operation: str, parameters: dict, affected_columns: list, source: str = "manual"):
    st.session_state.recipe.append({
        "step": len(st.session_state.recipe) + 1,
        "operation": operation,
        "parameters": parameters,
        "affected_columns": affected_columns,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
    })

def push_history():
    st.session_state.df_history.append(st.session_state.df_working.copy())

def reset_session():
    st.session_state.df_original = None
    st.session_state.df_working = None
    st.session_state.df_history = []
    st.session_state.recipe = []
    st.session_state.upload_key += 1
    st.session_state.pop("last_export_fig", None)
    st.session_state.pop("last_chart_fig", None)
    st.session_state.pop("last_chart_is_plotly", None)
    st.session_state.pop("ai_chat_history", None)
    st.session_state.pop("ai_pending_suggestions", None)

def reset_all():
    if st.session_state.df_original is not None:
        st.session_state.df_working = st.session_state.df_original.copy()
    st.session_state.df_history = []
    st.session_state.recipe = []
    st.session_state.pop("last_export_fig", None)
    st.session_state.pop("ai_chat_history", None)
    st.session_state.pop("ai_pending_suggestions", None)

def undo_last():
    if st.session_state.df_history:
        st.session_state.df_working = st.session_state.df_history.pop()
        if st.session_state.recipe:
            st.session_state.recipe.pop()

def safe_run(func, *args, **kwargs):
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, str(e)

def show_table(df, **kwargs):
    col_cfg = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_cfg[col] = st.column_config.NumberColumn(col, format="%.4g")
        else:
            col_cfg[col] = st.column_config.TextColumn(col)
    st.dataframe(df, column_config=col_cfg, hide_index=True,
                 use_container_width=True, **kwargs)

def numeric_cols(df):
    return df.select_dtypes(include="number").columns.tolist()

def categorical_cols(df):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()

def datetime_cols(df):
    return df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()


# ─────────────────────────────────────────────
# BEFORE / AFTER SUMMARY HELPERS
# ─────────────────────────────────────────────
def show_before_after_metrics(rows_before, rows_after, cols_before, cols_after,
                               extra_before=None, extra_after=None, extra_label=None):
    """
    Renders a clean before/after summary table.
    extra_before/after: optional scalar (e.g. missing count, outlier count).
    extra_label: label for the extra metric row.
    """
    st.markdown('<div class="before-after-box">', unsafe_allow_html=True)
    st.markdown('<h5>📊 Before / After Summary</h5>', unsafe_allow_html=True)

    rows_data = {
        "Metric": ["Rows", "Columns"],
        "Before": [f"{rows_before:,}", f"{cols_before:,}"],
        "After":  [f"{rows_after:,}",  f"{cols_after:,}"],
        "Change": [
            _change_str(rows_before, rows_after),
            _change_str(cols_before, cols_after),
        ],
    }
    if extra_label is not None and extra_before is not None and extra_after is not None:
        rows_data["Metric"].append(extra_label)
        rows_data["Before"].append(str(extra_before))
        rows_data["After"].append(str(extra_after))
        rows_data["Change"].append(_change_str(extra_before, extra_after) if isinstance(extra_before, (int, float)) else "—")

    summary_df = pd.DataFrame(rows_data)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def _change_str(before, after):
    """Return a coloured delta string."""
    try:
        delta = int(after) - int(before)
        if delta < 0:
            return f"▼ {abs(delta)}"
        elif delta > 0:
            return f"▲ {delta}"
        return "—"
    except Exception:
        return "—"


def show_column_changes(df_before, df_after):
    """Side-by-side column diff — used by Column Operations."""
    st.markdown('<div class="before-after-box">', unsafe_allow_html=True)
    st.markdown('<h5>📊 Column Changes</h5>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Before")
        st.dataframe(pd.DataFrame({"Columns Before": df_before.columns.tolist()}),
                     hide_index=True, use_container_width=True)
    with c2:
        st.caption("After")
        st.dataframe(pd.DataFrame({"Columns After": df_after.columns.tolist()}),
                     hide_index=True, use_container_width=True)

    before_set = set(df_before.columns)
    after_set  = set(df_after.columns)
    added   = after_set - before_set
    removed = before_set - after_set

    if added:
        st.success(f"✅ Added: {sorted(added)}")
    if removed:
        st.error(f"❌ Removed: {sorted(removed)}")
    if not added and not removed:
        st.info("No columns were added or removed — only names or values changed.")
    st.markdown('</div>', unsafe_allow_html=True)


def show_dtype_summary(col, dtype_before, dtype_after,
                        nonnull_before, nonnull_after,
                        null_before, null_after):
    """Before/after for data type conversions."""
    st.markdown('<div class="before-after-box">', unsafe_allow_html=True)
    st.markdown('<h5>📊 Before / After Summary</h5>', unsafe_allow_html=True)

    summary_df = pd.DataFrame({
        "Metric":  ["Data Type", "Non-null Values", "Missing Values"],
        "Before":  [dtype_before, f"{nonnull_before:,}", f"{null_before:,}"],
        "After":   [dtype_after,  f"{nonnull_after:,}",  f"{null_after:,}"],
        "Change":  [
            "—" if dtype_before == dtype_after else f"{dtype_before} → {dtype_after}",
            _change_str(nonnull_before, nonnull_after),
            _change_str(null_before, null_after),
        ],
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    coerced = null_after - null_before
    if coerced > 0:
        st.warning(f"⚠️ {coerced} value(s) could not be converted and became NaN/NaT.")
    st.markdown('</div>', unsafe_allow_html=True)


def show_categorical_summary(col, action, unique_before, unique_after,
                              top_before: pd.Series, top_after: pd.Series | None,
                              new_dummy_cols: list | None = None):
    """Before/after for categorical operations."""
    st.markdown('<div class="before-after-box">', unsafe_allow_html=True)
    st.markdown('<h5>📊 Before / After Summary</h5>', unsafe_allow_html=True)
    st.write(f"**Column:** `{col}` &nbsp;|&nbsp; **Operation:** `{action}`")

    if isinstance(unique_after, int):
        meta_df = pd.DataFrame({
            "Metric": ["Unique Categories"],
            "Before": [str(unique_before)],
            "After":  [str(unique_after)],
            "Change": [_change_str(unique_before, unique_after)],
        })
        st.dataframe(meta_df, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Top values — Before")
        st.dataframe(
            top_before.rename_axis("Category").reset_index(name="Count"),
            hide_index=True, use_container_width=True
        )
    with c2:
        if top_after is not None:
            st.caption("Top values — After")
            st.dataframe(
                top_after.rename_axis("Category").reset_index(name="Count"),
                hide_index=True, use_container_width=True
            )
        elif new_dummy_cols:
            st.caption("New dummy columns created")
            st.dataframe(pd.DataFrame({"Column": new_dummy_cols}),
                         hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_scaling_summary(cols, stats_before: pd.DataFrame, stats_after: pd.DataFrame):
    """Before/after stats for normalization/scaling."""
    st.markdown('<div class="before-after-box">', unsafe_allow_html=True)
    st.markdown('<h5>📊 Scaling — Before / After Statistics</h5>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Before")
        st.dataframe(stats_before.round(4), use_container_width=True)
    with c2:
        st.caption("After")
        st.dataframe(stats_after.round(4), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FILE LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_file(file_bytes, file_name):
    ext = file_name.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(file_bytes))
    elif ext == "json":
        return pd.read_json(io.BytesIO(file_bytes))
    raise ValueError(f"Unsupported file type: .{ext}")

@st.cache_data(show_spinner=False)
def load_gsheet(url: str):
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        raise ValueError("Could not extract sheet ID from URL.")
    sheet_id = match.group(1)
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    return pd.read_csv(csv_url)

# ─────────────────────────────────────────────
# PROFILING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def profile_missing(df):
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({"Column": df.columns, "Missing Count": missing.values, "Missing %": pct.values})

@st.cache_data(show_spinner=False)
def profile_numeric(df):
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()
    return num.describe().T.reset_index().rename(columns={"index": "Column"})

@st.cache_data(show_spinner=False)
def profile_categorical(df):
    cat = df.select_dtypes(include=["object", "category"])
    if cat.empty:
        return pd.DataFrame()
    rows = []
    for col in cat.columns:
        vc = df[col].value_counts()
        rows.append({
            "Column": col,
            "Count": df[col].count(),
            "Unique": df[col].nunique(),
            "Top Value": vc.index[0] if len(vc) else None,
            "Top Freq": int(vc.iloc[0]) if len(vc) else None,
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗂️ DataStudio")
    st.markdown("---")

    if st.session_state.recipe:
        st.markdown("### 📋 Transformation Log")
        for step in st.session_state.recipe:
            css_class = "recipe-step ai-step" if step.get("source") == "ai_suggested" else "recipe-step"
            ai_badge = " 🤖" if step.get("source") == "ai_suggested" else ""
            st.markdown(
                f'<div class="{css_class}"><b>#{step["step"]}</b> {step["operation"]}{ai_badge}<br>'
                f'<span style="color:#6b7280">{step["affected_columns"]}</span></div>',
                unsafe_allow_html=True
            )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↩ Undo", use_container_width=True):
                undo_last()
                st.rerun()
        with col2:
            if st.button("🔄 Reset All", use_container_width=True):
                reset_all()
                st.rerun()

    st.markdown("### ⚙️ Settings")
    ai_toggle = st.toggle("Enable AI Assistant", value=st.session_state.ai_enabled)
    st.session_state.ai_enabled = ai_toggle

    if ai_toggle:
        if not st.session_state.openai_key:
            st.session_state.openai_key = st.secrets.get("OPENAI_API_KEY", "")
        st.markdown('<div class="disclaimer">⚠️ AI suggestions may be imperfect. Always review before applying.</div>', unsafe_allow_html=True)

    st.markdown("---")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_a, tab_b, tab_c, tab_d = st.tabs([
    "📁 Upload and Overview",
    "🧹 Cleaning Studio",
    "📊 Visualization",
    "💾 Export and Report",
])

# ══════════════════════════════════════════════
# PAGE A — UPLOAD & OVERVIEW
# ══════════════════════════════════════════════
with tab_a:
    st.markdown("## Upload and Overview")

    _uk = st.session_state.upload_key
    col_upload, col_sheets = st.columns([1, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Drag & Drop your file here, or Browse Files",
            type=["csv", "xlsx", "xls", "json"],
            help="Supported formats: CSV, Excel (.xlsx/.xls), JSON",
            key=f"file_uploader_{_uk}"
        )
    with col_sheets:
        sheets_url = st.text_input(
            "Or paste Google Sheets public URL",
            placeholder="https://docs.google.com/spreadsheets/d/...",
            key=f"sheets_url_{_uk}"
        )

    load_error = None
    if uploaded_file:
        df_raw, load_error = safe_run(load_file, uploaded_file.read(), uploaded_file.name)
        if df_raw is not None and st.session_state.df_original is None:
            st.session_state.df_original = df_raw.copy()
            st.session_state.df_working = df_raw.copy()
    elif sheets_url:
        df_raw, load_error = safe_run(load_gsheet, sheets_url)
        if df_raw is not None and st.session_state.df_original is None:
            st.session_state.df_original = df_raw.copy()
            st.session_state.df_working = df_raw.copy()

    if load_error:
        st.error(f"❌ Could not load file: {load_error}")

    if st.session_state.df_working is not None:
        df = st.session_state.df_working

        col_reset = st.columns([4, 1])
        with col_reset[1]:
            if st.button("🔄 Reset Session", use_container_width=True):
                reset_session()
                st.rerun()

        st.markdown("---")

        st.markdown('<div class="section-header">1 · Dataset Information</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><h4>Rows</h4><p>{df.shape[0]:,}</p></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><h4>Columns</h4><p>{df.shape[1]}</p></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><h4>Total Cells</h4><p>{df.shape[0] * df.shape[1]:,}</p></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">2 · Column Names & Data Types</div>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype": df.dtypes.astype(str).values,
            "Non-Null Count": df.notnull().sum().values,
            "Sample Value": [str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0] > 0 else "—" for c in df.columns],
        })
        show_table(dtype_df)

        st.markdown('<div class="section-header">3 · Basic Summary Stats</div>', unsafe_allow_html=True)
        st3_tab1, st3_tab2 = st.tabs(["Numeric", "Categorical"])
        with st3_tab1:
            num_profile = profile_numeric(df)
            if num_profile.empty:
                st.info("No numeric columns found.")
            else:
                show_table(num_profile.round(3))
        with st3_tab2:
            cat_profile = profile_categorical(df)
            if cat_profile.empty:
                st.info("No categorical columns found.")
            else:
                show_table(cat_profile)

        st.markdown('<div class="section-header">4 · Missing Values by Column</div>', unsafe_allow_html=True)
        miss_df = profile_missing(df)
        miss_df_filtered = miss_df[miss_df["Missing Count"] > 0]
        if miss_df_filtered.empty:
            st.success("✅ No missing values detected.")
        else:
            show_table(miss_df_filtered)

        st.markdown('<div class="section-header">5 · Duplicates</div>', unsafe_allow_html=True)
        dup_count = df.duplicated().sum()
        st.markdown(f'<div class="metric-card"><h4>Duplicate Rows</h4><p>{dup_count:,}</p></div>', unsafe_allow_html=True)
        if dup_count > 0:
            if st.checkbox("Show duplicate rows"):
                show_table(df[df.duplicated(keep=False)].head(200))
    else:
        st.info("👆 Upload a file or paste a Google Sheets URL to get started.")


# ══════════════════════════════════════════════
# PAGE B — CLEANING STUDIO
# ══════════════════════════════════════════════
with tab_b:
    st.markdown("## 🧹 Cleaning & Preparation Studio")

    if st.session_state.df_working is None:
        st.info("Please upload a dataset on the Upload & Overview tab first.")
    else:
        df = st.session_state.df_working

        # ── Tool selector (matches first-draft selectbox navigation) ──────────
        cleaning_menu = st.selectbox(
            "Choose a cleaning tool",
            [
                "Missing Values",
                "Duplicates",
                "Data Types & Parsing",
                "Categorical Data Tools",
                "Numeric Cleaning",
                "Normalization / Scaling",
                "Column Operations",
                "Data Validation Rules",
            ],
            key="cleaning_menu",
        )

        st.markdown("---")

        # ── AI Assistant (if enabled) ──────────────────────────────────────────
        if st.session_state.ai_enabled:
            st.markdown('<div class="section-header">🤖 AI Cleaning Assistant</div>', unsafe_allow_html=True)
            st.markdown('<div class="disclaimer">⚠️ AI suggestions may be imperfect. Always review before applying.</div>', unsafe_allow_html=True)

            if "ai_chat_history" not in st.session_state:
                st.session_state["ai_chat_history"] = []
            if "ai_pending_suggestions" not in st.session_state:
                st.session_state["ai_pending_suggestions"] = []

            for msg in st.session_state["ai_chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if st.session_state["ai_pending_suggestions"]:
                with st.chat_message("assistant"):
                    st.markdown("Here are the suggested operations. Confirm or reject each one:")
                    for i, sug in enumerate(st.session_state["ai_pending_suggestions"]):
                        desc = sug.get("description", sug["operation"])
                        cols_str = ", ".join(sug.get("affected_columns", []))
                        params_str = ", ".join(f"{k}: {v}" for k, v in sug.get("parameters", {}).items())
                        with st.expander(f"**#{i+1} · {desc}**", expanded=True):
                            st.markdown(f"- **Operation:** `{sug['operation']}`")
                            st.markdown(f"- **Columns:** `{cols_str}`")
                            if params_str:
                                st.markdown(f"- **Parameters:** {params_str}")
                            c1, c2 = st.columns(2)
                            with c1:
                                if st.button(f"✅ Apply", key=f"ai_confirm_{i}"):
                                    push_history()
                                    try:
                                        op = sug["operation"]
                                        params = sug.get("parameters", {})
                                        affected = sug.get("affected_columns", [])
                                        wdf = st.session_state.df_working

                                        if op == "fill_missing":
                                            col = params.get("column") or (affected[0] if affected else None)
                                            method = params.get("method", "median")
                                            if col and col in wdf.columns:
                                                if method == "median" and pd.api.types.is_numeric_dtype(wdf[col]):
                                                    wdf[col] = wdf[col].fillna(wdf[col].median())
                                                elif method == "mean" and pd.api.types.is_numeric_dtype(wdf[col]):
                                                    wdf[col] = wdf[col].fillna(wdf[col].mean())
                                                elif method in ("mode", "most_frequent"):
                                                    wdf[col] = wdf[col].fillna(wdf[col].mode()[0])
                                                elif method == "ffill":
                                                    wdf[col] = wdf[col].ffill()
                                                elif method == "bfill":
                                                    wdf[col] = wdf[col].bfill()
                                                else:
                                                    wdf[col] = wdf[col].fillna(method)
                                                st.session_state.df_working = wdf

                                        elif op == "standardize_case":
                                            col = params.get("column") or (affected[0] if affected else None)
                                            case = params.get("case", "lower")
                                            if col and col in wdf.columns:
                                                if case == "lower":
                                                    wdf[col] = wdf[col].astype(str).str.lower().str.strip()
                                                elif case == "title":
                                                    wdf[col] = wdf[col].astype(str).str.title().str.strip()
                                                elif case == "upper":
                                                    wdf[col] = wdf[col].astype(str).str.upper().str.strip()
                                                else:
                                                    wdf[col] = wdf[col].astype(str).str.strip()
                                                st.session_state.df_working = wdf

                                        elif op == "drop_duplicates":
                                            subset = params.get("subset", None)
                                            keep = params.get("keep", "first")
                                            st.session_state.df_working = wdf.drop_duplicates(subset=subset, keep=keep)

                                        elif op == "convert_dtype":
                                            col = params.get("column") or (affected[0] if affected else None)
                                            target = params.get("target", "numeric")
                                            if col and col in wdf.columns:
                                                if target == "numeric":
                                                    wdf[col] = pd.to_numeric(wdf[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")
                                                elif target == "datetime":
                                                    wdf[col] = pd.to_datetime(wdf[col], errors="coerce")
                                                else:
                                                    wdf[col] = wdf[col].astype(str)
                                                st.session_state.df_working = wdf

                                        elif op == "rename_column":
                                            old_name = params.get("from") or params.get("old_name")
                                            new_name = params.get("to") or params.get("new_name")
                                            if old_name and new_name and old_name in wdf.columns:
                                                st.session_state.df_working = wdf.rename(columns={old_name: new_name})

                                        elif op == "drop_column":
                                            cols_to_drop = affected if affected else [params.get("column")]
                                            cols_to_drop = [c for c in cols_to_drop if c in wdf.columns]
                                            if cols_to_drop:
                                                st.session_state.df_working = wdf.drop(columns=cols_to_drop)

                                        elif op in ("scale_column", "normalize"):
                                            col = params.get("column") or (affected[0] if affected else None)
                                            method = params.get("method", "minmax")
                                            if col and col in wdf.columns and pd.api.types.is_numeric_dtype(wdf[col]):
                                                if method in ("minmax", "min_max"):
                                                    mn, mx = wdf[col].min(), wdf[col].max()
                                                    wdf[col] = (wdf[col] - mn) / (mx - mn) if mx != mn else 0
                                                else:
                                                    mean, std = wdf[col].mean(), wdf[col].std()
                                                    wdf[col] = (wdf[col] - mean) / std if std != 0 else 0
                                                st.session_state.df_working = wdf

                                        elif op in ("encode_categorical", "one_hot_encode"):
                                            col = params.get("column") or (affected[0] if affected else None)
                                            if col and col in wdf.columns:
                                                dummies = pd.get_dummies(wdf[col], prefix=col)
                                                st.session_state.df_working = pd.concat([wdf.drop(columns=[col]), dummies], axis=1)

                                        elif op in ("winsorize", "cap_outliers"):
                                            col = params.get("column") or (affected[0] if affected else None)
                                            lower_q = params.get("lower_quantile", 0.05)
                                            upper_q = params.get("upper_quantile", 0.95)
                                            if col and col in wdf.columns and pd.api.types.is_numeric_dtype(wdf[col]):
                                                lo = wdf[col].quantile(lower_q)
                                                hi = wdf[col].quantile(upper_q)
                                                st.session_state.df_working[col] = wdf[col].clip(lo, hi)

                                        log_step(op, params, affected, source="ai_suggested")
                                        st.session_state["ai_chat_history"].append({
                                            "role": "assistant",
                                            "content": f"✅ Applied: **{desc}** on `{cols_str}`"
                                        })
                                        st.session_state["ai_pending_suggestions"].pop(i)
                                    except Exception as e:
                                        st.error(f"Could not apply operation: {e}")
                                    st.rerun()
                            with c2:
                                if st.button(f"❌ Reject", key=f"ai_reject_{i}"):
                                    st.session_state["ai_chat_history"].append({
                                        "role": "assistant",
                                        "content": f"❌ Rejected: **{desc}**"
                                    })
                                    st.session_state["ai_pending_suggestions"].pop(i)
                                    st.rerun()

            ai_prompt = st.chat_input("Describe a cleaning operation, e.g. 'replace nulls in price with median'")
            if ai_prompt:
                st.session_state["ai_chat_history"].append({"role": "user", "content": ai_prompt})
                with st.spinner("Thinking..."):
                    try:
                        import openai
                        client = openai.OpenAI(api_key=st.session_state.openai_key)
                        schema_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
                        system_prompt = """You are a data cleaning assistant. Given a dataset schema and a natural language command, return ONLY a valid JSON array of operations.
Each operation must have:
- "operation": one of [fill_missing, drop_duplicates, convert_dtype, standardize_case, rename_column, drop_column, scale_column, encode_categorical, winsorize]
- "parameters": dict with relevant keys (e.g. column, method, case, target, subset, keep)
- "affected_columns": list of column names
- "description": short human-readable explanation (max 10 words)
Return ONLY the JSON array. No markdown, no backticks, no explanation text."""
                        user_prompt = f"Dataset columns and types: {json.dumps(schema_info)}\n\nUser command: {ai_prompt}"
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            max_tokens=1000,
                        )
                        raw = response.choices[0].message.content.strip()
                        raw = re.sub(r"```json|```", "", raw).strip()
                        suggestions = json.loads(raw)
                        st.session_state["ai_pending_suggestions"] = suggestions
                    except json.JSONDecodeError:
                        st.session_state["ai_chat_history"].append({
                            "role": "assistant",
                            "content": "⚠️ I couldn't parse a valid operation from that. Please try rephrasing."
                        })
                    except Exception as e:
                        st.session_state["ai_chat_history"].append({
                            "role": "assistant",
                            "content": f"⚠️ Error: {e}"
                        })
                st.rerun()

            st.markdown("---")

        # ══════════════════════════════════════
        # TOOL PANELS — driven by cleaning_menu
        # ══════════════════════════════════════

        # ─────────────────────────────────────
        # MISSING VALUES
        # ─────────────────────────────────────
        if cleaning_menu == "Missing Values":
            st.subheader("Handle Missing Values")

            missing_count   = df.isnull().sum()
            missing_percent = (df.isnull().sum() / len(df) * 100).round(2)
            missing_summary = pd.DataFrame({
                "Column":        df.columns,
                "Missing Count": missing_count.values,
                "Missing %":     missing_percent.values,
            })
            st.markdown('<div class="section-header">Missing Value Summary</div>', unsafe_allow_html=True)
            show_table(missing_summary)

            cols_with_missing = missing_summary[missing_summary["Missing Count"] > 0]["Column"].tolist()

            if not cols_with_missing:
                st.success("✅ No missing values found in this dataset.")
            else:
                mv_col    = st.selectbox("Select a column with missing values", cols_with_missing, key="mv_col")
                col_dtype = df[mv_col].dtype
                is_num    = pd.api.types.is_numeric_dtype(df[mv_col])

                st.subheader("Rows with Missing Values in Selected Column")
                show_table(df[df[mv_col].isnull()].head())

                if is_num:
                    method_options = [
                        "Fill with mean", "Fill with median", "Fill with mode",
                        "Fill with custom value", "Drop rows",
                        "Drop columns above threshold", "Forward fill", "Backward fill",
                    ]
                else:
                    method_options = [
                        "Fill with mode", "Fill with custom value", "Drop rows",
                        "Drop columns above threshold", "Forward fill", "Backward fill",
                    ]

                method = st.selectbox("Choose a method", method_options, key="mv_method")

                custom_value      = None
                threshold_percent = None
                if method == "Fill with custom value":
                    custom_value = st.text_input("Enter custom value", key="mv_const")
                if method == "Drop columns above threshold":
                    threshold_percent = st.number_input(
                        "Enter missing-value threshold (%)", 0.0, 100.0, 50.0, 1.0, key="mv_thresh")

                # Before metrics — shown before button (first-draft pattern)
                before_rows    = df.shape[0]
                before_cols    = df.shape[1]
                before_missing = int(df[mv_col].isnull().sum())

                st.write(f"Column data type: **{col_dtype}**")
                st.write(f"Rows before: **{before_rows:,}**")
                st.write(f"Missing values before: **{before_missing:,}**")

                if st.button("Apply Missing Value Treatment", key="mv_apply"):
                    push_history()
                    try:
                        wdf = st.session_state.df_working.copy()

                        if method == "Fill with mean":
                            wdf[mv_col] = wdf[mv_col].fillna(wdf[mv_col].mean())
                        elif method == "Fill with median":
                            wdf[mv_col] = wdf[mv_col].fillna(wdf[mv_col].median())
                        elif method == "Fill with mode":
                            wdf[mv_col] = wdf[mv_col].fillna(wdf[mv_col].mode()[0])
                        elif method == "Fill with custom value":
                            if custom_value is not None and custom_value != "":
                                wdf[mv_col] = wdf[mv_col].fillna(custom_value)
                            else:
                                st.warning("Please enter a custom value first.")
                                st.stop()
                        elif method == "Drop rows":
                            wdf = wdf.dropna(subset=[mv_col])
                        elif method == "Drop columns above threshold":
                            pcts        = (wdf.isnull().sum() / len(wdf)) * 100
                            cols_to_drop = pcts[pcts > threshold_percent].index.tolist()
                            if cols_to_drop:
                                wdf = wdf.drop(columns=cols_to_drop)
                                st.write("Dropped columns:", cols_to_drop)
                            else:
                                st.info("No columns exceeded the threshold.")
                        elif method == "Forward fill":
                            wdf[mv_col] = wdf[mv_col].ffill()
                        elif method == "Backward fill":
                            wdf[mv_col] = wdf[mv_col].bfill()

                        after_rows    = wdf.shape[0]
                        after_cols    = wdf.shape[1]
                        after_missing = int(wdf[mv_col].isnull().sum()) if mv_col in wdf.columns else 0

                        st.session_state.df_working = wdf
                        log_step("fill_missing", {"column": mv_col, "method": method}, [mv_col])

                        st.success(f"✅ Applied '{method}' to column '{mv_col}' successfully.")
                        show_before_after_metrics(
                            before_rows, after_rows,
                            before_cols, after_cols,
                            extra_before=before_missing,
                            extra_after=after_missing,
                            extra_label=f"Missing in '{mv_col}'",
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
                        undo_last()

        # ─────────────────────────────────────
        # DUPLICATES
        # ─────────────────────────────────────
        elif cleaning_menu == "Duplicates":
            st.subheader("Remove Duplicates")

            duplicate_type = st.radio(
                "Choose duplicate detection type",
                ["Full-row duplicates", "Duplicates by selected columns"],
                key="dup_type",
            )
            subset_columns = None
            if duplicate_type == "Duplicates by selected columns":
                subset_columns = st.multiselect(
                    "Select columns to check duplicates", df.columns.tolist(), key="dup_subset")

            keep_option = st.selectbox("Choose which duplicate to keep", ["first", "last"], key="dup_keep")

            if duplicate_type == "Full-row duplicates":
                dup_mask = df.duplicated(keep=False)
            else:
                dup_mask = df.duplicated(subset=subset_columns, keep=False) if subset_columns else pd.Series(False, index=df.index)

            dup_rows  = df[dup_mask]
            dup_count = len(dup_rows)

            st.write(f"Duplicate rows found: **{dup_count:,}**")

            if dup_count > 0:
                st.subheader("Duplicate Groups Preview")
                show_table(dup_rows.head(20))
                st.caption("Showing first 20 duplicate rows.")

                before_rows = df.shape[0]
                before_cols = df.shape[1]

                if st.button("Remove Duplicate Rows", key="dup_apply"):
                    push_history()
                    if duplicate_type == "Full-row duplicates":
                        wdf = st.session_state.df_working.drop_duplicates(keep=keep_option)
                    else:
                        if not subset_columns:
                            st.warning("Please select at least one column for subset detection.")
                            st.stop()
                        wdf = st.session_state.df_working.drop_duplicates(subset=subset_columns, keep=keep_option)

                    after_rows    = wdf.shape[0]
                    after_cols    = wdf.shape[1]
                    rows_removed  = before_rows - after_rows

                    st.session_state.df_working = wdf
                    log_step("drop_duplicates", {"subset": subset_columns, "keep": keep_option},
                             subset_columns or ["all columns"])

                    st.success("✅ Duplicate rows removed successfully.")
                    show_before_after_metrics(
                        before_rows, after_rows,
                        before_cols, after_cols,
                        extra_before=dup_count,
                        extra_after=0,
                        extra_label="Duplicate Rows",
                    )
                    st.write(f"Duplicate detection type: {duplicate_type}")
                    if subset_columns:
                        st.write("Subset columns used:", subset_columns)
                    st.write(f"Keep option: {keep_option}")
                    st.write(f"Duplicate rows removed: {rows_removed}")
            else:
                st.success("✅ No duplicate rows found for the selected criteria.")

        # ─────────────────────────────────────
        # DATA TYPES & PARSING
        # ─────────────────────────────────────
        elif cleaning_menu == "Data Types & Parsing":
            st.subheader("Data Types & Parsing")

            dt_col        = st.selectbox("Select a column", df.columns.tolist(), key="dt_col")
            current_dtype = str(df[dt_col].dtype)
            st.write(f"Current data type: **{current_dtype}**")

            target_type = st.selectbox(
                "Convert selected column to",
                ["string", "numeric", "datetime", "category"],
                key="dt_target",
            )

            clean_numeric_strings = False
            datetime_parse_mode   = None
            datetime_format       = None

            if target_type == "numeric":
                clean_numeric_strings = st.checkbox(
                    "Clean dirty numeric strings (remove commas, currency signs, spaces)",
                    value=True, key="dt_dirty")
            if target_type == "datetime":
                datetime_parse_mode = st.radio(
                    "Choose datetime parsing mode",
                    ["Auto parse", "Use custom format"],
                    key="dt_parse_mode",
                )
                if datetime_parse_mode == "Use custom format":
                    datetime_format = st.text_input(
                        "Enter datetime format (e.g. %Y-%m-%d)",
                        value="%Y-%m-%d", key="dt_fmt")

            # Before metrics — shown before button
            before_nonnull = int(df[dt_col].notnull().sum())
            before_null    = int(df[dt_col].isnull().sum())

            st.write(f"Non-null values before: **{before_nonnull:,}**")
            st.write(f"Missing values before: **{before_null:,}**")

            st.subheader("Preview Before Conversion")
            show_table(df[[dt_col]].head(10))

            if st.button("Apply Data Type Conversion", key="dt_apply"):
                push_history()
                try:
                    wdf = st.session_state.df_working.copy()

                    if target_type == "string":
                        wdf[dt_col] = wdf[dt_col].astype(str)
                    elif target_type == "numeric":
                        col_data = wdf[dt_col].astype(str)
                        if clean_numeric_strings:
                            col_data = (col_data
                                        .str.replace(",", "", regex=False)
                                        .str.replace("$", "", regex=False)
                                        .str.replace("€", "", regex=False)
                                        .str.replace("£", "", regex=False)
                                        .str.replace('"', "", regex=False)
                                        .str.strip())
                        wdf[dt_col] = pd.to_numeric(col_data, errors="coerce")
                    elif target_type == "datetime":
                        if datetime_parse_mode == "Auto parse":
                            wdf[dt_col] = pd.to_datetime(wdf[dt_col], errors="coerce", dayfirst=True)
                        else:
                            wdf[dt_col] = pd.to_datetime(
                                wdf[dt_col], format=datetime_format, errors="coerce")
                    elif target_type == "category":
                        wdf[dt_col] = wdf[dt_col].astype("category")

                    after_dtype   = str(wdf[dt_col].dtype)
                    after_nonnull = int(wdf[dt_col].notnull().sum())
                    after_null    = int(wdf[dt_col].isnull().sum())

                    st.session_state.df_working = wdf
                    log_step("convert_dtype", {"column": dt_col, "target": target_type}, [dt_col])

                    st.success(f"✅ Column '{dt_col}' converted successfully to {target_type}.")
                    show_dtype_summary(
                        dt_col,
                        current_dtype, after_dtype,
                        before_nonnull, after_nonnull,
                        before_null, after_null,
                    )

                    st.subheader("Preview After Conversion")
                    show_table(wdf[[dt_col]].head(10))
                except Exception as e:
                    st.error(f"Conversion failed: {e}")
                    undo_last()

        # ─────────────────────────────────────
        # CATEGORICAL DATA TOOLS
        # ─────────────────────────────────────
        elif cleaning_menu == "Categorical Data Tools":
            st.subheader("Categorical Data Tools")

            cat_cols_list = categorical_cols(df)
            if not cat_cols_list:
                st.warning("No categorical columns found in this dataset.")
            else:
                cat_col = st.selectbox("Select a categorical column", cat_cols_list, key="cat_col")
                st.write(f"Current data type: **{df[cat_col].dtype}**")

                st.subheader("Current Unique Values Preview")
                show_table(pd.DataFrame({"Unique Values": df[cat_col].astype(str).unique()[:20]}))

                cat_action = st.selectbox(
                    "Choose a categorical operation",
                    [
                        "Trim whitespace",
                        "Convert to lowercase",
                        "Convert to uppercase",
                        "Mapping / Replacement",
                        "Group rare categories",
                        "One-hot encoding",
                    ],
                    key="cat_action",
                )

                mapping_dict         = {}
                set_unmatched_to_other = False
                freq_threshold       = None

                if cat_action == "Mapping / Replacement":
                    st.write("Enter old value → new value pairs")
                    unique_vals = df[cat_col].dropna().astype(str).unique().tolist()
                    for val in unique_vals[:10]:
                        new_val = st.text_input(
                            f"Replace '{val}' with:", value=val,
                            key=f"map_{cat_col}_{val}")
                        mapping_dict[val] = new_val
                    set_unmatched_to_other = st.checkbox(
                        "Set unmatched values to 'Other'", value=False, key="cat_unmatched")

                if cat_action == "Group rare categories":
                    freq_threshold = st.number_input(
                        "Group categories with frequency below this threshold into 'Other'",
                        min_value=1, value=2, step=1, key="cat_rare_thresh")

                # Before metrics — shown before button
                before_unique     = int(df[cat_col].nunique(dropna=True))
                before_top_values = df[cat_col].value_counts(dropna=False).head(10)

                st.subheader("Before Summary")
                st.write(f"Unique categories before: **{before_unique}**")
                show_table(before_top_values.rename_axis("Category").reset_index(name="Count"))

                if st.button("Apply Categorical Transformation", key="cat_apply"):
                    push_history()
                    try:
                        wdf = st.session_state.df_working.copy()

                        if cat_action == "Trim whitespace":
                            wdf[cat_col] = wdf[cat_col].astype(str).str.strip()
                        elif cat_action == "Convert to lowercase":
                            wdf[cat_col] = wdf[cat_col].astype(str).str.lower()
                        elif cat_action == "Convert to uppercase":
                            wdf[cat_col] = wdf[cat_col].astype(str).str.upper()
                        elif cat_action == "Mapping / Replacement":
                            original = wdf[cat_col].astype(str)
                            if set_unmatched_to_other:
                                wdf[cat_col] = original.map(mapping_dict).fillna("Other")
                            else:
                                wdf[cat_col] = original.replace(mapping_dict)
                        elif cat_action == "Group rare categories":
                            vc   = wdf[cat_col].value_counts(dropna=False)
                            rare = vc[vc < freq_threshold].index
                            wdf[cat_col] = wdf[cat_col].apply(lambda x: "Other" if x in rare else x)
                        elif cat_action == "One-hot encoding":
                            dummies = pd.get_dummies(wdf[cat_col], prefix=cat_col)
                            wdf = pd.concat([wdf.drop(columns=[cat_col]), dummies], axis=1)

                        if cat_action != "One-hot encoding":
                            after_unique     = int(wdf[cat_col].nunique(dropna=True))
                            after_top_values = wdf[cat_col].value_counts(dropna=False).head(10)
                            new_dummy_cols   = None
                        else:
                            after_unique     = "N/A (new columns created)"
                            after_top_values = None
                            new_dummy_cols   = [c for c in wdf.columns if c.startswith(f"{cat_col}_")]

                        st.session_state.df_working = wdf
                        log_step("categorical_tool", {"column": cat_col, "action": cat_action}, [cat_col])

                        st.success(f"✅ Applied '{cat_action}' to column '{cat_col}' successfully.")
                        show_categorical_summary(
                            cat_col, cat_action,
                            before_unique, after_unique,
                            before_top_values, after_top_values,
                            new_dummy_cols=new_dummy_cols,
                        )

                        st.subheader("Preview After Transformation")
                        if cat_action != "One-hot encoding":
                            show_table(wdf[[cat_col]].head(10))
                        else:
                            show_table(wdf[new_dummy_cols].head(10))
                    except Exception as e:
                        st.error(f"Error: {e}")
                        undo_last()

        # ─────────────────────────────────────
        # NUMERIC CLEANING
        # ─────────────────────────────────────
        elif cleaning_menu == "Numeric Cleaning":
            st.subheader("Numeric Cleaning")

            num_cols_list = numeric_cols(df)
            if not num_cols_list:
                st.warning("No numeric columns found.")
            else:
                out_col  = st.selectbox("Select numeric column", num_cols_list, key="out_col")
                col_data = df[out_col].dropna()

                Q1  = col_data.quantile(0.25)
                Q3  = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (df[out_col] < lower_bound) | (df[out_col] > upper_bound)
                num_outliers = int(outlier_mask.sum())

                st.subheader("Outlier Detection Summary (IQR)")
                st.write(f"Q1: **{Q1:.2f}** | Q3: **{Q3:.2f}** | IQR: **{IQR:.2f}**")
                st.write(f"Lower bound: **{lower_bound:.2f}** | Upper bound: **{upper_bound:.2f}**")
                st.write(f"Number of outliers detected: **{num_outliers}**")

                out_action = st.selectbox(
                    "Choose action",
                    ["Do nothing", "Cap (Winsorize)", "Remove outlier rows"],
                    key="out_action",
                )

                if st.button("Apply Numeric Cleaning", key="out_apply"):
                    push_history()
                    wdf = st.session_state.df_working.copy()

                    if out_action == "Cap (Winsorize)":
                        before_vals  = wdf[out_col].copy()
                        wdf[out_col] = wdf[out_col].clip(lower_bound, upper_bound)
                        num_capped   = int(((before_vals != wdf[out_col]) & before_vals.notna()).sum())

                        st.session_state.df_working = wdf
                        log_step("outlier_handling",
                                 {"column": out_col, "method": "IQR", "action": "Cap"},
                                 [out_col])
                        st.success("✅ Outliers capped successfully.")
                        show_before_after_metrics(
                            df.shape[0], wdf.shape[0],
                            df.shape[1], wdf.shape[1],
                            extra_before=num_outliers,
                            extra_after=num_capped,
                            extra_label="Values Capped",
                        )
                        st.write(f"Values capped: **{num_capped}**")

                    elif out_action == "Remove outlier rows":
                        before_rows = wdf.shape[0]
                        wdf         = wdf[~outlier_mask]
                        after_rows  = wdf.shape[0]
                        removed     = before_rows - after_rows

                        st.session_state.df_working = wdf
                        log_step("outlier_handling",
                                 {"column": out_col, "method": "IQR", "action": "Remove rows"},
                                 [out_col])
                        st.success("✅ Outlier rows removed.")
                        show_before_after_metrics(
                            before_rows, after_rows,
                            df.shape[1], wdf.shape[1],
                            extra_before=num_outliers,
                            extra_after=0,
                            extra_label="Outlier Rows",
                        )
                        st.write(f"Rows removed: **{removed}**")
                    else:
                        st.info("No changes applied.")

                    st.subheader("Preview After Cleaning")
                    show_table(st.session_state.df_working.head(10))

        # ─────────────────────────────────────
        # NORMALIZATION / SCALING
        # ─────────────────────────────────────
        elif cleaning_menu == "Normalization / Scaling":
            st.subheader("Normalization / Scaling")

            num_cols_list = numeric_cols(df)
            if not num_cols_list:
                st.warning("No numeric columns found in this dataset.")
            else:
                scale_cols   = st.multiselect("Select numeric column(s) to scale", num_cols_list, key="scale_cols")
                scale_method = st.selectbox(
                    "Choose scaling method",
                    ["Min-Max Scaling", "Z-Score Standardization"],
                    key="scale_method",
                )

                if scale_cols:
                    before_stats = df[scale_cols].agg(["min", "max", "mean", "std"]).T
                    st.subheader("Before Scaling Statistics")
                    show_table(before_stats.round(4).reset_index().rename(columns={"index": "Column"}))

                    if st.button("Apply Scaling", key="scale_apply"):
                        push_history()
                        try:
                            wdf     = st.session_state.df_working.copy()
                            skipped = []
                            for col in scale_cols:
                                if scale_method == "Min-Max Scaling":
                                    col_min, col_max = wdf[col].min(), wdf[col].max()
                                    if col_max != col_min:
                                        wdf[col] = (wdf[col] - col_min) / (col_max - col_min)
                                    else:
                                        st.warning(f"Column '{col}' has constant values — Min-Max scaling skipped.")
                                        skipped.append(col)
                                else:
                                    col_mean, col_std = wdf[col].mean(), wdf[col].std()
                                    if col_std != 0:
                                        wdf[col] = (wdf[col] - col_mean) / col_std
                                    else:
                                        st.warning(f"Column '{col}' has zero std — Z-score scaling skipped.")
                                        skipped.append(col)

                            after_stats = wdf[scale_cols].agg(["min", "max", "mean", "std"]).T

                            st.session_state.df_working = wdf
                            log_step("scale_columns", {"method": scale_method, "columns": scale_cols}, scale_cols)

                            st.success(f"✅ Applied '{scale_method}' successfully.")
                            show_scaling_summary(
                                scale_cols,
                                before_stats.reset_index().rename(columns={"index": "Column"}),
                                after_stats.reset_index().rename(columns={"index": "Column"}),
                            )

                            st.subheader("Preview After Scaling")
                            show_table(wdf[scale_cols].head(10))
                        except Exception as e:
                            st.error(f"Error: {e}")
                            undo_last()
                else:
                    st.info("Please select at least one numeric column.")

        # ─────────────────────────────────────
        # COLUMN OPERATIONS
        # ─────────────────────────────────────
        elif cleaning_menu == "Column Operations":
            st.subheader("Column Operations")

            col_action = st.selectbox(
                "Choose a column operation",
                ["Rename Columns", "Drop Columns", "Create New Column", "Bin Numeric Column"],
                key="col_op",
            )

            # ── Rename ────────────────────────────────────────────────────────
            if col_action == "Rename Columns":
                rename_col     = st.selectbox("Select column to rename", df.columns.tolist(), key="rename_col")
                new_col_name   = st.text_input("Enter new column name", key="rename_new")

                if st.button("Apply Rename", key="rename_apply"):
                    if new_col_name.strip() == "":
                        st.warning("Please enter a valid new column name.")
                    elif new_col_name in df.columns:
                        st.warning("This column name already exists.")
                    else:
                        push_history()
                        df_before = st.session_state.df_working.copy()
                        st.session_state.df_working = df_before.rename(columns={rename_col: new_col_name})
                        log_step("rename_column", {"from": rename_col, "to": new_col_name}, [rename_col])
                        st.success(f"✅ Column '{rename_col}' renamed to '{new_col_name}'.")
                        show_column_changes(df_before, st.session_state.df_working)

            # ── Drop ──────────────────────────────────────────────────────────
            elif col_action == "Drop Columns":
                drop_cols = st.multiselect("Select column(s) to drop", df.columns.tolist(), key="drop_cols")

                if st.button("Apply Drop", key="drop_apply"):
                    if not drop_cols:
                        st.warning("Please select at least one column to drop.")
                    else:
                        push_history()
                        df_before = st.session_state.df_working.copy()
                        st.session_state.df_working = df_before.drop(columns=drop_cols)
                        log_step("drop_columns", {"columns": drop_cols}, drop_cols)
                        st.success("✅ Selected columns dropped successfully.")
                        show_column_changes(df_before, st.session_state.df_working)
                        st.write(f"Columns before: **{df_before.shape[1]}**")
                        st.write(f"Columns after: **{st.session_state.df_working.shape[1]}**")

            # ── Create ────────────────────────────────────────────────────────
            elif col_action == "Create New Column":
                num_cols_list = numeric_cols(df)

                formula_type = st.selectbox(
                    "Choose formula type",
                    [
                        "Add two columns", "Subtract two columns", "Divide two columns",
                        "Log of a column", "Center a column by its mean",
                    ],
                    key="formula_type",
                )
                new_col_name = st.text_input("Enter new column name", key="formula_name")

                if formula_type in ["Add two columns", "Subtract two columns", "Divide two columns"]:
                    fcol1 = st.selectbox("Select first numeric column",  num_cols_list, key="formula_col1")
                    fcol2 = st.selectbox("Select second numeric column", num_cols_list, key="formula_col2")
                else:
                    fcol1 = st.selectbox("Select numeric column", num_cols_list, key="single_formula_col")
                    fcol2 = None

                if st.button("Apply Formula", key="formula_apply"):
                    if new_col_name.strip() == "":
                        st.warning("Please enter a valid new column name.")
                    elif new_col_name in df.columns:
                        st.warning("This column name already exists.")
                    else:
                        push_history()
                        try:
                            df_before = st.session_state.df_working.copy()
                            if formula_type == "Add two columns":
                                df_before[new_col_name] = df_before[fcol1] + df_before[fcol2]
                            elif formula_type == "Subtract two columns":
                                df_before[new_col_name] = df_before[fcol1] - df_before[fcol2]
                            elif formula_type == "Divide two columns":
                                df_before[new_col_name] = df_before[fcol1] / df_before[fcol2]
                            elif formula_type == "Log of a column":
                                df_before[new_col_name] = np.log(df_before[fcol1])
                            elif formula_type == "Center a column by its mean":
                                df_before[new_col_name] = df_before[fcol1] - df_before[fcol1].mean()

                            st.session_state.df_working = df_before
                            log_step("create_column", {"name": new_col_name, "formula": formula_type}, [new_col_name])
                            st.success(f"✅ New column '{new_col_name}' created successfully.")
                            show_column_changes(df, st.session_state.df_working)
                            st.subheader("Preview of New Column")
                            show_table(df_before[[new_col_name]].head(10))
                        except Exception as e:
                            st.error(f"Could not create new column: {e}")
                            undo_last()

            # ── Bin ───────────────────────────────────────────────────────────
            elif col_action == "Bin Numeric Column":
                num_cols_list = numeric_cols(df)
                bin_col       = st.selectbox("Select numeric column to bin", num_cols_list, key="bin_col")
                bin_method    = st.selectbox(
                    "Choose binning method", ["Equal-width bins", "Quantile bins"], key="bin_method")
                num_bins      = st.number_input(
                    "Number of bins", min_value=2, max_value=10, value=4, step=1, key="bin_count")
                bin_new_name  = st.text_input("Enter new binned column name", key="bin_new_name")

                if st.button("Apply Binning", key="bin_apply"):
                    if bin_new_name.strip() == "":
                        st.warning("Please enter a valid new column name.")
                    elif bin_new_name in df.columns:
                        st.warning("This column name already exists.")
                    else:
                        push_history()
                        try:
                            df_before = st.session_state.df_working.copy()
                            if bin_method == "Equal-width bins":
                                df_before[bin_new_name] = pd.cut(
                                    df_before[bin_col], bins=num_bins)
                            else:
                                df_before[bin_new_name] = pd.qcut(
                                    df_before[bin_col], q=num_bins, duplicates="drop")

                            st.session_state.df_working = df_before
                            log_step("bin_column",
                                     {"column": bin_col, "method": bin_method, "bins": num_bins},
                                     [bin_new_name])
                            st.success(f"✅ Binned column '{bin_new_name}' created successfully.")
                            show_column_changes(df, st.session_state.df_working)
                            st.subheader("Preview of Binned Column")
                            show_table(df_before[[bin_col, bin_new_name]].head(10))
                        except Exception as e:
                            st.error(f"Binning failed: {e}")
                            undo_last()

        # ─────────────────────────────────────
        # DATA VALIDATION RULES
        # ─────────────────────────────────────
        elif cleaning_menu == "Data Validation Rules":
            st.subheader("Data Validation Rules")

            validation_type = st.selectbox(
                "Choose validation rule",
                ["Numeric Range Check", "Allowed Categories Check", "Non-Null Check"],
                key="val_rule",
            )

            violations_df = pd.DataFrame()

            if validation_type == "Numeric Range Check":
                num_cols_list = numeric_cols(df)
                if not num_cols_list:
                    st.warning("No numeric columns found in this dataset.")
                else:
                    val_col   = st.selectbox("Select numeric column", num_cols_list, key="val_num_col")
                    min_value = st.number_input("Minimum allowed value", value=float(df[val_col].min()), key="val_min")
                    max_value = st.number_input("Maximum allowed value", value=float(df[val_col].max()), key="val_max")
                    if st.button("Run Numeric Range Check", key="val_num_check"):
                        violations_df = df[(df[val_col] < min_value) | (df[val_col] > max_value)]
                        st.subheader("Violations Table")
                        st.write(f"Number of violating rows: **{len(violations_df)}**")
                        show_table(violations_df)

            elif validation_type == "Allowed Categories Check":
                cat_cols_list = categorical_cols(df)
                if not cat_cols_list:
                    st.warning("No categorical columns found in this dataset.")
                else:
                    val_col       = st.selectbox("Select categorical column", cat_cols_list, key="val_cat_col")
                    st.write("Enter allowed categories separated by commas")
                    allowed_input = st.text_input("Allowed categories", value="", key="val_allowed")
                    if st.button("Run Allowed Categories Check", key="val_cat_check"):
                        allowed_values = [x.strip() for x in allowed_input.split(",") if x.strip()]
                        if not allowed_values:
                            st.warning("Please enter at least one allowed category.")
                        else:
                            violations_df = df[~df[val_col].isin(allowed_values)]
                            st.subheader("Violations Table")
                            st.write(f"Allowed categories: {allowed_values}")
                            st.write(f"Number of violating rows: **{len(violations_df)}**")
                            show_table(violations_df)

            elif validation_type == "Non-Null Check":
                val_cols = st.multiselect(
                    "Select column(s) that must not contain missing values",
                    df.columns.tolist(), key="val_null_cols")
                if st.button("Run Non-Null Check", key="val_null_check"):
                    if not val_cols:
                        st.warning("Please select at least one column.")
                    else:
                        violations_df = df[df[val_cols].isnull().any(axis=1)]
                        st.subheader("Violations Table")
                        st.write(f"Checked columns: {val_cols}")
                        st.write(f"Number of violating rows: **{len(violations_df)}**")
                        show_table(violations_df)

            if not violations_df.empty:
                st.markdown(
                    f'<span class="violation-badge">⚠️ {len(violations_df)} violations found</span>',
                    unsafe_allow_html=True)
                csv_data = violations_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Violations as CSV", csv_data,
                    "validation_violations.csv", "text/csv")

        # ─────────────────────────────────────
        # Dataset preview — always at the bottom
        # ─────────────────────────────────────
        st.markdown("---")
        st.subheader("Preview of Current Dataset")
        st.caption(f"Shape: {st.session_state.df_working.shape[0]:,} rows × {st.session_state.df_working.shape[1]} cols")
        show_table(st.session_state.df_working.head(100))


# ══════════════════════════════════════════════
# PAGE C — VISUALIZATION STUDIO
# ══════════════════════════════════════════════
with tab_c:
    st.markdown("## 📊 Visualization Studio")

    if st.session_state.df_working is None:
        st.info("Please upload a dataset first.")
    else:
        df = st.session_state.df_working
        num_cols_list = numeric_cols(df)
        cat_cols_list = categorical_cols(df)
        dt_cols_list  = datetime_cols(df)

        PALETTE = [
            "#0066CC", "#4CB140", "#009596", "#F0AB00", "#EC7A08",
            "#C9190B", "#519DE9", "#7CC674", "#73C5C5", "#F6D173",
        ]
        BG, GRID, TEXT, MUTED = "#FAFAFA", "#E8E8E8", "#151515", "#6A6E73"

        def _styled_ax(ax, title, xlabel, ylabel):
            ax.set_title(title, fontsize=14, fontweight="bold", pad=14, color=TEXT)
            ax.set_xlabel(xlabel, fontweight="bold", fontsize=12, color=TEXT, labelpad=8)
            ax.set_ylabel(ylabel, fontweight="bold", fontsize=12, color=TEXT, labelpad=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#D2D2D2")
            ax.set_facecolor(BG)
            ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="--", zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=10, colors=MUTED)

        def _style_legend(ax, title=None):
            leg = ax.legend(title=title, fontsize=9, framealpha=0.92,
                            edgecolor="#D2D2D2", facecolor="white", title_fontsize=9)
            if leg:
                leg.get_frame().set_linewidth(0.8)

        def _apply_pf_axes(fig, *, showlegend=True):
            fig.update_layout(
                paper_bgcolor="white", plot_bgcolor=BG,
                font=dict(color=TEXT, family="DM Sans", size=12),
                title=dict(font=dict(size=15, color=TEXT)),
                legend=dict(bgcolor="white", bordercolor="#D2D2D2", borderwidth=1,
                            font=dict(size=11, color=TEXT),
                            title=dict(font=dict(size=11, color=TEXT))),
                margin=dict(l=60, r=40, t=70, b=60),
                showlegend=showlegend,
            )
            fig.update_xaxes(title_font=dict(size=13, color=TEXT),
                             tickfont=dict(size=11, color=MUTED),
                             gridcolor=GRID, zerolinecolor=GRID, linecolor="#D2D2D2")
            fig.update_yaxes(title_font=dict(size=13, color=TEXT),
                             tickfont=dict(size=11, color=MUTED),
                             gridcolor=GRID, zerolinecolor=GRID, linecolor="#D2D2D2")
            return fig

        def _order_cats(series):
            return list(dict.fromkeys(series.dropna().astype(str).tolist()))

        def _sort_line(frame, x_name):
            if x_name not in frame.columns:
                return frame
            if pd.api.types.is_numeric_dtype(frame[x_name]) or pd.api.types.is_datetime64_any_dtype(frame[x_name]):
                return frame.sort_values(x_name)
            cats = _order_cats(frame[x_name])
            tmp = frame.copy()
            tmp["__ord"] = pd.Categorical(tmp[x_name].astype(str), categories=cats, ordered=True)
            return tmp.sort_values("__ord").drop(columns="__ord")

        def _auto_rot(ax, labels):
            n, mx = len(labels), max((len(str(l)) for l in labels), default=0)
            if n > 10 or mx > 8:
                ax.set_xticklabels([str(l) for l in labels], rotation=40, ha="right", fontsize=9)
            elif n > 6 or mx > 5:
                ax.set_xticklabels([str(l) for l in labels], rotation=20, ha="right", fontsize=10)
            else:
                ax.set_xticklabels([str(l) for l in labels], rotation=0, ha="center", fontsize=10)

        def _make_mpl_fig(n_cats=1, n_groups=1):
            w = max(10, n_cats * n_groups * 0.75 + 3)
            return plt.subplots(figsize=(min(w, 32), 5))

        left_col, right_col = st.columns([3, 7], gap="large")

        with left_col:
            st.markdown('<div class="section-header">① Chart Type</div>', unsafe_allow_html=True)
            chart_type = st.selectbox("Select chart type", [
                "Histogram", "Box Plot", "Scatter Plot",
                "Line Chart", "Bar Chart (Grouped)", "Heatmap / Correlation Matrix",
            ], key="chart_type")

            st.markdown('<div style="height:0.2rem;border-top:1px solid #e5e7eb;margin:0.4rem 0 0.3rem 0"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">② Axes & Options</div>', unsafe_allow_html=True)

            x_col = y_col = color_col = agg_method = None
            top_n = 10
            bins = 20
            heatmap_cols = []
            chart_ok = True
            chart_warn = ""

            if chart_type == "Histogram":
                if not num_cols_list:
                    chart_ok, chart_warn = False, "No numeric columns found."
                else:
                    a1, a2 = st.columns(2)
                    with a1: x_col = st.selectbox("Column", num_cols_list, key="hist_x")
                    with a2: bins  = st.slider("Bins", 5, 100, 20, key="hist_bins")

            elif chart_type == "Box Plot":
                if not num_cols_list:
                    chart_ok, chart_warn = False, "No numeric columns found."
                else:
                    a1, a2 = st.columns(2)
                    with a1: y_col = st.selectbox("Y (numeric)", num_cols_list, key="box_y")
                    with a2: x_col = st.selectbox("Group by (optional)", ["(none)"] + cat_cols_list, key="box_x")

            elif chart_type == "Scatter Plot":
                if len(num_cols_list) < 2:
                    chart_ok, chart_warn = False, "Need ≥ 2 numeric columns."
                else:
                    a1, a2 = st.columns(2)
                    with a1: x_col = st.selectbox("X", num_cols_list, key="scatter_x")
                    with a2: y_col = st.selectbox("Y", [c for c in num_cols_list if c != x_col] or num_cols_list, key="scatter_y")
                    a3, a4 = st.columns(2)
                    with a3: color_col  = st.selectbox("Color by", ["(none)"] + cat_cols_list, key="scatter_color")
                    with a4: agg_method = st.selectbox("Aggregation", ["None (raw)", "mean", "sum", "count", "median"], key="scatter_agg")

            elif chart_type == "Line Chart":
                all_x = dt_cols_list + num_cols_list + cat_cols_list
                if not all_x or not num_cols_list:
                    chart_ok, chart_warn = False, "Need at least one X column and one numeric Y."
                else:
                    a1, a2 = st.columns(2)
                    with a1: x_col = st.selectbox("X axis", all_x, key="line_x")
                    with a2: y_col = st.selectbox("Y (numeric)", num_cols_list, key="line_y")
                    a3, a4 = st.columns(2)
                    with a3: color_col  = st.selectbox("Group by", ["(none)"] + cat_cols_list, key="line_color")
                    with a4: agg_method = st.selectbox("Aggregation", ["sum", "mean", "count", "median"], key="line_agg")

            elif chart_type == "Bar Chart (Grouped)":
                if not num_cols_list:
                    chart_ok, chart_warn = False, "No numeric columns found."
                else:
                    x_options = cat_cols_list + num_cols_list
                    a1, a2 = st.columns(2)
                    with a1: x_col = st.selectbox("X (category)", x_options, key="bar_x")
                    with a2: y_col = st.selectbox("Y (numeric)", num_cols_list, key="bar_y")
                    a3, a4 = st.columns(2)
                    with a3: color_col  = st.selectbox("Group by (optional)", ["(none)"] + cat_cols_list, key="bar_color")
                    with a4: agg_method = st.selectbox("Aggregation", ["sum", "mean", "count", "median"], key="bar_agg")
                    top_n = st.slider("Top N categories", 3, 50, 10, key="bar_topn")

            elif chart_type == "Heatmap / Correlation Matrix":
                if len(num_cols_list) < 2:
                    chart_ok, chart_warn = False, "Need ≥ 2 numeric columns."
                else:
                    heatmap_cols = st.multiselect("Columns", num_cols_list,
                                                  default=num_cols_list[:min(10, len(num_cols_list))],
                                                  key="heatmap_cols")
                    if len(heatmap_cols) < 2:
                        chart_ok, chart_warn = False, "Select ≥ 2 columns."

            chart_title = st.text_input("Chart title (optional)", key="chart_title")

            st.markdown('<div style="height:0.2rem;border-top:1px solid #e5e7eb;margin:0.4rem 0 0.3rem 0"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">③ Filters</div>', unsafe_allow_html=True)
            show_filters = st.toggle("⚙ Show Filters", value=False, key="show_filters")

            plot_df = df.copy()
            filter_suffix_parts = []
            filter_cat_col  = "(none)"
            filter_cat_vals = []

            if show_filters:
                f1, f2 = st.columns(2)
                with f1:
                    filter_cat_col = st.selectbox("Filter by category", ["(none)"] + cat_cols_list, key="filter_cat_col")
                    if filter_cat_col != "(none)":
                        filter_cat_vals = st.multiselect(
                            "Keep values",
                            sorted(df[filter_cat_col].dropna().unique().tolist()),
                            key="filter_cat_vals"
                        )
                        if filter_cat_vals:
                            plot_df = plot_df[plot_df[filter_cat_col].isin(filter_cat_vals)]
                            lbl = ", ".join(str(v) for v in filter_cat_vals[:3])
                            if len(filter_cat_vals) > 3:
                                lbl += f" +{len(filter_cat_vals)-3}"
                            filter_suffix_parts.append(f"{filter_cat_col}: {lbl}")
                with f2:
                    filter_num_col = st.selectbox("Filter by numeric range", ["(none)"] + num_cols_list, key="filter_num_col")
                    if filter_num_col != "(none)":
                        col_min = float(df[filter_num_col].min())
                        col_max = float(df[filter_num_col].max())
                        if col_min < col_max:
                            filter_range = st.slider("Range", col_min, col_max, (col_min, col_max), key="filter_range")
                            plot_df = plot_df[(plot_df[filter_num_col] >= filter_range[0]) & (plot_df[filter_num_col] <= filter_range[1])]
                            filter_suffix_parts.append(f"{filter_num_col}: {filter_range[0]:.1f}–{filter_range[1]:.1f}")

            filter_suffix = (" · " + " | ".join(filter_suffix_parts)) if filter_suffix_parts else ""

            st.markdown('<div style="height:0.3rem"></div>', unsafe_allow_html=True)
            generate_clicked = st.button("📊 Generate Chart", use_container_width=True, key="generate_chart", type="primary")

        with right_col:
            st.markdown('<div class="section-header">Chart Output</div>', unsafe_allow_html=True)

            if not generate_clicked and "last_export_fig" not in st.session_state:
                st.info("Configure your chart on the left and click **📊 Generate Chart**.")

            if generate_clicked and not chart_ok:
                st.warning(f"⚠️ {chart_warn}")

            elif generate_clicked and chart_ok:
                st.session_state.pop("last_export_fig", None)

                def make_title(base):
                    t = chart_title if chart_title else base
                    return t + filter_suffix if filter_suffix else t

                if plot_df.empty:
                    st.warning("⚠️ No data after applying filters. Adjust or clear them.")
                else:
                    st.caption(f"{len(plot_df):,} rows shown out of {len(df):,}")
                    export_fig = None

                    hist_group_col = (filter_cat_col
                                      if filter_cat_col != "(none)" and len(filter_cat_vals) > 1
                                      else None)

                    try:
                        if chart_type == "Histogram":
                            hist_df = plot_df[[x_col] + ([hist_group_col] if hist_group_col else [])].dropna(subset=[x_col]).copy()
                            _all_vals = hist_df[x_col].dropna()
                            _bin_edges = np.linspace(_all_vals.min(), _all_vals.max(), bins + 1)

                            if hist_group_col:
                                pfig = px.histogram(hist_df, x=x_col, color=hist_group_col,
                                                    nbins=bins, barmode="overlay", opacity=0.68,
                                                    color_discrete_sequence=PALETTE,
                                                    title=make_title(f"Distribution of {x_col}"),
                                                    category_orders={hist_group_col: _order_cats(hist_df[hist_group_col])})
                                pfig.update_traces(marker_line_color="white", marker_line_width=0.7)
                                pfig.update_layout(legend_title_text=hist_group_col, bargap=0.08)
                                _apply_pf_axes(pfig, showlegend=True)
                            else:
                                pfig = px.histogram(hist_df, x=x_col, nbins=bins,
                                                    color_discrete_sequence=PALETTE,
                                                    title=make_title(f"Distribution of {x_col}"))
                                pfig.update_traces(marker_line_color="white", marker_line_width=0.7)
                                pfig.update_layout(showlegend=False, bargap=0.08)
                                _apply_pf_axes(pfig, showlegend=False)
                            st.plotly_chart(pfig, use_container_width=True)

                            fig_e, ax_e = plt.subplots(figsize=(10, 5))
                            if hist_group_col:
                                for idx, gv in enumerate(_order_cats(hist_df[hist_group_col])):
                                    gdata = hist_df.loc[hist_df[hist_group_col].astype(str) == gv, x_col].dropna()
                                    ax_e.hist(gdata, bins=_bin_edges, color=PALETTE[idx % len(PALETTE)],
                                              alpha=0.68, edgecolor="white", linewidth=0.8, label=str(gv), zorder=3)
                                _style_legend(ax_e, title=hist_group_col)
                            else:
                                ax_e.hist(_all_vals, bins=_bin_edges, color=PALETTE[0],
                                          alpha=0.88, edgecolor="white", linewidth=0.8, zorder=3)
                            _styled_ax(ax_e, make_title(f"Distribution of {x_col}"), x_col, "Count")
                            fig_e.patch.set_facecolor("white")
                            fig_e.tight_layout(pad=1.5)
                            export_fig = fig_e

                        elif chart_type == "Box Plot":
                            grp_col = x_col if (x_col and x_col != "(none)") else None
                            if grp_col:
                                cat_ord = _order_cats(plot_df[grp_col])
                                pfig = px.box(plot_df, x=grp_col, y=y_col,
                                              title=make_title(f"{y_col} by {grp_col}"),
                                              color=grp_col, color_discrete_sequence=PALETTE,
                                              category_orders={grp_col: cat_ord})
                                pfig.update_traces(line=dict(width=1.1), marker=dict(size=5))
                            else:
                                pfig = px.box(plot_df, y=y_col, title=make_title(f"Box Plot: {y_col}"))
                                pfig.update_traces(marker_color=PALETTE[0], line=dict(width=1.1))
                            _apply_pf_axes(pfig, showlegend=bool(grp_col))
                            st.plotly_chart(pfig, use_container_width=True)

                            _bp_groups = ([g for g in plot_df[grp_col].dropna().unique()] if grp_col else None)
                            _n_box = len(_bp_groups) if _bp_groups else 1
                            fig_e, ax_e = _make_mpl_fig(_n_box, 1)
                            if grp_col and _bp_groups:
                                data_list = [plot_df[plot_df[grp_col] == g][y_col].dropna().values for g in _bp_groups]
                                bp = ax_e.boxplot(data_list, patch_artist=True,
                                                  labels=[str(g) for g in _bp_groups],
                                                  medianprops=dict(color=TEXT, linewidth=2))
                                for patch, col in zip(bp["boxes"], PALETTE):
                                    patch.set_facecolor(col); patch.set_alpha(0.82)
                                for w in bp["whiskers"]: w.set_color("#6A6E73")
                                for c in bp["caps"]:     c.set_color("#6A6E73")
                                _auto_rot(ax_e, [str(g) for g in _bp_groups])
                            else:
                                bp = ax_e.boxplot(plot_df[y_col].dropna().values, patch_artist=True,
                                                  medianprops=dict(color=TEXT, linewidth=2))
                                bp["boxes"][0].set_facecolor(PALETTE[0]); bp["boxes"][0].set_alpha(0.82)
                            _styled_ax(ax_e, make_title(f"Box Plot: {y_col}"), grp_col or "", y_col)
                            fig_e.patch.set_facecolor("white"); fig_e.tight_layout(pad=1.5)
                            export_fig = fig_e

                        elif chart_type == "Scatter Plot":
                            sc_df = plot_df.copy()
                            c = color_col if color_col and color_col != "(none)" else None
                            if agg_method and agg_method != "None (raw)":
                                grp = [x_col] + ([c] if c else [])
                                sc_df = sc_df.groupby(grp, dropna=False)[y_col].agg(agg_method).reset_index()
                            pfig = px.scatter(sc_df, x=x_col, y=y_col, color=c,
                                              title=make_title(f"{x_col} vs {y_col}"),
                                              opacity=0.78, color_discrete_sequence=PALETTE)
                            pfig.update_traces(marker=dict(size=8, line=dict(width=0.6, color="white")))
                            _apply_pf_axes(pfig, showlegend=bool(c))
                            st.plotly_chart(pfig, use_container_width=True)

                            fig_e, ax_e = plt.subplots(figsize=(10, 5))
                            if c and c in sc_df.columns:
                                for idx, (gv, gdf) in enumerate(sc_df.groupby(c, dropna=False)):
                                    ax_e.scatter(gdf[x_col], gdf[y_col],
                                                 color=PALETTE[idx % len(PALETTE)], alpha=0.78,
                                                 label=str(gv), s=55, edgecolors="white", linewidths=0.5, zorder=3)
                                _style_legend(ax_e, title=c)
                            else:
                                ax_e.scatter(sc_df[x_col], sc_df[y_col], color=PALETTE[0],
                                             alpha=0.78, s=55, edgecolors="white", linewidths=0.5, zorder=3)
                            _styled_ax(ax_e, make_title(f"{x_col} vs {y_col}"), x_col, y_col)
                            fig_e.patch.set_facecolor("white"); fig_e.tight_layout(pad=1.5)
                            export_fig = fig_e

                        elif chart_type == "Line Chart":
                            c = color_col if color_col and color_col != "(none)" else None
                            grp = [x_col] + ([c] if c else [])
                            try:
                                ln_df = plot_df.groupby(grp, dropna=False)[y_col].agg(agg_method).reset_index()
                                ln_df = _sort_line(ln_df, x_col)
                            except Exception:
                                ln_df = plot_df.copy()
                            pfig = px.line(ln_df, x=x_col, y=y_col, color=c,
                                           title=make_title(f"{y_col} over {x_col}"),
                                           markers=True, color_discrete_sequence=PALETTE)
                            pfig.update_traces(line=dict(width=2.5), marker=dict(size=7))
                            _apply_pf_axes(pfig, showlegend=bool(c))
                            st.plotly_chart(pfig, use_container_width=True)

                            fig_e, ax_e = plt.subplots(figsize=(10, 5))
                            if c and c in ln_df.columns:
                                for idx, (gv, gdf) in enumerate(ln_df.groupby(c, dropna=False)):
                                    gdf_s = _sort_line(gdf, x_col)
                                    ax_e.plot(gdf_s[x_col].astype(str), gdf_s[y_col],
                                              color=PALETTE[idx % len(PALETTE)], marker="o",
                                              linewidth=2.5, markersize=7, label=str(gv), zorder=3)
                                _style_legend(ax_e, title=c)
                                _all_x_vals = ln_df.drop_duplicates(subset=[x_col]).pipe(lambda f: _sort_line(f, x_col))[x_col].astype(str).tolist()
                            else:
                                ld = _sort_line(ln_df, x_col)
                                ax_e.plot(ld[x_col].astype(str), ld[y_col], color=PALETTE[0],
                                          marker="o", linewidth=2.5, markersize=7, zorder=3)
                                _all_x_vals = ld[x_col].astype(str).tolist()
                            _auto_rot(ax_e, _all_x_vals)
                            _styled_ax(ax_e, make_title(f"{y_col} over {x_col}"), x_col, y_col)
                            fig_e.patch.set_facecolor("white"); fig_e.tight_layout(pad=1.5)
                            export_fig = fig_e

                        elif chart_type == "Bar Chart (Grouped)":
                            c = color_col if color_col and color_col != "(none)" else None
                            grp = [x_col] + ([c] if c else [])
                            bar_df = plot_df.groupby(grp, dropna=False)[y_col].agg(agg_method).reset_index()

                            avail  = bar_df[x_col].nunique()
                            eff_n  = min(top_n, avail)
                            top_vals = bar_df.groupby(x_col, dropna=False)[y_col].sum().nlargest(eff_n).index
                            bar_df   = bar_df[bar_df[x_col].isin(top_vals)].copy()
                            x_order  = bar_df.groupby(x_col, dropna=False)[y_col].sum().sort_values(ascending=False).index.tolist()
                            t = make_title(f"{agg_method.title()} of {y_col} by {x_col} (Top {eff_n})")

                            pfig = px.bar(bar_df, x=x_col, y=y_col, color=c, barmode="group",
                                          title=t, color_discrete_sequence=PALETTE,
                                          category_orders={x_col: [str(v) for v in x_order]})
                            pfig.update_traces(marker_line_width=0)
                            if eff_n > 12:
                                pfig.update_layout(bargap=0.15, bargroupgap=0.05)
                            _apply_pf_axes(pfig, showlegend=bool(c))
                            st.plotly_chart(pfig, use_container_width=True)

                            cats   = [str(v) for v in x_order]
                            g_list = (list(dict.fromkeys(bar_df[c].dropna().astype(str).tolist()))
                                      if c and c in bar_df.columns else [])
                            n_g    = max(len(g_list), 1)
                            fig_e, ax_e = _make_mpl_fig(len(cats), n_g)
                            x_idx  = np.arange(len(cats))

                            if c and g_list:
                                bar_w        = 0.75 / n_g
                                offset_start = -0.75 / 2 + bar_w / 2
                                for i, gv in enumerate(g_list):
                                    gdf = bar_df[bar_df[c].astype(str) == gv].copy()
                                    gdf["__xk"] = gdf[x_col].astype(str)
                                    gs   = gdf.groupby("__xk")[y_col].sum()
                                    vals = [float(gs.get(cat, 0)) for cat in cats]
                                    ax_e.bar(x_idx + offset_start + i * bar_w, vals,
                                             width=bar_w * 0.92, color=PALETTE[i % len(PALETTE)],
                                             alpha=0.90, label=str(gv), zorder=3)
                                ax_e.set_xticks(x_idx)
                                _auto_rot(ax_e, cats)
                                _style_legend(ax_e, title=c)
                            else:
                                vals = bar_df.groupby(x_col, dropna=False)[y_col].sum().reindex(x_order, fill_value=0).tolist()
                                ax_e.bar(cats, vals, color=PALETTE[0], alpha=0.90, zorder=3, width=0.6)
                                _auto_rot(ax_e, cats)
                            _styled_ax(ax_e, t, x_col, y_col)
                            fig_e.patch.set_facecolor("white"); fig_e.tight_layout(pad=1.5)
                            export_fig = fig_e

                        elif chart_type == "Heatmap / Correlation Matrix":
                            corr = plot_df[heatmap_cols].corr()
                            sz   = max(8, len(heatmap_cols))
                            fig_e, ax_d = plt.subplots(figsize=(sz, max(6, sz - 1)))
                            cmap = sns.diverging_palette(220, 10, as_cmap=True)
                            sns.heatmap(corr, annot=(len(heatmap_cols) <= 12), fmt=".2f",
                                        cmap=cmap, center=0, ax=ax_d,
                                        linewidths=0.6, linecolor="#E8E8E8",
                                        annot_kws={"size": 9, "color": TEXT},
                                        cbar_kws={"shrink": 0.8, "label": "Correlation"})
                            ax_d.set_title(make_title("Correlation Matrix"), fontsize=14,
                                           fontweight="bold", pad=14, color=TEXT)
                            ax_d.set_xticklabels(ax_d.get_xticklabels(), fontweight="bold",
                                                 fontsize=10, rotation=40, ha="right", color=TEXT)
                            ax_d.set_yticklabels(ax_d.get_yticklabels(), fontweight="bold",
                                                 fontsize=10, rotation=0, color=TEXT)
                            fig_e.patch.set_facecolor("white"); fig_e.tight_layout(pad=1.5)
                            st.pyplot(fig_e, use_container_width=True)
                            export_fig = fig_e

                        if export_fig is not None:
                            st.session_state["last_export_fig"] = export_fig

                    except Exception as e:
                        st.error(f"⚠️ Could not generate chart: {e}")
                    finally:
                        plt.close("all")

            if "last_export_fig" in st.session_state:
                buf = io.BytesIO()
                try:
                    st.session_state["last_export_fig"].savefig(
                        buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
                    buf.seek(0)
                    st.download_button("📥 Export Chart as PNG", buf, "chart.png",
                                       "image/png", key="chart_png")
                except Exception:
                    pass


# ══════════════════════════════════════════════
# PAGE D — EXPORT & REPORT
# ══════════════════════════════════════════════
with tab_d:
    st.markdown("## 💾 Export & Report")

    if st.session_state.df_working is None:
        st.info("Please upload a dataset first.")
    else:
        df_export = st.session_state.df_working

        st.markdown('<div class="section-header">Dataset Export</div>', unsafe_allow_html=True)
        st.caption(f"Current working dataset: **{df_export.shape[0]:,} rows × {df_export.shape[1]} cols**")

        d1, d2 = st.columns(2)
        with d1:
            csv_bytes = df_export.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv_bytes, "cleaned_data.csv", "text/csv", use_container_width=True)
        with d2:
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                df_export.to_excel(writer, index=False)
            excel_buf.seek(0)
            st.download_button("📥 Download Excel", excel_buf, "cleaned_data.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True)

        st.markdown("---")

        st.markdown('<div class="section-header">Transformation Report</div>', unsafe_allow_html=True)

        if not st.session_state.recipe:
            st.info("No transformations applied yet.")
        else:
            st.caption(f"{len(st.session_state.recipe)} transformation(s) applied")
            report_rows = []
            for step in st.session_state.recipe:
                source_label = "🤖 AI" if step.get("source") == "ai_suggested" else "👤 Manual"
                report_rows.append({
                    "Step": step["step"],
                    "Operation": step["operation"],
                    "Parameters": json.dumps(step["parameters"]),
                    "Affected Columns": ", ".join(str(c) for c in step["affected_columns"]),
                    "Timestamp": step["timestamp"],
                    "Source": source_label,
                })
            report_df = pd.DataFrame(report_rows)
            show_table(report_df)

            st.markdown("---")
            st.markdown('<div class="section-header">Recipe Export (JSON)</div>', unsafe_allow_html=True)
            recipe_json = json.dumps(st.session_state.recipe, indent=2)
            st.code(recipe_json, language="json")
            st.download_button(
                "📥 Download recipe.json",
                recipe_json.encode("utf-8"),
                "recipe.json",
                "application/json",
                use_container_width=True
            )

            st.markdown("---")
            st.markdown('<div class="section-header">Full Report Export</div>', unsafe_allow_html=True)
            report_text = f"DataStudio Transformation Report\nGenerated: {datetime.now().isoformat(timespec='seconds')}\n"
            report_text += f"Dataset shape: {df_export.shape[0]} rows × {df_export.shape[1]} cols\n"
            report_text += "=" * 60 + "\n"
            for step in st.session_state.recipe:
                report_text += (
                    f"\nStep {step['step']}: {step['operation']}\n"
                    f"  Parameters : {json.dumps(step['parameters'])}\n"
                    f"  Columns    : {step['affected_columns']}\n"
                    f"  Timestamp  : {step['timestamp']}\n"
                    f"  Source     : {step.get('source', 'manual')}\n"
                )
            st.download_button(
                "📥 Download Full Report (.txt)",
                report_text.encode("utf-8"),
                "transformation_report.txt",
                "text/plain",
                use_container_width=True
            )
