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

/* push tab bar down so labels are never clipped */
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
/* Tighter spacing inside left viz panel */
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

/* ── Dataframe global alignment ── */
/* Left-align all header cells */
[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] [data-testid="glideDataEditor"] .header-cell,
.stDataFrame thead th {
    text-align: left !important;
    justify-content: flex-start !important;
}
/* Left-align text/string cells */
[data-testid="stDataFrame"] td,
.stDataFrame tbody td {
    text-align: left !important;
}
/* Right-align numeric cells (Streamlit adds dvn-* class for number columns) */
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
    """Full reset — clears everything including uploaded file, returns to blank page."""
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
    """Reset All — restores working df to original, clears actions. Keeps file loaded."""
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
    """Render a dataframe with consistent alignment:
    text columns left-aligned, numeric columns right-aligned."""
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

    # Transformation Log
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

   
        # AI Toggle
    st.markdown("### ⚙️ Settings")
    ai_toggle = st.toggle("Enable AI Assistant", value=st.session_state.ai_enabled)
    st.session_state.ai_enabled = ai_toggle

    if ai_toggle:
        # API key from secrets; fallback to session state for local dev
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

        # ── Panel 1: Shape
        st.markdown('<div class="section-header">1 · Shape</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><h4>Rows</h4><p>{df.shape[0]:,}</p></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><h4>Columns</h4><p>{df.shape[1]}</p></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><h4>Total Cells</h4><p>{df.shape[0] * df.shape[1]:,}</p></div>', unsafe_allow_html=True)

        # ── Panel 2: Column Names & Dtypes
        st.markdown('<div class="section-header">2 · Column Names & Inferred Dtypes</div>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype": df.dtypes.astype(str).values,
            "Non-Null Count": df.notnull().sum().values,
            "Sample Value": [str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0] > 0 else "—" for c in df.columns],
        })
        show_table(dtype_df)

        # ── Panel 3: Basic Summary Stats
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

        # ── Panel 4: Missing Values
        st.markdown('<div class="section-header">4 · Missing Values by Column</div>', unsafe_allow_html=True)
        miss_df = profile_missing(df)
        miss_df_filtered = miss_df[miss_df["Missing Count"] > 0]
        if miss_df_filtered.empty:
            st.success("✅ No missing values detected.")
        else:
            show_table(miss_df_filtered)

        # ── Panel 5: Duplicates
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

        # ── AI Assistant (if enabled)
        if st.session_state.ai_enabled:
            st.markdown('<div class="section-header">🤖 AI Cleaning Assistant</div>', unsafe_allow_html=True)
            st.markdown('<div class="disclaimer">⚠️ AI suggestions may be imperfect. Always review before applying.</div>', unsafe_allow_html=True)

            # Init chat history in session state
            if "ai_chat_history" not in st.session_state:
                st.session_state["ai_chat_history"] = []
            if "ai_pending_suggestions" not in st.session_state:
                st.session_state["ai_pending_suggestions"] = []

            # Render chat history
            for msg in st.session_state["ai_chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Render pending suggestions (confirm / reject per item)
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
                                    # Execute the operation directly
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

            # Chat input box
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
                        # Strip markdown fences if model added them
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

        # ─────────────────────────────────────
        # 4.1 MISSING VALUES
        # ─────────────────────────────────────
        with st.expander("4.1 · Missing Values", expanded=False):
            miss_summary = profile_missing(df)
            cols_with_missing = miss_summary[miss_summary["Missing Count"] > 0]["Column"].tolist()

            show_table(miss_summary[miss_summary["Missing Count"] > 0])

            if not cols_with_missing:
                st.success("No missing values.")
            else:
                mv_col = st.selectbox("Select column", cols_with_missing, key="mv_col")
                col_dtype = str(df[mv_col].dtype)
                is_numeric = pd.api.types.is_numeric_dtype(df[mv_col])
                is_datetime = pd.api.types.is_datetime64_any_dtype(df[mv_col])

                action = st.selectbox("Action", [
                    "Drop rows with missing",
                    "Drop column if missing > threshold %",
                    "Replace with constant",
                    "Replace with mean",
                    "Replace with median",
                    "Replace with mode",
                    "Replace with most frequent",
                    "Forward fill",
                    "Backward fill",
                ], key="mv_action")

                extra = {}
                if action == "Replace with constant":
                    extra["value"] = st.text_input("Constant value", key="mv_const")
                if action == "Drop column if missing > threshold %":
                    extra["threshold"] = st.slider("Threshold %", 0, 100, 50, key="mv_thresh")

                # Before preview
                before_rows = df.shape[0]
                before_missing = int(df[mv_col].isnull().sum())

                if st.button("Apply", key="mv_apply"):
                    push_history()
                    try:
                        wdf = st.session_state.df_working
                        if action == "Drop rows with missing":
                            st.session_state.df_working = wdf.dropna(subset=[mv_col])
                        elif action == "Drop column if missing > threshold %":
                            pct = wdf[mv_col].isnull().mean() * 100
                            if pct > extra["threshold"]:
                                st.session_state.df_working = wdf.drop(columns=[mv_col])
                        elif action == "Replace with constant":
                            st.session_state.df_working[mv_col] = wdf[mv_col].fillna(extra["value"])
                        elif action == "Replace with mean" and is_numeric:
                            st.session_state.df_working[mv_col] = wdf[mv_col].fillna(wdf[mv_col].mean())
                        elif action == "Replace with median" and is_numeric:
                            st.session_state.df_working[mv_col] = wdf[mv_col].fillna(wdf[mv_col].median())
                        elif action in ("Replace with mode", "Replace with most frequent"):
                            st.session_state.df_working[mv_col] = wdf[mv_col].fillna(wdf[mv_col].mode()[0])
                        elif action == "Forward fill":
                            st.session_state.df_working[mv_col] = wdf[mv_col].ffill()
                        elif action == "Backward fill":
                            st.session_state.df_working[mv_col] = wdf[mv_col].bfill()

                        after_rows = st.session_state.df_working.shape[0]
                        after_missing = int(st.session_state.df_working[mv_col].isnull().sum()) if mv_col in st.session_state.df_working.columns else 0
                        log_step("fill_missing", {"column": mv_col, "action": action, **extra}, [mv_col])
                        st.success(f"✅ Done. Rows: {before_rows} → {after_rows} | Missing in '{mv_col}': {before_missing} → {after_missing}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        undo_last()

        # ─────────────────────────────────────
        # 4.2 DUPLICATES
        # ─────────────────────────────────────
        with st.expander("4.2 · Duplicates", expanded=False):
            dup_mode = st.radio("Detect duplicates by", ["All columns (full-row)", "Subset of columns"], key="dup_mode")
            subset = None
            if dup_mode == "Subset of columns":
                subset = st.multiselect("Select key columns", df.columns.tolist(), key="dup_subset")

            dup_count = df.duplicated(subset=subset).sum()
            st.metric("Duplicate rows detected", dup_count)

            if dup_count > 0:
                if st.checkbox("Show duplicate groups"):
                    show_table(df[df.duplicated(subset=subset, keep=False)].sort_values(
                        by=subset or df.columns.tolist()))

                keep = st.selectbox("Keep which duplicate?", ["first", "last"], key="dup_keep")
                if st.button("Remove Duplicates", key="dup_apply"):
                    push_history()
                    st.session_state.df_working = st.session_state.df_working.drop_duplicates(subset=subset, keep=keep)
                    log_step("drop_duplicates", {"subset": subset, "keep": keep}, subset or ["all columns"])
                    st.success(f"✅ Removed {dup_count} duplicate rows.")
                    st.rerun()

        # ─────────────────────────────────────
        # 4.3 DATA TYPES & PARSING
        # ─────────────────────────────────────
        with st.expander("4.3 · Data Types & Parsing", expanded=False):
            dt_col = st.selectbox("Select column", df.columns.tolist(), key="dt_col")
            current_dtype = str(df[dt_col].dtype)
            st.write(f"Current dtype: **{current_dtype}**")

            target_type = st.selectbox("Convert to", ["numeric", "categorical (string)", "datetime"], key="dt_target")

            dt_fmt = None
            dirty_clean = False
            if target_type == "datetime":
                dt_fmt = st.text_input("Datetime format (leave blank for auto)", placeholder="%Y-%m-%d", key="dt_fmt")
            if target_type == "numeric":
                dirty_clean = st.checkbox("Clean dirty numerics (remove $, commas, etc.)", key="dt_dirty")

            if st.button("Convert", key="dt_apply"):
                push_history()
                try:
                    wdf = st.session_state.df_working
                    if target_type == "numeric":
                        col_data = wdf[dt_col].astype(str)
                        if dirty_clean:
                            col_data = col_data.str.replace(r"[^\d.\-]", "", regex=True)
                        st.session_state.df_working[dt_col] = pd.to_numeric(col_data, errors="coerce")
                    elif target_type == "categorical (string)":
                        st.session_state.df_working[dt_col] = wdf[dt_col].astype(str)
                    elif target_type == "datetime":
                        fmt = dt_fmt if dt_fmt else None
                        st.session_state.df_working[dt_col] = pd.to_datetime(wdf[dt_col], format=fmt, errors="coerce")
                    log_step("convert_dtype", {"column": dt_col, "target": target_type, "format": dt_fmt}, [dt_col])
                    st.success(f"✅ Converted '{dt_col}' to {target_type}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    undo_last()

        # ─────────────────────────────────────
        # 4.4 CATEGORICAL TOOLS
        # ─────────────────────────────────────
        with st.expander("4.4 · Categorical Tools", expanded=False):
            cat_cols_list = categorical_cols(df)
            if not cat_cols_list:
                st.info("No categorical columns detected.")
            else:
                cat_col = st.selectbox("Select column", cat_cols_list, key="cat_col")
                cat_action = st.selectbox("Action", [
                    "Trim whitespace",
                    "Lowercase",
                    "Title case",
                    "Map / Replace values",
                    "Group rare categories into 'Other'",
                    "One-hot encode",
                ], key="cat_action")

                params = {}
                if cat_action == "Map / Replace values":
                    st.write("Enter mapping (one per line: old_value=new_value)")
                    mapping_text = st.text_area("Mapping", key="cat_map_text", placeholder="Apt=Apartment\nHse=House")
                    unmatched = st.selectbox("Unmatched values", ["Keep as-is", "Set to Other"], key="cat_unmatched")
                    params = {"mapping_text": mapping_text, "unmatched": unmatched}

                if cat_action == "Group rare categories into 'Other'":
                    freq_thresh = st.slider("Frequency threshold (%)", 1, 20, 5, key="cat_rare_thresh")
                    params = {"threshold": freq_thresh}

                if st.button("Apply", key="cat_apply"):
                    push_history()
                    try:
                        wdf = st.session_state.df_working
                        if cat_action == "Trim whitespace":
                            wdf[cat_col] = wdf[cat_col].astype(str).str.strip()
                        elif cat_action == "Lowercase":
                            wdf[cat_col] = wdf[cat_col].astype(str).str.lower()
                        elif cat_action == "Title case":
                            wdf[cat_col] = wdf[cat_col].astype(str).str.title()
                        elif cat_action == "Map / Replace values":
                            mapping = {}
                            for line in mapping_text.strip().split("\n"):
                                if "=" in line:
                                    k, v = line.split("=", 1)
                                    mapping[k.strip()] = v.strip()
                            if params["unmatched"] == "Set to Other":
                                wdf[cat_col] = wdf[cat_col].apply(lambda x: mapping.get(str(x), "Other"))
                            else:
                                wdf[cat_col] = wdf[cat_col].map(lambda x: mapping.get(str(x), x))
                        elif cat_action == "Group rare categories into 'Other'":
                            freq = wdf[cat_col].value_counts(normalize=True) * 100
                            rare = freq[freq < freq_thresh].index
                            wdf[cat_col] = wdf[cat_col].apply(lambda x: "Other" if x in rare else x)
                        elif cat_action == "One-hot encode":
                            dummies = pd.get_dummies(wdf[cat_col], prefix=cat_col)
                            wdf = pd.concat([wdf.drop(columns=[cat_col]), dummies], axis=1)

                        st.session_state.df_working = wdf
                        log_step("categorical_tool", {"column": cat_col, "action": cat_action, **params}, [cat_col])
                        st.success(f"✅ Applied '{cat_action}' to '{cat_col}'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        undo_last()

        # ─────────────────────────────────────
        # 4.5 NUMERIC CLEANING / OUTLIERS
        # ─────────────────────────────────────
        with st.expander("4.5 · Numeric Cleaning & Outliers", expanded=False):
            num_cols_list = numeric_cols(df)
            if not num_cols_list:
                st.info("No numeric columns detected.")
            else:
                out_col = st.selectbox("Select column", num_cols_list, key="out_col")
                out_method = st.radio("Detection method", ["IQR", "Z-score"], key="out_method")
                out_action = st.selectbox("Action", ["Do nothing (just show)", "Cap / Winsorize at quantiles", "Remove outlier rows"], key="out_action")

                col_data = df[out_col].dropna()
                if out_method == "IQR":
                    Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                    outliers = col_data[(col_data < lower) | (col_data > upper)]
                else:
                    z = np.abs(stats.zscore(col_data))
                    outliers = col_data[z > 3]
                    lower, upper = col_data.mean() - 3 * col_data.std(), col_data.mean() + 3 * col_data.std()

                st.metric(f"Outliers detected ({out_method})", len(outliers))
                st.write(f"Bounds: **{lower:.2f}** — **{upper:.2f}**")

                q_low, q_high = 0.05, 0.95
                if out_action == "Cap / Winsorize at quantiles":
                    q_low = st.slider("Lower quantile", 0.0, 0.2, 0.05, 0.01, key="out_qlow")
                    q_high = st.slider("Upper quantile", 0.8, 1.0, 0.95, 0.01, key="out_qhigh")

                if out_action != "Do nothing (just show)":
                    if st.button("Apply", key="out_apply"):
                        push_history()
                        try:
                            wdf = st.session_state.df_working
                            before = wdf.shape[0]
                            if out_action == "Remove outlier rows":
                                if out_method == "IQR":
                                    mask = (wdf[out_col] >= lower) & (wdf[out_col] <= upper)
                                else:
                                    z = np.abs(stats.zscore(wdf[out_col].dropna()))
                                    valid_idx = wdf[out_col].dropna().index[z <= 3]
                                    mask = wdf.index.isin(valid_idx) | wdf[out_col].isnull()
                                st.session_state.df_working = wdf[mask]
                                removed = before - st.session_state.df_working.shape[0]
                                st.success(f"✅ Removed {removed} outlier rows.")
                            elif out_action == "Cap / Winsorize at quantiles":
                                lo = wdf[out_col].quantile(q_low)
                                hi = wdf[out_col].quantile(q_high)
                                st.session_state.df_working[out_col] = wdf[out_col].clip(lo, hi)
                                st.success(f"✅ Capped '{out_col}' to [{lo:.2f}, {hi:.2f}].")
                            log_step("outlier_handling", {"column": out_col, "method": out_method, "action": out_action}, [out_col])
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                            undo_last()

        # ─────────────────────────────────────
        # 4.6 NORMALIZATION / SCALING
        # ─────────────────────────────────────
        with st.expander("4.6 · Normalization & Scaling", expanded=False):
            num_cols_list = numeric_cols(df)
            if not num_cols_list:
                st.info("No numeric columns detected.")
            else:
                scale_cols = st.multiselect("Select columns to scale", num_cols_list, key="scale_cols")
                scale_method = st.selectbox("Method", ["Min-Max Scaling", "Z-score Standardization"], key="scale_method")

                if scale_cols:
                    before_stats = df[scale_cols].describe().T[["mean", "std", "min", "max"]].round(3)
                    st.write("**Before stats:**")
                    show_table(before_stats)

                if scale_cols and st.button("Apply Scaling", key="scale_apply"):
                    push_history()
                    try:
                        wdf = st.session_state.df_working
                        for col in scale_cols:
                            if scale_method == "Min-Max Scaling":
                                mn, mx = wdf[col].min(), wdf[col].max()
                                wdf[col] = (wdf[col] - mn) / (mx - mn) if mx != mn else 0
                            else:
                                mean, std = wdf[col].mean(), wdf[col].std()
                                wdf[col] = (wdf[col] - mean) / std if std != 0 else 0
                        st.session_state.df_working = wdf
                        after_stats = wdf[scale_cols].describe().T[["mean", "std", "min", "max"]].round(3)
                        st.write("**After stats:**")
                        show_table(after_stats)
                        log_step("scale_columns", {"method": scale_method, "columns": scale_cols}, scale_cols)
                        st.success(f"✅ Scaled {len(scale_cols)} column(s).")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        undo_last()

        # ─────────────────────────────────────
        # 4.7 COLUMN OPERATIONS
        # ─────────────────────────────────────
        with st.expander("4.7 · Column Operations", expanded=False):
            col_op = st.selectbox("Operation", [
                "Rename column",
                "Drop columns",
                "Create new column (formula)",
                "Bin numeric column",
            ], key="col_op")

            if col_op == "Rename column":
                rename_col = st.selectbox("Column to rename", df.columns.tolist(), key="rename_col")
                new_name = st.text_input("New name", key="rename_new")
                if st.button("Rename", key="rename_apply") and new_name:
                    push_history()
                    st.session_state.df_working = st.session_state.df_working.rename(columns={rename_col: new_name})
                    log_step("rename_column", {"from": rename_col, "to": new_name}, [rename_col])
                    st.success(f"✅ Renamed '{rename_col}' → '{new_name}'.")
                    st.rerun()

            elif col_op == "Drop columns":
                drop_cols = st.multiselect("Select columns to drop", df.columns.tolist(), key="drop_cols")
                if drop_cols and st.button("Drop", key="drop_apply"):
                    push_history()
                    st.session_state.df_working = st.session_state.df_working.drop(columns=drop_cols)
                    log_step("drop_columns", {"columns": drop_cols}, drop_cols)
                    st.success(f"✅ Dropped {len(drop_cols)} column(s).")
                    st.rerun()

            elif col_op == "Create new column (formula)":
                new_col_name = st.text_input("New column name", key="formula_name")
                formula = st.text_input("Formula (use column names as variables)",
                                         placeholder="price / area   or   price - price.mean()", key="formula_expr")
                st.caption("Available: all column names, numpy as np, pandas as pd")
                if new_col_name and formula and st.button("Create", key="formula_apply"):
                    push_history()
                    try:
                        local_vars = {col: st.session_state.df_working[col] for col in st.session_state.df_working.columns}
                        local_vars["np"] = np
                        local_vars["pd"] = pd
                        st.session_state.df_working[new_col_name] = eval(formula, {"__builtins__": {}}, local_vars)
                        log_step("create_column", {"name": new_col_name, "formula": formula}, [new_col_name])
                        st.success(f"✅ Created column '{new_col_name}'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Formula error: {e}")
                        undo_last()

            elif col_op == "Bin numeric column":
                num_cols_list = numeric_cols(df)
                bin_col = st.selectbox("Column to bin", num_cols_list, key="bin_col")
                bin_method = st.radio("Binning method", ["Equal-width", "Quantile"], key="bin_method")
                bin_count = st.slider("Number of bins", 2, 20, 5, key="bin_count")
                bin_new_name = st.text_input("New column name", value=f"{bin_col}_binned", key="bin_new_name")
                if st.button("Bin", key="bin_apply"):
                    push_history()
                    try:
                        if bin_method == "Equal-width":
                            st.session_state.df_working[bin_new_name] = pd.cut(
                                st.session_state.df_working[bin_col], bins=bin_count, include_lowest=True).astype(str)
                        else:
                            st.session_state.df_working[bin_new_name] = pd.qcut(
                                st.session_state.df_working[bin_col], q=bin_count, duplicates="drop").astype(str)
                        log_step("bin_column", {"column": bin_col, "method": bin_method, "bins": bin_count}, [bin_new_name])
                        st.success(f"✅ Created binned column '{bin_new_name}'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        undo_last()

        # ─────────────────────────────────────
        # 4.8 DATA VALIDATION RULES
        # ─────────────────────────────────────
        with st.expander("4.8 · Data Validation Rules", expanded=False):
            val_rule = st.selectbox("Rule type", [
                "Numeric range check",
                "Allowed categories list",
                "Non-null constraint",
            ], key="val_rule")

            violations_df = pd.DataFrame()

            if val_rule == "Numeric range check":
                num_cols_list = numeric_cols(df)
                if num_cols_list:
                    val_col = st.selectbox("Column", num_cols_list, key="val_num_col")
                    v_min = st.number_input("Min allowed", value=float(df[val_col].min()), key="val_min")
                    v_max = st.number_input("Max allowed", value=float(df[val_col].max()), key="val_max")
                    if st.button("Check", key="val_num_check"):
                        violations_df = df[(df[val_col] < v_min) | (df[val_col] > v_max)]
                else:
                    st.info("No numeric columns.")

            elif val_rule == "Allowed categories list":
                cat_cols_list = categorical_cols(df)
                if cat_cols_list:
                    val_col = st.selectbox("Column", cat_cols_list, key="val_cat_col")
                    allowed_text = st.text_area("Allowed values (one per line)", key="val_allowed")
                    if st.button("Check", key="val_cat_check") and allowed_text:
                        allowed = [v.strip() for v in allowed_text.split("\n") if v.strip()]
                        violations_df = df[~df[val_col].isin(allowed)]
                else:
                    st.info("No categorical columns.")

            elif val_rule == "Non-null constraint":
                val_cols = st.multiselect("Columns that must be non-null", df.columns.tolist(), key="val_null_cols")
                if st.button("Check", key="val_null_check") and val_cols:
                    violations_df = df[df[val_cols].isnull().any(axis=1)]

            if not violations_df.empty:
                st.markdown(f'<span class="violation-badge">⚠️ {len(violations_df)} violations found</span>', unsafe_allow_html=True)
                show_table(violations_df.head(200))
                viol_csv = violations_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Export Violations CSV", viol_csv, "violations.csv", "text/csv")
            elif "val_num_check" in st.session_state or "val_cat_check" in st.session_state or "val_null_check" in st.session_state:
                st.success("✅ No violations found.")

        # Current data preview
        st.markdown("---")
        st.markdown('<div class="section-header">Current Working Dataset</div>', unsafe_allow_html=True)
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
        dt_cols_list = datetime_cols(df)
        all_cols = df.columns.tolist()

        # ── VISUALIZATION STUDIO — 30/70 split
        left_col, right_col = st.columns([3, 7], gap="large")

        with left_col:
            # ── Section 1: Chart Type
            st.markdown('<div class="section-header">① Chart Type</div>', unsafe_allow_html=True)
            chart_type = st.selectbox("Select chart type", [
                "Histogram",
                "Box Plot",
                "Scatter Plot",
                "Line Chart",
                "Bar Chart (Grouped)",
                "Heatmap / Correlation Matrix",
            ], key="chart_type")

            # ── Section 2: Axes & Options
            st.markdown('<div style="height:0.2rem;border-top:1px solid #e5e7eb;margin:0.4rem 0 0.3rem 0"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">② Axes & Options</div>', unsafe_allow_html=True)

            x_col = y_col = color_col = agg_method = None
            top_n = None
            bins = 20
            heatmap_cols = []
            chart_ok = True
            chart_warn = ""

            if chart_type == "Histogram":
                if not num_cols_list:
                    chart_ok = False
                    chart_warn = "Histogram requires at least one numeric column. None found."
                else:
                    a1, a2 = st.columns(2)
                    with a1:
                        x_col = st.selectbox("Column (numeric)", num_cols_list, key="hist_x")
                    with a2:
                        bins = st.slider("Bins", 5, 100, 20, key="hist_bins")

            elif chart_type == "Box Plot":
                if not num_cols_list:
                    chart_ok = False
                    chart_warn = "Box Plot requires at least one numeric column."
                else:
                    a1, a2 = st.columns(2)
                    with a1:
                        y_col = st.selectbox("Y (numeric)", num_cols_list, key="box_y")
                    with a2:
                        x_col = st.selectbox("Group by X (optional)", ["(none)"] + cat_cols_list, key="box_x")

            elif chart_type == "Scatter Plot":
                if len(num_cols_list) < 2:
                    chart_ok = False
                    chart_warn = "Scatter Plot requires at least 2 numeric columns."
                else:
                    a1, a2 = st.columns(2)
                    with a1:
                        x_col = st.selectbox("X (numeric)", num_cols_list, key="scatter_x")
                    with a2:
                        y_col = st.selectbox("Y (numeric)", [c for c in num_cols_list if c != x_col] or num_cols_list, key="scatter_y")
                    a3, a4 = st.columns(2)
                    with a3:
                        color_col = st.selectbox("Color by (optional)", ["(none)"] + cat_cols_list, key="scatter_color")
                    with a4:
                        agg_method = st.selectbox("Aggregation", ["None (raw)", "mean", "sum", "count", "median"], key="scatter_agg")

            elif chart_type == "Line Chart":
                all_x_cols = dt_cols_list + num_cols_list + cat_cols_list
                if not all_x_cols or not num_cols_list:
                    chart_ok = False
                    chart_warn = "Line Chart requires at least one X column and one numeric Y column."
                else:
                    a1, a2 = st.columns(2)
                    with a1:
                        x_col = st.selectbox("X axis", all_x_cols, key="line_x")
                    with a2:
                        y_col = st.selectbox("Y (numeric)", num_cols_list, key="line_y")
                    a3, a4 = st.columns(2)
                    with a3:
                        color_col = st.selectbox("Group by (optional)", ["(none)"] + cat_cols_list, key="line_color")
                    with a4:
                        agg_method = st.selectbox("Aggregation", ["sum", "mean", "count", "median"], key="line_agg", index=0)

            elif chart_type == "Bar Chart (Grouped)":
                if not num_cols_list:
                    chart_ok = False
                    chart_warn = "Bar Chart requires at least one numeric column."
                else:
                    x_options = cat_cols_list + num_cols_list
                    a1, a2 = st.columns(2)
                    with a1:
                        x_col = st.selectbox("X (category)", x_options, key="bar_x")
                    with a2:
                        y_col = st.selectbox("Y (numeric)", num_cols_list, key="bar_y")
                    a3, a4 = st.columns(2)
                    with a3:
                        color_col = st.selectbox("Group by (optional)", ["(none)"] + cat_cols_list, key="bar_color")
                    with a4:
                        agg_method = st.selectbox("Aggregation", ["sum", "mean", "count", "median"], key="bar_agg", index=0)
                    top_n = st.slider("Top N categories", 3, 50, 10, key="bar_topn")

            elif chart_type == "Heatmap / Correlation Matrix":
                if len(num_cols_list) < 2:
                    chart_ok = False
                    chart_warn = "Heatmap requires at least 2 numeric columns."
                else:
                    heatmap_cols = st.multiselect(
                        "Select numeric columns",
                        num_cols_list,
                        default=num_cols_list[:min(10, len(num_cols_list))],
                        key="heatmap_cols"
                    )
                    if len(heatmap_cols) < 2:
                        chart_ok = False
                        chart_warn = "Select at least 2 columns for the correlation matrix."

            chart_title = st.text_input("Chart title (optional)", key="chart_title")

            # ── Section 3: Filters (toggle)
            st.markdown('<div style="height:0.2rem;border-top:1px solid #e5e7eb;margin:0.4rem 0 0.3rem 0"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">③ Filters</div>', unsafe_allow_html=True)
            show_filters = st.toggle("⚙ Show Filters", value=False, key="show_filters")

            plot_df = df.copy()
            filter_suffix_parts = []
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
                            vals_label = ", ".join(str(v) for v in filter_cat_vals[:3])
                            if len(filter_cat_vals) > 3:
                                vals_label += f" +{len(filter_cat_vals)-3}"
                            filter_suffix_parts.append(f"{filter_cat_col}: {vals_label}")
                with f2:
                    filter_num_col = st.selectbox("Filter by numeric range", ["(none)"] + num_cols_list, key="filter_num_col")
                    if filter_num_col != "(none)":
                        col_min = float(df[filter_num_col].min())
                        col_max = float(df[filter_num_col].max())
                        if col_min < col_max:
                            filter_range = st.slider("Range", col_min, col_max, (col_min, col_max), key="filter_range")
                            plot_df = plot_df[(plot_df[filter_num_col] >= filter_range[0]) & (plot_df[filter_num_col] <= filter_range[1])]
                            filter_suffix_parts.append(f"{filter_num_col}: {filter_range[0]:.1f}–{filter_range[1]:.1f}")

            # Build effective title (user title takes priority; auto-suffix added when filtered)
            filter_suffix = (" · " + " | ".join(filter_suffix_parts)) if filter_suffix_parts else ""
            effective_title = (chart_title + filter_suffix) if chart_title else None
            # effective_title is None when no manual title → each chart builds its own + suffix

            st.markdown('<div style="height:0.3rem"></div>', unsafe_allow_html=True)
            generate_clicked = st.button("📊 Generate Chart", use_container_width=True, key="generate_chart", type="primary")

        # ── RIGHT COLUMN — Chart Output (70%)
        with right_col:
            st.markdown('<div class="section-header">Chart Output</div>', unsafe_allow_html=True)

            if not generate_clicked and "last_chart_fig" not in st.session_state:
                st.info("Configure your chart on the left and click **📊 Generate Chart**.")

            if not chart_ok and generate_clicked:
                st.warning(f"⚠️ Cannot generate this chart: {chart_warn}")

            elif generate_clicked and chart_ok:
                st.session_state.pop("last_chart_params", None)
                st.session_state.pop("last_export_fig", None)

                # ── PatternFly-inspired chart theme
                # Ordered palette aligned to PatternFly-style categorical colors
                PALETTE = [
                    "#0066CC",  # blue
                    "#4CB140",  # green
                    "#009596",  # cyan
                    "#F0AB00",  # gold
                    "#EC7A08",  # orange
                    "#C9190B",  # red
                    "#519DE9",  # light blue
                    "#7CC674",  # light green
                    "#73C5C5",  # light cyan
                    "#F6D173",  # light gold
                ]

                PF_PLOTLY = PALETTE
                BG = "#FAFAFA"
                GRID = "#E8E8E8"
                TEXT = "#151515"
                MUTED = "#6A6E73"

                def styled_ax(ax, title, xlabel, ylabel):
                    ax.set_title(title, fontsize=14, fontweight="bold", pad=14, color=TEXT)
                    ax.set_xlabel(xlabel, fontweight="bold", fontsize=12, color=TEXT, labelpad=8)
                    ax.set_ylabel(ylabel, fontweight="bold", fontsize=12, color=TEXT, labelpad=8)
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.spines[["left", "bottom"]].set_color("#D2D2D2")
                    ax.set_facecolor(BG)
                    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="--", zorder=0)
                    ax.set_axisbelow(True)
                    ax.tick_params(labelsize=10, colors=MUTED)

                def style_legend(ax, title=None):
                    leg = ax.legend(
                        title=title,
                        fontsize=9,
                        framealpha=0.92,
                        edgecolor="#D2D2D2",
                        facecolor="white",
                        title_fontsize=9,
                    )
                    if leg:
                        leg.get_frame().set_linewidth(0.8)

                def apply_pf_axes(fig, *, showlegend=True):
                    fig.update_layout(
                        paper_bgcolor="white",
                        plot_bgcolor=BG,
                        font=dict(color=TEXT, family="DM Sans", size=12),
                        title=dict(font=dict(size=15, color=TEXT)),
                        legend=dict(
                            bgcolor="white",
                            bordercolor="#D2D2D2",
                            borderwidth=1,
                            font=dict(size=11, color=TEXT),
                            title=dict(font=dict(size=11, color=TEXT)),
                        ),
                        margin=dict(l=60, r=40, t=70, b=60),
                        showlegend=showlegend,
                    )
                    fig.update_xaxes(
                        title_font=dict(size=13, color=TEXT),
                        tickfont=dict(size=11, color=MUTED),
                        gridcolor=GRID,
                        zerolinecolor=GRID,
                        linecolor="#D2D2D2",
                    )
                    fig.update_yaxes(
                        title_font=dict(size=13, color=TEXT),
                        tickfont=dict(size=11, color=MUTED),
                        gridcolor=GRID,
                        zerolinecolor=GRID,
                        linecolor="#D2D2D2",
                    )
                    return fig

                def row_count_text(frame):
                    return f"{len(frame):,} rows shown out of {len(df):,}"

                def order_categories(series):
                    return list(dict.fromkeys(series.dropna().astype(str).tolist()))

                def sort_for_line(frame, x_name):
                    if x_name not in frame.columns:
                        return frame
                    if pd.api.types.is_numeric_dtype(frame[x_name]) or pd.api.types.is_datetime64_any_dtype(frame[x_name]):
                        return frame.sort_values(x_name)
                    ordered = pd.Categorical(frame[x_name].astype(str), categories=order_categories(frame[x_name]), ordered=True)
                    tmp = frame.copy()
                    tmp["_x_order"] = ordered
                    return tmp.sort_values("_x_order").drop(columns="_x_order")

                try:
                    export_fig = None

                    def make_title(base):
                        return f"{base}{filter_suffix}" if filter_suffix else base

                    # Clear chart summary for the user
                    st.caption(row_count_text(plot_df))

                    # Decide whether the active category filter should become a grouped view
                    active_cat_col = locals().get("filter_cat_col", "(none)") if show_filters else "(none)"
                    active_cat_vals = locals().get("filter_cat_vals", []) if show_filters else []
                    hist_group_col = active_cat_col if (active_cat_col != "(none)" and len(active_cat_vals) > 1) else None

                    # ── Histogram ─────────────────────────────────────────
                    if chart_type == "Histogram":
                        hist_df = plot_df[[x_col] + ([hist_group_col] if hist_group_col else [])].dropna(subset=[x_col]).copy()

                        # Display: Plotly for a clearer, more polished view
                        if hist_group_col:
                            pfig = px.histogram(
                                hist_df,
                                x=x_col,
                                color=hist_group_col,
                                nbins=bins,
                                barmode="overlay",
                                opacity=0.68,
                                color_discrete_sequence=PF_PLOTLY,
                                title=make_title(f"Distribution of {x_col}"),
                                category_orders={hist_group_col: order_categories(hist_df[hist_group_col])},
                            )
                            pfig.update_traces(marker_line_color="white", marker_line_width=0.7)
                            pfig.update_layout(legend_title_text=hist_group_col, bargap=0.08)
                            apply_pf_axes(pfig, showlegend=True)
                        else:
                            pfig = px.histogram(
                                hist_df,
                                x=x_col,
                                nbins=bins,
                                color_discrete_sequence=PF_PLOTLY,
                                title=make_title(f"Distribution of {x_col}"),
                            )
                            pfig.update_traces(marker_line_color="white", marker_line_width=0.7)
                            pfig.update_layout(showlegend=False, bargap=0.08)
                            apply_pf_axes(pfig, showlegend=False)
                        st.plotly_chart(pfig, use_container_width=True)

                        # Export: Matplotlib
                        fig_e, ax_e = plt.subplots(figsize=(10, 5))
                        if hist_group_col:
                            groups = order_categories(hist_df[hist_group_col])
                            for idx, grp_val in enumerate(groups):
                                grp_data = hist_df.loc[hist_df[hist_group_col].astype(str) == grp_val, x_col].dropna()
                                ax_e.hist(
                                    grp_data,
                                    bins=bins,
                                    color=PALETTE[idx % len(PALETTE)],
                                    alpha=0.68,
                                    edgecolor="white",
                                    linewidth=0.8,
                                    label=str(grp_val),
                                    zorder=3,
                                )
                            style_legend(ax_e, title=hist_group_col)
                        else:
                            ax_e.hist(
                                hist_df[x_col].dropna(),
                                bins=bins,
                                color=PALETTE[0],
                                alpha=0.88,
                                edgecolor="white",
                                linewidth=0.8,
                                zorder=3,
                            )
                        styled_ax(ax_e, make_title(f"Distribution of {x_col}"), x_col, "Count")
                        fig_e.patch.set_facecolor("white")
                        fig_e.tight_layout()
                        export_fig = fig_e

                    # ── Box Plot ──────────────────────────────────────────
                    elif chart_type == "Box Plot":
                        grp_col = x_col if (x_col and x_col != "(none)") else None
                        if grp_col:
                            cat_order = order_categories(plot_df[grp_col])
                            pfig = px.box(
                                plot_df,
                                x=grp_col,
                                y=y_col,
                                title=make_title(f"{y_col} by {grp_col}"),
                                color=grp_col,
                                color_discrete_sequence=PF_PLOTLY,
                                category_orders={grp_col: cat_order},
                            )
                            pfig.update_traces(line=dict(width=1.1), marker=dict(size=5))
                        else:
                            pfig = px.box(plot_df, y=y_col, title=make_title(f"Box Plot: {y_col}"))
                            pfig.update_traces(marker_color=PF_PLOTLY[0], line=dict(width=1.1))
                        apply_pf_axes(pfig, showlegend=bool(grp_col))
                        st.plotly_chart(pfig, use_container_width=True)

                        fig_e, ax_e = plt.subplots(figsize=(10, 5))
                        if grp_col:
                            groups = [g for g in plot_df[grp_col].dropna().unique().tolist()]
                            data_list = [plot_df[plot_df[grp_col] == g][y_col].dropna().values for g in groups]
                            bp = ax_e.boxplot(
                                data_list,
                                patch_artist=True,
                                labels=[str(g) for g in groups],
                                medianprops=dict(color=TEXT, linewidth=2),
                            )
                            for patch, col in zip(bp["boxes"], PALETTE):
                                patch.set_facecolor(col)
                                patch.set_alpha(0.82)
                            for whisker in bp["whiskers"]:
                                whisker.set_color("#6A6E73")
                            for cap in bp["caps"]:
                                cap.set_color("#6A6E73")
                        else:
                            bp = ax_e.boxplot(
                                plot_df[y_col].dropna().values,
                                patch_artist=True,
                                medianprops=dict(color=TEXT, linewidth=2),
                            )
                            bp["boxes"][0].set_facecolor(PALETTE[0])
                            bp["boxes"][0].set_alpha(0.82)
                        styled_ax(ax_e, make_title(f"Box Plot: {y_col}"), grp_col or "", y_col)
                        fig_e.patch.set_facecolor("white")
                        fig_e.tight_layout()
                        export_fig = fig_e

                    # ── Scatter Plot ──────────────────────────────────────
                    elif chart_type == "Scatter Plot":
                        scatter_df = plot_df.copy()
                        c = color_col if color_col and color_col != "(none)" else None
                        if agg_method and agg_method != "None (raw)":
                            grp = [x_col] + ([c] if c else [])
                            scatter_df = scatter_df.groupby(grp, dropna=False)[y_col].agg(agg_method).reset_index()

                        pfig = px.scatter(
                            scatter_df,
                            x=x_col,
                            y=y_col,
                            color=c,
                            title=make_title(f"{x_col} vs {y_col}"),
                            opacity=0.78,
                            color_discrete_sequence=PF_PLOTLY,
                        )
                        pfig.update_traces(marker=dict(size=8, line=dict(width=0.6, color="white")))
                        apply_pf_axes(pfig, showlegend=bool(c))
                        st.plotly_chart(pfig, use_container_width=True)

                        fig_e, ax_e = plt.subplots(figsize=(10, 5))
                        if c and c in scatter_df.columns:
                            for idx, (gv, gdf) in enumerate(scatter_df.groupby(c, dropna=False)):
                                ax_e.scatter(
                                    gdf[x_col],
                                    gdf[y_col],
                                    color=PALETTE[idx % len(PALETTE)],
                                    alpha=0.78,
                                    label=str(gv),
                                    s=55,
                                    edgecolors="white",
                                    linewidths=0.5,
                                    zorder=3,
                                )
                            style_legend(ax_e, title=c)
                        else:
                            ax_e.scatter(
                                scatter_df[x_col],
                                scatter_df[y_col],
                                color=PALETTE[0],
                                alpha=0.78,
                                s=55,
                                edgecolors="white",
                                linewidths=0.5,
                                zorder=3,
                            )
                        styled_ax(ax_e, make_title(f"{x_col} vs {y_col}"), x_col, y_col)
                        fig_e.patch.set_facecolor("white")
                        fig_e.tight_layout()
                        export_fig = fig_e

                    # ── Line Chart ────────────────────────────────────────
                    elif chart_type == "Line Chart":
                        line_df = plot_df.copy()
                        c = color_col if color_col and color_col != "(none)" else None
                        grp = [x_col] + ([c] if c else [])
                        try:
                            line_df = line_df.groupby(grp, dropna=False)[y_col].agg(agg_method).reset_index()
                            line_df = sort_for_line(line_df, x_col)
                        except Exception:
                            pass

                        pfig = px.line(
                            line_df,
                            x=x_col,
                            y=y_col,
                            color=c,
                            title=make_title(f"{y_col} over {x_col}"),
                            markers=True,
                            color_discrete_sequence=PF_PLOTLY,
                        )
                        pfig.update_traces(line=dict(width=2.5), marker=dict(size=7))
                        apply_pf_axes(pfig, showlegend=bool(c))
                        st.plotly_chart(pfig, use_container_width=True)

                        fig_e, ax_e = plt.subplots(figsize=(10, 5))
                        if c and c in line_df.columns:
                            for idx, (gv, gdf) in enumerate(line_df.groupby(c, dropna=False)):
                                gdf_s = sort_for_line(gdf, x_col)
                                ax_e.plot(
                                    gdf_s[x_col].astype(str),
                                    gdf_s[y_col],
                                    color=PALETTE[idx % len(PALETTE)],
                                    marker="o",
                                    linewidth=2.5,
                                    markersize=7,
                                    label=str(gv),
                                    zorder=3,
                                )
                            style_legend(ax_e, title=c)
                        else:
                            ld = sort_for_line(line_df, x_col)
                            ax_e.plot(
                                ld[x_col].astype(str),
                                ld[y_col],
                                color=PALETTE[0],
                                marker="o",
                                linewidth=2.5,
                                markersize=7,
                                zorder=3,
                            )
                        ax_e.tick_params(axis="x", rotation=30)
                        styled_ax(ax_e, make_title(f"{y_col} over {x_col}"), x_col, y_col)
                        fig_e.patch.set_facecolor("white")
                        fig_e.tight_layout()
                        export_fig = fig_e

                    # ── Bar Chart ─────────────────────────────────────────
                    elif chart_type == "Bar Chart (Grouped)":
                        bar_df = plot_df.copy()
                        c = color_col if color_col and color_col != "(none)" else None
                        grp = [x_col] + ([c] if c else [])
                        bar_df = bar_df.groupby(grp, dropna=False)[y_col].agg(agg_method).reset_index()

                        top_vals = bar_df.groupby(x_col, dropna=False)[y_col].sum().nlargest(top_n).index
                        bar_df = bar_df[bar_df[x_col].isin(top_vals)].copy()
                        bar_df = bar_df.sort_values(y_col, ascending=False)
                        t = make_title(f"{agg_method.title()} of {y_col} by {x_col} (Top {top_n})")

                        x_order = bar_df.groupby(x_col, dropna=False)[y_col].sum().sort_values(ascending=False).index.tolist()
                        pfig = px.bar(
                            bar_df,
                            x=x_col,
                            y=y_col,
                            color=c,
                            barmode="group",
                            title=t,
                            color_discrete_sequence=PF_PLOTLY,
                            category_orders={x_col: [str(v) for v in x_order]},
                        )
                        pfig.update_traces(marker_line_width=0)
                        apply_pf_axes(pfig, showlegend=bool(c))
                        st.plotly_chart(pfig, use_container_width=True)

                        fig_e, ax_e = plt.subplots(figsize=(12, 5))
                        if c and c in bar_df.columns:
                            groups = list(dict.fromkeys(bar_df[c].dropna().astype(str).tolist()))
                            cats = [str(v) for v in x_order]
                            x_idx = np.arange(len(cats))
                            w = 0.8 / max(len(groups), 1)
                            for i, gv in enumerate(groups):
                                gdf = bar_df[bar_df[c].astype(str) == gv].copy()
                                gdf["_x_key"] = gdf[x_col].astype(str)
                                gseries = gdf.groupby("_x_key")[y_col].sum()
                                vals = [float(gseries.get(cat, 0)) for cat in cats]
                                ax_e.bar(
                                    x_idx + i * w,
                                    vals,
                                    width=w,
                                    color=PALETTE[i % len(PALETTE)],
                                    alpha=0.90,
                                    label=str(gv),
                                    zorder=3,
                                )
                            ax_e.set_xticks(x_idx + w * (len(groups) - 1) / 2)
                            ax_e.set_xticklabels(cats, rotation=30, ha="right")
                            style_legend(ax_e, title=c)
                        else:
                            cats = [str(v) for v in x_order]
                            vals = bar_df.groupby(x_col, dropna=False)[y_col].sum().reindex(x_order, fill_value=0).tolist()
                            cols = [PALETTE[i % len(PALETTE)] for i in range(len(cats))]
                            ax_e.bar(cats, vals, color=cols, alpha=0.90, zorder=3)
                            ax_e.tick_params(axis="x", rotation=30)
                        styled_ax(ax_e, t, x_col, y_col)
                        fig_e.patch.set_facecolor("white")
                        fig_e.tight_layout()
                        export_fig = fig_e

                    # ── Heatmap ───────────────────────────────────────────
                    elif chart_type == "Heatmap / Correlation Matrix":
                        corr = plot_df[heatmap_cols].corr()
                        size = max(8, len(heatmap_cols))
                        fig_d, ax_d = plt.subplots(figsize=(size, max(6, size - 1)))
                        cmap = sns.diverging_palette(220, 10, as_cmap=True)
                        annot = len(heatmap_cols) <= 12
                        sns.heatmap(
                            corr,
                            annot=annot,
                            fmt=".2f",
                            cmap=cmap,
                            center=0,
                            ax=ax_d,
                            linewidths=0.6,
                            linecolor="#E8E8E8",
                            annot_kws={"size": 9, "color": TEXT},
                            cbar_kws={"shrink": 0.8, "label": "Correlation"},
                        )
                        ax_d.set_title(make_title("Correlation Matrix"), fontsize=14, fontweight="bold", pad=14, color=TEXT)
                        ax_d.set_xticklabels(ax_d.get_xticklabels(), fontweight="bold", fontsize=10, rotation=30, ha="right", color=TEXT)
                        ax_d.set_yticklabels(ax_d.get_yticklabels(), fontweight="bold", fontsize=10, rotation=0, color=TEXT)
                        fig_d.patch.set_facecolor("white")
                        fig_d.tight_layout()
                        st.pyplot(fig_d, use_container_width=True)
                        export_fig = fig_d

                    if export_fig is not None:
                        st.session_state["last_export_fig"] = export_fig

                except Exception as e:
                    st.error(f"⚠️ Could not generate chart: {e}. Try a different column combination.")
                finally:
                    plt.close("all")

            # PNG export — always matplotlib, always full color
            if "last_export_fig" in st.session_state:
                buf = io.BytesIO()
                try:
                    st.session_state["last_export_fig"].savefig(
                        buf, format="png", bbox_inches="tight",
                        dpi=150, facecolor="white")
                    buf.seek(0)
                    st.download_button("📥 Export Chart as PNG", buf, "chart.png",
                                       "image/png", key="chart_png")
                except Exception as e:
                    st.caption(f"PNG export unavailable: {e}")


# ══════════════════════════════════════════════
# PAGE D — EXPORT & REPORT
# ══════════════════════════════════════════════
with tab_d:
    st.markdown("## 💾 Export & Report")

    if st.session_state.df_working is None:
        st.info("Please upload a dataset first.")
    else:
        df_export = st.session_state.df_working

        # ── Dataset Export
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

        # ── Transformation Report
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

            # ── Recipe JSON Export
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

            # ── Report text export
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
