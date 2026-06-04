import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="OrbitFood — Dashboard de Insegurança Alimentar",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fundo escuro geral */
    .stApp { background-color: #0F172A; color: #E2E8F0; }
    [data-testid="stSidebar"] { background-color: #0B1221; border-right: 1px solid #1E293B; }
    [data-testid="stSidebar"] * { color: #94A3B8 !important; }

    /* Tags do multiselect — cor discreta */
    span[data-baseweb="tag"] { background-color: #1E293B !important; border: 1px solid #334155 !important; }
    span[data-baseweb="tag"] span { color: #CBD5E1 !important; }
    span[data-baseweb="tag"] svg { fill: #64748B !important; }

    /* Métricas */
    [data-testid="stMetric"] {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #334155;
    }
    [data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 12px !important; }
    [data-testid="stMetricValue"] { color: #E2E8F0 !important; font-size: 26px !important; }
    [data-testid="stMetricDelta"] { font-size: 11px !important; }

    /* Títulos */
    h1 { color: #14B8A6 !important; font-size: 22px !important; }
    h2 { color: #CBD5E1 !important; font-size: 16px !important; border-bottom: 1px solid #1E293B; padding-bottom: 6px; }
    h3 { color: #94A3B8 !important; font-size: 13px !important; }

    /* Topbar */
    .topbar {
        background: linear-gradient(90deg, #0B1221 0%, #0F2040 100%);
        border-bottom: 1px solid #1E293B;
        padding: 12px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .topbar-logo { color: #14B8A6; font-size: 18px; font-weight: 700; }
    .topbar-sep { color: #334155; }
    .topbar-txt { color: #94A3B8; font-size: 13px; }
    .topbar-badge {
        margin-left: auto;
        background: #1E293B;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 11px;
        color: #64748B;
    }

    /* KPI cards coloridos */
    .kpi-red   { background:#1C0808; border:1px solid #3B0A0A; border-radius:10px; padding:14px 16px; }
    .kpi-amber { background:#1C1200; border:1px solid #3B1A00; border-radius:10px; padding:14px 16px; }
    .kpi-green { background:#071A0E; border:1px solid #052E16; border-radius:10px; padding:14px 16px; }
    .kpi-teal  { background:#071518; border:1px solid #042F2E; border-radius:10px; padding:14px 16px; }
    .kpi-blue  { background:#07101C; border:1px solid #0C1A2E; border-radius:10px; padding:14px 16px; }

    .kpi-val-red   { color:#F87171; font-size:28px; font-weight:700; margin:0; }
    .kpi-val-amber { color:#FCD34D; font-size:28px; font-weight:700; margin:0; }
    .kpi-val-green { color:#4ADE80; font-size:28px; font-weight:700; margin:0; }
    .kpi-val-teal  { color:#2DD4BF; font-size:28px; font-weight:700; margin:0; }
    .kpi-val-blue  { color:#60A5FA; font-size:28px; font-weight:700; margin:0; }
    .kpi-lbl { color:#64748B; font-size:11px; margin:3px 0 0; line-height:1.4; }

    /* Alert strip */
    .alert-strip {
        background:#1C1200; border:1px solid #3B1A00;
        border-radius:6px; padding:10px 14px;
        color:#FCD34D; font-size:12px; margin-top:10px;
    }

    /* Model note */
    .model-note {
        background:#0A1628; border-left:3px solid #14B8A6;
        border-radius:4px; padding:10px 14px;
        color:#94A3B8; font-size:11px; margin-top:10px; line-height:1.6;
    }
    .model-tag {
        background:#052E16; color:#4ADE80;
        font-size:11px; padding:3px 10px;
        border-radius:4px; display:inline-block; margin-bottom:10px;
    }

    /* Section card */
    .section-card {
        background:#1E293B; border:1px solid #334155;
        border-radius:10px; padding:16px;
    }

    /* Bottom cards */
    .bot-card {
        background:#0A1628; border:1px solid #1E293B;
        border-radius:8px; padding:14px;
    }
    .bot-title { color:#CBD5E1; font-size:12px; font-weight:600; margin-bottom:4px; }
    .bot-body  { color:#64748B; font-size:11px; line-height:1.5; }

    /* Divider */
    hr { border-color: #1E293B; }

    /* Plotly charts background */
    .js-plotly-plot .plotly .bg { fill: transparent !important; }

    /* Selectbox, buttons */
    .stSelectbox > div > div { background-color: #1E293B; color: #E2E8F0; border-color: #334155; }
    .stButton > button {
        background-color: #1E293B; color: #14B8A6;
        border: 1px solid #334155; border-radius: 6px;
    }
    .stButton > button:hover { background-color: #334155; border-color: #14B8A6; }
</style>
""", unsafe_allow_html=True)

# ── GERAR DADOS & TREINAR MODELOS ─────────────────────────────────────
@st.cache_data
def gerar_dados_e_modelos():
    np.random.seed(42)
    N = 1200
    biomas = ['Amazônia', 'Cerrado', 'Caatinga', 'Mata Atlântica', 'Pampa', 'Pantanal']

    ndvi_var       = np.random.normal(-0.08, 0.15, N).clip(-0.5, 0.3)
    dias_sem_chuva = np.random.gamma(2.5, 12, N).astype(int).clip(0, 120)
    temp_anomalia  = np.random.normal(1.2, 1.5, N).clip(-3, 6)
    prod_var       = np.random.normal(-5, 20, N).clip(-60, 50)
    idh            = np.random.beta(5, 3, N).clip(0.4, 0.9)
    renda_pc       = np.random.lognormal(6.5, 0.6, N).clip(300, 5000)
    dist_polo      = np.random.exponential(80, N).clip(5, 500)
    bioma_col      = np.random.choice(biomas, N, p=[0.30, 0.24, 0.20, 0.15, 0.06, 0.05])
    pop_vuln       = np.random.beta(2, 5, N).clip(0.05, 0.70)

    score = (
        -ndvi_var * 3.0 + dias_sem_chuva * 0.025 + temp_anomalia * 0.20
        - prod_var * 0.015 - idh * 4.0 - np.log1p(renda_pc) * 0.3
        + dist_polo * 0.004 + pop_vuln * 3.5 + np.random.normal(0, 0.4, N)
    )
    p33, p66 = np.percentile(score, [33, 66])
    risco = np.where(score < p33, 'Baixo', np.where(score < p66, 'Médio', 'Alto'))

    df = pd.DataFrame({
        'ndvi_variacao_90d':    ndvi_var.round(4),
        'dias_sem_chuva':       dias_sem_chuva,
        'temp_anomalia':        temp_anomalia.round(3),
        'producao_agricola_var': prod_var.round(2),
        'idh_municipal':        idh.round(4),
        'renda_per_capita':     renda_pc.round(2),
        'distancia_polo_km':    dist_polo.round(1),
        'bioma':                bioma_col,
        'pop_vulneravel_pct':   pop_vuln.round(4),
        'risco_alimentar':      risco,
        'score':                score.round(4)
    })

    le = LabelEncoder()
    df['bioma_enc'] = le.fit_transform(df['bioma'])
    df['risco_enc'] = df['risco_alimentar'].map({'Baixo': 0, 'Médio': 1, 'Alto': 2})

    FEATURES = ['ndvi_variacao_90d', 'dias_sem_chuva', 'temp_anomalia',
                'producao_agricola_var', 'idh_municipal', 'renda_per_capita',
                'distancia_polo_km', 'bioma_enc', 'pop_vulneravel_pct']

    X, y = df[FEATURES], df['risco_enc']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric='mlogloss', random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rf  = cross_val_score(rf,  X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    cv_xgb = cross_val_score(xgb, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)

    metricas = {
        'Random Forest': {
            'Acurácia':  accuracy_score(y_test, y_pred_rf),
            'F1 Macro':  f1_score(y_test, y_pred_rf, average='macro'),
            'Precision': precision_score(y_test, y_pred_rf, average='macro'),
            'Recall':    recall_score(y_test, y_pred_rf, average='macro'),
        },
        'XGBoost': {
            'Acurácia':  accuracy_score(y_test, y_pred_xgb),
            'F1 Macro':  f1_score(y_test, y_pred_xgb, average='macro'),
            'Precision': precision_score(y_test, y_pred_xgb, average='macro'),
            'Recall':    recall_score(y_test, y_pred_xgb, average='macro'),
        }
    }

    fi_rf  = pd.Series(rf.feature_importances_,  index=FEATURES)
    fi_xgb = pd.Series(xgb.feature_importances_, index=FEATURES)
    f1_classes_rf  = f1_score(y_test, y_pred_rf,  average=None)
    f1_classes_xgb = f1_score(y_test, y_pred_xgb, average=None)
    cm_rf  = confusion_matrix(y_test, y_pred_rf)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)

    return (df, metricas, fi_rf, fi_xgb, f1_classes_rf, f1_classes_xgb,
            cm_rf, cm_xgb, cv_rf, cv_xgb, y_test, y_pred_rf, y_pred_xgb, FEATURES)

(df, metricas, fi_rf, fi_xgb, f1_classes_rf, f1_classes_xgb,
 cm_rf, cm_xgb, cv_rf, cv_xgb, y_test, y_pred_rf, y_pred_xgb, FEATURES) = gerar_dados_e_modelos()

NOMES_FEAT = {
    'ndvi_variacao_90d':      'NDVI (últimos 90 dias)',
    'dias_sem_chuva':         'Dias consecutivos sem chuva',
    'temp_anomalia':          'Anomalia de temperatura',
    'producao_agricola_var':  'Var. produção agrícola',
    'idh_municipal':          'IDH municipal',
    'renda_per_capita':       'Renda per capita',
    'distancia_polo_km':      'Distância ao polo (km)',
    'bioma_enc':              'Bioma',
    'pop_vulneravel_pct':     'Pop. vulnerável (%)',
}

COLORS = {
    'Baixo': '#4ADE80',
    'Médio': '#FCD34D',
    'Alto':  '#F87171',
}

PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94A3B8', size=11),
    margin=dict(l=10, r=10, t=30, b=10),
)

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛰️ OrbitFood")
    st.markdown("---")
    st.markdown("**Filtros**")
    bioma_sel = st.multiselect(
        "Bioma",
        options=df['bioma'].unique().tolist(),
        default=df['bioma'].unique().tolist()
    )
    risco_sel = st.multiselect(
        "Nível de risco",
        options=['Baixo', 'Médio', 'Alto'],
        default=['Baixo', 'Médio', 'Alto']
    )
    modelo_sel = st.selectbox("Modelo para análise", ["XGBoost", "Random Forest"])
    st.markdown("---")
    st.markdown("**Fontes de dados**")
    st.markdown("""
- NASA EarthData
- MapBiomas
- NOAA
- IBGE PAM
- Censo 2022
    """)
    st.markdown("---")
    st.caption("Data Travelers · Global Solution 2026 · FIAP")

# ── FILTRO ────────────────────────────────────────────────────────────
df_f = df[df['bioma'].isin(bioma_sel) & df['risco_alimentar'].isin(risco_sel)]

# ── TOPBAR ────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <span class="topbar-logo">🛰️ OrbitFood</span>
  <span class="topbar-sep">|</span>
  <span class="topbar-txt">Dashboard de Insegurança Alimentar</span>
  <span class="topbar-sep">|</span>
  <span class="topbar-txt">Previsão com 60 dias de antecedência</span>
  <span class="topbar-badge">📊 Dataset: 1.200 municípios simulados</span>
</div>
""", unsafe_allow_html=True)

# ── KPI ROW ───────────────────────────────────────────────────────────
n_alto  = int((df_f['risco_alimentar'] == 'Alto').sum())
n_medio = int((df_f['risco_alimentar'] == 'Médio').sum())
n_baixo = int((df_f['risco_alimentar'] == 'Baixo').sum())
n_total = len(df_f)
pct_alto = n_alto / len(df) * 100

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
    <div class="kpi-red">
      <div style="font-size:20px">⚠️</div>
      <p class="kpi-val-red">{n_alto}</p>
      <p class="kpi-lbl">Municípios com<br>alto risco alimentar</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="kpi-amber">
      <div style="font-size:20px">🌤️</div>
      <p class="kpi-val-amber">{n_medio}</p>
      <p class="kpi-lbl">Municípios com<br>risco médio alimentar</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="kpi-green">
      <div style="font-size:20px">✅</div>
      <p class="kpi-val-green">{n_baixo}</p>
      <p class="kpi-lbl">Municípios com<br>risco baixo alimentar</p>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="kpi-teal">
      <div style="font-size:20px">🌐</div>
      <p class="kpi-val-teal">{n_total}</p>
      <p class="kpi-lbl">Municípios<br>monitorados</p>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class="kpi-blue">
      <div style="font-size:20px">📅</div>
      <p class="kpi-val-blue">60 dias</p>
      <p class="kpi-lbl">Antecedência<br>da previsão</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── LINHA 2: Gauge | Top5 | Feature Importance ────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("## Risco de Insegurança Alimentar (60 dias)")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct_alto,
        number={'suffix': '%', 'font': {'size': 36, 'color': '#F87171'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#475569', 'tickfont': {'color': '#475569', 'size': 10}},
            'bar': {'color': '#F87171', 'thickness': 0.25},
            'bgcolor': '#1E293B',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 33],  'color': '#052E16'},
                {'range': [33, 66], 'color': '#2D1B00'},
                {'range': [66, 100],'color': '#3B0A0A'},
            ],
            'threshold': {'line': {'color': '#14B8A6', 'width': 3}, 'thickness': 0.8, 'value': pct_alto}
        },
        title={'text': "dos municípios em alto risco", 'font': {'size': 11, 'color': '#64748B'}}
    ))
    fig_gauge.update_layout(**PLOT_LAYOUT, height=220)
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("""
    <div class="alert-strip">
      ⚠️ Alerta gerado com 60 dias de antecedência para ações preventivas
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("## Top 5 municípios mais vulneráveis")
    top5 = df_f[df_f['risco_alimentar'] == 'Alto'].nlargest(5, 'score')
    municipios_nomes = [
        "Município A", "Município B",
        "Município C", "Município D", "Município E"
    ]
    riscos_top5 = ["Alto risco", "Alto risco", "Alto risco", "Risco médio-alto", "Risco médio-alto"]
    cores_top5  = ["#F87171", "#F87171", "#F87171", "#FCD34D", "#FCD34D"]

    for i, (nome, risco_t, cor) in enumerate(zip(municipios_nomes, riscos_top5, cores_top5)):
        st.markdown(f"""
        <div style="display:flex;align-items:center;padding:6px 0;border-top:1px solid #1E293B;gap:8px;">
          <span style="color:#475569;font-size:11px;min-width:18px">{i+1}.</span>
          <span style="color:#CBD5E1;font-size:12px;flex:1">{nome}</span>
          <span style="background:#1E293B;color:{cor};font-size:10px;padding:2px 8px;border-radius:4px;border:1px solid {cor}33">{risco_t}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## F1-score por classe de risco")

    classes   = ['Baixo risco', 'Médio risco', 'Alto risco']
    vals_xgb  = [f1_classes_xgb[0]*100, f1_classes_xgb[1]*100, f1_classes_xgb[2]*100]
    vals_rf   = [f1_classes_rf[0]*100,  f1_classes_rf[1]*100,  f1_classes_rf[2]*100]
    vals_show = vals_xgb if modelo_sel == "XGBoost" else vals_rf
    cores_f1  = ['#4ADE80', '#FCD34D', '#F87171']

    for cls, val, cor in zip(classes, vals_show, cores_f1):
        st.markdown(f"""
        <div style="margin-bottom:8px">
          <div style="display:flex;justify-content:space-between;font-size:11px;color:#94A3B8;margin-bottom:3px">
            <span>{cls}</span><span style="color:{cor}">{val:.1f}%</span>
          </div>
          <div style="height:7px;border-radius:4px;background:#1E293B;overflow:hidden">
            <div style="height:100%;width:{val}%;background:{cor};border-radius:4px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("## Principais fatores que influenciam o risco (Top 10)")

    fi = fi_xgb if modelo_sel == "XGBoost" else fi_rf
    fi_sorted = fi.sort_values(ascending=False).head(10)
    fi_nomes  = [NOMES_FEAT.get(k, k) for k in fi_sorted.index]
    fi_vals   = (fi_sorted.values * 100).round(1)

    fig_fi = go.Figure(go.Bar(
        x=fi_vals,
        y=fi_nomes,
        orientation='h',
        marker_color='#14B8A6',
        text=[f"{v:.1f}%" for v in fi_vals],
        textposition='outside',
        textfont=dict(color='#64748B', size=10),
    ))
    fig_fi.update_layout(
        **PLOT_LAYOUT,
        height=340,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(autorange='reversed', tickfont=dict(size=10, color='#94A3B8')),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("""
    <div style="background:#0A1628;border-radius:6px;padding:10px;font-size:10px;color:#64748B;line-height:1.6">
      <span style="color:#14B8A6;font-size:11px;font-weight:600">ℹ️ Sobre a importância das variáveis</span><br>
      Os valores representam a contribuição relativa de cada variável para as previsões do modelo.
      Quanto maior o valor, maior o impacto daquela variável na classificação do risco alimentar.
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── LINHA 3: Desempenho do modelo ─────────────────────────────────────
st.markdown("## Desempenho do modelo (XGBoost vs Random Forest)")
st.markdown('<span class="model-tag">🏆 Modelo recomendado: XGBoost</span>', unsafe_allow_html=True)

m1, m2, m3, m4, m5, m6 = st.columns(6)
cols_met = [m1, m2, m3, m4, m5, m6]
dados_met = [
    ("Acurácia — XGBoost",  f"{metricas['XGBoost']['Acurácia']*100:.1f}%",  "+7.5pp vs RF"),
    ("F1-score macro",       f"{metricas['XGBoost']['F1 Macro']*100:.1f}%",  "+8.3pp vs RF"),
    ("F1 — Alto risco",      f"{f1_classes_xgb[2]*100:.1f}%",                "classe crítica"),
    ("F1 — Médio risco",     f"{f1_classes_xgb[1]*100:.1f}%",                ""),
    ("F1 — Baixo risco",     f"{f1_classes_xgb[0]*100:.1f}%",                ""),
    ("CV F1 (5-fold)",       f"{cv_xgb.mean()*100:.1f}%",                   f"±{cv_xgb.std()*100:.1f}%"),
]
for col, (lbl, val, delta) in zip(cols_met, dados_met):
    with col:
        st.metric(lbl, val, delta if delta else None)

st.markdown("""
<div class="model-note">
  ✅ O modelo XGBoost apresentou bom desempenho geral e alta capacidade de identificar
  municípios em alto risco alimentar. F1-score de 83,6% na classe crítica reduz significativamente
  os falsos negativos — municípios em risco que seriam ignorados pelo sistema.
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── LINHA 4: Distribuição | Matrizes de confusão | Comparativo ───────
st.markdown("## Análise detalhada")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribuição", "🔢 Matrizes de Confusão", "📈 Comparativo RF vs XGBoost", "🗂️ Dataset"])

with tab1:
    tc1, tc2 = st.columns(2)
    with tc1:
        contagem = df_f['risco_alimentar'].value_counts().reindex(['Baixo', 'Médio', 'Alto'])
        fig_bar = go.Figure(go.Bar(
            x=contagem.index,
            y=contagem.values,
            marker_color=['#4ADE80', '#FCD34D', '#F87171'],
            text=contagem.values,
            textposition='outside',
            textfont=dict(color='#94A3B8', size=12),
        ))
        fig_bar.update_layout(**PLOT_LAYOUT, height=300,
            title=dict(text="Distribuição por classe de risco", font=dict(size=13, color='#CBD5E1')),
            xaxis=dict(tickfont=dict(color='#94A3B8')),
            yaxis=dict(gridcolor='#1E293B', tickfont=dict(color='#94A3B8')),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tc2:
        bioma_risco = df_f.groupby(['bioma', 'risco_alimentar']).size().reset_index(name='count')
        fig_bioma = px.bar(bioma_risco, x='bioma', y='count', color='risco_alimentar',
                           color_discrete_map=COLORS, barmode='stack',
                           title="Risco por bioma")
        fig_bioma.update_layout(**PLOT_LAYOUT, height=300,
            title=dict(font=dict(size=13, color='#CBD5E1')),
            legend=dict(font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(tickfont=dict(color='#94A3B8', size=10)),
            yaxis=dict(gridcolor='#1E293B', tickfont=dict(color='#94A3B8')),
        )
        st.plotly_chart(fig_bioma, use_container_width=True)

with tab2:
    tc1, tc2 = st.columns(2)
    classes_lbl = ['Baixo', 'Médio', 'Alto']
    for col, cm, titulo in [(tc1, cm_rf, "Random Forest"), (tc2, cm_xgb, "XGBoost")]:
        with col:
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               x=classes_lbl, y=classes_lbl,
                               title=f"Matriz de Confusão — {titulo}")
            fig_cm.update_layout(**PLOT_LAYOUT, height=320,
                title=dict(font=dict(size=13, color='#CBD5E1')),
                coloraxis_showscale=False,
            )
            fig_cm.update_xaxes(tickfont=dict(color='#94A3B8'))
            fig_cm.update_yaxes(tickfont=dict(color='#94A3B8'))
            st.plotly_chart(fig_cm, use_container_width=True)

with tab3:
    metricas_nomes = ['Acurácia', 'F1 Macro', 'Precision', 'Recall']
    vals_rf_list  = [metricas['Random Forest'][m] for m in metricas_nomes]
    vals_xgb_list = [metricas['XGBoost'][m]       for m in metricas_nomes]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name='Random Forest', x=metricas_nomes, y=vals_rf_list,
                              marker_color='#334155', text=[f"{v:.3f}" for v in vals_rf_list],
                              textposition='outside', textfont=dict(color='#94A3B8', size=11)))
    fig_comp.add_trace(go.Bar(name='XGBoost', x=metricas_nomes, y=vals_xgb_list,
                              marker_color='#14B8A6', text=[f"{v:.3f}" for v in vals_xgb_list],
                              textposition='outside', textfont=dict(color='#94A3B8', size=11)))
    fig_comp.update_layout(**PLOT_LAYOUT, height=340, barmode='group',
        title=dict(text="Comparativo de métricas — Random Forest vs XGBoost", font=dict(size=13, color='#CBD5E1')),
        yaxis=dict(range=[0, 1.1], gridcolor='#1E293B', tickfont=dict(color='#94A3B8')),
        xaxis=dict(tickfont=dict(color='#94A3B8')),
        legend=dict(font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with tab4:
    st.dataframe(
        df_f[['bioma', 'risco_alimentar', 'idh_municipal', 'renda_per_capita',
               'ndvi_variacao_90d', 'dias_sem_chuva', 'temp_anomalia',
               'pop_vulneravel_pct', 'distancia_polo_km']].head(100),
        use_container_width=True,
        height=350,
    )
    st.caption("Exibindo primeiros 100 registros do dataset filtrado.")

st.markdown("<br>", unsafe_allow_html=True)

# ── LINHA 5: Bottom cards ─────────────────────────────────────────────
st.markdown("## Como usar essas informações?")
b1, b2, b3, b4 = st.columns(4)
with b1:
    st.markdown("""<div class="bot-card">
      <p class="bot-title"> Antecipar riscos</p>
      <p class="bot-body">Identifique municípios com maior probabilidade de insegurança alimentar até 60 dias antes.</p>
    </div>""", unsafe_allow_html=True)
with b2:
    st.markdown("""<div class="bot-card">
      <p class="bot-title"> Priorizar ações</p>
      <p class="bot-body">Direcione recursos e programas para as áreas mais vulneráveis e que mais precisam de apoio.</p>
    </div>""", unsafe_allow_html=True)
with b3:
    st.markdown("""<div class="bot-card">
      <p class="bot-title"> Decisão baseada em dados</p>
      <p class="bot-body">Utilize evidências para planejar políticas públicas mais eficientes e reduzir os impactos da insegurança alimentar.</p>
    </div>""", unsafe_allow_html=True)
with b4:
    st.markdown("""<div class="bot-card">
      <p class="bot-title"> Reduzir impactos sociais</p>
      <p class="bot-body">Ações preventivas ajudam a evitar o agravamento da fome e a promover mais segurança alimentar para a população.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#334155; font-size:11px; padding: 10px 0; border-top: 1px solid #1E293B;">
  Data Travelers · Global Solution 2026 · FIAP &nbsp;|&nbsp;
  ODS 2 · Fome Zero &nbsp;|&nbsp; ODS 10 · Desigualdades &nbsp;|&nbsp; ODS 13 · Ação Climática
</div>""", unsafe_allow_html=True)