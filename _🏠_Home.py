import streamlit as st

st.set_page_config(
    page_title="OrbitFood — Insegurança Alimentar",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS GLOBAL ───────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0F172A; color: #E2E8F0; }
  [data-testid="stSidebar"] { background-color: #0B1221; border-right: 1px solid #1E293B; }
  [data-testid="stSidebar"] * { color: #94A3B8 !important; }
  span[data-baseweb="tag"] { background-color: #1E293B !important; border: 1px solid #334155 !important; }
  span[data-baseweb="tag"] span { color: #CBD5E1 !important; }
  span[data-baseweb="tag"] svg { fill: #64748B !important; }
  h1, h2, h3 { color: #CBD5E1 !important; }
  [data-testid="stMetric"] { background-color: #1E293B; border-radius: 10px; padding: 14px; border: 1px solid #334155; }
  [data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 12px !important; }
  [data-testid="stMetricValue"] { color: #E2E8F0 !important; }
  .stTabs [data-baseweb="tab"] { background-color: #1E293B; color: #94A3B8; border-radius: 6px 6px 0 0; }
  .stTabs [aria-selected="true"] { background-color: #14B8A6 !important; color: #0F172A !important; }
  .stButton > button { background-color: #1E293B; color: #14B8A6; border: 1px solid #334155; border-radius: 6px; }
  .stButton > button:hover { background-color: #334155; border-color: #14B8A6; }
  .stSelectbox > div > div { background-color: #1E293B; color: #E2E8F0; border-color: #334155; }
  .stNumberInput > div > div > input { background-color: #1E293B; color: #E2E8F0; border-color: #334155; }
  .stSlider > div > div { color: #94A3B8; }
  div[data-testid="stDataFrame"] { background-color: #1E293B; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR LOGO ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
      <div style='font-size:32px'>🛰️</div>
      <div style='font-size:18px; font-weight:700; color:#14B8A6'>OrbitFood</div>
      <div style='font-size:11px; color:#475569; margin-top:4px'>Insegurança Alimentar · ML</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px; color:#334155; text-align:center; margin-top:20px; line-height:1.8'>
      OrbitFood<br>Global Solution 2026<br>FIAP
    </div>
    """, unsafe_allow_html=True)

# ── HOME PAGE ─────────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #0B1221 0%, #0F2040 100%);
     border-radius: 12px; padding: 40px 32px; margin-bottom: 24px;
     border: 1px solid #1E293B;'>
  <div style='font-size:13px; color:#14B8A6; letter-spacing:3px; margin-bottom:8px'>
    GLOBAL SOLUTION 2026
  </div>
  <div style='font-size:42px; font-weight:800; color:#E2E8F0; line-height:1.1; margin-bottom:12px'>
    OrbitFood
  </div>
  <div style='font-size:18px; color:#14B8A6; font-style:italic; margin-bottom:16px'>
    Inteligência espacial contra a fome
  </div>
  <div style='font-size:13px; color:#64748B; max-width:600px; line-height:1.7'>
    Plataforma de previsão de insegurança alimentar municipal com dados de satélite e machine learning.
    Classifica o risco de cada município com até <strong style='color:#CBD5E1'>60 dias de antecedência</strong>,
    permitindo que gestores públicos ajam <strong style='color:#CBD5E1'>antes</strong> da crise chegar.
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (
        "🏠",
        "Home",
        "Visão geral da plataforma OrbitFood e acesso rápido aos módulos analíticos.",
        None
    ),
    (
        "📊",
        "Dashboard Estratégico",
        "Indicadores executivos, distribuição de risco e análise territorial dos municípios.",
        "page_1_dashboard.py"
    ),
    (
        "🧪",
        "Simulador de Decisão",
        "Simule cenários e estime a probabilidade de insegurança alimentar utilizando Machine Learning.",
        "page_2_simulador.py"
    ),
    (
        "📈",
        "Análise Comparativa",
        "Compare municípios, explore distribuições estatísticas e identifique padrões relevantes.",
        "page_3_analise.py"
    ),
    (
        "🧠",
        "Desempenho do Modelo",
        "Avalie métricas, matriz de confusão e importância das variáveis preditivas.",
        "page_4_modelo.py"
    ),
]
for col, (icon, titulo, desc, _) in zip([c1, c2, c3, c4, c5], cards):
    with col:
        st.markdown(f"""
        <div style='background:#1E293B; border:1px solid #334155; border-radius:10px;
             padding:18px; height:160px;'>
          <div style='font-size:28px; margin-bottom:8px'>{icon}</div>
          <div style='font-size:13px; font-weight:600; color:#CBD5E1; margin-bottom:6px'>{titulo}</div>
          <div style='font-size:11px; color:#64748B; line-height:1.5'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#334155; font-size:11px; padding:10px 0; border-top:1px solid #1E293B'>
  ODS 2 · Fome Zero &nbsp;|&nbsp; ODS 10 · Redução das Desigualdades &nbsp;|&nbsp; ODS 13 · Ação Climática
</div>
""", unsafe_allow_html=True)
