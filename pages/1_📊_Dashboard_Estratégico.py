import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import gerar_dados_e_modelos, PLOT_BG, COLORS

dados = gerar_dados_e_modelos()
df = dados['df']

# ── SIDEBAR FILTROS ───────────────────────────────────────────────────
st.sidebar.markdown("### 🔍 Filtros")
bioma_sel = st.sidebar.multiselect(
    "Bioma", df['bioma'].unique().tolist(),
    default=df['bioma'].unique().tolist()
)
risco_sel = st.sidebar.multiselect(
    "Nível de risco", ['Baixo', 'Médio', 'Alto'],
    default=['Baixo', 'Médio', 'Alto']
)

df_f = df[df['bioma'].isin(bioma_sel) & df['risco_alimentar'].isin(risco_sel)]

# ── HEADER ────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#0B1221;border-radius:10px;padding:16px 20px;margin-bottom:20px;
     border:1px solid #1E293B;display:flex;align-items:center;gap:10px;'>
  <span style='font-size:20px'>📊</span>
  <span style='font-size:18px;font-weight:700;color:#14B8A6'>Dashboard Estratégico</span>
  <span style='color:#334155'>|</span>
  <span style='font-size:13px;color:#64748B'>Monitorização de Risco Alimentar</span>
  <span style='margin-left:auto;background:#1E293B;border-radius:5px;padding:3px 10px;
        font-size:11px;color:#475569'>Dataset: 1.200 municípios simulados</span>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────
n_alto  = int((df_f['risco_alimentar'] == 'Alto').sum())
n_medio = int((df_f['risco_alimentar'] == 'Médio').sum())
n_baixo = int((df_f['risco_alimentar'] == 'Baixo').sum())
n_total = len(df_f)
pct_alto = round(n_alto / len(df) * 100, 1)

c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (c1, "🔴",  str(n_alto),   "Municípios\nalto risco",    "#F87171", "#1C0808", "#3B0A0A"),
    (c2, "🟡",  str(n_medio),  "Municípios\nrisco médio",   "#FCD34D", "#1C1200", "#3B1A00"),
    (c3, "🟢",  str(n_baixo),  "Municípios\nrisco baixo",   "#4ADE80", "#071A0E", "#052E16"),
    (c4, "🌐",  str(n_total),  "Total\nmonitorados",        "#2DD4BF", "#071518", "#042F2E"),
    (c5, "📅",  "60 dias",     "Antecedência\nda previsão", "#60A5FA", "#07101C", "#0C1A2E"),
]
for col, icon, val, lbl, cor, bg, brd in kpis:
    with col:
        st.markdown(f"""
        <div style='background:{bg};border:1px solid {brd};border-radius:10px;padding:14px 12px;'>
          <div style='font-size:18px'>{icon}</div>
          <div style='font-size:24px;font-weight:700;color:{cor};margin:4px 0 2px'>{val}</div>
          <div style='font-size:10px;color:#64748B;line-height:1.4'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── GRÁFICOS LINHA 1 ──────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Distribuição por classe de risco")
    contagem = df_f['risco_alimentar'].value_counts().reindex(['Baixo', 'Médio', 'Alto'])
    fig = go.Figure(go.Bar(
        x=contagem.index, y=contagem.values,
        marker_color=['#4ADE80', '#FCD34D', '#F87171'],
        text=contagem.values, textposition='outside',
        textfont=dict(color='#94A3B8', size=12)
    ))
    fig.update_layout(**PLOT_BG, height=280,
        xaxis=dict(tickfont=dict(color='#94A3B8')),
        yaxis=dict(gridcolor='#1E293B', tickfont=dict(color='#94A3B8')),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Risco por bioma")
    bioma_risco = df_f.groupby(['bioma', 'risco_alimentar']).size().reset_index(name='count')
    fig2 = px.bar(bioma_risco, x='bioma', y='count', color='risco_alimentar',
                  color_discrete_map=COLORS, barmode='stack')
    fig2.update_layout(**PLOT_BG, height=280,
        legend=dict(font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(tickfont=dict(color='#94A3B8', size=10)),
        yaxis=dict(gridcolor='#1E293B', tickfont=dict(color='#94A3B8')),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── SCATTER SOCIOECONÔMICO ────────────────────────────────────────────
st.markdown("#### Análise geoespacial e socioeconômica")
fig3 = px.scatter(
    df_f, x="renda_per_capita", y="dias_sem_chuva",
    color="risco_alimentar", size="pop_vulneravel_pct",
    color_discrete_map=COLORS,
    labels={
        "renda_per_capita": "Renda per capita (R$)",
        "dias_sem_chuva": "Dias sem chuva",
        "risco_alimentar": "Risco",
        "pop_vulneravel_pct": "Pop. vulnerável"
    },
    opacity=0.7
)
fig3.update_layout(**PLOT_BG, height=340,
    legend=dict(font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
    xaxis=dict(gridcolor='#1E293B', tickfont=dict(color='#94A3B8')),
    yaxis=dict(gridcolor='#1E293B', tickfont=dict(color='#94A3B8')),
)
st.plotly_chart(fig3, use_container_width=True)

# ── ALERTA ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:#1C1200;border:1px solid #3B1A00;border-radius:8px;
     padding:12px 16px;color:#FCD34D;font-size:12px;margin-top:4px'>
  ⚠️ <strong>{pct_alto}%</strong> dos municípios monitorados estão em zona de alto risco alimentar.
  Alerta gerado com <strong>60 dias de antecedência</strong> para ações preventivas.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("OrbitFood · Global Solution 2026 · FIAP")
