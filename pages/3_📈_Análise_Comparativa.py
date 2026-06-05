import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_loader import gerar_dados_e_modelos, PLOT_BG, COLORS

dados = gerar_dados_e_modelos()
df    = dados['df']

# ── HEADER ────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#0B1221;border-radius:10px;padding:16px 20px;margin-bottom:20px;border:1px solid #1E293B;'>
  <span style='font-size:20px'>📈</span>
  <span style='font-size:18px;font-weight:700;color:#14B8A6;margin-left:8px'>Análise Comparativa</span>
  <span style='color:#334155;margin:0 8px'>|</span>
  <span style='font-size:13px;color:#64748B'>Insights detalhados por bioma e variável</span>
</div>
""", unsafe_allow_html=True)

# ── FILTROS ────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🔍 Filtros")
bioma_sel = st.sidebar.selectbox("Selecione o Bioma:", df['bioma'].unique().tolist())
risco_sel = st.sidebar.multiselect(
    "Nível de risco", ['Baixo', 'Médio', 'Alto'],
    default=['Baixo', 'Médio', 'Alto']
)

df_tab = df[(df['bioma'] == bioma_sel) & (df['risco_alimentar'].isin(risco_sel))]

# ── KPIs DO BIOMA ─────────────────────────────────────────────────────
st.markdown(f"#### Resumo — {bioma_sel}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total municípios", len(df_tab))
c2.metric("Alto risco", int((df_tab['risco_alimentar'] == 'Alto').sum()))
c3.metric("IDH médio", f"{df_tab['idh_municipal'].mean():.3f}")
c4.metric("Renda média (R$)", f"{df_tab['renda_per_capita'].mean():,.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# ── GRÁFICOS ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"#### Distribuição de risco — {bioma_sel}")
    fig1 = px.histogram(
        df_tab, x="risco_alimentar", color="risco_alimentar",
        color_discrete_map=COLORS,
        category_orders={"risco_alimentar": ["Baixo", "Médio", "Alto"]}
    )
    fig1.update_layout(**PLOT_BG, height=300, showlegend=False,
        xaxis=dict(tickfont=dict(color='#94A3B8')),
        yaxis=dict(gridcolor='#1E293B', tickfont=dict(color='#94A3B8')),
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown(f"""
        #### Renda per capita vs Risco — {bioma_sel}
        <p style="color: #64748B; font-size: 0.85em; margin-top: -10px; margin-bottom: 10px;">
        As caixas mostram a renda da maioria dos municípios. Pontos isolados representam casos de rendas excepcionalmente altas.
        </p>
    """, unsafe_allow_html=True)
    
    fig2 = px.box(
        df_tab, x="risco_alimentar", y="renda_per_capita",
        color="risco_alimentar", color_discrete_map=COLORS,
        category_orders={"risco_alimentar": ["Baixo", "Médio", "Alto"]}
    )
    
    estilo_box = PLOT_BG.copy()
    estilo_box['margin'] = dict(l=20, r=20, t=10, b=20)
    
    fig2.update_layout(
        **estilo_box, 
        height=270, 
        showlegend=False,
        xaxis=dict(title="", tickfont=dict(color='#94A3B8')),
        yaxis=dict(title="Renda (R$)", gridcolor='#1E293B', tickfont=dict(color='#94A3B8'))
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── GRÁFICOS ──────────────────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    # Título e subtítulo com formatação HTML integrada
    st.markdown(f"""
        #### Dias sem chuva vs Risco — {bioma_sel}
        <p style="color: #64748B; font-size: 0.9em; margin-top: -10px; margin-bottom: 10px;">
        A largura da forma indica a concentração de municípios; o centro marca a mediana.
        </p>
    """, unsafe_allow_html=True)
    
    # Configuração do gráfico de violino
    fig3 = px.violin(
        df_tab, x="risco_alimentar", y="dias_sem_chuva",
        color="risco_alimentar", color_discrete_map=COLORS,
        box=True, 
        category_orders={"risco_alimentar": ["Baixo", "Médio", "Alto"]}
    )
    
    # Limpeza do que aparece ao passar o mouse
    fig3.update_traces(
        meanline_visible=True,
        hovertemplate='<b>%{x}</b><br>Dias sem chuva: %{y:.0f}<extra></extra>'
    )
    
    # Criação de um estilo customizado para evitar o conflito de margens
    estilo_grafico = PLOT_BG.copy()
    estilo_grafico['margin'] = dict(l=20, r=20, t=10, b=20)
    
    # Atualização do layout usando o estilo customizado
    fig3.update_layout(
        **estilo_grafico, 
        height=270, 
        showlegend=False,
        xaxis=dict(title="", tickfont=dict(color='#94A3B8')),
        yaxis=dict(title="Dias sem chuva", gridcolor='#1E293B', tickfont=dict(color='#94A3B8'))
    )
    
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.markdown(f"#### IDH vs Pop. vulnerável — {bioma_sel}")
    fig4 = px.scatter(
        df_tab, x="idh_municipal", y="pop_vulneravel_pct",
        color="risco_alimentar", color_discrete_map=COLORS,
        opacity=0.7,
        labels={"idh_municipal": "IDH", "pop_vulneravel_pct": "Pop. vulnerável (%)"}
    )
    fig4.update_layout(**PLOT_BG, height=300,
        legend=dict(font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(gridcolor='#1E293B', tickfont=dict(color='#94A3B8')),
        yaxis=dict(gridcolor='#1E293B', tickfont=dict(color='#94A3B8')),
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── TABELA ────────────────────────────────────────────────────────────
st.markdown(f"#### 📋 Dados detalhados — {bioma_sel}")
st.dataframe(
    df_tab[['bioma', 'risco_alimentar', 'renda_per_capita', 'dias_sem_chuva',
             'idh_municipal', 'pop_vulneravel_pct', 'ndvi_variacao_90d',
             'temp_anomalia', 'distancia_polo_km']].reset_index(drop=True),
    use_container_width=True, height=320
)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("OrbitFood · Global Solution 2026 · FIAP")
