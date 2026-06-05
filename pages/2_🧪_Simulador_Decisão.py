import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_loader import gerar_dados_e_modelos

dados = gerar_dados_e_modelos()
xgb   = dados['xgb']
le_bioma_map = {'Amazônia': 0, 'Caatinga': 1, 'Cerrado': 2,
                'Mata Atlântica': 3, 'Pampa': 4, 'Pantanal': 5}

# ── HEADER ────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#0B1221;border-radius:10px;padding:16px 20px;margin-bottom:20px;border:1px solid #1E293B;'>
  <span style='font-size:20px'>🧪</span>
  <span style='font-size:18px;font-weight:700;color:#14B8A6;margin-left:8px'>Simulador de Decisão</span>
  <span style='color:#334155;margin:0 8px'>|</span>
  <span style='font-size:13px;color:#64748B'>Preveja o risco alimentar de um município</span>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ Como utilizar o Simulador", expanded=True):
    st.markdown("""
    Ajuste os parâmetros abaixo para simular o perfil de um município e obter a previsão de risco alimentar com o modelo **XGBoost**:

    1. **Renda Per Capita**: renda média da população local em R$
    2. **Dias sem chuva**: período de estiagem atual na região
    3. **IDH Municipal**: Índice de Desenvolvimento Humano (0.40 a 0.90)
    4. **Variação Produção Agrícola (%)**: aumento ou redução da produção agrícola local
    5. **NDVI (variação 90 dias)**: indicador da saúde da vegetação; valores negativos sugerem degradação ambiental
    6. **População vulnerável (%)**: percentual da população em situação de vulnerabilidade social
    7. **Anomalia de temperatura (°C)**: diferença entre a temperatura observada e a média histórica da região
    8. **Distância ao polo (km)**: distância até o principal centro regional de abastecimento e distribuição de alimentos
    """)

# ── INPUTS ────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 💰 Indicadores Socioeconômicos")
    renda = st.number_input("Renda Per Capita (R$):", min_value=300.0, max_value=5000.0, value=800.0, step=50.0)
    st.markdown(
    """
    <div style='color: #94A3B8; font-size: 12px; margin-top: -10px;'>
        ℹ️ Faixa utilizada no treinamento do modelo: R$ 300,00 até R$ 5.000,00.
    </div>
    """, 
    unsafe_allow_html=True
)
    idh     = st.slider("IDH Municipal:", 0.40, 0.90, 0.60, step=0.01)
    pop_vuln= st.slider("Pop. vulnerável (%):", 0.05, 0.70, 0.28, step=0.01)

with col2:
    st.markdown("##### 🌾 Indicadores Agrícolas")
    prod    = st.number_input("Variação Produção Agrícola (%):", min_value=-60.0, max_value=50.0, value=0.0, step=1.0)
    ndvi    = st.slider("Variação NDVI 90 dias:", -0.50, 0.30, -0.08, step=0.01)
    dist    = st.slider("Distância ao polo (km):", 5, 500, 80)

with col3:
    st.markdown("##### 🌤️ Indicadores Climáticos")
    seca    = st.slider("Dias sem chuva:", 0, 120, 30)
    temp    = st.slider("Anomalia de temperatura (°C):", -3.0, 6.0, 1.2, step=0.1)
    bioma_s = st.selectbox("Bioma:", list(le_bioma_map.keys()))

st.markdown("<br>", unsafe_allow_html=True)

# ── BOTÃO PREVER ──────────────────────────────────────────────────────
col_btn = st.columns([1, 2, 1])[1]
with col_btn:
    prever = st.button("🚀 Prever Risco Alimentar", use_container_width=True)

if prever:
    bioma_enc = le_bioma_map[bioma_s]
    input_df  = pd.DataFrame([[
        ndvi, seca, temp, prod, idh, renda, dist, bioma_enc, pop_vuln
    ]], columns=[
        'ndvi_variacao_90d', 'dias_sem_chuva', 'temp_anomalia',
        'producao_agricola_var', 'idh_municipal', 'renda_per_capita',
        'distancia_polo_km', 'bioma_enc', 'pop_vulneravel_pct'
    ])

    pred    = xgb.predict(input_df)[0]
    proba   = xgb.predict_proba(input_df)[0]
    label   = {0: 'Baixo', 1: 'Médio', 2: 'Alto'}[pred]
    cores   = {'Baixo': ('#4ADE80', '#071A0E', '#052E16'),
               'Médio': ('#FCD34D', '#1C1200', '#3B1A00'),
               'Alto':  ('#F87171', '#1C0808', '#3B0A0A')}
    cor, bg, brd = cores[label]
    icones  = {'Baixo': '🟢', 'Médio': '🟡', 'Alto': '🔴'}

    st.markdown(f"""
    <div style='background:{bg};border:2px solid {brd};border-radius:12px;
         padding:24px;text-align:center;margin:16px 0;'>
      <div style='font-size:40px'>{icones[label]}</div>
      <div style='font-size:14px;color:#94A3B8;margin:8px 0 4px'>Risco previsto pelo modelo XGBoost</div>
      <div style='font-size:36px;font-weight:800;color:{cor}'>{label} Risco</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Probabilidade por classe")
    c1p, c2p, c3p = st.columns(3)
    for col, (cls, prob) in zip([c1p, c2p, c3p],
        [('Baixo', proba[0]), ('Médio', proba[1]), ('Alto', proba[2])]):
        cor_p = cores[cls][0]
        with col:
            st.markdown(f"""
            <div style='background:#1E293B;border-radius:8px;padding:14px;text-align:center;'>
              <div style='font-size:26px;font-weight:700;color:{cor_p}'>{prob*100:.1f}%</div>
              <div style='font-size:11px;color:#64748B'>{cls} Risco</div>
              <div style='height:6px;border-radius:3px;background:#0F172A;margin-top:8px;overflow:hidden'>
                <div style='height:100%;width:{prob*100}%;background:{cor_p};border-radius:3px'></div>
              </div>
            </div>""", unsafe_allow_html=True)

    ações = {
        'Baixo': "🟢 <strong>Situação estável.</strong> Mantenha o monitoramento regular e acompanhe indicadores mensalmente.",
        'Médio': "🟡 <strong>Atenção necessária.</strong> Revise estoques alimentares e prepare plano de contingência.",
        'Alto':  "🔴 <strong>Ação imediata.</strong> Acione programas sociais, contate a CONAB e inicie distribuição emergencial."
    }
    st.markdown(f"""
    <div style='background:#0A1628;border-left:3px solid #14B8A6;border-radius:4px;
         padding:12px 16px;font-size:13px;color:#94A3B8;margin-top:12px'>
      <strong style='color:#14B8A6'>Recomendação:</strong> {ações[label]}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("OrbitFood · Global Solution 2026 · FIAP")
