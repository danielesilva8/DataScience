import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_loader import gerar_dados_e_modelos, PLOT_BG

dados      = gerar_dados_e_modelos()
metricas   = dados['metricas']
fi_rf      = dados['fi_rf']
fi_xgb     = dados['fi_xgb']
f1_rf      = dados['f1_rf']
f1_xgb     = dados['f1_xgb']
cm_rf      = dados['cm_rf']
cm_xgb     = dados['cm_xgb']
cv_rf      = dados['cv_rf']
cv_xgb     = dados['cv_xgb']
NOMES_FEAT = dados['NOMES_FEAT']

# ── HEADER ────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#0B1221;border-radius:10px;padding:16px 20px;margin-bottom:20px;border:1px solid #1E293B;'>
  <span style='font-size:20px'>🤖</span>
  <span style='font-size:18px;font-weight:700;color:#14B8A6;margin-left:8px'>Desempenho do Modelo</span>
  <span style='color:#334155;margin:0 8px'>|</span>
  <span style='font-size:13px;color:#64748B'>Métricas, matrizes de confusão e feature importance</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<span style="background:#052E16;color:#4ADE80;font-size:11px;padding:3px 10px;border-radius:4px">🏆 Modelo recomendado: XGBoost</span>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── MÉTRICAS ──────────────────────────────────────────────────────────
st.markdown("#### Comparativo de métricas — XGBoost vs Random Forest")
c1, c2, c3, c4 = st.columns(4)
for col, met in zip([c1, c2, c3, c4], ['Acurácia', 'F1 Macro', 'Precision', 'Recall']):
    xgb_v = metricas['XGBoost'][met]
    rf_v  = metricas['Random Forest'][met]
    delta = f"+{(xgb_v - rf_v)*100:.1f}pp vs RF"
    col.metric(f"XGBoost — {met}", f"{xgb_v*100:.1f}%", delta)

st.markdown("<br>", unsafe_allow_html=True)

# ── CV ────────────────────────────────────────────────────────────────
col_cv1, col_cv2 = st.columns(2)
with col_cv1:
    st.markdown(f"""
    <div style='background:#1E293B;border-radius:8px;padding:14px;text-align:center'>
      <div style='font-size:11px;color:#64748B;margin-bottom:4px'>CV F1-macro — XGBoost (5-fold)</div>
      <div style='font-size:28px;font-weight:700;color:#4ADE80'>{cv_xgb.mean()*100:.1f}%</div>
      <div style='font-size:11px;color:#475569'>± {cv_xgb.std()*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with col_cv2:
    st.markdown(f"""
    <div style='background:#1E293B;border-radius:8px;padding:14px;text-align:center'>
      <div style='font-size:11px;color:#64748B;margin-bottom:4px'>CV F1-macro — Random Forest (5-fold)</div>
      <div style='font-size:28px;font-weight:700;color:#94A3B8'>{cv_rf.mean()*100:.1f}%</div>
      <div style='font-size:11px;color:#475569'>± {cv_rf.std()*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── COMPARATIVO MÉTRICAS ──────────────────────────────────────────────
st.markdown("#### Gráfico comparativo de métricas")
mets  = ['Acurácia', 'F1 Macro', 'Precision', 'Recall']
v_rf  = [metricas['Random Forest'][m] for m in mets]
v_xgb = [metricas['XGBoost'][m]       for m in mets]

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(name='Random Forest', x=mets, y=v_rf,
    marker_color='#334155', text=[f"{v:.3f}" for v in v_rf],
    textposition='outside', textfont=dict(color='#94A3B8', size=11)))
fig_comp.add_trace(go.Bar(name='XGBoost', x=mets, y=v_xgb,
    marker_color='#14B8A6', text=[f"{v:.3f}" for v in v_xgb],
    textposition='outside', textfont=dict(color='#E2E8F0', size=11)))
fig_comp.update_layout(**PLOT_BG, height=320, barmode='group',
    yaxis=dict(range=[0, 1.12], gridcolor='#1E293B', tickfont=dict(color='#94A3B8')),
    xaxis=dict(tickfont=dict(color='#94A3B8')),
    legend=dict(font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
)
st.plotly_chart(fig_comp, use_container_width=True)

# ── F1 POR CLASSE ─────────────────────────────────────────────────────
st.markdown("#### F1-score por classe de risco")
classes = ['Baixo risco', 'Médio risco', 'Alto risco']
fig_f1 = go.Figure()
fig_f1.add_trace(go.Bar(name='Random Forest', x=classes, y=f1_rf,
    marker_color='#334155', text=[f"{v:.3f}" for v in f1_rf],
    textposition='outside', textfont=dict(color='#94A3B8', size=11)))
fig_f1.add_trace(go.Bar(name='XGBoost', x=classes, y=f1_xgb,
    marker_color=['#4ADE80', '#FCD34D', '#F87171'],
    text=[f"{v:.3f}" for v in f1_xgb],
    textposition='outside', textfont=dict(color='#E2E8F0', size=11)))
fig_f1.update_layout(**PLOT_BG, height=300, barmode='group',
    yaxis=dict(range=[0, 1.12], gridcolor='#1E293B', tickfont=dict(color='#94A3B8')),
    xaxis=dict(tickfont=dict(color='#94A3B8')),
    legend=dict(font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
)
st.plotly_chart(fig_f1, use_container_width=True)

# ── MATRIZES DE CONFUSÃO ──────────────────────────────────────────────
st.markdown("#### Matrizes de confusão")
col1, col2 = st.columns(2)
for col, cm, titulo in [(col1, cm_rf, "Random Forest"), (col2, cm_xgb, "XGBoost")]:
    with col:
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           x=['Baixo', 'Médio', 'Alto'],
                           y=['Baixo', 'Médio', 'Alto'],
                           title=titulo)
        fig_cm.update_layout(**PLOT_BG, height=320,
            title=dict(font=dict(size=13, color='#CBD5E1')),
            coloraxis_showscale=False,
        )
        fig_cm.update_xaxes(tickfont=dict(color='#94A3B8'))
        fig_cm.update_yaxes(tickfont=dict(color='#94A3B8'))
        st.plotly_chart(fig_cm, use_container_width=True)

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────
st.markdown("#### Feature importance — Top 9 variáveis")
col1, col2 = st.columns(2)
for col, fi, titulo, cor in [
    (col1, fi_rf,  "Random Forest", '#334155'),
    (col2, fi_xgb, "XGBoost",       '#14B8A6')
]:
    with col:
        fi_s = fi.sort_values(ascending=True)
        nomes = [NOMES_FEAT.get(k, k) for k in fi_s.index]
        fig_fi = go.Figure(go.Bar(
            x=(fi_s.values * 100).round(1), y=nomes,
            orientation='h', marker_color=cor,
            text=[f"{v:.1f}%" for v in fi_s.values * 100],
            textposition='outside', textfont=dict(color='#64748B', size=10),
        ))
        fig_fi.update_layout(**PLOT_BG, height=300,
            title=dict(text=titulo, font=dict(size=13, color='#CBD5E1')),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(autorange='reversed', tickfont=dict(size=10, color='#94A3B8')),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

st.markdown("""
<div style='background:#0A1628;border-left:3px solid #14B8A6;border-radius:4px;
     padding:12px 16px;font-size:11px;color:#94A3B8;line-height:1.7'>
  <strong style='color:#14B8A6'>✅ Conclusão:</strong> O XGBoost superou o Random Forest em todas as métricas,
  com destaque para a classe Alto Risco (F1 = {:.1f}% vs {:.1f}%). O modelo é recomendado para produção
  por reduzir falsos negativos — municípios em risco que seriam ignorados pelo sistema.
</div>
""".format(f1_xgb[2]*100, f1_rf[2]*100), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("OrbitFood · Global Solution 2026 · FIAP")
