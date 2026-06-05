import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier

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
        'ndvi_variacao_90d':     ndvi_var.round(4),
        'dias_sem_chuva':        dias_sem_chuva,
        'temp_anomalia':         temp_anomalia.round(3),
        'producao_agricola_var': prod_var.round(2),
        'idh_municipal':         idh.round(4),
        'renda_per_capita':      renda_pc.round(2),
        'distancia_polo_km':     dist_polo.round(1),
        'bioma':                 bioma_col,
        'pop_vulneravel_pct':    pop_vuln.round(4),
        'risco_alimentar':       risco,
        'score':                 score.round(4)
    })

    le = LabelEncoder()
    df['bioma_enc'] = le.fit_transform(df['bioma'])
    df['risco_enc'] = df['risco_alimentar'].map({'Baixo': 0, 'Médio': 1, 'Alto': 2})

    FEATURES = ['ndvi_variacao_90d', 'dias_sem_chuva', 'temp_anomalia',
                'producao_agricola_var', 'idh_municipal', 'renda_per_capita',
                'distancia_polo_km', 'bioma_enc', 'pop_vulneravel_pct']

    X, y = df[FEATURES], df['risco_enc']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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
    f1_rf  = f1_score(y_test, y_pred_rf,  average=None)
    f1_xgb = f1_score(y_test, y_pred_xgb, average=None)
    cm_rf  = confusion_matrix(y_test, y_pred_rf)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)

    return dict(
        df=df, metricas=metricas, fi_rf=fi_rf, fi_xgb=fi_xgb,
        f1_rf=f1_rf, f1_xgb=f1_xgb, cm_rf=cm_rf, cm_xgb=cm_xgb,
        cv_rf=cv_rf, cv_xgb=cv_xgb,
        y_test=y_test, y_pred_rf=y_pred_rf, y_pred_xgb=y_pred_xgb,
        FEATURES=FEATURES, rf=rf, xgb=xgb,
        NOMES_FEAT={
            'ndvi_variacao_90d':     'NDVI (últimos 90 dias)',
            'dias_sem_chuva':        'Dias sem chuva',
            'temp_anomalia':         'Anomalia temperatura',
            'producao_agricola_var': 'Var. produção agrícola',
            'idh_municipal':         'IDH municipal',
            'renda_per_capita':      'Renda per capita',
            'distancia_polo_km':     'Distância ao polo (km)',
            'bioma_enc':             'Bioma',
            'pop_vulneravel_pct':    'Pop. vulnerável (%)',
        }
    )

PLOT_BG = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94A3B8', size=11),
    margin=dict(l=10, r=10, t=36, b=10),
)

COLORS = {'Baixo': '#4ADE80', 'Médio': '#FCD34D', 'Alto': '#F87171'}
