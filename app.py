# -*- coding: utf-8 -*-
"""
Follow Here - Edição Otimizada para Mobile e Cadastro em Lote
=============================================================
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO GERAL DA PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Follow Here",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# PERSISTÊNCIA EM DISCO / MEMÓRIA
# ---------------------------------------------------------------------------
DIRETORIO_BASE = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_MUSCULACAO = os.path.join(DIRETORIO_BASE, "dados_musculacao.csv")
ARQUIVO_CORRIDA = os.path.join(DIRETORIO_BASE, "dados_corrida.csv")
ARQUIVO_DIETA = os.path.join(DIRETORIO_BASE, "dados_dieta.csv")

COLUNAS_MUSCULACAO = [
    "data", "exercicio",
    "series_prescritas", "reps_prescritas", "carga_prescrita",
    "series_realizadas", "reps_realizadas", "carga_realizada",
]
COLUNAS_CORRIDA = [
    "data", "distancia_prescrita_km", "tempo_prescrito_min",
    "distancia_real_km", "tempo_real_min", "pace_real",
]
COLUNAS_DIETA = [
    "data", "refeicao", "prescrito", "alimento_consumido",
    "qtd_g", "calorias_prescritas", "calorias_consumidas",
]


def carregar_csv(caminho, colunas):
    if os.path.exists(caminho):
        try:
            df = pd.read_csv(caminho)
            if "data" in df.columns and not df.empty:
                df["data"] = pd.to_datetime(df["data"]).dt.date
            return df
        except Exception:
            return pd.DataFrame(columns=colunas)
    return pd.DataFrame(columns=colunas)


def salvar_dados_disco():
    try:
        st.session_state["musculacao"].to_csv(ARQUIVO_MUSCULACAO, index=False)
        st.session_state["corrida"].to_csv(ARQUIVO_CORRIDA, index=False)
        st.session_state["dieta"].to_csv(ARQUIVO_DIETA, index=False)
    except Exception:
        pass


def inicializar_estado():
    if "musculacao" not in st.session_state:
        st.session_state["musculacao"] = carregar_csv(ARQUIVO_MUSCULACAO, COLUNAS_MUSCULACAO)
    if "corrida" not in st.session_state:
        st.session_state["corrida"] = carregar_csv(ARQUIVO_CORRIDA, COLUNAS_CORRIDA)
    if "dieta" not in st.session_state:
        st.session_state["dieta"] = carregar_csv(ARQUIVO_DIETA, COLUNAS_DIETA)


inicializar_estado()

# ---------------------------------------------------------------------------
# NAVEGAÇÃO
# ---------------------------------------------------------------------------
st.sidebar.title("🎯 Follow Here")
pagina = st.sidebar.radio(
    "Navegação",
    ["🏠 Início", "📔 Diário", "📈 Evolução"],
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# PÁGINA 1 - INÍCIO
# ---------------------------------------------------------------------------
if pagina == "🏠 Início":
    st.title("🎯 Follow Here")
    st.caption("Acompanhamento inteligente de treino, corrida e dieta")

    df_musc = st.session_state["musculacao"]
    df_corrida = st.session_state["corrida"]
    df_dieta = st.session_state["dieta"]

    total_treinos = df_musc["carga_realizada"].notna().sum() if not df_musc.empty else 0
    total_refeicoes = df_dieta["calorias_consumidas"].notna().sum() if not df_dieta.empty else 0
    total_km = df_corrida["distancia_real_km"].dropna().sum() if not df_corrida.empty else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("🏋️ Exercícios executados", int(total_treinos))
    c2.metric("🥗 Refeições salvas", int(total_refeicoes))
    c3.metric("🏃 Km rodados", f"{total_km:.1f} km")

    st.info("💡 **Dica:** Vá na aba **Diário** para cadastrar seus treinos em lote (10+ exercícios de uma só vez).")

# ---------------------------------------------------------------------------
# PÁGINA 2 - DIÁRIO (Otimizado com Cadastro em Lote)
# ---------------------------------------------------------------------------
elif pagina == "📔 Diário":
    st.title("📔 Diário de Registro")

    data_selecionada = st.date_input("Data do Registro", value=date.today())

    aba_musc, aba_corrida, aba_dieta = st.tabs(["🏋️ Musculação", "🏃 Corrida", "🥗 Dieta"])

    # --- SUB-ABA MUSCULAÇÃO ---
    with aba_musc:
        st.subheader("🏋️ Treino em Lote (Múltiplos Exercícios)")
        
        modo_musc = st.radio("Selecione o que deseja cadastrar:", ["📋 Prescrição do Dia", "✅ Execução Real"], horizontal=True)

        if modo_musc == "📋 Prescrição do Dia":
            st.markdown("#### Cadastrar Prescrição em Lote")
            qtd_ex = st.number_input("Quantos exercícios deseja prescrever?", min_value=1, max_value=20, value=10, step=1)
            
            # Tabela base editável
            df_template = pd.DataFrame({
                "exercicio": [f"Exercício {i+1}" for i in range(qtd_ex)],
                "series_prescritas": [4] * qtd_ex,
                "reps_prescritas": [10] * qtd_ex,
                "carga_prescrita": [0.0] * qtd_ex,
            })

            edited_df = st.data_editor(
                df_template,
                column_config={
                    "exercicio": st.column_config.TextColumn("Exercício", required=True),
                    "series_prescritas": st.column_config.NumberColumn("Séries", min_value=1, step=1),
                    "reps_prescritas": st.column_config.NumberColumn("Reps", min_value=1, step=1),
                    "carga_prescrita": st.column_config.NumberColumn("Carga (kg)", min_value=0.0, step=0.5),
                },
                use_container_width=True,
                hide_index=True,
                key="editor_prescricao"
            )

            if st.button("💾 Salvar Toda Prescrição", use_container_width=True, type="primary"):
                df_atual = st.session_state["musculacao"]
                for _, row in edited_df.iterrows():
                    ex = str(row["exercicio"]).strip()
                    if ex:
                        filtro = (df_atual["data"] == data_selecionada) & (df_atual["exercicio"] == ex)
                        if filtro.any():
                            df_atual.loc[filtro, ["series_prescritas", "reps_prescritas", "carga_prescrita"]] = [
                                row["series_prescritas"], row["reps_prescritas"], row["carga_prescrita"]
                            ]
                        else:
                            nova = {
                                "data": data_selecionada, "exercicio": ex,
                                "series_prescritas": row["series_prescritas"],
                                "reps_prescritas": row["reps_prescritas"],
                                "carga_prescrita": row["carga_prescrita"],
                                "series_realizadas": np.nan, "reps_realizadas": np.nan, "carga_realizada": np.nan
                            }
                            df_atual = pd.concat([df_atual, pd.DataFrame([nova])], ignore_index=True)
                
                st.session_state["musculacao"] = df_atual
                salvar_dados_disco()
                st.success(f"Prescrição de {qtd_ex} exercícios salva para {data_selecionada}!")
                st.rerun()

        else:
            st.markdown("#### Registrar Execução Real em Lote")
            df_atual = st.session_state["musculacao"]
            prescritos = df_atual[df_atual["data"] == data_selecionada]

            if not prescritos.empty:
                df_exec = prescritos[["exercicio", "series_prescritas", "reps_prescritas", "carga_prescrita"]].copy()
                df_exec["series_realizadas"] = df_exec["series_prescritas"]
                df_exec["reps_realizadas"] = df_exec["reps_prescritas"]
                df_exec["carga_realizada"] = df_exec["carga_prescrita"]
            else:
                qtd_ex_r = st.number_input("Quantos exercícios você realizou?", min_value=1, max_value=20, value=10, step=1)
                df_exec = pd.DataFrame({
                    "exercicio": [f"Exercício {i+1}" for i in range(qtd_ex_r)],
                    "series_realizadas": [4] * qtd_ex_r,
                    "reps_realizadas": [10] * qtd_ex_r,
                    "carga_realizada": [0.0] * qtd_ex_r,
                })

            edited_exec = st.data_editor(
                df_exec,
                column_config={
                    "exercicio": st.column_config.TextColumn("Exercício", required=True),
                    "series_realizadas": st.column_config.NumberColumn("Séries Realizadas", min_value=0, step=1),
                    "reps_realizadas": st.column_config.NumberColumn("Reps Realizadas", min_value=0, step=1),
                    "carga_realizada": st.column_config.NumberColumn("Carga Executada (kg)", min_value=0.0, step=0.5),
                },
                use_container_width=True,
                hide_index=True,
                key="editor_execucao"
            )

            if st.button("✅ Salvar Execução do Treino", use_container_width=True, type="primary"):
                df_atual = st.session_state["musculacao"]
                for _, row in edited_exec.iterrows():
                    ex = str(row["exercicio"]).strip()
                    if ex:
                        filtro = (df_atual["data"] == data_selecionada) & (df_atual["exercicio"] == ex)
                        if filtro.any():
                            df_atual.loc[filtro, ["series_realizadas", "reps_realizadas", "carga_realizada"]] = [
                                row["series_realizadas"], row["reps_realizadas"], row["carga_realizada"]
                            ]
                        else:
                            nova = {
                                "data": data_selecionada, "exercicio": ex,
                                "series_prescritas": np.nan, "reps_prescritas": np.nan, "carga_prescrita": np.nan,
                                "series_realizadas": row["series_realizadas"],
                                "reps_realizadas": row["reps_realizadas"],
                                "carga_realizada": row["carga_realizada"]
                            }
                            df_atual = pd.concat([df_atual, pd.DataFrame([nova])], ignore_index=True)

                st.session_state["musculacao"] = df_atual
                salvar_dados_disco()
                st.success("Execução do treino completa gravada com sucesso!")
                st.rerun()

        st.markdown("---")
        st.markdown("#### Resumo do Dia")
        registros_dia = st.session_state["musculacao"][st.session_state["musculacao"]["data"] == data_selecionada]
        if not registros_dia.empty:
            st.dataframe(
                registros_dia[["exercicio", "carga_prescrita", "carga_realizada", "series_realizadas", "reps_realizadas"]].rename(
                    columns={"exercicio": "Exercício", "carga_prescrita": "Meta (kg)", "carga_realizada": "Feito (kg)", "series_realizadas": "Séries", "reps_realizadas": "Reps"}
                ),
                use_container_width=True, hide_index=True
            )

    # --- SUB-ABA CORRIDA ---
    with aba_corrida:
        st.subheader("🏃 Corrida")
        col1, col2 = st.columns(2)
        dist_p = col1.number_input("Distância Meta (km)", min_value=0.0, step=0.5, value=0.0)
        tempo_p = col2.number_input("Tempo Meta (min)", min_value=0.0, step=1.0, value=0.0)
        
        col3, col4 = st.columns(2)
        dist_r = col3.number_input("Distância Executada (km)", min_value=0.0, step=0.1, value=0.0)
        tempo_r = col4.number_input("Tempo Gasto (min)", min_value=0.0, step=1.0, value=0.0)

        if st.button("🏃 Salvar Registro de Corrida", use_container_width=True):
            pace = tempo_r / dist_r if dist_r > 0 else np.nan
            df_c = st.session_state["corrida"]
            nova_corrida = {
                "data": data_selecionada, "distancia_prescrita_km": dist_p,
                "tempo_prescrito_min": tempo_p, "distancia_real_km": dist_r,
                "tempo_real_min": tempo_r, "pace_real": pace
            }
            st.session_state["corrida"] = pd.concat([df_c, pd.DataFrame([nova_corrida])], ignore_index=True)
            salvar_dados_disco()
            st.success("Corrida gravada!")
            st.rerun()

    # --- SUB-ABA DIETA ---
    with aba_dieta:
        st.subheader("🥗 Dieta do Dia")
        
        refeicoes = ["Café da Manhã", "Almoço", "Lanche", "Jantar"]
        df_dieta_temp = pd.DataFrame({
            "refeicao": refeicoes,
            "prescrito": [""] * 4,
            "calorias_prescritas": [0.0] * 4,
            "alimento_consumido": [""] * 4,
            "calorias_consumidas": [0.0] * 4,
        })

        edited_dieta = st.data_editor(
            df_dieta_temp,
            column_config={
                "refeicao": st.column_config.TextColumn("Refeição", disabled=True),
                "prescrito": st.column_config.TextColumn("Planejado"),
                "calorias_prescritas": st.column_config.NumberColumn("Meta Kcal", step=50.0),
                "alimento_consumido": st.column_config.TextColumn("Consumido Real"),
                "calorias_consumidas": st.column_config.NumberColumn("Kcal Consumidas", step=50.0),
            },
            use_container_width=True,
            hide_index=True,
            key="editor_dieta"
        )

        if st.button("🥗 Salvar Dieta do Dia", use_container_width=True):
            df_d = st.session_state["dieta"]
            for _, row in edited_dieta.iterrows():
                nova_d = {
                    "data": data_selecionada, "refeicao": row["refeicao"],
                    "prescrito": row["prescrito"], "qtd_g": 0,
                    "calorias_prescritas": row["calorias_prescritas"],
                    "alimento_consumido": row["alimento_consumido"],
                    "calorias_consumidas": row["calorias_consumidas"]
                }
                df_d = pd.concat([df_d, pd.DataFrame([nova_d])], ignore_index=True)
            st.session_state["dieta"] = df_d
            salvar_dados_disco()
            st.success("Dieta salva!")
            st.rerun()

# ---------------------------------------------------------------------------
# PÁGINA 3 - EVOLUÇÃO
# ---------------------------------------------------------------------------
elif pagina == "📈 Evolução":
    st.title("📈 Evolução e Análises")

    df_musc = st.session_state["musculacao"]
    df_corrida = st.session_state["corrida"]
    df_dieta = st.session_state["dieta"]

    # KPIs Principais
    c1, c2, c3 = st.columns(3)
    c1.metric("Treinos Realizados", len(df_musc["data"].unique()) if not df_musc.empty else 0)
    c2.metric("Km Acumulados", f"{df_corrida['distancia_real_km'].sum():.1f} km" if not df_corrida.empty else "0.0 km")
    c3.metric("Média Kcal/Dia", f"{df_dieta['calorias_consumidas'].mean():.0f} kcal" if not df_dieta.empty and df_dieta['calorias_consumidas'].mean() > 0 else "0 kcal")

    st.markdown("---")

    # Gráfico de Evolução de Carga
    st.subheader("🏋️ Progresso de Carga por Exercício")
    if not df_musc.empty and df_musc["carga_realizada"].notna().any():
        dados_carga = df_musc.dropna(subset=["carga_realizada"]).sort_values("data")
        fig = px.line(
            dados_carga, x="data", y="carga_realizada", color="exercicio", markers=True,
            labels={"data": "Data", "carga_realizada": "Carga (kg)", "exercicio": "Exercício"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Cadastre execuções no Diário para liberar o gráfico de carga.")
