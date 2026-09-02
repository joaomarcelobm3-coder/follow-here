# -*- coding: utf-8 -*-
"""
Follow Here
=============
Aplicação Streamlit para acompanhamento diário de treinos de musculação,
corrida e dieta, com comparativo entre o que foi PRESCRITO e o que foi
REALIZADO, além de um painel de evolução com gráficos, indicadores (KPIs),
previsão de tendência (regressão linear) e interpretação automática dos
resultados.

Os dados vivem em st.session_state durante a execução, mas também são
persistidos automaticamente em disco (arquivos CSV locais), de forma que
o histórico não se perca ao reiniciar a aplicação ou fechar o navegador.
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
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# PERSISTÊNCIA EM DISCO (ARQUIVOS CSV LOCAIS)
# ---------------------------------------------------------------------------
# Caminhos dos arquivos CSV usados para persistir cada tabela. Ficam na mesma
# pasta do script, permitindo que os dados sobrevivam a reinícios do app.
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
    """
    Carrega um DataFrame a partir de um arquivo CSV, se ele existir no disco.
    Caso contrário, retorna um DataFrame vazio com as colunas corretas.
    A coluna 'data' é convertida para datetime.date, garantindo compatibilidade
    com os seletores de data do Streamlit (st.date_input).
    """
    if os.path.exists(caminho):
        df = pd.read_csv(caminho)
        if "data" in df.columns and not df.empty:
            df["data"] = pd.to_datetime(df["data"]).dt.date
        return df
    return pd.DataFrame(columns=colunas)


def salvar_dados_disco():
    """
    Persiste as três tabelas atuais do st.session_state em arquivos CSV
    locais. Deve ser chamada ao final de toda gravação (prescrição ou
    execução real) de musculação, corrida ou dieta, garantindo que nada
    seja perdido ao reiniciar a aplicação.
    """
    st.session_state["musculacao"].to_csv(ARQUIVO_MUSCULACAO, index=False)
    st.session_state["corrida"].to_csv(ARQUIVO_CORRIDA, index=False)
    st.session_state["dieta"].to_csv(ARQUIVO_DIETA, index=False)


# ---------------------------------------------------------------------------
# INICIALIZAÇÃO DO ESTADO DA SESSÃO
# ---------------------------------------------------------------------------
def inicializar_estado():
    """
    Garante que todas as estruturas de dados existam em st.session_state
    antes de qualquer interação do usuário. Na primeira execução da sessão,
    os dados são carregados automaticamente dos arquivos CSV em disco (se
    existirem), permitindo que o histórico persista entre reinícios do app.
    """

    # Histórico de musculação: cada linha representa um exercício em uma
    # data específica, podendo conter dados prescritos, realizados, ou ambos.
    if "musculacao" not in st.session_state:
        st.session_state["musculacao"] = carregar_csv(ARQUIVO_MUSCULACAO, COLUNAS_MUSCULACAO)

    # Histórico de corrida: uma linha por data, com meta prescrita e o que
    # foi de fato realizado.
    if "corrida" not in st.session_state:
        st.session_state["corrida"] = carregar_csv(ARQUIVO_CORRIDA, COLUNAS_CORRIDA)

    # Histórico de dieta: uma linha por refeição/data, podendo ter apenas a
    # prescrição, apenas o consumo real, ou ambos.
    if "dieta" not in st.session_state:
        st.session_state["dieta"] = carregar_csv(ARQUIVO_DIETA, COLUNAS_DIETA)


inicializar_estado()


# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES - MUSCULAÇÃO
# ---------------------------------------------------------------------------
def salvar_prescricao_musculacao(data_ref, exercicio, series, reps, carga):
    """Cria ou atualiza a prescrição de um exercício em uma data."""
    df = st.session_state["musculacao"]
    filtro = (df["data"] == data_ref) & (df["exercicio"] == exercicio)

    if filtro.any():
        df.loc[filtro, ["series_prescritas", "reps_prescritas", "carga_prescrita"]] = [series, reps, carga]
    else:
        nova_linha = {
            "data": data_ref, "exercicio": exercicio,
            "series_prescritas": series, "reps_prescritas": reps, "carga_prescrita": carga,
            "series_realizadas": np.nan, "reps_realizadas": np.nan, "carga_realizada": np.nan,
        }
        st.session_state["musculacao"] = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)

    salvar_dados_disco()


def salvar_realizado_musculacao(data_ref, exercicio, series, reps, carga):
    """Registra a execução real de um exercício, casando com a prescrição se existir."""
    df = st.session_state["musculacao"]
    filtro = (df["data"] == data_ref) & (df["exercicio"] == exercicio)

    if filtro.any():
        # Atualiza a primeira ocorrência sem execução ainda registrada, senão a primeira encontrada
        idx = df[filtro].index[0]
        df.loc[idx, ["series_realizadas", "reps_realizadas", "carga_realizada"]] = [series, reps, carga]
    else:
        nova_linha = {
            "data": data_ref, "exercicio": exercicio,
            "series_prescritas": np.nan, "reps_prescritas": np.nan, "carga_prescrita": np.nan,
            "series_realizadas": series, "reps_realizadas": reps, "carga_realizada": carga,
        }
        st.session_state["musculacao"] = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)

    salvar_dados_disco()


# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES - CORRIDA
# ---------------------------------------------------------------------------
def salvar_prescricao_corrida(data_ref, distancia, tempo):
    """Cria ou atualiza a meta de corrida (distância/tempo) de uma data."""
    df = st.session_state["corrida"]
    filtro = df["data"] == data_ref

    if filtro.any():
        df.loc[filtro, ["distancia_prescrita_km", "tempo_prescrito_min"]] = [distancia, tempo]
    else:
        nova_linha = {
            "data": data_ref,
            "distancia_prescrita_km": distancia, "tempo_prescrito_min": tempo,
            "distancia_real_km": np.nan, "tempo_real_min": np.nan, "pace_real": np.nan,
        }
        st.session_state["corrida"] = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)

    salvar_dados_disco()


def salvar_realizado_corrida(data_ref, distancia, tempo):
    """Registra a corrida realizada, calculando o pace (min/km) automaticamente."""
    pace = tempo / distancia if distancia > 0 else np.nan
    df = st.session_state["corrida"]
    filtro = df["data"] == data_ref

    if filtro.any():
        idx = df[filtro].index[0]
        df.loc[idx, ["distancia_real_km", "tempo_real_min", "pace_real"]] = [distancia, tempo, pace]
    else:
        nova_linha = {
            "data": data_ref,
            "distancia_prescrita_km": np.nan, "tempo_prescrito_min": np.nan,
            "distancia_real_km": distancia, "tempo_real_min": tempo, "pace_real": pace,
        }
        st.session_state["corrida"] = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)

    salvar_dados_disco()
    return pace


# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES - DIETA
# ---------------------------------------------------------------------------
def salvar_prescricao_dieta(data_ref, refeicao, prescrito_texto, calorias_prescritas):
    """Cria ou atualiza a prescrição de uma refeição em uma data."""
    df = st.session_state["dieta"]
    filtro = (df["data"] == data_ref) & (df["refeicao"] == refeicao) & (df["alimento_consumido"].isna())

    if filtro.any():
        idx = df[filtro].index[0]
        df.loc[idx, ["prescrito", "calorias_prescritas"]] = [prescrito_texto, calorias_prescritas]
    else:
        nova_linha = {
            "data": data_ref, "refeicao": refeicao,
            "prescrito": prescrito_texto, "alimento_consumido": np.nan,
            "qtd_g": np.nan, "calorias_prescritas": calorias_prescritas, "calorias_consumidas": np.nan,
        }
        st.session_state["dieta"] = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)

    salvar_dados_disco()


def salvar_realizado_dieta(data_ref, refeicao, alimento, qtd, calorias):
    """Registra o consumo real de uma refeição, reaproveitando a prescrição pendente se existir."""
    df = st.session_state["dieta"]
    filtro = (df["data"] == data_ref) & (df["refeicao"] == refeicao) & (df["alimento_consumido"].isna())

    if filtro.any():
        idx = df[filtro].index[0]
        df.loc[idx, ["alimento_consumido", "qtd_g", "calorias_consumidas"]] = [alimento, qtd, calorias]
    else:
        nova_linha = {
            "data": data_ref, "refeicao": refeicao,
            "prescrito": np.nan, "alimento_consumido": alimento,
            "qtd_g": qtd, "calorias_prescritas": np.nan, "calorias_consumidas": calorias,
        }
        st.session_state["dieta"] = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)

    salvar_dados_disco()


# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES - ANÁLISE / PREVISÃO
# ---------------------------------------------------------------------------
def calcular_erro(prescrito, realizado):
    """Calcula MAE e RMSE entre valores prescritos e realizados (pares completos)."""
    pares = pd.DataFrame({"prescrito": prescrito, "realizado": realizado}).dropna()
    if len(pares) < 3:
        return None, None, len(pares)
    mae = mean_absolute_error(pares["prescrito"], pares["realizado"])
    rmse = np.sqrt(mean_squared_error(pares["prescrito"], pares["realizado"]))
    return mae, rmse, len(pares)


def prever_tendencia(datas, valores, dias_futuros=30):
    """
    Ajusta uma regressão linear simples (dia como X, valor como Y) e projeta
    a tendência para os próximos `dias_futuros` dias. Retorna um DataFrame
    com as datas futuras e os valores previstos, além do coeficiente angular.
    """
    dados = pd.DataFrame({"data": pd.to_datetime(datas), "valor": valores}).dropna()
    dados = dados.sort_values("data")

    if len(dados) < 3:
        return None, None

    dia_zero = dados["data"].min()
    dados["dia_num"] = (dados["data"] - dia_zero).dt.days

    modelo = LinearRegression()
    X = dados[["dia_num"]].values
    y = dados["valor"].values
    modelo.fit(X, y)

    ultimo_dia = dados["dia_num"].max()
    dias_previstos = np.arange(ultimo_dia + 1, ultimo_dia + dias_futuros + 1).reshape(-1, 1)
    valores_previstos = modelo.predict(dias_previstos)
    datas_previstas = [dia_zero + timedelta(days=int(d)) for d in dias_previstos.flatten()]

    df_previsao = pd.DataFrame({"data": datas_previstas, "valor_previsto": valores_previstos})
    coeficiente = modelo.coef_[0]
    return df_previsao, coeficiente


# ---------------------------------------------------------------------------
# BARRA LATERAL - NAVEGAÇÃO
# ---------------------------------------------------------------------------
st.sidebar.title("🎯 Follow Here")
st.sidebar.caption("Diário de treino, corrida e dieta com evolução inteligente")
pagina = st.sidebar.radio(
    "Navegação",
    ["🏠 Início", "📔 Diário", "📈 Evolução"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.info("Seus registros são salvos automaticamente em arquivos CSV locais "
                 "(dados_musculacao.csv, dados_corrida.csv, dados_dieta.csv), "
                 "permanecendo disponíveis mesmo após reiniciar o aplicativo.")


# ===========================================================================
# PÁGINA 1 - INÍCIO
# ===========================================================================
def pagina_inicio():
    """Landing page com apresentação do app e métricas rápidas."""
    st.title("🎯 Follow Here")
    st.subheader("Acompanhe seu treino, sua corrida e sua dieta — tudo em um só lugar")

    st.markdown("""
    O **Follow Here** foi criado para ajudar você a comparar, todos os dias,
    o que foi **planejado** com o que foi **realmente executado** — seja na
    academia, na pista de corrida ou no prato. Assim, fica fácil enxergar
    onde você está evoluindo e onde precisa ajustar a rota.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📔 Diário de Registro")
        st.write(
            "Selecione uma data e registre, em três sub-abas dedicadas, sua "
            "execução real de musculação, corrida e dieta, comparando lado a "
            "lado com o que foi prescrito para o dia."
        )
    with col2:
        st.markdown("#### 📈 Acompanhamento de Evolução")
        st.write(
            "Veja seu histórico completo, indicadores de adesão, gráficos "
            "interativos de progresso e uma previsão de tendência para os "
            "próximos 30 dias, com interpretação automática dos resultados."
        )

    st.markdown("---")
    st.markdown("### 📊 Resumo geral")

    df_musc = st.session_state["musculacao"]
    df_corrida = st.session_state["corrida"]
    df_dieta = st.session_state["dieta"]

    total_treinos = df_musc["carga_realizada"].notna().sum()
    total_refeicoes = df_dieta["calorias_consumidas"].notna().sum()
    total_km = df_corrida["distancia_real_km"].dropna().sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("🏋️ Treinos registrados", int(total_treinos))
    c2.metric("🥗 Refeições registradas", int(total_refeicoes))
    c3.metric("🏃 Km rodados (total)", f"{total_km:.1f} km")

    if total_treinos == 0 and total_refeicoes == 0 and total_km == 0:
        st.info("Você ainda não tem registros. Vá até a aba **Diário** para começar!")


# ===========================================================================
# PÁGINA 2 - DIÁRIO
# ===========================================================================
def pagina_diario():
    """Página de registro diário, com comparativo prescrito x realizado."""
    st.title("📔 Diário de Registro")

    data_selecionada = st.date_input("Selecione a data do registro", value=date.today())

    aba_musc, aba_corrida, aba_dieta = st.tabs(["🏋️ Musculação", "🏃 Corrida", "🥗 Dieta"])

    # ------------------------------------------------------------------
    # SUB-ABA MUSCULAÇÃO
    # ------------------------------------------------------------------
    with aba_musc:
        st.markdown("#### Prescrição do dia")
        df_musc = st.session_state["musculacao"]
        prescritos_dia = df_musc[
            (df_musc["data"] == data_selecionada) & (df_musc["carga_prescrita"].notna())
        ]

        if prescritos_dia.empty:
            st.warning("Nenhum treino prescrito para esta data ainda.")
        else:
            st.dataframe(
                prescritos_dia[["exercicio", "series_prescritas", "reps_prescritas", "carga_prescrita"]]
                .rename(columns={
                    "exercicio": "Exercício", "series_prescritas": "Séries",
                    "reps_prescritas": "Repetições", "carga_prescrita": "Carga (kg)",
                }),
                use_container_width=True, hide_index=True,
            )

        with st.expander("➕ Cadastrar / editar prescrição de exercício"):
            with st.form("form_prescricao_musc", clear_on_submit=True):
                col1, col2, col3 = st.columns(3)
                ex_nome = col1.text_input("Nome do exercício")
                ex_series = col2.number_input("Séries prescritas", min_value=0, step=1, value=0)
                ex_reps = col3.number_input("Repetições prescritas", min_value=0, step=1, value=0)
                ex_carga = st.number_input("Carga prescrita (kg)", min_value=0.0, step=0.5, value=0.0)
                enviado = st.form_submit_button("Salvar prescrição")
                if enviado:
                    if ex_nome.strip() == "":
                        st.error("Informe o nome do exercício.")
                    else:
                        salvar_prescricao_musculacao(data_selecionada, ex_nome.strip(), ex_series, ex_reps, ex_carga)
                        st.success(f"Prescrição de '{ex_nome}' salva para {data_selecionada}.")
                        st.rerun()

        st.markdown("---")
        st.markdown("#### Registrar execução real")
        with st.form("form_realizado_musc", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            ex_nome_r = col1.text_input("Nome do exercício realizado")
            ex_series_r = col2.number_input("Séries realizadas", min_value=0, step=1, value=0, key="sr")
            ex_reps_r = col3.number_input("Repetições realizadas", min_value=0, step=1, value=0, key="rr")
            ex_carga_r = st.number_input("Carga realizada (kg)", min_value=0.0, step=0.5, value=0.0, key="cr")
            enviado_r = st.form_submit_button("Registrar execução")
            if enviado_r:
                if ex_nome_r.strip() == "":
                    st.error("Informe o nome do exercício.")
                else:
                    salvar_realizado_musculacao(data_selecionada, ex_nome_r.strip(), ex_series_r, ex_reps_r, ex_carga_r)
                    st.success(f"Execução de '{ex_nome_r}' registrada.")
                    st.rerun()

        st.markdown("---")
        st.markdown("#### Comparativo: Prescrito vs. Realizado")
        registros_dia = df_musc[df_musc["data"] == data_selecionada]
        if registros_dia.empty:
            st.info("Nenhum dado de musculação para esta data.")
        else:
            tabela = registros_dia[[
                "exercicio", "series_prescritas", "series_realizadas",
                "reps_prescritas", "reps_realizadas", "carga_prescrita", "carga_realizada",
            ]].rename(columns={
                "exercicio": "Exercício",
                "series_prescritas": "Séries (Presc.)", "series_realizadas": "Séries (Real.)",
                "reps_prescritas": "Reps (Presc.)", "reps_realizadas": "Reps (Real.)",
                "carga_prescrita": "Carga Presc. (kg)", "carga_realizada": "Carga Real. (kg)",
            })
            st.dataframe(tabela, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # SUB-ABA CORRIDA
    # ------------------------------------------------------------------
    with aba_corrida:
        st.markdown("#### Meta prescrita do dia")
        df_corrida = st.session_state["corrida"]
        meta_dia = df_corrida[df_corrida["data"] == data_selecionada]

        if meta_dia.empty or meta_dia["distancia_prescrita_km"].isna().all():
            st.warning("Nenhuma meta de corrida prescrita para esta data ainda.")
        else:
            linha = meta_dia.iloc[0]
            c1, c2 = st.columns(2)
            c1.metric("Distância prescrita", f"{linha['distancia_prescrita_km']:.1f} km")
            c2.metric("Tempo prescrito", f"{linha['tempo_prescrito_min']:.0f} min")

        with st.expander("➕ Cadastrar / editar meta de corrida"):
            with st.form("form_prescricao_corrida", clear_on_submit=True):
                col1, col2 = st.columns(2)
                dist_p = col1.number_input("Distância prescrita (km)", min_value=0.0, step=0.5, value=0.0)
                tempo_p = col2.number_input("Tempo prescrito (min)", min_value=0.0, step=1.0, value=0.0)
                enviado = st.form_submit_button("Salvar meta")
                if enviado:
                    salvar_prescricao_corrida(data_selecionada, dist_p, tempo_p)
                    st.success("Meta de corrida salva.")
                    st.rerun()

        st.markdown("---")
        st.markdown("#### Registrar corrida realizada")
        with st.form("form_realizado_corrida", clear_on_submit=True):
            col1, col2 = st.columns(2)
            dist_r = col1.number_input("Distância realizada (km)", min_value=0.0, step=0.1, value=0.0, key="dr")
            tempo_r = col2.number_input("Tempo realizado (min)", min_value=0.0, step=1.0, value=0.0, key="tr")
            enviado_r = st.form_submit_button("Registrar corrida")
            if enviado_r:
                if dist_r <= 0:
                    st.error("Informe uma distância maior que zero.")
                else:
                    pace = salvar_realizado_corrida(data_selecionada, dist_r, tempo_r)
                    st.success(f"Corrida registrada! Pace calculado: {pace:.2f} min/km")
                    st.rerun()

        st.markdown("---")
        st.markdown("#### Comparativo: Meta vs. Executado")
        registro_corrida_dia = df_corrida[df_corrida["data"] == data_selecionada]
        if registro_corrida_dia.empty:
            st.info("Nenhum dado de corrida para esta data.")
        else:
            linha = registro_corrida_dia.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Distância", f"{linha.get('distancia_real_km', np.nan):.2f} km" if pd.notna(linha.get("distancia_real_km")) else "—",
                       delta=(f"{(linha['distancia_real_km'] - linha['distancia_prescrita_km']):.2f} km"
                              if pd.notna(linha.get("distancia_real_km")) and pd.notna(linha.get("distancia_prescrita_km")) else None))
            c2.metric("Tempo", f"{linha.get('tempo_real_min', np.nan):.0f} min" if pd.notna(linha.get("tempo_real_min")) else "—")
            c3.metric("Pace", f"{linha.get('pace_real', np.nan):.2f} min/km" if pd.notna(linha.get("pace_real")) else "—")

    # ------------------------------------------------------------------
    # SUB-ABA DIETA
    # ------------------------------------------------------------------
    with aba_dieta:
        tipos_refeicao = ["Café da manhã", "Almoço", "Jantar", "Lanche"]

        st.markdown("#### Dieta prescrita do dia")
        df_dieta = st.session_state["dieta"]
        prescricoes_dia = df_dieta[
            (df_dieta["data"] == data_selecionada) & (df_dieta["prescrito"].notna())
        ]
        if prescricoes_dia.empty:
            st.warning("Nenhuma refeição prescrita para esta data ainda.")
        else:
            st.dataframe(
                prescricoes_dia[["refeicao", "prescrito", "calorias_prescritas"]]
                .rename(columns={"refeicao": "Refeição", "prescrito": "Planejado", "calorias_prescritas": "Calorias (kcal)"}),
                use_container_width=True, hide_index=True,
            )

        with st.expander("➕ Cadastrar / editar refeição prescrita"):
            with st.form("form_prescricao_dieta", clear_on_submit=True):
                refeicao_p = st.selectbox("Refeição", tipos_refeicao, key="ref_p")
                planejado = st.text_area("Alimentos planejados")
                cal_p = st.number_input("Calorias prescritas (kcal)", min_value=0.0, step=10.0, value=0.0)
                enviado = st.form_submit_button("Salvar prescrição de dieta")
                if enviado:
                    salvar_prescricao_dieta(data_selecionada, refeicao_p, planejado, cal_p)
                    st.success(f"Prescrição de '{refeicao_p}' salva.")
                    st.rerun()

        st.markdown("---")
        st.markdown("#### Registrar consumo real")
        with st.form("form_realizado_dieta", clear_on_submit=True):
            refeicao_r = st.selectbox("Tipo de refeição", tipos_refeicao, key="ref_r")
            alimento_r = st.text_input("Alimentos consumidos")
            qtd_r = st.number_input("Quantidade (g/unid.)", min_value=0.0, step=10.0, value=0.0)
            cal_r = st.number_input("Calorias consumidas (kcal)", min_value=0.0, step=10.0, value=0.0, key="calr")
            enviado_r = st.form_submit_button("Registrar consumo")
            if enviado_r:
                if alimento_r.strip() == "":
                    st.error("Informe os alimentos consumidos.")
                else:
                    salvar_realizado_dieta(data_selecionada, refeicao_r, alimento_r.strip(), qtd_r, cal_r)
                    st.success(f"Consumo de '{refeicao_r}' registrado.")
                    st.rerun()

        st.markdown("---")
        st.markdown("#### Tabela comparativa do dia")
        registros_dieta_dia = df_dieta[df_dieta["data"] == data_selecionada]
        if registros_dieta_dia.empty:
            st.info("Nenhum dado de dieta para esta data.")
        else:
            tabela = registros_dieta_dia[[
                "refeicao", "prescrito", "calorias_prescritas", "alimento_consumido", "qtd_g", "calorias_consumidas",
            ]].rename(columns={
                "refeicao": "Refeição", "prescrito": "Planejado", "calorias_prescritas": "Kcal Prescritas",
                "alimento_consumido": "Consumido", "qtd_g": "Qtd (g)", "calorias_consumidas": "Kcal Consumidas",
            })
            st.dataframe(tabela, use_container_width=True, hide_index=True)


# ===========================================================================
# PÁGINA 3 - EVOLUÇÃO
# ===========================================================================
def pagina_evolucao():
    """Dashboard com histórico completo, KPIs, gráficos e previsão de tendência."""
    st.title("📈 Evolução e Análises")

    df_musc = st.session_state["musculacao"].copy()
    df_corrida = st.session_state["corrida"].copy()
    df_dieta = st.session_state["dieta"].copy()

    # ------------------------------------------------------------------
    # HISTÓRICO COMPLETO EM TABELAS
    # ------------------------------------------------------------------
    with st.expander("📋 Histórico completo (tabelas)", expanded=False):
        st.markdown("**Musculação**")
        st.dataframe(df_musc, use_container_width=True, hide_index=True)
        st.markdown("**Corrida**")
        st.dataframe(df_corrida, use_container_width=True, hide_index=True)
        st.markdown("**Dieta**")
        st.dataframe(df_dieta, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # KPIs DE ADESÃO
    # ------------------------------------------------------------------
    st.markdown("### 🎯 Indicadores de desempenho (KPIs)")

    dias_com_treino_prescrito = df_musc.loc[df_musc["carga_prescrita"].notna(), "data"].nunique()
    dias_com_treino_cumprido = df_musc.loc[
        df_musc["carga_prescrita"].notna() & df_musc["carga_realizada"].notna(), "data"
    ].nunique()
    adesao_treino = (dias_com_treino_cumprido / dias_com_treino_prescrito * 100) if dias_com_treino_prescrito > 0 else 0

    refeicoes_prescritas = df_dieta["calorias_prescritas"].notna().sum()
    refeicoes_cumpridas = (df_dieta["calorias_prescritas"].notna() & df_dieta["calorias_consumidas"].notna()).sum()
    adesao_dieta = (refeicoes_cumpridas / refeicoes_prescritas * 100) if refeicoes_prescritas > 0 else 0

    corridas_prescritas = df_corrida["distancia_prescrita_km"].notna().sum()
    corridas_cumpridas = (df_corrida["distancia_prescrita_km"].notna() & df_corrida["distancia_real_km"].notna()).sum()
    adesao_corrida = (corridas_cumpridas / corridas_prescritas * 100) if corridas_prescritas > 0 else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Adesão à Musculação", f"{adesao_treino:.0f}%")
    k2.metric("Adesão à Corrida", f"{adesao_corrida:.0f}%")
    k3.metric("Adesão à Dieta", f"{adesao_dieta:.0f}%")

    st.markdown("---")

    # ------------------------------------------------------------------
    # GRÁFICO 1: EVOLUÇÃO DE CARGA POR EXERCÍCIO
    # ------------------------------------------------------------------
    st.markdown("### 🏋️ Evolução de carga (kg) por exercício")
    dados_carga = df_musc.dropna(subset=["carga_realizada"]).sort_values("data")
    if dados_carga.empty:
        st.info("Ainda não há execuções de musculação registradas.")
    else:
        fig_carga = px.line(
            dados_carga, x="data", y="carga_realizada", color="exercicio", markers=True,
            labels={"data": "Data", "carga_realizada": "Carga realizada (kg)", "exercicio": "Exercício"},
        )
        st.plotly_chart(fig_carga, use_container_width=True)

    # ------------------------------------------------------------------
    # GRÁFICO 2: VOLUME SEMANAL DE CORRIDA E PACE
    # ------------------------------------------------------------------
    st.markdown("### 🏃 Volume semanal de corrida e ritmo (pace)")
    dados_corrida = df_corrida.dropna(subset=["distancia_real_km"]).copy()
    if dados_corrida.empty:
        st.info("Ainda não há corridas registradas.")
    else:
        dados_corrida["data"] = pd.to_datetime(dados_corrida["data"])
        dados_corrida = dados_corrida.set_index("data")
        semanal = dados_corrida.resample("W").agg({"distancia_real_km": "sum", "pace_real": "mean"}).reset_index()

        fig_corrida = go.Figure()
        fig_corrida.add_trace(go.Bar(x=semanal["data"], y=semanal["distancia_real_km"], name="Km na semana", yaxis="y1"))
        fig_corrida.add_trace(go.Scatter(x=semanal["data"], y=semanal["pace_real"], name="Pace médio (min/km)",
                                          yaxis="y2", mode="lines+markers"))
        fig_corrida.update_layout(
            xaxis=dict(title="Semana"),
            yaxis=dict(title="Distância (km)"),
            yaxis2=dict(title="Pace (min/km)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_corrida, use_container_width=True)

    # ------------------------------------------------------------------
    # GRÁFICO 3: CONSUMO CALÓRICO DIÁRIO VS META
    # ------------------------------------------------------------------
    st.markdown("### 🥗 Consumo calórico diário vs. meta")
    if df_dieta.empty:
        st.info("Ainda não há dados de dieta registrados.")
    else:
        cal_dia = df_dieta.groupby("data").agg(
            calorias_prescritas=("calorias_prescritas", "sum"),
            calorias_consumidas=("calorias_consumidas", "sum"),
        ).reset_index().sort_values("data")

        if cal_dia.empty:
            st.info("Ainda não há dados suficientes de dieta para exibir o gráfico.")
        else:
            fig_dieta = go.Figure()
            fig_dieta.add_trace(go.Scatter(x=cal_dia["data"], y=cal_dia["calorias_prescritas"],
                                            name="Meta (kcal)", mode="lines+markers"))
            fig_dieta.add_trace(go.Scatter(x=cal_dia["data"], y=cal_dia["calorias_consumidas"],
                                            name="Consumido (kcal)", mode="lines+markers"))
            fig_dieta.update_layout(xaxis_title="Data", yaxis_title="Calorias (kcal)",
                                     legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_dieta, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # PREVISÃO DE TENDÊNCIA (REGRESSÃO LINEAR)
    # ------------------------------------------------------------------
    st.markdown("### 🔮 Previsão de tendência (próximos 30 dias)")

    col_previsao_musc, col_previsao_corrida = st.columns(2)

    # --- Previsão de carga por exercício ---
    with col_previsao_musc:
        st.markdown("**Musculação — projeção de carga**")
        exercicios_disponiveis = dados_carga["exercicio"].unique().tolist() if not dados_carga.empty else []
        if not exercicios_disponiveis:
            st.info("Sem dados suficientes para projeção de musculação.")
        else:
            exercicio_escolhido = st.selectbox("Escolha o exercício", exercicios_disponiveis)
            dados_ex = dados_carga[dados_carga["exercicio"] == exercicio_escolhido]
            previsao, coef = prever_tendencia(dados_ex["data"], dados_ex["carga_realizada"])
            if previsao is None:
                st.warning("São necessários pelo menos 3 registros deste exercício para gerar a previsão.")
            else:
                fig_prev = go.Figure()
                fig_prev.add_trace(go.Scatter(x=dados_ex["data"], y=dados_ex["carga_realizada"],
                                               mode="markers+lines", name="Histórico"))
                fig_prev.add_trace(go.Scatter(x=previsao["data"], y=previsao["valor_previsto"],
                                               mode="lines", name="Tendência prevista", line=dict(dash="dash")))
                fig_prev.update_layout(xaxis_title="Data", yaxis_title="Carga (kg)")
                st.plotly_chart(fig_prev, use_container_width=True)
                tendencia_texto = "crescente 📈" if coef > 0 else ("decrescente 📉" if coef < 0 else "estável ➡️")
                st.caption(f"Tendência {tendencia_texto} de aproximadamente {coef:.3f} kg/dia.")

    # --- Previsão de distância de corrida ---
    with col_previsao_corrida:
        st.markdown("**Corrida — projeção de distância**")
        if dados_corrida.empty or len(dados_corrida) < 3:
            st.info("São necessários pelo menos 3 registros de corrida para gerar a previsão.")
        else:
            dados_corrida_reset = dados_corrida.reset_index()
            previsao_corrida, coef_corrida = prever_tendencia(
                dados_corrida_reset["data"], dados_corrida_reset["distancia_real_km"]
            )
            if previsao_corrida is None:
                st.info("Dados insuficientes para gerar a previsão de corrida.")
            else:
                fig_prev_corrida = go.Figure()
                fig_prev_corrida.add_trace(go.Scatter(x=dados_corrida_reset["data"], y=dados_corrida_reset["distancia_real_km"],
                                                        mode="markers+lines", name="Histórico"))
                fig_prev_corrida.add_trace(go.Scatter(x=previsao_corrida["data"], y=previsao_corrida["valor_previsto"],
                                                        mode="lines", name="Tendência prevista", line=dict(dash="dash")))
                fig_prev_corrida.update_layout(xaxis_title="Data", yaxis_title="Distância (km)")
                st.plotly_chart(fig_prev_corrida, use_container_width=True)
                tendencia_texto_c = "crescente 📈" if coef_corrida > 0 else ("decrescente 📉" if coef_corrida < 0 else "estável ➡️")
                st.caption(f"Tendência {tendencia_texto_c} de aproximadamente {coef_corrida:.3f} km/dia.")

    st.markdown("---")

    # ------------------------------------------------------------------
    # MÉTRICAS DE ERRO (MAE / RMSE)
    # ------------------------------------------------------------------
    st.markdown("### 📐 Métricas de erro (Prescrito vs. Realizado)")

    mae_carga, rmse_carga, n_carga = calcular_erro(df_musc["carga_prescrita"], df_musc["carga_realizada"])
    mae_dist, rmse_dist, n_dist = calcular_erro(df_corrida["distancia_prescrita_km"], df_corrida["distancia_real_km"])
    mae_cal, rmse_cal, n_cal = calcular_erro(df_dieta["calorias_prescritas"], df_dieta["calorias_consumidas"])

    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("**Carga (musculação)**")
        if mae_carga is None:
            st.caption(f"Registros pareados insuficientes ({n_carga}/3).")
        else:
            st.write(f"MAE: {mae_carga:.2f} kg  \nRMSE: {rmse_carga:.2f} kg")
    with e2:
        st.markdown("**Distância (corrida)**")
        if mae_dist is None:
            st.caption(f"Registros pareados insuficientes ({n_dist}/3).")
        else:
            st.write(f"MAE: {mae_dist:.2f} km  \nRMSE: {rmse_dist:.2f} km")
    with e3:
        st.markdown("**Calorias (dieta)**")
        if mae_cal is None:
            st.caption(f"Registros pareados insuficientes ({n_cal}/3).")
        else:
            st.write(f"MAE: {mae_cal:.0f} kcal  \nRMSE: {rmse_cal:.0f} kcal")

    st.markdown("---")

    # ------------------------------------------------------------------
    # PAINEL DE INTERPRETAÇÃO GERENCIAL (FEEDBACK AUTOMÁTICO)
    # ------------------------------------------------------------------
    st.markdown("### 🧭 Interpretação gerencial")

    mensagens = []

    # Interpretação da progressão de carga
    if not dados_carga.empty:
        for ex in dados_carga["exercicio"].unique():
            serie = dados_carga[dados_carga["exercicio"] == ex]
            if len(serie) >= 3:
                _, coef = prever_tendencia(serie["data"], serie["carga_realizada"])
                if coef is not None:
                    if coef > 0.05:
                        mensagens.append(("success", f"Sua progressão de carga em **{ex}** está crescente — ótimo trabalho!"))
                    elif coef < -0.05:
                        mensagens.append(("warning", f"Atenção: sua carga em **{ex}** apresenta tendência de queda."))
                    else:
                        mensagens.append(("info", f"Sua progressão de carga em **{ex}** está estável."))

    # Interpretação de adesão à dieta
    if refeicoes_prescritas > 0:
        soma_prescrita = df_dieta["calorias_prescritas"].sum()
        soma_consumida = df_dieta["calorias_consumidas"].sum()
        if soma_prescrita > 0:
            variacao = (soma_consumida - soma_prescrita) / soma_prescrita * 100
            if variacao > 10:
                mensagens.append(("warning", f"Atenção: seu consumo calórico está {variacao:.0f}% acima do prescrito."))
            elif variacao < -10:
                mensagens.append(("warning", f"Atenção: seu consumo calórico está {abs(variacao):.0f}% abaixo do prescrito."))
            else:
                mensagens.append(("success", "Seu consumo calórico está alinhado com o prescrito."))

    # Interpretação de adesão a treino e corrida
    if dias_com_treino_prescrito > 0:
        if adesao_treino >= 80:
            mensagens.append(("success", f"Excelente adesão à musculação: {adesao_treino:.0f}% dos treinos prescritos foram cumpridos."))
        elif adesao_treino >= 50:
            mensagens.append(("info", f"Adesão moderada à musculação: {adesao_treino:.0f}% dos treinos prescritos foram cumpridos."))
        else:
            mensagens.append(("warning", f"Baixa adesão à musculação: apenas {adesao_treino:.0f}% dos treinos prescritos foram cumpridos."))

    if corridas_prescritas > 0:
        if adesao_corrida >= 80:
            mensagens.append(("success", f"Excelente adesão à corrida: {adesao_corrida:.0f}% das metas cumpridas."))
        elif adesao_corrida >= 50:
            mensagens.append(("info", f"Adesão moderada à corrida: {adesao_corrida:.0f}% das metas cumpridas."))
        else:
            mensagens.append(("warning", f"Baixa adesão à corrida: apenas {adesao_corrida:.0f}% das metas cumpridas."))

    if not mensagens:
        st.info("Ainda não há dados suficientes para gerar interpretações. Registre mais dias no Diário.")
    else:
        for tipo, texto in mensagens:
            if tipo == "success":
                st.success(texto)
            elif tipo == "warning":
                st.warning(texto)
            else:
                st.info(texto)


# ---------------------------------------------------------------------------
# ROTEAMENTO ENTRE PÁGINAS
# ---------------------------------------------------------------------------
if pagina == "🏠 Início":
    pagina_inicio()
elif pagina == "📔 Diário":
    pagina_diario()
elif pagina == "📈 Evolução":
    pagina_evolucao()
