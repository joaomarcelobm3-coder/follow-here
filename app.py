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

Cadastro em lote: musculação e dieta usam st.data_editor, permitindo
preencher vários exercícios ou refeições de uma só vez, com um único
clique para salvar.
"""

import os
import html
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
    page_icon=":material/target:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# IDENTIDADE VISUAL - CSS CUSTOMIZADO (DARK MODE ELEGANTE)
# ---------------------------------------------------------------------------
CSS_APP = """
<style>
    :root {
        --bg-primary: #0E1117;
        --bg-card: #161B22;
        --bg-card-hover: #1C2128;
        --border-subtle: #2A2F3A;
        --accent-primary: #39FF88;
        --accent-secondary: #3E8BFF;
        --text-primary: #F2F4F8;
        --text-muted: #8B93A1;
        --danger: #FF5C7A;
        --warning: #FFB454;
    }

    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    }

    /* Cabeçalho principal */
    .fh-hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(57,255,136,0.10), rgba(62,139,255,0.08));
        border: 1px solid var(--border-subtle);
        margin-bottom: 1.4rem;
    }
    .fh-hero h1 {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 0.2rem 0;
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .fh-hero p {
        color: var(--text-muted);
        font-size: 0.98rem;
        margin: 0;
    }

    /* Cartões de conteúdo */
    .fh-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
    }
    .fh-card h4 {
        margin: 0 0 0.4rem 0;
        font-size: 1.02rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .fh-card p {
        color: var(--text-muted);
        font-size: 0.92rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Rótulo de seção */
    .fh-section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 1.6rem 0 0.6rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .fh-section-title .fh-tag {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        background: rgba(57,255,136,0.12);
        color: var(--accent-primary);
        border: 1px solid rgba(57,255,136,0.35);
    }
    .fh-section-sub {
        color: var(--text-muted);
        font-size: 0.86rem;
        margin: -0.3rem 0 0.8rem 0;
    }

    /* Cartões de KPI */
    .fh-kpi {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        text-align: left;
    }
    .fh-kpi .fh-kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        font-weight: 600;
    }
    .fh-kpi .fh-kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: var(--text-primary);
        margin-top: 0.15rem;
    }
    .fh-kpi .fh-kpi-value.accent { color: var(--accent-primary); }
    .fh-kpi .fh-kpi-value.blue { color: var(--accent-secondary); }
    .fh-kpi .fh-kpi-value.warn { color: var(--warning); }

    /* Pílulas de status */
    .fh-pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .fh-pill.ok { background: rgba(57,255,136,0.14); color: var(--accent-primary); }
    .fh-pill.mid { background: rgba(62,139,255,0.14); color: var(--accent-secondary); }
    .fh-pill.low { background: rgba(255,92,122,0.14); color: var(--danger); }

    div[data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 0.85rem 1rem;
    }

    button[kind="primary"], button[kind="secondary"] {
        border-radius: 10px !important;
    }

    hr { border-color: var(--border-subtle) !important; }
</style>
"""
st.markdown(CSS_APP, unsafe_allow_html=True)


def cartao_kpi(label, valor, tom="accent"):
    """Renderiza um cartão de KPI estilizado em HTML/CSS customizado."""
    st.markdown(
        f"""
        <div class="fh-kpi">
            <div class="fh-kpi-label">{label}</div>
            <div class="fh-kpi-value {tom}">{valor}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def titulo_secao(texto, tag=None, subtitulo=None):
    """Renderiza um título de seção estilizado, com tag opcional e subtítulo curto."""
    tag_html = f'<span class="fh-tag">{tag}</span>' if tag else ""
    st.markdown(f'<div class="fh-section-title">{texto} {tag_html}</div>', unsafe_allow_html=True)
    if subtitulo:
        st.markdown(f'<div class="fh-section-sub">{subtitulo}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PERSISTÊNCIA EM DISCO (ARQUIVOS CSV LOCAIS)
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
    "distancia_real_km", "tempo_real_min", "pace_real", "comentarios",
]
COLUNAS_DIETA = [
    "data", "refeicao", "prescrito", "alimento_consumido",
    "qtd_g", "calorias_prescritas", "calorias_consumidas",
]

TIPOS_REFEICAO = ["Café da manhã", "Almoço", "Lanche", "Jantar"]


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
        for col in colunas:
            if col not in df.columns:
                df[col] = np.nan
        return df[colunas]
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
    if "musculacao" not in st.session_state:
        st.session_state["musculacao"] = carregar_csv(ARQUIVO_MUSCULACAO, COLUNAS_MUSCULACAO)

    if "corrida" not in st.session_state:
        st.session_state["corrida"] = carregar_csv(ARQUIVO_CORRIDA, COLUNAS_CORRIDA)

    if "dieta" not in st.session_state:
        st.session_state["dieta"] = carregar_csv(ARQUIVO_DIETA, COLUNAS_DIETA)


inicializar_estado()


# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES - MUSCULAÇÃO (CADASTRO EM LOTE)
# ---------------------------------------------------------------------------
def salvar_lote_prescricao_musculacao(data_ref, tabela_editada):
    """
    Recebe o DataFrame preenchido no data_editor (um exercício por linha) e
    grava/atualiza a prescrição de todos os exercícios da data de uma vez.
    Linhas sem nome de exercício são ignoradas.
    """
    df = st.session_state["musculacao"]
    tabela_editada = tabela_editada[tabela_editada["Exercício"].astype(str).str.strip() != ""]

    for _, linha in tabela_editada.iterrows():
        exercicio = str(linha["Exercício"]).strip()
        filtro = (df["data"] == data_ref) & (df["exercicio"] == exercicio)
        valores = [linha["Séries"], linha["Repetições"], linha["Carga (kg)"]]

        if filtro.any():
            df.loc[filtro, ["series_prescritas", "reps_prescritas", "carga_prescrita"]] = valores
        else:
            nova_linha = {
                "data": data_ref, "exercicio": exercicio,
                "series_prescritas": linha["Séries"], "reps_prescritas": linha["Repetições"],
                "carga_prescrita": linha["Carga (kg)"],
                "series_realizadas": np.nan, "reps_realizadas": np.nan, "carga_realizada": np.nan,
            }
            df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)

    st.session_state["musculacao"] = df
    salvar_dados_disco()


def salvar_lote_realizado_musculacao(data_ref, tabela_editada):
    """
    Recebe o DataFrame preenchido no data_editor com a execução real de
    vários exercícios e grava/atualiza todos de uma vez, casando com a
    prescrição existente sempre que possível.
    """
    df = st.session_state["musculacao"]
    tabela_editada = tabela_editada[tabela_editada["Exercício"].astype(str).str.strip() != ""]

    for _, linha in tabela_editada.iterrows():
        exercicio = str(linha["Exercício"]).strip()
        filtro = (df["data"] == data_ref) & (df["exercicio"] == exercicio)
        valores = [linha["Séries"], linha["Repetições"], linha["Carga (kg)"]]

        if filtro.any():
            idx = df[filtro].index[0]
            df.loc[idx, ["series_realizadas", "reps_realizadas", "carga_realizada"]] = valores
        else:
            nova_linha = {
                "data": data_ref, "exercicio": exercicio,
                "series_prescritas": np.nan, "reps_prescritas": np.nan, "carga_prescrita": np.nan,
                "series_realizadas": linha["Séries"], "reps_realizadas": linha["Repetições"],
                "carga_realizada": linha["Carga (kg)"],
            }
            df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)

    st.session_state["musculacao"] = df
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
            "comentarios": np.nan,
        }
        st.session_state["corrida"] = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)

    salvar_dados_disco()


def salvar_realizado_corrida(data_ref, distancia, tempo, comentarios=""):
    """
    Registra a corrida realizada, calculando o pace (min/km) automaticamente
    e salvando a descrição/comentários livres do treino (sensação, terreno,
    tipo de treino etc.).
    """
    pace = tempo / distancia if distancia > 0 else np.nan
    comentarios = comentarios.strip() if isinstance(comentarios, str) else comentarios
    df = st.session_state["corrida"]
    filtro = df["data"] == data_ref

    if filtro.any():
        idx = df[filtro].index[0]
        df.loc[idx, ["distancia_real_km", "tempo_real_min", "pace_real", "comentarios"]] = [
            distancia, tempo, pace, comentarios,
        ]
    else:
        nova_linha = {
            "data": data_ref,
            "distancia_prescrita_km": np.nan, "tempo_prescrito_min": np.nan,
            "distancia_real_km": distancia, "tempo_real_min": tempo, "pace_real": pace,
            "comentarios": comentarios,
        }
        st.session_state["corrida"] = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)

    salvar_dados_disco()
    return pace


# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES - DIETA (CADASTRO EM LOTE)
# ---------------------------------------------------------------------------
def salvar_lote_prescricao_dieta(data_ref, tabela_editada):
    """
    Recebe o DataFrame do data_editor com uma linha por refeição e grava a
    prescrição (alimentos + calorias) de todas as refeições de uma vez.
    """
    df = st.session_state["dieta"]

    for _, linha in tabela_editada.iterrows():
        refeicao = str(linha["Refeição"]).strip()
        filtro = (df["data"] == data_ref) & (df["refeicao"] == refeicao) & (df["alimento_consumido"].isna())

        if filtro.any():
            idx = df[filtro].index[0]
            df.loc[idx, ["prescrito", "calorias_prescritas"]] = [linha["Alimentos planejados"], linha["Calorias (kcal)"]]
        else:
            nova_linha = {
                "data": data_ref, "refeicao": refeicao,
                "prescrito": linha["Alimentos planejados"], "alimento_consumido": np.nan,
                "qtd_g": np.nan, "calorias_prescritas": linha["Calorias (kcal)"], "calorias_consumidas": np.nan,
            }
            df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)

    st.session_state["dieta"] = df
    salvar_dados_disco()


def salvar_lote_realizado_dieta(data_ref, tabela_editada):
    """
    Recebe o DataFrame do data_editor com uma linha por refeição e grava o
    consumo real (alimentos, quantidade, calorias) de todas de uma vez.
    """
    df = st.session_state["dieta"]

    for _, linha in tabela_editada.iterrows():
        refeicao = str(linha["Refeição"]).strip()
        filtro = (df["data"] == data_ref) & (df["refeicao"] == refeicao) & (df["alimento_consumido"].isna())

        if filtro.any():
            idx = df[filtro].index[0]
            df.loc[idx, ["alimento_consumido", "qtd_g", "calorias_consumidas"]] = [
                linha["Alimentos consumidos"], linha["Qtd (g)"], linha["Calorias (kcal)"],
            ]
        else:
            nova_linha = {
                "data": data_ref, "refeicao": refeicao,
                "prescrito": np.nan, "alimento_consumido": linha["Alimentos consumidos"],
                "qtd_g": linha["Qtd (g)"], "calorias_prescritas": np.nan,
                "calorias_consumidas": linha["Calorias (kcal)"],
            }
            df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)

    st.session_state["dieta"] = df
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
st.sidebar.markdown(
    """
    <div style="padding:0.4rem 0 1rem 0;">
        <div style="font-size:1.5rem; font-weight:800; letter-spacing:-0.02em;
                    background:linear-gradient(90deg,#39FF88,#3E8BFF);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Follow Here
        </div>
        <div style="font-size:0.8rem; color:#8B93A1;">
            Treino, corrida e dieta — prescrito vs. realizado
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
pagina = st.sidebar.radio(
    "Navegação",
    ["Início", "Diário", "Evolução"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Seus registros são salvos automaticamente em arquivos CSV locais "
    "(dados_musculacao.csv, dados_corrida.csv, dados_dieta.csv) e "
    "permanecem disponíveis mesmo após reiniciar o aplicativo."
)


# ===========================================================================
# PÁGINA 1 - INÍCIO
# ===========================================================================
def pagina_inicio():
    """Landing page com apresentação do app e métricas rápidas."""
    st.markdown(
        """
        <div class="fh-hero">
            <h1>Follow Here</h1>
            <p>Compare, todos os dias, o que foi planejado com o que foi realmente executado —
            na academia, na pista e no prato.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="fh-card">
                <h4>Diário de Registro</h4>
                <p>Escolha uma data e registre, em lote, sua execução real de musculação,
                corrida e dieta, comparando lado a lado com o que foi prescrito.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="fh-card">
                <h4>Painel de Evolução</h4>
                <p>Histórico completo, indicadores de adesão, gráficos interativos e
                previsão de tendência para os próximos 30 dias.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    titulo_secao("Resumo geral", subtitulo="Um retrato rápido de tudo o que você já registrou.")

    df_musc = st.session_state["musculacao"]
    df_corrida = st.session_state["corrida"]
    df_dieta = st.session_state["dieta"]

    total_treinos = df_musc["carga_realizada"].notna().sum()
    total_refeicoes = df_dieta["calorias_consumidas"].notna().sum()
    total_km = df_corrida["distancia_real_km"].dropna().sum()

    c1, c2, c3 = st.columns(3)
    with c1:
        cartao_kpi("Treinos registrados", int(total_treinos), "accent")
    with c2:
        cartao_kpi("Refeições registradas", int(total_refeicoes), "blue")
    with c3:
        cartao_kpi("Km rodados (total)", f"{total_km} km", "warn")

    if total_treinos == 0 and total_refeicoes == 0 and total_km == 0:
        st.info("Você ainda não tem registros. Acesse o **Diário** para começar.")


# ===========================================================================
# PÁGINA 2 - DIÁRIO
# ===========================================================================
def pagina_diario():
    """Página de registro diário, com cadastro em lote e comparativo prescrito x realizado."""
    st.markdown(
        """
        <div class="fh-hero">
            <h1>Diário de Registro</h1>
            <p>Preencha tudo de uma vez em tabelas editáveis e salve com um único clique.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_selecionada = st.date_input("Data do registro", value=date.today())

    aba_musc, aba_corrida, aba_dieta = st.tabs(["Musculação", "Corrida", "Dieta"])

    # ------------------------------------------------------------------
    # SUB-ABA MUSCULAÇÃO
    # ------------------------------------------------------------------
    with aba_musc:
        titulo_secao("Prescrição do dia", tag="Planejado")
        df_musc = st.session_state["musculacao"]
        prescritos_dia = df_musc[
            (df_musc["data"] == data_selecionada) & (df_musc["carga_prescrita"].notna())
        ]

        if prescritos_dia.empty:
            st.info("Nenhum treino prescrito para esta data ainda.")
        else:
            st.dataframe(
                prescritos_dia[["exercicio", "series_prescritas", "reps_prescritas", "carga_prescrita"]]
                .rename(columns={
                    "exercicio": "Exercício", "series_prescritas": "Séries",
                    "reps_prescritas": "Reps", "carga_prescrita": "Carga (kg)",
                }),
                use_container_width=True, hide_index=True,
            )

        with st.expander("Cadastrar treino prescrito em lote"):
            st.caption("Defina quantos exercícios compõem o treino e preencha tudo de uma vez.")
            qtd_ex_p = st.number_input(
                "Quantidade de exercícios", min_value=1, max_value=30, value=5, step=1, key="qtd_ex_prescrito",
            )
            modelo_prescrito = pd.DataFrame({
                "Exercício": [""] * qtd_ex_p,
                "Séries": [0] * qtd_ex_p,
                "Repetições": [0] * qtd_ex_p,
                "Carga (kg)": [0.0] * qtd_ex_p,
            })
            tabela_prescrito = st.data_editor(
                modelo_prescrito,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                key="editor_musc_prescrito",
                column_config={
                    "Séries": st.column_config.NumberColumn(min_value=0, step=1),
                    "Repetições": st.column_config.NumberColumn(min_value=0, step=1),
                    "Carga (kg)": st.column_config.NumberColumn(min_value=0.0, step=0.5),
                },
            )
            if st.button("Salvar treino prescrito", use_container_width=True, key="btn_salvar_prescrito_musc"):
                validos = tabela_prescrito[tabela_prescrito["Exercício"].astype(str).str.strip() != ""]
                if validos.empty:
                    st.error("Preencha ao menos o nome de um exercício.")
                else:
                    salvar_lote_prescricao_musculacao(data_selecionada, tabela_prescrito)
                    st.success(f"{len(validos)} exercício(s) prescrito(s) salvos para {data_selecionada}.")
                    st.rerun()

        st.markdown("---")
        titulo_secao("Execução real", tag="Realizado")

        with st.expander("Registrar execução em lote", expanded=True):
            st.caption("Registre séries, repetições e carga de todos os exercícios treinados hoje.")
            qtd_ex_r = st.number_input(
                "Quantidade de exercícios", min_value=1, max_value=30, value=5, step=1, key="qtd_ex_realizado",
            )
            modelo_realizado = pd.DataFrame({
                "Exercício": [""] * qtd_ex_r,
                "Séries": [0] * qtd_ex_r,
                "Repetições": [0] * qtd_ex_r,
                "Carga (kg)": [0.0] * qtd_ex_r,
            })
            tabela_realizado = st.data_editor(
                modelo_realizado,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                key="editor_musc_realizado",
                column_config={
                    "Séries": st.column_config.NumberColumn(min_value=0, step=1),
                    "Repetições": st.column_config.NumberColumn(min_value=0, step=1),
                    "Carga (kg)": st.column_config.NumberColumn(min_value=0.0, step=0.5),
                },
            )
            if st.button("Salvar treino realizado", use_container_width=True, key="btn_salvar_realizado_musc"):
                validos = tabela_realizado[tabela_realizado["Exercício"].astype(str).str.strip() != ""]
                if validos.empty:
                    st.error("Preencha ao menos o nome de um exercício.")
                else:
                    salvar_lote_realizado_musculacao(data_selecionada, tabela_realizado)
                    st.success(f"{len(validos)} exercício(s) registrados para {data_selecionada}.")
                    st.rerun()

        st.markdown("---")
        titulo_secao("Comparativo do dia", subtitulo="Planejado, realizado e a diferença entre os dois, exercício por exercício.")
        registros_dia = df_musc[df_musc["data"] == data_selecionada]
        if registros_dia.empty:
            st.info("Nenhum dado de musculação para esta data.")
        else:
            tabela = registros_dia[["exercicio", "carga_prescrita", "carga_realizada"]].copy()
            tabela["diferenca"] = tabela["carga_realizada"] - tabela["carga_prescrita"]
            tabela = tabela.rename(columns={
                "exercicio": "Exercício",
                "carga_prescrita": "Planejado (kg)", "carga_realizada": "Realizado (kg)",
                "diferenca": "Diferença (kg)",
            })
            st.dataframe(tabela, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # SUB-ABA CORRIDA
    # ------------------------------------------------------------------
    with aba_corrida:
        titulo_secao("Meta do dia", tag="Planejado")
        df_corrida = st.session_state["corrida"]
        meta_dia = df_corrida[df_corrida["data"] == data_selecionada]

        if meta_dia.empty or meta_dia["distancia_prescrita_km"].isna().all():
            st.info("Nenhuma meta de corrida prescrita para esta data ainda.")
        else:
            linha = meta_dia.iloc[0]
            c1, c2 = st.columns(2)
            c1.metric("Distância prescrita", f"{linha['distancia_prescrita_km']} km")
            c2.metric("Tempo prescrito", f"{linha['tempo_prescrito_min']} min")

        with st.expander("Cadastrar / editar meta de corrida"):
            with st.form("form_prescricao_corrida", clear_on_submit=True):
                col1, col2 = st.columns(2)
                dist_p = col1.number_input("Distância prescrita (km)", min_value=0.0, step=0.5, value=0.0)
                tempo_p = col2.number_input("Tempo prescrito (min)", min_value=0.0, step=1.0, value=0.0)
                enviado = st.form_submit_button("Salvar meta", use_container_width=True)
                if enviado:
                    salvar_prescricao_corrida(data_selecionada, dist_p, tempo_p)
                    st.success("Meta de corrida salva.")
                    st.rerun()

        st.markdown("---")
        titulo_secao("Corrida realizada", tag="Realizado")
        with st.form("form_realizado_corrida", clear_on_submit=True):
            col1, col2 = st.columns(2)
            dist_r = col1.number_input("Distância realizada (km)", min_value=0.0, step=0.1, value=0.0, key="dr")
            tempo_r = col2.number_input("Tempo realizado (min)", min_value=0.0, step=1.0, value=0.0, key="tr")
            comentarios_r = st.text_area(
                "Descrição do Treino / Comentários",
                placeholder="Ex: Treino de tiros, sensação de cansaço, terreno com aclive...",
                key="comentarios_corrida",
            )
            enviado_r = st.form_submit_button("Registrar corrida", use_container_width=True)
            if enviado_r:
                if dist_r <= 0:
                    st.error("Informe uma distância maior que zero.")
                else:
                    pace = salvar_realizado_corrida(data_selecionada, dist_r, tempo_r, comentarios_r)
                    st.success(f"Corrida registrada. Pace calculado: {pace} min/km")
                    st.rerun()

        st.markdown("---")
        titulo_secao("Comparativo do dia", subtitulo="Planejado, realizado e a diferença exata entre os dois.")
        registro_corrida_dia = df_corrida[df_corrida["data"] == data_selecionada]
        if registro_corrida_dia.empty:
            st.info("Nenhum dado de corrida para esta data.")
        else:
            linha = registro_corrida_dia.iloc[0]
            dist_p, dist_r = linha.get("distancia_prescrita_km"), linha.get("distancia_real_km")
            tempo_p_v, tempo_r_v = linha.get("tempo_prescrito_min"), linha.get("tempo_real_min")

            tabela_corrida = pd.DataFrame([
                {
                    "Métrica": "Distância (km)",
                    "Planejado": dist_p if pd.notna(dist_p) else "—",
                    "Realizado": dist_r if pd.notna(dist_r) else "—",
                    "Diferença": (dist_r - dist_p) if pd.notna(dist_r) and pd.notna(dist_p) else "—",
                },
                {
                    "Métrica": "Tempo (min)",
                    "Planejado": tempo_p_v if pd.notna(tempo_p_v) else "—",
                    "Realizado": tempo_r_v if pd.notna(tempo_r_v) else "—",
                    "Diferença": (tempo_r_v - tempo_p_v) if pd.notna(tempo_r_v) and pd.notna(tempo_p_v) else "—",
                },
            ])
            st.dataframe(tabela_corrida, use_container_width=True, hide_index=True)

            pace_v = linha.get("pace_real")
            st.metric("Pace realizado", f"{pace_v} min/km" if pd.notna(pace_v) else "—")

            comentario_txt = linha.get("comentarios", "")
            if isinstance(comentario_txt, str) and comentario_txt.strip():
                st.markdown(
                    f"""
                    <div class="fh-card" style="margin-top:0.8rem;">
                        <h4>Descrição do treino</h4>
                        <p style="white-space:pre-wrap;">{html.escape(comentario_txt.strip())}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ------------------------------------------------------------------
    # SUB-ABA DIETA
    # ------------------------------------------------------------------
    with aba_dieta:
        titulo_secao("Dieta prescrita do dia", tag="Planejado")
        df_dieta = st.session_state["dieta"]
        prescricoes_dia = df_dieta[
            (df_dieta["data"] == data_selecionada) & (df_dieta["prescrito"].notna())
        ]
        if prescricoes_dia.empty:
            st.info("Nenhuma refeição prescrita para esta data ainda.")
        else:
            st.dataframe(
                prescricoes_dia[["refeicao", "prescrito", "calorias_prescritas"]]
                .rename(columns={"refeicao": "Refeição", "prescrito": "Planejado", "calorias_prescritas": "Kcal"}),
                use_container_width=True, hide_index=True,
            )

        with st.expander("Cadastrar dieta prescrita em lote", expanded=True):
            st.caption("Preencha os alimentos e calorias planejados para cada refeição do dia.")
            modelo_dieta_p = pd.DataFrame({
                "Refeição": TIPOS_REFEICAO,
                "Alimentos planejados": [""] * len(TIPOS_REFEICAO),
                "Calorias (kcal)": [0.0] * len(TIPOS_REFEICAO),
            })
            tabela_dieta_p = st.data_editor(
                modelo_dieta_p,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="editor_dieta_prescrito",
                disabled=["Refeição"],
                column_config={
                    "Calorias (kcal)": st.column_config.NumberColumn(min_value=0.0, step=10.0),
                },
            )
            if st.button("Salvar dieta prescrita", use_container_width=True, key="btn_salvar_prescrito_dieta"):
                salvar_lote_prescricao_dieta(data_selecionada, tabela_dieta_p)
                st.success("Dieta prescrita salva para todas as refeições.")
                st.rerun()

        st.markdown("---")
        titulo_secao("Consumo real do dia", tag="Realizado")
        with st.expander("Registrar consumo em lote", expanded=True):
            st.caption("Registre o que foi consumido em cada refeição.")
            modelo_dieta_r = pd.DataFrame({
                "Refeição": TIPOS_REFEICAO,
                "Alimentos consumidos": [""] * len(TIPOS_REFEICAO),
                "Qtd (g)": [0.0] * len(TIPOS_REFEICAO),
                "Calorias (kcal)": [0.0] * len(TIPOS_REFEICAO),
            })
            tabela_dieta_r = st.data_editor(
                modelo_dieta_r,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="editor_dieta_realizado",
                disabled=["Refeição"],
                column_config={
                    "Qtd (g)": st.column_config.NumberColumn(min_value=0.0, step=10.0),
                    "Calorias (kcal)": st.column_config.NumberColumn(min_value=0.0, step=10.0),
                },
            )
            if st.button("Salvar consumo real", use_container_width=True, key="btn_salvar_realizado_dieta"):
                salvar_lote_realizado_dieta(data_selecionada, tabela_dieta_r)
                st.success("Consumo real salvo para todas as refeições.")
                st.rerun()

        st.markdown("---")
        titulo_secao("Comparativo do dia", subtitulo="Planejado, realizado e a diferença exata entre os dois.")
        registros_dieta_dia = df_dieta[df_dieta["data"] == data_selecionada]
        if registros_dieta_dia.empty:
            st.info("Nenhum dado de dieta para esta data.")
        else:
            tabela = registros_dieta_dia[["refeicao", "calorias_prescritas", "calorias_consumidas"]].copy()
            tabela["diferenca"] = tabela["calorias_consumidas"] - tabela["calorias_prescritas"]
            tabela = tabela.rename(columns={
                "refeicao": "Refeição",
                "calorias_prescritas": "Planejado (kcal)", "calorias_consumidas": "Realizado (kcal)",
                "diferenca": "Diferença (kcal)",
            })
            st.dataframe(tabela, use_container_width=True, hide_index=True)


# ===========================================================================
# PÁGINA 3 - EVOLUÇÃO
# ===========================================================================
def pagina_evolucao():
    """Dashboard com histórico completo, KPIs, gráficos e previsão de tendência."""
    st.markdown(
        """
        <div class="fh-hero">
            <h1>Evolução e Análises</h1>
            <p>Indicadores de adesão, progressão e previsão de tendência para os próximos 30 dias.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_musc = st.session_state["musculacao"].copy()
    df_corrida = st.session_state["corrida"].copy()
    df_dieta = st.session_state["dieta"].copy()

    # ------------------------------------------------------------------
    # HISTÓRICO COMPARATIVO: PLANEJADO x REALIZADO x DIFERENÇA
    # ------------------------------------------------------------------
    with st.expander("Histórico completo — Planejado vs. Realizado", expanded=False):
        st.markdown("**Musculação**")
        tabela_musc_hist = df_musc[["data", "exercicio", "carga_prescrita", "carga_realizada"]].copy()
        tabela_musc_hist["diferenca"] = tabela_musc_hist["carga_realizada"] - tabela_musc_hist["carga_prescrita"]
        tabela_musc_hist = tabela_musc_hist.sort_values("data", ascending=False).rename(columns={
            "data": "Data", "exercicio": "Exercício",
            "carga_prescrita": "Planejado (kg)", "carga_realizada": "Realizado (kg)",
            "diferenca": "Diferença (kg)",
        })
        st.dataframe(tabela_musc_hist, use_container_width=True, hide_index=True)

        st.markdown("**Corrida**")
        tabela_corrida_hist = df_corrida[[
            "data", "distancia_prescrita_km", "distancia_real_km",
            "tempo_prescrito_min", "tempo_real_min", "pace_real", "comentarios",
        ]].copy()
        tabela_corrida_hist["diferenca_distancia"] = (
            tabela_corrida_hist["distancia_real_km"] - tabela_corrida_hist["distancia_prescrita_km"]
        )
        tabela_corrida_hist["diferenca_tempo"] = (
            tabela_corrida_hist["tempo_real_min"] - tabela_corrida_hist["tempo_prescrito_min"]
        )
        tabela_corrida_hist = tabela_corrida_hist.sort_values("data", ascending=False).rename(columns={
            "data": "Data",
            "distancia_prescrita_km": "Distância Planejada (km)", "distancia_real_km": "Distância Realizada (km)",
            "diferenca_distancia": "Diferença Distância (km)",
            "tempo_prescrito_min": "Tempo Planejado (min)", "tempo_real_min": "Tempo Realizado (min)",
            "diferenca_tempo": "Diferença Tempo (min)",
            "pace_real": "Pace (min/km)", "comentarios": "Descrição do Treino",
        })
        st.dataframe(
            tabela_corrida_hist[[
                "Data", "Distância Planejada (km)", "Distância Realizada (km)", "Diferença Distância (km)",
                "Tempo Planejado (min)", "Tempo Realizado (min)", "Diferença Tempo (min)",
                "Pace (min/km)", "Descrição do Treino",
            ]],
            use_container_width=True, hide_index=True,
        )

        st.markdown("**Dieta**")
        tabela_dieta_hist = df_dieta[["data", "refeicao", "calorias_prescritas", "calorias_consumidas"]].copy()
        tabela_dieta_hist["diferenca"] = (
            tabela_dieta_hist["calorias_consumidas"] - tabela_dieta_hist["calorias_prescritas"]
        )
        tabela_dieta_hist = tabela_dieta_hist.sort_values("data", ascending=False).rename(columns={
            "data": "Data", "refeicao": "Refeição",
            "calorias_prescritas": "Planejado (kcal)", "calorias_consumidas": "Realizado (kcal)",
            "diferenca": "Diferença (kcal)",
        })
        st.dataframe(tabela_dieta_hist, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # KPIs DE ADESÃO
    # ------------------------------------------------------------------
    titulo_secao("Indicadores de desempenho", tag="KPIs")

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
    with k1:
        cartao_kpi("Adesão à Musculação", f"{adesao_treino}%", "accent")
    with k2:
        cartao_kpi("Adesão à Corrida", f"{adesao_corrida}%", "blue")
    with k3:
        cartao_kpi("Adesão à Dieta", f"{adesao_dieta}%", "warn")

    st.markdown("---")

    # ------------------------------------------------------------------
    # GRÁFICO 1: EVOLUÇÃO DE CARGA POR EXERCÍCIO
    # ------------------------------------------------------------------
    titulo_secao("Evolução de carga por exercício", subtitulo="Progressão da carga real ao longo do tempo.")
    dados_carga = df_musc.dropna(subset=["carga_realizada"]).sort_values("data")
    if dados_carga.empty:
        st.info("Ainda não há execuções de musculação registradas.")
    else:
        fig_carga = px.line(
            dados_carga, x="data", y="carga_realizada", color="exercicio", markers=True,
            labels={"data": "Data", "carga_realizada": "Carga realizada (kg)", "exercicio": "Exercício"},
            color_discrete_sequence=["#39FF88", "#3E8BFF", "#FFB454", "#FF5C7A", "#B18CFF", "#4FE0D4"],
        )
        fig_carga.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#F2F4F8", legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_carga, use_container_width=True)

    # ------------------------------------------------------------------
    # GRÁFICO 2: VOLUME SEMANAL DE CORRIDA E PACE
    # ------------------------------------------------------------------
    titulo_secao("Volume semanal de corrida e ritmo", subtitulo="Quilometragem semanal comparada ao pace médio.")
    dados_corrida = df_corrida.dropna(subset=["distancia_real_km"]).copy()
    if dados_corrida.empty:
        st.info("Ainda não há corridas registradas.")
    else:
        dados_corrida["data"] = pd.to_datetime(dados_corrida["data"])
        dados_corrida = dados_corrida.set_index("data")
        semanal = dados_corrida.resample("W").agg({"distancia_real_km": "sum", "pace_real": "mean"}).reset_index()

        fig_corrida = go.Figure()
        fig_corrida.add_trace(go.Bar(x=semanal["data"], y=semanal["distancia_real_km"], name="Km na semana",
                                      yaxis="y1", marker_color="#3E8BFF"))
        fig_corrida.add_trace(go.Scatter(x=semanal["data"], y=semanal["pace_real"], name="Pace médio (min/km)",
                                          yaxis="y2", mode="lines+markers", line=dict(color="#39FF88")))
        fig_corrida.update_layout(
            xaxis=dict(title="Semana"),
            yaxis=dict(title="Distância (km)"),
            yaxis2=dict(title="Pace (min/km)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#F2F4F8",
        )
        st.plotly_chart(fig_corrida, use_container_width=True)

    # ------------------------------------------------------------------
    # COMENTÁRIOS DOS TREINOS DE CORRIDA
    # ------------------------------------------------------------------
    titulo_secao("Descrições de treino", subtitulo="Comentários registrados nas corridas mais recentes.")
    comentarios_corrida = df_corrida[
        df_corrida["comentarios"].notna() & (df_corrida["comentarios"].astype(str).str.strip() != "")
    ].sort_values("data", ascending=False)

    if comentarios_corrida.empty:
        st.info("Nenhum comentário de treino registrado ainda.")
    else:
        for _, linha_c in comentarios_corrida.head(10).iterrows():
            data_fmt = pd.to_datetime(linha_c["data"]).strftime("%d/%m/%Y")
            dist_fmt = f"{linha_c['distancia_real_km']} km" if pd.notna(linha_c.get("distancia_real_km")) else "—"
            st.markdown(
                f"""
                <div class="fh-card">
                    <h4>{data_fmt} · {dist_fmt}</h4>
                    <p style="white-space:pre-wrap;">{html.escape(str(linha_c["comentarios"]).strip())}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------
    # GRÁFICO 3: CONSUMO CALÓRICO DIÁRIO VS META
    # ------------------------------------------------------------------
    titulo_secao("Consumo calórico diário vs. meta")
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
                                            name="Meta (kcal)", mode="lines+markers", line=dict(color="#3E8BFF")))
            fig_dieta.add_trace(go.Scatter(x=cal_dia["data"], y=cal_dia["calorias_consumidas"],
                                            name="Consumido (kcal)", mode="lines+markers", line=dict(color="#39FF88")))
            fig_dieta.update_layout(
                xaxis_title="Data", yaxis_title="Calorias (kcal)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#F2F4F8",
            )
            st.plotly_chart(fig_dieta, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # PREVISÃO DE TENDÊNCIA (REGRESSÃO LINEAR)
    # ------------------------------------------------------------------
    titulo_secao("Previsão de tendência", tag="30 dias", subtitulo="Projeção estatística com base no seu histórico.")

    col_previsao_musc, col_previsao_corrida = st.columns(2)

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
                                               mode="markers+lines", name="Histórico", line=dict(color="#3E8BFF")))
                fig_prev.add_trace(go.Scatter(x=previsao["data"], y=previsao["valor_previsto"],
                                               mode="lines", name="Tendência prevista",
                                               line=dict(dash="dash", color="#39FF88")))
                fig_prev.update_layout(
                    xaxis_title="Data", yaxis_title="Carga (kg)",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#F2F4F8",
                )
                st.plotly_chart(fig_prev, use_container_width=True)
                tendencia_texto = "crescente" if coef > 0 else ("decrescente" if coef < 0 else "estável")
                st.caption(f"Tendência {tendencia_texto} de aproximadamente {coef} kg/dia.")

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
                                                        mode="markers+lines", name="Histórico", line=dict(color="#3E8BFF")))
                fig_prev_corrida.add_trace(go.Scatter(x=previsao_corrida["data"], y=previsao_corrida["valor_previsto"],
                                                        mode="lines", name="Tendência prevista",
                                                        line=dict(dash="dash", color="#39FF88")))
                fig_prev_corrida.update_layout(
                    xaxis_title="Data", yaxis_title="Distância (km)",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#F2F4F8",
                )
                st.plotly_chart(fig_prev_corrida, use_container_width=True)
                tendencia_texto_c = "crescente" if coef_corrida > 0 else ("decrescente" if coef_corrida < 0 else "estável")
                st.caption(f"Tendência {tendencia_texto_c} de aproximadamente {coef_corrida} km/dia.")

    st.markdown("---")

    # ------------------------------------------------------------------
    # MÉTRICAS DE ERRO (MAE / RMSE)
    # ------------------------------------------------------------------
    titulo_secao("Métricas de erro", subtitulo="Diferença média entre o prescrito e o realizado.")

    mae_carga, rmse_carga, n_carga = calcular_erro(df_musc["carga_prescrita"], df_musc["carga_realizada"])
    mae_dist, rmse_dist, n_dist = calcular_erro(df_corrida["distancia_prescrita_km"], df_corrida["distancia_real_km"])
    mae_cal, rmse_cal, n_cal = calcular_erro(df_dieta["calorias_prescritas"], df_dieta["calorias_consumidas"])

    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("**Carga (musculação)**")
        if mae_carga is None:
            st.caption(f"Registros pareados insuficientes ({n_carga}/3).")
        else:
            st.write(f"MAE: {mae_carga} kg  \nRMSE: {rmse_carga} kg")
    with e2:
        st.markdown("**Distância (corrida)**")
        if mae_dist is None:
            st.caption(f"Registros pareados insuficientes ({n_dist}/3).")
        else:
            st.write(f"MAE: {mae_dist} km  \nRMSE: {rmse_dist} km")
    with e3:
        st.markdown("**Calorias (dieta)**")
        if mae_cal is None:
            st.caption(f"Registros pareados insuficientes ({n_cal}/3).")
        else:
            st.write(f"MAE: {mae_cal} kcal  \nRMSE: {rmse_cal} kcal")

    st.markdown("---")

    # ------------------------------------------------------------------
    # PAINEL DE INTERPRETAÇÃO GERENCIAL (FEEDBACK AUTOMÁTICO)
    # ------------------------------------------------------------------
    titulo_secao("Interpretação gerencial", subtitulo="Leitura automática dos seus resultados recentes.")

    mensagens = []

    if not dados_carga.empty:
        for ex in dados_carga["exercicio"].unique():
            serie = dados_carga[dados_carga["exercicio"] == ex]
            if len(serie) >= 3:
                _, coef = prever_tendencia(serie["data"], serie["carga_realizada"])
                if coef is not None:
                    if coef > 0.05:
                        mensagens.append(("success", f"Sua progressão de carga em **{ex}** está crescente."))
                    elif coef < -0.05:
                        mensagens.append(("warning", f"Atenção: sua carga em **{ex}** apresenta tendência de queda."))
                    else:
                        mensagens.append(("info", f"Sua progressão de carga em **{ex}** está estável."))

    if refeicoes_prescritas > 0:
        soma_prescrita = df_dieta["calorias_prescritas"].sum()
        soma_consumida = df_dieta["calorias_consumidas"].sum()
        if soma_prescrita > 0:
            variacao = (soma_consumida - soma_prescrita) / soma_prescrita * 100
            if variacao > 10:
                mensagens.append(("warning", f"Seu consumo calórico está {variacao}% acima do prescrito."))
            elif variacao < -10:
                mensagens.append(("warning", f"Seu consumo calórico está {abs(variacao)}% abaixo do prescrito."))
            else:
                mensagens.append(("success", "Seu consumo calórico está alinhado com o prescrito."))

    if dias_com_treino_prescrito > 0:
        if adesao_treino >= 80:
            mensagens.append(("success", f"Excelente adesão à musculação: {adesao_treino}% dos treinos cumpridos."))
        elif adesao_treino >= 50:
            mensagens.append(("info", f"Adesão moderada à musculação: {adesao_treino}% dos treinos cumpridos."))
        else:
            mensagens.append(("warning", f"Baixa adesão à musculação: apenas {adesao_treino}% dos treinos cumpridos."))

    if corridas_prescritas > 0:
        if adesao_corrida >= 80:
            mensagens.append(("success", f"Excelente adesão à corrida: {adesao_corrida}% das metas cumpridas."))
        elif adesao_corrida >= 50:
            mensagens.append(("info", f"Adesão moderada à corrida: {adesao_corrida}% das metas cumpridas."))
        else:
            mensagens.append(("warning", f"Baixa adesão à corrida: apenas {adesao_corrida}% das metas cumpridas."))

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
if pagina == "Início":
    pagina_inicio()
elif pagina == "Diário":
    pagina_diario()
elif pagina == "Evolução":
    pagina_evolucao()
