"""
Follow Here — Diário de Treinos de Corrida (Integração Runna / Google Calendar)
=================================================================================
Fluxo de persistência progressiva:
  - Evento PLANEJADO no iCal -> grava/atualiza apenas os campos de META (prescrito)
    na base persistente (CSV), para a data do evento.
  - Evento REALIZADO no iCal (o Runna SUBSTITUI o evento planejado pelo resumo
    assim que o treino é sincronizado do Strava/Garmin) -> grava/atualiza apenas
    os campos REAIS na mesma linha daquela data, sem apagar a meta já gravada.
  - Isso garante que, mesmo o Google Calendar não exibindo mais o planejado
    depois que o treino é concluído, o app nunca perde o dado de prescrição.
"""

import os
import re
from datetime import datetime, date

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from icalendar import Calendar

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================
DATA_INICIO_PLANO = date(2026, 8, 17)
DATA_PROVA = date(2026, 11, 20)
NOME_PROVA = "Meia Maratona"

TOLERANCIA_DISTANCIA_PCT = 0.10   # 10% de tolerância na distância
TOLERANCIA_PACE_PCT = 0.05        # 5% de tolerância no pace

# Caminho do "banco de dados" persistente (CSV). Em deploy na nuvem, troque
# por um disco persistente ou por SQLite/Google Sheets conforme sua infra.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "treinos_db.csv")

COLUNAS_BASE = [
    "data", "nome_treino",
    "distancia_prescrita_km", "tempo_prescrito_min", "pace_prescrito_min_km",
    "distancia_real_km", "tempo_real_min", "pace_real_min_km",
    "status",
]

st.set_page_config(page_title="Follow Here — Corrida", layout="wide", page_icon="🏃")


# ============================================================
# FUNÇÕES DE CONVERSÃO / FORMATAÇÃO
# ============================================================
def parse_distancia_br(texto: str):
    """Converte string de distância (aceita vírgula ou ponto) para float."""
    if texto is None:
        return None
    texto = texto.strip().replace(",", ".")
    try:
        return float(texto)
    except ValueError:
        return None


def mmss_para_minutos(mm: str, ss: str):
    """Converte MM:SS (strings) para minutos decimais. Ex.: 38:05 -> 38.083..."""
    try:
        return int(mm) + int(ss) / 60
    except (TypeError, ValueError):
        return None


def minutos_para_mmss(minutos_decimais) -> str:
    """Converte minutos decimais para string 'MM:SS'. Ex.: 38.083 -> '38:05'"""
    if minutos_decimais is None or pd.isna(minutos_decimais):
        return "-"
    total_segundos = int(round(float(minutos_decimais) * 60))
    mm, ss = divmod(total_segundos, 60)
    return f"{mm}:{ss:02d}"


def calcular_pace_min_km(tempo_min, distancia_km):
    """Calcula pace (min/km) a partir de tempo total (min) e distância (km)."""
    if not tempo_min or not distancia_km or pd.isna(tempo_min) or pd.isna(distancia_km) or distancia_km == 0:
        return None
    return tempo_min / distancia_km


def formatar_pace(pace_min_km) -> str:
    """Formata pace decimal (min/km) para 'M:SS /km'."""
    if pace_min_km is None or pd.isna(pace_min_km):
        return "-"
    return f"{minutos_para_mmss(pace_min_km)} /km"


# ============================================================
# REGEX - PADRÕES DE PARSING DO RUNNA
# ============================================================

# --- EVENTO PLANEJADO (prescrição, treino futuro / ainda não sincronizado) ---
# Descrição típica: "Treino de ritmo • 5,5km • 35m - 40m\n2km de aquecimento..."
REGEX_PRESCRICAO_FAIXA = re.compile(
    r"(?P<nome>[^•\n]+?)\s*•\s*(?P<distancia>[\d.,]+)\s*km\s*•\s*"
    r"(?P<tempo_min>\d+)\s*m\s*-\s*(?P<tempo_max>\d+)\s*m",
    re.IGNORECASE,
)

# Variante sem faixa, com tempo único: "Treino longo • 12km • 70m"
REGEX_PRESCRICAO_UNICA = re.compile(
    r"(?P<nome>[^•\n]+?)\s*•\s*(?P<distancia>[\d.,]+)\s*km\s*•\s*(?P<tempo>\d+)\s*m(?!\s*-)",
    re.IGNORECASE,
)

# --- EVENTO REALIZADO (resumo sincronizado via Strava/Garmin, substitui o planejado) ---
REGEX_REAL_DISTANCIA = re.compile(r"Dist[âa]ncia:\s*([\d.,]+)\s*km", re.IGNORECASE)
REGEX_REAL_TEMPO = re.compile(r"Hor[áa]rio:\s*(\d+):(\d{2})", re.IGNORECASE)
REGEX_REAL_PACE = re.compile(r"Ritmo\s*m[ée]dio:\s*(\d+):(\d{2})\s*/\s*km", re.IGNORECASE)

# Marcador que identifica que a descrição é um "resumo de execução" (contém "Resumo:")
MARCADOR_REALIZADO = re.compile(r"(Resumo|Dist[âa]ncia:|Hor[áa]rio:)", re.IGNORECASE)


def eh_evento_realizado(descricao: str) -> bool:
    """
    Detecta se a descrição corresponde a um treino REALIZADO.
    Regra: o Runna, sem integração direta ao Strava, SUBSTITUI o evento
    planejado pelo resumo assim que o treino é concluído — então a presença
    de "Resumo:"/"Distância:"/"Horário:" é o sinal decisivo.
    """
    if not descricao:
        return False
    return bool(MARCADOR_REALIZADO.search(descricao)) and bool(REGEX_REAL_DISTANCIA.search(descricao))


def extrair_prescricao(titulo: str, descricao: str):
    """Extrai nome, distância prescrita, tempo médio prescrito e pace prescrito (META)."""
    texto = descricao or titulo or ""

    match = REGEX_PRESCRICAO_FAIXA.search(texto)
    if match:
        nome = match.group("nome").strip()
        distancia = parse_distancia_br(match.group("distancia"))
        t_min = float(match.group("tempo_min"))
        t_max = float(match.group("tempo_max"))
        tempo_medio = (t_min + t_max) / 2
    else:
        match2 = REGEX_PRESCRICAO_UNICA.search(texto)
        if not match2:
            return None
        nome = match2.group("nome").strip()
        distancia = parse_distancia_br(match2.group("distancia"))
        tempo_medio = float(match2.group("tempo"))

    pace = calcular_pace_min_km(tempo_medio, distancia)
    return {
        "nome_treino": nome,
        "distancia_prescrita_km": distancia,
        "tempo_prescrito_min": tempo_medio,
        "pace_prescrito_min_km": pace,
    }


def extrair_realizado(titulo: str, descricao: str):
    """Extrai distância, tempo e pace REAIS a partir do resumo Strava/Garmin."""
    texto = descricao or ""

    m_dist = REGEX_REAL_DISTANCIA.search(texto)
    m_tempo = REGEX_REAL_TEMPO.search(texto)
    m_pace = REGEX_REAL_PACE.search(texto)

    distancia = parse_distancia_br(m_dist.group(1)) if m_dist else None
    tempo = mmss_para_minutos(m_tempo.group(1), m_tempo.group(2)) if m_tempo else None
    pace = mmss_para_minutos(m_pace.group(1), m_pace.group(2)) if m_pace else calcular_pace_min_km(tempo, distancia)

    nome = titulo.replace("🏃", "").strip() if titulo else "Treino realizado"
    return {
        "nome_treino": nome or "Treino realizado",
        "distancia_real_km": distancia,
        "tempo_real_min": tempo,
        "pace_real_min_km": pace,
    }


# ============================================================
# STATUS DA SESSÃO (Prescrito x Real)
# ============================================================
def classificar_status(linha) -> str:
    """
    Compara prescrito x real e retorna um status textual.
    Ajuste TOLERANCIA_DISTANCIA_PCT / TOLERANCIA_PACE_PCT conforme necessário.
    """
    d_plan = linha.get("distancia_prescrita_km")
    d_real = linha.get("distancia_real_km")
    p_plan = linha.get("pace_prescrito_min_km")
    p_real = linha.get("pace_real_min_km")

    if pd.isna(d_plan) or pd.isna(d_real):
        return "Sem Registro"

    delta_dist_pct = (d_real - d_plan) / d_plan if d_plan else 0

    delta_pace_pct = 0
    if p_plan and p_real and not pd.isna(p_plan) and not pd.isna(p_real):
        delta_pace_pct = (p_plan - p_real) / p_plan  # positivo = correu mais rápido que o prescrito

    if delta_dist_pct < -TOLERANCIA_DISTANCIA_PCT:
        return "Abaixo do Alvo"
    if delta_dist_pct > TOLERANCIA_DISTANCIA_PCT or delta_pace_pct > TOLERANCIA_PACE_PCT:
        return "Superado"
    return "Dentro do Alvo"


# ============================================================
# BASE PERSISTENTE (CSV) — GRAVAÇÃO PROGRESSIVA
# ============================================================
def carregar_base() -> pd.DataFrame:
    """Carrega a base persistente do disco (ou cria uma vazia se ainda não existir)."""
    if os.path.exists(DB_PATH):
        df = pd.read_csv(DB_PATH)
        df["data"] = pd.to_datetime(df["data"]).dt.date
        for coluna in COLUNAS_BASE:
            if coluna not in df.columns:
                df[coluna] = None
        return df[COLUNAS_BASE]
    return pd.DataFrame(columns=COLUNAS_BASE)


def salvar_base(df: pd.DataFrame) -> None:
    """Persiste a base no disco em CSV."""
    df.to_csv(DB_PATH, index=False)


def upsert_linha(df_base: pd.DataFrame, data_evento: date, tipo: str, dados: dict) -> pd.DataFrame:
    """
    Insere ou atualiza a linha de `data_evento` na base persistente.

    - tipo == "prescricao": grava apenas os campos de META, sem tocar nos campos REAIS
      já existentes.
    - tipo == "realizado": grava apenas os campos REAIS, sem apagar a META já gravada
      anteriormente (é exatamente isso que evita perder o planejado quando o Runna
      substitui o evento pelo resumo).
    """
    idx_existente = df_base.index[df_base["data"] == data_evento]

    if len(idx_existente) == 0:
        nova_linha = {coluna: None for coluna in COLUNAS_BASE}
        nova_linha["data"] = data_evento
        df_base = pd.concat([df_base, pd.DataFrame([nova_linha])], ignore_index=True)
        idx = df_base.index[df_base["data"] == data_evento][0]
    else:
        idx = idx_existente[0]

    if tipo == "prescricao":
        df_base.at[idx, "nome_treino"] = dados["nome_treino"]
        df_base.at[idx, "distancia_prescrita_km"] = dados["distancia_prescrita_km"]
        df_base.at[idx, "tempo_prescrito_min"] = dados["tempo_prescrito_min"]
        df_base.at[idx, "pace_prescrito_min_km"] = dados["pace_prescrito_min_km"]
    else:  # "realizado"
        # Só usa o nome do resumo se ainda não existir um nome de prescrição salvo.
        nome_atual = df_base.at[idx, "nome_treino"]
        if pd.isna(nome_atual) or not str(nome_atual).strip():
            df_base.at[idx, "nome_treino"] = dados["nome_treino"]
        df_base.at[idx, "distancia_real_km"] = dados["distancia_real_km"]
        df_base.at[idx, "tempo_real_min"] = dados["tempo_real_min"]
        df_base.at[idx, "pace_real_min_km"] = dados["pace_real_min_km"]

    df_base.at[idx, "status"] = classificar_status(df_base.loc[idx])
    return df_base


# ============================================================
# DOWNLOAD E SINCRONIZAÇÃO DO ICAL -> BASE PERSISTENTE
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def baixar_ical(url: str) -> bytes:
    """Baixa o conteúdo do iCal a partir da URL pública do Google Calendar (Runna)."""
    resposta = requests.get(url, timeout=20)
    resposta.raise_for_status()
    return resposta.content


def sincronizar_calendario(url: str, data_inicio: date) -> pd.DataFrame:
    """
    Baixa o iCal, classifica cada evento (prescrição ou realizado) e grava
    progressivamente na base persistente, preservando os dados já existentes.
    """
    conteudo = baixar_ical(url)
    calendario = Calendar.from_ical(conteudo)
    df_base = carregar_base()

    for componente in calendario.walk():
        if componente.name != "VEVENT":
            continue

        dtstart = componente.get("dtstart")
        if dtstart is None:
            continue

        data_evento = dtstart.dt
        if isinstance(data_evento, datetime):
            data_evento = data_evento.date()

        if data_evento < data_inicio or data_evento > date.today():
            continue

        titulo = str(componente.get("summary", ""))
        descricao = str(componente.get("description", ""))

        if eh_evento_realizado(descricao):
            dados = extrair_realizado(titulo, descricao)
            if dados:
                df_base = upsert_linha(df_base, data_evento, "realizado", dados)
        else:
            dados = extrair_prescricao(titulo, descricao)
            if dados:
                df_base = upsert_linha(df_base, data_evento, "prescricao", dados)

    df_base = df_base.sort_values("data").reset_index(drop=True)
    salvar_base(df_base)
    return df_base


def salvar_edicao_manual(data_evento: date, campos_prescritos: dict, campos_reais: dict) -> pd.DataFrame:
    """Aplica um ajuste manual diretamente na base persistente e recalcula pace/status."""
    df_base = carregar_base()

    if campos_prescritos:
        df_base = upsert_linha(df_base, data_evento, "prescricao", {
            "nome_treino": campos_prescritos.get("nome_treino") or "Treino",
            "distancia_prescrita_km": campos_prescritos["distancia_prescrita_km"],
            "tempo_prescrito_min": campos_prescritos["tempo_prescrito_min"],
            "pace_prescrito_min_km": calcular_pace_min_km(
                campos_prescritos["tempo_prescrito_min"], campos_prescritos["distancia_prescrita_km"]
            ),
        })
    if campos_reais:
        df_base = upsert_linha(df_base, data_evento, "realizado", {
            "nome_treino": campos_reais.get("nome_treino") or "Treino realizado",
            "distancia_real_km": campos_reais["distancia_real_km"],
            "tempo_real_min": campos_reais["tempo_real_min"],
            "pace_real_min_km": calcular_pace_min_km(
                campos_reais["tempo_real_min"], campos_reais["distancia_real_km"]
            ),
        })

    df_base = df_base.sort_values("data").reset_index(drop=True)
    salvar_base(df_base)
    return df_base


# ============================================================
# BANNER DE CONTAGEM REGRESSIVA
# ============================================================
def renderizar_banner_contagem(data_prova: date, nome_prova: str) -> None:
    dias_restantes = (data_prova - date.today()).days
    if dias_restantes > 0:
        texto_dias = f"Faltam {dias_restantes} dias"
    elif dias_restantes == 0:
        texto_dias = "É hoje! 🎉"
    else:
        texto_dias = f"Prova realizada há {abs(dias_restantes)} dias"

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #FF4B4B, #FF8C42);
            padding: 14px 22px; border-radius: 12px; color: white;
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 18px;">
            <div>
                <div style="font-size: 0.9rem; opacity: 0.9;">🏁 {nome_prova} — {data_prova.strftime('%d/%m/%Y')}</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{texto_dias}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ESTADO DA SESSÃO
# ============================================================
if "df_treinos" not in st.session_state:
    st.session_state.df_treinos = carregar_base()
if "url_ical" not in st.session_state:
    st.session_state.url_ical = ""


# ============================================================
# SIDEBAR — SINCRONIZAÇÃO
# ============================================================
with st.sidebar:
    st.header("⚙️ Sincronização")
    url_ical = st.text_input(
        "URL do iCal (Google Calendar / Runna)",
        value=st.session_state.url_ical,
        placeholder="https://calendar.google.com/calendar/ical/.../basic.ics",
    )
    st.caption(f"Dados considerados a partir de **{DATA_INICIO_PLANO.strftime('%d/%m/%Y')}**")
    st.caption(f"Base persistente: `{os.path.basename(DB_PATH)}`")

    if st.button("🔄 Sincronizar Calendário", use_container_width=True, type="primary"):
        if not url_ical:
            st.error("Informe a URL do iCal antes de sincronizar.")
        else:
            with st.spinner("Baixando, classificando e gravando os treinos..."):
                try:
                    df = sincronizar_calendario(url_ical, DATA_INICIO_PLANO)
                    st.session_state.df_treinos = df
                    st.session_state.url_ical = url_ical
                    st.success(f"{len(df)} dias sincronizados e gravados na base!")
                except Exception as e:
                    st.error(f"Erro ao sincronizar: {e}")

df_treinos = st.session_state.df_treinos.copy()


# ============================================================
# LAYOUT PRINCIPAL
# ============================================================
st.title("🏃 Follow Here — Diário de Treinos")

aba_diario, aba_evolucao = st.tabs(["📅 Diário do Dia", "📈 Evolução"])

# ------------------------------------------------------------
# ABA 1: DIÁRIO DO DIA — sub-aba Corrida
# ------------------------------------------------------------
with aba_diario:
    (sub_corrida,) = st.tabs(["🏃 Corrida"])

    with sub_corrida:
        renderizar_banner_contagem(DATA_PROVA, NOME_PROVA)

        if df_treinos.empty:
            st.info("Nenhum dado sincronizado ainda. Clique em **Sincronizar Calendário** na barra lateral.")
        else:
            datas_disponiveis = sorted(df_treinos["data"].unique())
            valor_padrao = date.today() if date.today() in datas_disponiveis else datas_disponiveis[-1]

            data_selecionada = st.date_input(
                "Data do registro",
                value=valor_padrao,
                min_value=DATA_INICIO_PLANO,
                max_value=date.today(),
            )

            linha_sel = df_treinos[df_treinos["data"] == data_selecionada]

            if linha_sel.empty:
                st.warning("Nenhum treino registrado para esta data.")
                treino = pd.Series({c: None for c in COLUNAS_BASE})
            else:
                treino = linha_sel.iloc[0]

                # --------- CARD DE CABEÇALHO ---------
                st.subheader(f"🎯 {treino['nome_treino'] or 'Treino'}")
                st.caption(
                    f"{data_selecionada.strftime('%A, %d/%m/%Y')} — "
                    f"Status: **{treino.get('status') or 'Sem Registro'}**"
                )

                # --------- COMPARATIVO LADO A LADO ---------
                col1, col2, col3 = st.columns(3)

                with col1:
                    delta_dist = None
                    if pd.notna(treino["distancia_prescrita_km"]) and pd.notna(treino["distancia_real_km"]):
                        delta_dist = treino["distancia_real_km"] - treino["distancia_prescrita_km"]
                    st.metric(
                        "Distância (Real)",
                        f"{treino['distancia_real_km']:.2f} km" if pd.notna(treino["distancia_real_km"]) else "-",
                        delta=f"{delta_dist:+.2f} km" if delta_dist is not None else None,
                        help=(
                            f"Planejado: {treino['distancia_prescrita_km']:.2f} km"
                            if pd.notna(treino["distancia_prescrita_km"]) else "Sem meta registrada para o dia"
                        ),
                    )

                with col2:
                    delta_tempo = None
                    if pd.notna(treino["tempo_prescrito_min"]) and pd.notna(treino["tempo_real_min"]):
                        delta_tempo = treino["tempo_real_min"] - treino["tempo_prescrito_min"]
                    st.metric(
                        "Tempo (Real)",
                        minutos_para_mmss(treino["tempo_real_min"]),
                        delta=f"{delta_tempo:+.1f} min" if delta_tempo is not None else None,
                        delta_color="inverse",  # menos tempo do que o previsto = melhor
                        help=f"Planejado: {minutos_para_mmss(treino['tempo_prescrito_min'])}",
                    )

                with col3:
                    delta_pace = None
                    if pd.notna(treino["pace_prescrito_min_km"]) and pd.notna(treino["pace_real_min_km"]):
                        delta_pace = treino["pace_real_min_km"] - treino["pace_prescrito_min_km"]
                    st.metric(
                        "Pace (Real)",
                        formatar_pace(treino["pace_real_min_km"]),
                        delta=f"{delta_pace:+.2f} min/km" if delta_pace is not None else None,
                        delta_color="inverse",  # pace menor do que o previsto = melhor
                        help=f"Planejado: {formatar_pace(treino['pace_prescrito_min_km'])}",
                    )

                st.divider()

            # --------- EDIÇÃO MANUAL (ÚNICO EXPANDER, DISCRETO) ---------
            with st.expander("✏️ Editar registro manualmente"):
                st.caption("Os ajustes gravam direto na base persistente, sem apagar os demais dados do dia.")
                with st.form(key=f"form_edicao_{data_selecionada}"):
                    st.markdown("**Meta (prescrito)**")
                    c1, c2 = st.columns(2)
                    nova_dist_prescrita = c1.number_input(
                        "Distância prescrita (km)",
                        value=float(treino["distancia_prescrita_km"]) if pd.notna(treino.get("distancia_prescrita_km")) else 0.0,
                        step=0.01, format="%.2f",
                    )
                    novo_tempo_prescrito = c2.number_input(
                        "Tempo prescrito (min, decimal)",
                        value=float(treino["tempo_prescrito_min"]) if pd.notna(treino.get("tempo_prescrito_min")) else 0.0,
                        step=0.1, format="%.2f",
                    )

                    st.markdown("**Realizado**")
                    c3, c4 = st.columns(2)
                    nova_dist_real = c3.number_input(
                        "Distância realizada (km)",
                        value=float(treino["distancia_real_km"]) if pd.notna(treino.get("distancia_real_km")) else 0.0,
                        step=0.01, format="%.2f",
                    )
                    novo_tempo_real = c4.number_input(
                        "Tempo realizado (min, decimal — ex.: 38:05 = 38.08)",
                        value=float(treino["tempo_real_min"]) if pd.notna(treino.get("tempo_real_min")) else 0.0,
                        step=0.1, format="%.2f",
                    )

                    salvar = st.form_submit_button("Salvar ajuste")
                    if salvar:
                        campos_prescritos = (
                            {
                                "nome_treino": treino.get("nome_treino"),
                                "distancia_prescrita_km": nova_dist_prescrita,
                                "tempo_prescrito_min": novo_tempo_prescrito,
                            }
                            if nova_dist_prescrita > 0 or novo_tempo_prescrito > 0 else None
                        )
                        campos_reais = (
                            {
                                "nome_treino": treino.get("nome_treino"),
                                "distancia_real_km": nova_dist_real,
                                "tempo_real_min": novo_tempo_real,
                            }
                            if nova_dist_real > 0 or novo_tempo_real > 0 else None
                        )
                        df_atualizado = salvar_edicao_manual(data_selecionada, campos_prescritos, campos_reais)
                        st.session_state.df_treinos = df_atualizado
                        st.success("Registro atualizado manualmente na base persistente.")
                        st.rerun()

# ------------------------------------------------------------
# ABA 2: EVOLUÇÃO — panorama geral do plano
# ------------------------------------------------------------
with aba_evolucao:
    if df_treinos.empty:
        st.info("Sincronize o calendário para visualizar a evolução do plano.")
    else:
        st.subheader("📊 Panorama Geral do Plano")

        dist_total_real = df_treinos["distancia_real_km"].sum(skipna=True)
        dist_total_plan = df_treinos["distancia_prescrita_km"].sum(skipna=True)
        horas_total_real = df_treinos["tempo_real_min"].sum(skipna=True) / 60
        pct_cumprimento = (dist_total_real / dist_total_plan * 100) if dist_total_plan else 0

        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Distância Total Realizada",
            f"{dist_total_real:.1f} km",
            delta=f"{dist_total_real - dist_total_plan:+.1f} km vs {dist_total_plan:.1f} km planejados",
        )
        k2.metric("Horas Totais de Treino", f"{horas_total_real:.1f} h")
        k3.metric("% de Cumprimento do Plano", f"{pct_cumprimento:.0f}%")

        st.divider()

        # --------- GRÁFICO 1: VOLUME SEMANAL ---------
        df_semanal = df_treinos.copy()
        df_semanal["semana"] = (
            pd.to_datetime(df_semanal["data"]).dt.to_period("W-SUN").apply(lambda p: p.start_time.date())
        )
        agrupado = df_semanal.groupby("semana", as_index=False).agg(
            km_prescrito=("distancia_prescrita_km", "sum"),
            km_real=("distancia_real_km", "sum"),
        )

        fig_semanal = go.Figure()
        fig_semanal.add_bar(x=agrupado["semana"], y=agrupado["km_prescrito"], name="Planejado (km)")
        fig_semanal.add_bar(x=agrupado["semana"], y=agrupado["km_real"], name="Realizado (km)")
        fig_semanal.update_layout(
            barmode="group",
            title="Volume Semanal: Planejado vs Realizado",
            xaxis_title="Semana",
            yaxis_title="Km",
        )
        st.plotly_chart(fig_semanal, use_container_width=True)

        # --------- GRÁFICO 2: TENDÊNCIA DE PACE ---------
        df_pace_real = df_treinos.dropna(subset=["pace_real_min_km"]).sort_values("data")
        fig_pace = go.Figure()
        fig_pace.add_scatter(
            x=df_pace_real["data"], y=df_pace_real["pace_real_min_km"],
            mode="lines+markers", name="Pace Realizado",
        )
        df_pace_plan = df_treinos.dropna(subset=["pace_prescrito_min_km"]).sort_values("data")
        if not df_pace_plan.empty:
            fig_pace.add_scatter(
                x=df_pace_plan["data"], y=df_pace_plan["pace_prescrito_min_km"],
                mode="lines+markers", name="Pace Planejado", line=dict(dash="dot"),
            )
        fig_pace.update_layout(
            title="Evolução do Pace ao Longo do Tempo",
            xaxis_title="Data",
            yaxis_title="Pace (min/km)",
        )
        st.plotly_chart(fig_pace, use_container_width=True)

        st.divider()

        # --------- TABELA GERAL ---------
        st.subheader("📋 Histórico Detalhado")
        tabela = df_treinos.copy()
        tabela["Pace Planejado"] = tabela["pace_prescrito_min_km"].apply(formatar_pace)
        tabela["Pace Realizado"] = tabela["pace_real_min_km"].apply(formatar_pace)
        tabela["Tempo Planejado"] = tabela["tempo_prescrito_min"].apply(minutos_para_mmss)
        tabela["Tempo Realizado"] = tabela["tempo_real_min"].apply(minutos_para_mmss)

        tabela_exibicao = tabela[[
            "data", "nome_treino", "distancia_prescrita_km", "distancia_real_km",
            "Tempo Planejado", "Tempo Realizado", "Pace Planejado", "Pace Realizado", "status",
        ]].rename(columns={
            "data": "Data",
            "nome_treino": "Treino",
            "distancia_prescrita_km": "Dist. Planejada (km)",
            "distancia_real_km": "Dist. Realizada (km)",
            "status": "Status",
        })

        st.dataframe(
            tabela_exibicao.sort_values("Data", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
