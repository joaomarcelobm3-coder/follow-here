"""
Diário de Treinos de Corrida — Integração Runna (Google Calendar iCal)
========================================================================
Refatoração completa:
  - Sincronização automática do iCal (Runna via Google Calendar)
  - Parsing via Regex de eventos PLANEJADOS e REALIZADOS
  - Aba "Diário do Dia" (sub-aba Corrida) com card comparativo
  - Aba "Evolução" com KPIs, gráficos semanais e tabela histórica
"""

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
TOLERANCIA_DISTANCIA_PCT = 0.10   # 10% de tolerância na distância
TOLERANCIA_PACE_PCT = 0.05        # 5% de tolerância no pace

st.set_page_config(page_title="Diário de Treinos - Corrida", layout="wide", page_icon="🏃")


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
    if not tempo_min or not distancia_km or distancia_km == 0:
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

# --- EVENTO PLANEJADO ---
# Descrição típica: "Treino de ritmo • 5,5km • 35m - 40m\n2km de aquecimento..."
REGEX_PLANEJADO_FAIXA = re.compile(
    r"(?P<nome>[^•\n]+?)\s*•\s*(?P<distancia>[\d.,]+)\s*km\s*•\s*"
    r"(?P<tempo_min>\d+)\s*m\s*-\s*(?P<tempo_max>\d+)\s*m",
    re.IGNORECASE,
)

# Variante sem faixa, com tempo único: "Treino longo • 12km • 70m"
REGEX_PLANEJADO_UNICO = re.compile(
    r"(?P<nome>[^•\n]+?)\s*•\s*(?P<distancia>[\d.,]+)\s*km\s*•\s*(?P<tempo>\d+)\s*m(?!\s*-)",
    re.IGNORECASE,
)

# --- EVENTO REALIZADO (sincronizado via Strava/Garmin no Runna) ---
REGEX_REALIZADO_DISTANCIA = re.compile(r"Dist[âa]ncia:\s*([\d.,]+)\s*km", re.IGNORECASE)
REGEX_REALIZADO_TEMPO = re.compile(r"Hor[áa]rio:\s*(\d+):(\d{2})", re.IGNORECASE)
REGEX_REALIZADO_PACE = re.compile(r"Ritmo\s*m[ée]dio:\s*(\d+):(\d{2})\s*/\s*km", re.IGNORECASE)

# Marcador para identificar que a descrição é um "resumo de execução"
MARCADOR_REALIZADO = re.compile(r"(Resumo|Dist[âa]ncia:|Hor[áa]rio:)", re.IGNORECASE)


def eh_evento_realizado(descricao: str) -> bool:
    """Detecta se a descrição corresponde a um treino REALIZADO (execução sincronizada)."""
    if not descricao:
        return False
    return bool(MARCADOR_REALIZADO.search(descricao)) and bool(REGEX_REALIZADO_DISTANCIA.search(descricao))


def extrair_planejado(titulo: str, descricao: str):
    """Extrai nome, distância planejada, tempo médio planejado e pace planejado."""
    texto = descricao or titulo or ""

    match = REGEX_PLANEJADO_FAIXA.search(texto)
    if match:
        nome = match.group("nome").strip()
        distancia = parse_distancia_br(match.group("distancia"))
        t_min = float(match.group("tempo_min"))
        t_max = float(match.group("tempo_max"))
        tempo_medio = (t_min + t_max) / 2
    else:
        match2 = REGEX_PLANEJADO_UNICO.search(texto)
        if not match2:
            return None
        nome = match2.group("nome").strip()
        distancia = parse_distancia_br(match2.group("distancia"))
        tempo_medio = float(match2.group("tempo"))

    pace = calcular_pace_min_km(tempo_medio, distancia)
    return {
        "nome_treino": nome,
        "distancia_planejada_km": distancia,
        "tempo_planejado_min": tempo_medio,
        "pace_planejado_min_km": pace,
    }


def extrair_realizado(titulo: str, descricao: str):
    """Extrai distância, tempo e pace REALIZADOS a partir do resumo Strava/Garmin."""
    texto = descricao or ""

    m_dist = REGEX_REALIZADO_DISTANCIA.search(texto)
    m_tempo = REGEX_REALIZADO_TEMPO.search(texto)
    m_pace = REGEX_REALIZADO_PACE.search(texto)

    distancia = parse_distancia_br(m_dist.group(1)) if m_dist else None
    tempo = mmss_para_minutos(m_tempo.group(1), m_tempo.group(2)) if m_tempo else None
    pace = mmss_para_minutos(m_pace.group(1), m_pace.group(2)) if m_pace else calcular_pace_min_km(tempo, distancia)

    nome = titulo.replace("🏃", "").strip() if titulo else "Treino realizado"
    return {
        "nome_treino": nome or "Treino realizado",
        "distancia_realizada_km": distancia,
        "tempo_realizado_min": tempo,
        "pace_realizado_min_km": pace,
    }


# ============================================================
# STATUS DA SESSÃO (Planejado x Realizado)
# ============================================================
def classificar_status(linha) -> str:
    """
    Compara planejado x realizado e retorna um status textual.
    Ajuste TOLERANCIA_DISTANCIA_PCT / TOLERANCIA_PACE_PCT conforme necessário.
    """
    d_plan = linha.get("distancia_planejada_km")
    d_real = linha.get("distancia_realizada_km")
    p_plan = linha.get("pace_planejado_min_km")
    p_real = linha.get("pace_realizado_min_km")

    if pd.isna(d_plan) or pd.isna(d_real):
        return "Sem Registro"

    delta_dist_pct = (d_real - d_plan) / d_plan if d_plan else 0

    delta_pace_pct = 0
    if p_plan and p_real and not pd.isna(p_plan) and not pd.isna(p_real):
        delta_pace_pct = (p_plan - p_real) / p_plan  # positivo = correu mais rápido que o planejado

    if delta_dist_pct < -TOLERANCIA_DISTANCIA_PCT:
        return "Abaixo do Alvo"
    if delta_dist_pct > TOLERANCIA_DISTANCIA_PCT or delta_pace_pct > TOLERANCIA_PACE_PCT:
        return "Superado"
    return "Dentro do Alvo"


# ============================================================
# DOWNLOAD E PROCESSAMENTO DO ICAL
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def baixar_ical(url: str) -> bytes:
    """Baixa o conteúdo do iCal a partir da URL pública do Google Calendar (Runna)."""
    resposta = requests.get(url, timeout=20)
    resposta.raise_for_status()
    return resposta.content


def processar_ical(conteudo_ical: bytes, data_inicio: date) -> pd.DataFrame:
    """
    Lê o iCal e consolida uma linha por dia com os dados planejados e
    realizados (quando existirem), a partir de `data_inicio` até hoje.
    """
    calendario = Calendar.from_ical(conteudo_ical)

    registros_planejados = {}
    registros_realizados = {}

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
                registros_realizados[data_evento] = dados
        else:
            dados = extrair_planejado(titulo, descricao)
            if dados:
                registros_planejados[data_evento] = dados

    todas_datas = sorted(set(registros_planejados) | set(registros_realizados))

    linhas = []
    for d in todas_datas:
        planejado = registros_planejados.get(d, {})
        realizado = registros_realizados.get(d, {})
        linhas.append({
            "data": d,
            "nome_treino": planejado.get("nome_treino") or realizado.get("nome_treino") or "Treino",
            "distancia_planejada_km": planejado.get("distancia_planejada_km"),
            "tempo_planejado_min": planejado.get("tempo_planejado_min"),
            "pace_planejado_min_km": planejado.get("pace_planejado_min_km"),
            "distancia_realizada_km": realizado.get("distancia_realizada_km"),
            "tempo_realizado_min": realizado.get("tempo_realizado_min"),
            "pace_realizado_min_km": realizado.get("pace_realizado_min_km"),
        })

    df = pd.DataFrame(linhas)
    if not df.empty:
        df["status"] = df.apply(classificar_status, axis=1)
    return df


# ============================================================
# ESTADO DA SESSÃO
# ============================================================
if "df_treinos" not in st.session_state:
    st.session_state.df_treinos = pd.DataFrame()
if "edicoes_manuais" not in st.session_state:
    st.session_state.edicoes_manuais = {}  # {data: {"distancia_realizada_km": x, "tempo_realizado_min": y}}
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

    if st.button("🔄 Sincronizar Calendário", use_container_width=True, type="primary"):
        if not url_ical:
            st.error("Informe a URL do iCal antes de sincronizar.")
        else:
            with st.spinner("Baixando e processando o calendário..."):
                try:
                    conteudo = baixar_ical(url_ical)
                    df = processar_ical(conteudo, DATA_INICIO_PLANO)
                    st.session_state.df_treinos = df
                    st.session_state.url_ical = url_ical
                    st.success(f"{len(df)} sessões sincronizadas com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao sincronizar: {e}")

# DataFrame de trabalho (aplica edições manuais por cima do que veio do iCal)
df_treinos = st.session_state.df_treinos.copy()

for data_edicao, campos in st.session_state.edicoes_manuais.items():
    if not df_treinos.empty and data_edicao in df_treinos["data"].values:
        idx = df_treinos.index[df_treinos["data"] == data_edicao][0]
        for campo, valor in campos.items():
            df_treinos.at[idx, campo] = valor
        df_treinos.at[idx, "pace_realizado_min_km"] = calcular_pace_min_km(
            df_treinos.at[idx, "tempo_realizado_min"], df_treinos.at[idx, "distancia_realizada_km"]
        )
        df_treinos.at[idx, "status"] = classificar_status(df_treinos.loc[idx])


# ============================================================
# LAYOUT PRINCIPAL
# ============================================================
st.title("🏃 Diário de Treinos de Corrida")

aba_diario, aba_evolucao = st.tabs(["📅 Diário do Dia", "📈 Evolução"])

# ------------------------------------------------------------
# ABA 1: DIÁRIO DO DIA — sub-aba Corrida
# ------------------------------------------------------------
with aba_diario:
    (sub_corrida,) = st.tabs(["🏃 Corrida"])

    with sub_corrida:
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
            else:
                treino = linha_sel.iloc[0]

                # --------- CARD DE CABEÇALHO ---------
                st.subheader(f"🎯 {treino['nome_treino']}")
                st.caption(
                    f"{data_selecionada.strftime('%A, %d/%m/%Y')} — "
                    f"Status: **{treino.get('status', 'Sem Registro')}**"
                )

                # --------- COMPARATIVO LADO A LADO ---------
                col1, col2, col3 = st.columns(3)

                with col1:
                    delta_dist = None
                    if pd.notna(treino["distancia_planejada_km"]) and pd.notna(treino["distancia_realizada_km"]):
                        delta_dist = treino["distancia_realizada_km"] - treino["distancia_planejada_km"]
                    st.metric(
                        "Distância (Realizada)",
                        f"{treino['distancia_realizada_km']:.2f} km" if pd.notna(treino["distancia_realizada_km"]) else "-",
                        delta=f"{delta_dist:+.2f} km" if delta_dist is not None else None,
                        help=(
                            f"Planejado: {treino['distancia_planejada_km']:.2f} km"
                            if pd.notna(treino["distancia_planejada_km"]) else "Sem planejamento para o dia"
                        ),
                    )

                with col2:
                    delta_tempo = None
                    if pd.notna(treino["tempo_planejado_min"]) and pd.notna(treino["tempo_realizado_min"]):
                        delta_tempo = treino["tempo_realizado_min"] - treino["tempo_planejado_min"]
                    st.metric(
                        "Tempo (Realizado)",
                        minutos_para_mmss(treino["tempo_realizado_min"]),
                        delta=f"{delta_tempo:+.1f} min" if delta_tempo is not None else None,
                        delta_color="inverse",  # menos tempo do que o previsto = melhor
                        help=f"Planejado: {minutos_para_mmss(treino['tempo_planejado_min'])}",
                    )

                with col3:
                    delta_pace = None
                    if pd.notna(treino["pace_planejado_min_km"]) and pd.notna(treino["pace_realizado_min_km"]):
                        delta_pace = treino["pace_realizado_min_km"] - treino["pace_planejado_min_km"]
                    st.metric(
                        "Pace (Realizado)",
                        formatar_pace(treino["pace_realizado_min_km"]),
                        delta=f"{delta_pace:+.2f} min/km" if delta_pace is not None else None,
                        delta_color="inverse",  # pace menor do que o previsto = melhor
                        help=f"Planejado: {formatar_pace(treino['pace_planejado_min_km'])}",
                    )

                st.divider()

                # --------- EDIÇÃO MANUAL (ÚNICO EXPANDER, DISCRETO) ---------
                with st.expander("✏️ Editar registro manualmente"):
                    with st.form(key=f"form_edicao_{data_selecionada}"):
                        nova_distancia = st.number_input(
                            "Distância realizada (km)",
                            value=float(treino["distancia_realizada_km"]) if pd.notna(treino["distancia_realizada_km"]) else 0.0,
                            step=0.01,
                            format="%.2f",
                        )
                        novo_tempo = st.number_input(
                            "Tempo realizado (min, decimal — ex.: 38:05 = 38.08)",
                            value=float(treino["tempo_realizado_min"]) if pd.notna(treino["tempo_realizado_min"]) else 0.0,
                            step=0.1,
                            format="%.2f",
                        )
                        salvar = st.form_submit_button("Salvar ajuste")
                        if salvar:
                            st.session_state.edicoes_manuais[data_selecionada] = {
                                "distancia_realizada_km": nova_distancia,
                                "tempo_realizado_min": novo_tempo,
                            }
                            st.success("Registro atualizado manualmente.")
                            st.rerun()

# ------------------------------------------------------------
# ABA 2: EVOLUÇÃO — panorama geral do plano
# ------------------------------------------------------------
with aba_evolucao:
    if df_treinos.empty:
        st.info("Sincronize o calendário para visualizar a evolução do plano.")
    else:
        st.subheader("📊 Panorama Geral do Plano")

        dist_total_real = df_treinos["distancia_realizada_km"].sum(skipna=True)
        dist_total_plan = df_treinos["distancia_planejada_km"].sum(skipna=True)
        horas_total_real = df_treinos["tempo_realizado_min"].sum(skipna=True) / 60
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
            km_planejado=("distancia_planejada_km", "sum"),
            km_realizado=("distancia_realizada_km", "sum"),
        )

        fig_semanal = go.Figure()
        fig_semanal.add_bar(x=agrupado["semana"], y=agrupado["km_planejado"], name="Planejado (km)")
        fig_semanal.add_bar(x=agrupado["semana"], y=agrupado["km_realizado"], name="Realizado (km)")
        fig_semanal.update_layout(
            barmode="group",
            title="Volume Semanal: Planejado vs Realizado",
            xaxis_title="Semana",
            yaxis_title="Km",
        )
        st.plotly_chart(fig_semanal, use_container_width=True)

        # --------- GRÁFICO 2: TENDÊNCIA DE PACE ---------
        df_pace_real = df_treinos.dropna(subset=["pace_realizado_min_km"]).sort_values("data")
        fig_pace = go.Figure()
        fig_pace.add_scatter(
            x=df_pace_real["data"], y=df_pace_real["pace_realizado_min_km"],
            mode="lines+markers", name="Pace Realizado",
        )
        df_pace_plan = df_treinos.dropna(subset=["pace_planejado_min_km"]).sort_values("data")
        if not df_pace_plan.empty:
            fig_pace.add_scatter(
                x=df_pace_plan["data"], y=df_pace_plan["pace_planejado_min_km"],
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
        tabela["Pace Planejado"] = tabela["pace_planejado_min_km"].apply(formatar_pace)
        tabela["Pace Realizado"] = tabela["pace_realizado_min_km"].apply(formatar_pace)
        tabela["Tempo Planejado"] = tabela["tempo_planejado_min"].apply(minutos_para_mmss)
        tabela["Tempo Realizado"] = tabela["tempo_realizado_min"].apply(minutos_para_mmss)

        tabela_exibicao = tabela[[
            "data", "nome_treino", "distancia_planejada_km", "distancia_realizada_km",
            "Tempo Planejado", "Tempo Realizado", "Pace Planejado", "Pace Realizado", "status",
        ]].rename(columns={
            "data": "Data",
            "nome_treino": "Treino",
            "distancia_planejada_km": "Dist. Planejada (km)",
            "distancia_realizada_km": "Dist. Realizada (km)",
            "status": "Status",
        })

        st.dataframe(
            tabela_exibicao.sort_values("Data", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
"""
Diário de Treinos de Corrida — Integração Runna (Google Calendar iCal)
========================================================================
Refatoração completa:
  - Sincronização automática do iCal (Runna via Google Calendar)
  - Parsing via Regex de eventos PLANEJADOS e REALIZADOS
  - Aba "Diário do Dia" (sub-aba Corrida) com card comparativo
  - Aba "Evolução" com KPIs, gráficos semanais e tabela histórica
"""

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
TOLERANCIA_DISTANCIA_PCT = 0.10   # 10% de tolerância na distância
TOLERANCIA_PACE_PCT = 0.05        # 5% de tolerância no pace

st.set_page_config(page_title="Diário de Treinos - Corrida", layout="wide", page_icon="🏃")


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
    if not tempo_min or not distancia_km or distancia_km == 0:
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

# --- EVENTO PLANEJADO ---
# Descrição típica: "Treino de ritmo • 5,5km • 35m - 40m\n2km de aquecimento..."
REGEX_PLANEJADO_FAIXA = re.compile(
    r"(?P<nome>[^•\n]+?)\s*•\s*(?P<distancia>[\d.,]+)\s*km\s*•\s*"
    r"(?P<tempo_min>\d+)\s*m\s*-\s*(?P<tempo_max>\d+)\s*m",
    re.IGNORECASE,
)

# Variante sem faixa, com tempo único: "Treino longo • 12km • 70m"
REGEX_PLANEJADO_UNICO = re.compile(
    r"(?P<nome>[^•\n]+?)\s*•\s*(?P<distancia>[\d.,]+)\s*km\s*•\s*(?P<tempo>\d+)\s*m(?!\s*-)",
    re.IGNORECASE,
)

# --- EVENTO REALIZADO (sincronizado via Strava/Garmin no Runna) ---
REGEX_REALIZADO_DISTANCIA = re.compile(r"Dist[âa]ncia:\s*([\d.,]+)\s*km", re.IGNORECASE)
REGEX_REALIZADO_TEMPO = re.compile(r"Hor[áa]rio:\s*(\d+):(\d{2})", re.IGNORECASE)
REGEX_REALIZADO_PACE = re.compile(r"Ritmo\s*m[ée]dio:\s*(\d+):(\d{2})\s*/\s*km", re.IGNORECASE)

# Marcador para identificar que a descrição é um "resumo de execução"
MARCADOR_REALIZADO = re.compile(r"(Resumo|Dist[âa]ncia:|Hor[áa]rio:)", re.IGNORECASE)


def eh_evento_realizado(descricao: str) -> bool:
    """Detecta se a descrição corresponde a um treino REALIZADO (execução sincronizada)."""
    if not descricao:
        return False
    return bool(MARCADOR_REALIZADO.search(descricao)) and bool(REGEX_REALIZADO_DISTANCIA.search(descricao))


def extrair_planejado(titulo: str, descricao: str):
    """Extrai nome, distância planejada, tempo médio planejado e pace planejado."""
    texto = descricao or titulo or ""

    match = REGEX_PLANEJADO_FAIXA.search(texto)
    if match:
        nome = match.group("nome").strip()
        distancia = parse_distancia_br(match.group("distancia"))
        t_min = float(match.group("tempo_min"))
        t_max = float(match.group("tempo_max"))
        tempo_medio = (t_min + t_max) / 2
    else:
        match2 = REGEX_PLANEJADO_UNICO.search(texto)
        if not match2:
            return None
        nome = match2.group("nome").strip()
        distancia = parse_distancia_br(match2.group("distancia"))
        tempo_medio = float(match2.group("tempo"))

    pace = calcular_pace_min_km(tempo_medio, distancia)
    return {
        "nome_treino": nome,
        "distancia_planejada_km": distancia,
        "tempo_planejado_min": tempo_medio,
        "pace_planejado_min_km": pace,
    }


def extrair_realizado(titulo: str, descricao: str):
    """Extrai distância, tempo e pace REALIZADOS a partir do resumo Strava/Garmin."""
    texto = descricao or ""

    m_dist = REGEX_REALIZADO_DISTANCIA.search(texto)
    m_tempo = REGEX_REALIZADO_TEMPO.search(texto)
    m_pace = REGEX_REALIZADO_PACE.search(texto)

    distancia = parse_distancia_br(m_dist.group(1)) if m_dist else None
    tempo = mmss_para_minutos(m_tempo.group(1), m_tempo.group(2)) if m_tempo else None
    pace = mmss_para_minutos(m_pace.group(1), m_pace.group(2)) if m_pace else calcular_pace_min_km(tempo, distancia)

    nome = titulo.replace("🏃", "").strip() if titulo else "Treino realizado"
    return {
        "nome_treino": nome or "Treino realizado",
        "distancia_realizada_km": distancia,
        "tempo_realizado_min": tempo,
        "pace_realizado_min_km": pace,
    }


# ============================================================
# STATUS DA SESSÃO (Planejado x Realizado)
# ============================================================
def classificar_status(linha) -> str:
    """
    Compara planejado x realizado e retorna um status textual.
    Ajuste TOLERANCIA_DISTANCIA_PCT / TOLERANCIA_PACE_PCT conforme necessário.
    """
    d_plan = linha.get("distancia_planejada_km")
    d_real = linha.get("distancia_realizada_km")
    p_plan = linha.get("pace_planejado_min_km")
    p_real = linha.get("pace_realizado_min_km")

    if pd.isna(d_plan) or pd.isna(d_real):
        return "Sem Registro"

    delta_dist_pct = (d_real - d_plan) / d_plan if d_plan else 0

    delta_pace_pct = 0
    if p_plan and p_real and not pd.isna(p_plan) and not pd.isna(p_real):
        delta_pace_pct = (p_plan - p_real) / p_plan  # positivo = correu mais rápido que o planejado

    if delta_dist_pct < -TOLERANCIA_DISTANCIA_PCT:
        return "Abaixo do Alvo"
    if delta_dist_pct > TOLERANCIA_DISTANCIA_PCT or delta_pace_pct > TOLERANCIA_PACE_PCT:
        return "Superado"
    return "Dentro do Alvo"


# ============================================================
# DOWNLOAD E PROCESSAMENTO DO ICAL
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def baixar_ical(url: str) -> bytes:
    """Baixa o conteúdo do iCal a partir da URL pública do Google Calendar (Runna)."""
    resposta = requests.get(url, timeout=20)
    resposta.raise_for_status()
    return resposta.content


def processar_ical(conteudo_ical: bytes, data_inicio: date) -> pd.DataFrame:
    """
    Lê o iCal e consolida uma linha por dia com os dados planejados e
    realizados (quando existirem), a partir de `data_inicio` até hoje.
    """
    calendario = Calendar.from_ical(conteudo_ical)

    registros_planejados = {}
    registros_realizados = {}

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
                registros_realizados[data_evento] = dados
        else:
            dados = extrair_planejado(titulo, descricao)
            if dados:
                registros_planejados[data_evento] = dados

    todas_datas = sorted(set(registros_planejados) | set(registros_realizados))

    linhas = []
    for d in todas_datas:
        planejado = registros_planejados.get(d, {})
        realizado = registros_realizados.get(d, {})
        linhas.append({
            "data": d,
            "nome_treino": planejado.get("nome_treino") or realizado.get("nome_treino") or "Treino",
            "distancia_planejada_km": planejado.get("distancia_planejada_km"),
            "tempo_planejado_min": planejado.get("tempo_planejado_min"),
            "pace_planejado_min_km": planejado.get("pace_planejado_min_km"),
            "distancia_realizada_km": realizado.get("distancia_realizada_km"),
            "tempo_realizado_min": realizado.get("tempo_realizado_min"),
            "pace_realizado_min_km": realizado.get("pace_realizado_min_km"),
        })

    df = pd.DataFrame(linhas)
    if not df.empty:
        df["status"] = df.apply(classificar_status, axis=1)
    return df


# ============================================================
# ESTADO DA SESSÃO
# ============================================================
if "df_treinos" not in st.session_state:
    st.session_state.df_treinos = pd.DataFrame()
if "edicoes_manuais" not in st.session_state:
    st.session_state.edicoes_manuais = {}  # {data: {"distancia_realizada_km": x, "tempo_realizado_min": y}}
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

    if st.button("🔄 Sincronizar Calendário", use_container_width=True, type="primary"):
        if not url_ical:
            st.error("Informe a URL do iCal antes de sincronizar.")
        else:
            with st.spinner("Baixando e processando o calendário..."):
                try:
                    conteudo = baixar_ical(url_ical)
                    df = processar_ical(conteudo, DATA_INICIO_PLANO)
                    st.session_state.df_treinos = df
                    st.session_state.url_ical = url_ical
                    st.success(f"{len(df)} sessões sincronizadas com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao sincronizar: {e}")

# DataFrame de trabalho (aplica edições manuais por cima do que veio do iCal)
df_treinos = st.session_state.df_treinos.copy()

for data_edicao, campos in st.session_state.edicoes_manuais.items():
    if not df_treinos.empty and data_edicao in df_treinos["data"].values:
        idx = df_treinos.index[df_treinos["data"] == data_edicao][0]
        for campo, valor in campos.items():
            df_treinos.at[idx, campo] = valor
        df_treinos.at[idx, "pace_realizado_min_km"] = calcular_pace_min_km(
            df_treinos.at[idx, "tempo_realizado_min"], df_treinos.at[idx, "distancia_realizada_km"]
        )
        df_treinos.at[idx, "status"] = classificar_status(df_treinos.loc[idx])


# ============================================================
# LAYOUT PRINCIPAL
# ============================================================
st.title("🏃 Diário de Treinos de Corrida")

aba_diario, aba_evolucao = st.tabs(["📅 Diário do Dia", "📈 Evolução"])

# ------------------------------------------------------------
# ABA 1: DIÁRIO DO DIA — sub-aba Corrida
# ------------------------------------------------------------
with aba_diario:
    (sub_corrida,) = st.tabs(["🏃 Corrida"])

    with sub_corrida:
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
            else:
                treino = linha_sel.iloc[0]

                # --------- CARD DE CABEÇALHO ---------
                st.subheader(f"🎯 {treino['nome_treino']}")
                st.caption(
                    f"{data_selecionada.strftime('%A, %d/%m/%Y')} — "
                    f"Status: **{treino.get('status', 'Sem Registro')}**"
                )

                # --------- COMPARATIVO LADO A LADO ---------
                col1, col2, col3 = st.columns(3)

                with col1:
                    delta_dist = None
                    if pd.notna(treino["distancia_planejada_km"]) and pd.notna(treino["distancia_realizada_km"]):
                        delta_dist = treino["distancia_realizada_km"] - treino["distancia_planejada_km"]
                    st.metric(
                        "Distância (Realizada)",
                        f"{treino['distancia_realizada_km']:.2f} km" if pd.notna(treino["distancia_realizada_km"]) else "-",
                        delta=f"{delta_dist:+.2f} km" if delta_dist is not None else None,
                        help=(
                            f"Planejado: {treino['distancia_planejada_km']:.2f} km"
                            if pd.notna(treino["distancia_planejada_km"]) else "Sem planejamento para o dia"
                        ),
                    )

                with col2:
                    delta_tempo = None
                    if pd.notna(treino["tempo_planejado_min"]) and pd.notna(treino["tempo_realizado_min"]):
                        delta_tempo = treino["tempo_realizado_min"] - treino["tempo_planejado_min"]
                    st.metric(
                        "Tempo (Realizado)",
                        minutos_para_mmss(treino["tempo_realizado_min"]),
                        delta=f"{delta_tempo:+.1f} min" if delta_tempo is not None else None,
                        delta_color="inverse",  # menos tempo do que o previsto = melhor
                        help=f"Planejado: {minutos_para_mmss(treino['tempo_planejado_min'])}",
                    )

                with col3:
                    delta_pace = None
                    if pd.notna(treino["pace_planejado_min_km"]) and pd.notna(treino["pace_realizado_min_km"]):
                        delta_pace = treino["pace_realizado_min_km"] - treino["pace_planejado_min_km"]
                    st.metric(
                        "Pace (Realizado)",
                        formatar_pace(treino["pace_realizado_min_km"]),
                        delta=f"{delta_pace:+.2f} min/km" if delta_pace is not None else None,
                        delta_color="inverse",  # pace menor do que o previsto = melhor
                        help=f"Planejado: {formatar_pace(treino['pace_planejado_min_km'])}",
                    )

                st.divider()

                # --------- EDIÇÃO MANUAL (ÚNICO EXPANDER, DISCRETO) ---------
                with st.expander("✏️ Editar registro manualmente"):
                    with st.form(key=f"form_edicao_{data_selecionada}"):
                        nova_distancia = st.number_input(
                            "Distância realizada (km)",
                            value=float(treino["distancia_realizada_km"]) if pd.notna(treino["distancia_realizada_km"]) else 0.0,
                            step=0.01,
                            format="%.2f",
                        )
                        novo_tempo = st.number_input(
                            "Tempo realizado (min, decimal — ex.: 38:05 = 38.08)",
                            value=float(treino["tempo_realizado_min"]) if pd.notna(treino["tempo_realizado_min"]) else 0.0,
                            step=0.1,
                            format="%.2f",
                        )
                        salvar = st.form_submit_button("Salvar ajuste")
                        if salvar:
                            st.session_state.edicoes_manuais[data_selecionada] = {
                                "distancia_realizada_km": nova_distancia,
                                "tempo_realizado_min": novo_tempo,
                            }
                            st.success("Registro atualizado manualmente.")
                            st.rerun()

# ------------------------------------------------------------
# ABA 2: EVOLUÇÃO — panorama geral do plano
# ------------------------------------------------------------
with aba_evolucao:
    if df_treinos.empty:
        st.info("Sincronize o calendário para visualizar a evolução do plano.")
    else:
        st.subheader("📊 Panorama Geral do Plano")

        dist_total_real = df_treinos["distancia_realizada_km"].sum(skipna=True)
        dist_total_plan = df_treinos["distancia_planejada_km"].sum(skipna=True)
        horas_total_real = df_treinos["tempo_realizado_min"].sum(skipna=True) / 60
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
            km_planejado=("distancia_planejada_km", "sum"),
            km_realizado=("distancia_realizada_km", "sum"),
        )

        fig_semanal = go.Figure()
        fig_semanal.add_bar(x=agrupado["semana"], y=agrupado["km_planejado"], name="Planejado (km)")
        fig_semanal.add_bar(x=agrupado["semana"], y=agrupado["km_realizado"], name="Realizado (km)")
        fig_semanal.update_layout(
            barmode="group",
            title="Volume Semanal: Planejado vs Realizado",
            xaxis_title="Semana",
            yaxis_title="Km",
        )
        st.plotly_chart(fig_semanal, use_container_width=True)

        # --------- GRÁFICO 2: TENDÊNCIA DE PACE ---------
        df_pace_real = df_treinos.dropna(subset=["pace_realizado_min_km"]).sort_values("data")
        fig_pace = go.Figure()
        fig_pace.add_scatter(
            x=df_pace_real["data"], y=df_pace_real["pace_realizado_min_km"],
            mode="lines+markers", name="Pace Realizado",
        )
        df_pace_plan = df_treinos.dropna(subset=["pace_planejado_min_km"]).sort_values("data")
        if not df_pace_plan.empty:
            fig_pace.add_scatter(
                x=df_pace_plan["data"], y=df_pace_plan["pace_planejado_min_km"],
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
        tabela["Pace Planejado"] = tabela["pace_planejado_min_km"].apply(formatar_pace)
        tabela["Pace Realizado"] = tabela["pace_realizado_min_km"].apply(formatar_pace)
        tabela["Tempo Planejado"] = tabela["tempo_planejado_min"].apply(minutos_para_mmss)
        tabela["Tempo Realizado"] = tabela["tempo_realizado_min"].apply(minutos_para_mmss)

        tabela_exibicao = tabela[[
            "data", "nome_treino", "distancia_planejada_km", "distancia_realizada_km",
            "Tempo Planejado", "Tempo Realizado", "Pace Planejado", "Pace Realizado", "status",
        ]].rename(columns={
            "data": "Data",
            "nome_treino": "Treino",
            "distancia_planejada_km": "Dist. Planejada (km)",
            "distancia_realizada_km": "Dist. Realizada (km)",
            "status": "Status",
        })

        st.dataframe(
            tabela_exibicao.sort_values("Data", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
