import os
import json
import re
import pandas as pd
import streamlit as st

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from databricks import sql
from databricks.sdk.core import Config


st.set_page_config(
    page_title="World Bank Findex AI Assistant",
    page_icon="🌍",
    layout="wide"
)

# ----------------------------
# Config
# ----------------------------

VECTOR_SEARCH_ENDPOINT_NAME = os.getenv("VECTOR_SEARCH_ENDPOINT_NAME", "worldbank_findex_vs_endpoint")
VECTOR_SEARCH_INDEX_NAME = os.getenv("VECTOR_SEARCH_INDEX_NAME", "worldbank_demo.findex.findex_report_chunks_index")
LLM_ENDPOINT_NAME = os.getenv("LLM_ENDPOINT_NAME", "databricks-meta-llama-3-3-70b-instruct")

CATALOG = os.getenv("CATALOG", "worldbank_demo")
SCHEMA = os.getenv("SCHEMA", "findex")

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_findex_indicators"

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")

# ----------------------------
# Clients
# ----------------------------

@st.cache_resource
def get_workspace_client():
    return WorkspaceClient()


@st.cache_resource
def get_vector_index():
    cfg = Config()

    vsc = VectorSearchClient(
        workspace_url=cfg.host,
        service_principal_client_id=cfg.client_id,
        service_principal_client_secret=cfg.client_secret
    )

    return vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=VECTOR_SEARCH_INDEX_NAME
    )


w = get_workspace_client()
index = get_vector_index()


# ----------------------------
# LLM call
# ----------------------------

def call_llm(prompt: str) -> str:
    response = w.serving_endpoints.query(
        name=LLM_ENDPOINT_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant for the World Bank Global Findex 2025 demo."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1200,
        temperature=0.1
    )

    # Databricks SDK response shape can vary slightly by endpoint type.
    try:
        return response.choices[0].message.content
    except Exception:
        return str(response)


# ----------------------------
# SQL helper
# ----------------------------

def run_sql(query: str) -> pd.DataFrame:
    if not DATABRICKS_WAREHOUSE_ID:
        raise ValueError("DATABRICKS_WAREHOUSE_ID environment variable is not set.")

    with sql.connect(
        server_hostname=DATABRICKS_HOST.replace("https://", ""),
        http_path=f"/sql/1.0/warehouses/{DATABRICKS_WAREHOUSE_ID}",
        credentials_provider=lambda: w.config.authenticate
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description]
            return pd.DataFrame(rows, columns=cols)


# ----------------------------
# Router
# ----------------------------

def classify_question(question: str) -> dict:
    prompt = f"""
You are a router for a World Bank Global Findex 2025 AI assistant.

Classify the user question into:
1. route
2. group
3. subgroup
4. year

ROUTE VALUES:
- mobile_money_leaders
- low_account_ownership
- region_summary
- report_only
- combined_general

GROUP/SUBGROUP VALUES:
- all / all
- gender / men
- gender / women
- income / richest 60%
- income / poorest 40%
- age_cat / age 25+
- age_cat / ages 15-24
- urbanicity / rural
- urbanicity / urban
- laborforce / out of laborforce
- laborforce / in laborforce
- education / secondary edu or more
- education / prim edu or less

YEAR RULES:
- If the user mentions a specific year, return that year.
- Valid years: 2011, 2014, 2017, 2021, 2024.
- If user says latest/current/recent or no year, use 2024.
- If user asks for trend, over time, change over years, or compare years, use null.

ROUTE RULES:
- mobile_money_leaders: top/highest/ranking/leaders/mobile money ownership
- low_account_ownership: lowest account ownership/unbanked/low financial access
- region_summary: regions/regional comparison/average by region
- report_only: explanation, causes, barriers, recommendations, summary, narrative only
- combined_general: compare countries, combine data with explanation, both "which" and "why", or unsure

Return only valid JSON:
{{
  "route": "route_value",
  "group": "group_value",
  "subgroup": "subgroup_value",
  "year": 2024,
  "reason": "brief reason"
}}

Question:
{question}
"""

    raw = call_llm(prompt)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {
            "route": "combined_general",
            "group": "all",
            "subgroup": "all",
            "year": 2024,
            "reason": "Fallback because router did not return JSON."
        }

    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return {
            "route": "combined_general",
            "group": "all",
            "subgroup": "all",
            "year": 2024,
            "reason": "Fallback because router JSON could not be parsed."
        }

    valid_routes = {
        "mobile_money_leaders",
        "low_account_ownership",
        "region_summary",
        "report_only",
        "combined_general"
    }

    valid_groups = {
        ("all", "all"),
        ("gender", "men"),
        ("gender", "women"),
        ("income", "richest 60%"),
        ("income", "poorest 40%"),
        ("age_cat", "age 25+"),
        ("age_cat", "ages 15-24"),
        ("urbanicity", "rural"),
        ("urbanicity", "urban"),
        ("laborforce", "out of laborforce"),
        ("laborforce", "in laborforce"),
        ("education", "secondary edu or more"),
        ("education", "prim edu or less")
    }

    valid_years = {2011, 2014, 2017, 2021, 2024}

    route = parsed.get("route", "combined_general")
    group = parsed.get("group", "all")
    subgroup = parsed.get("subgroup", "all")
    year = parsed.get("year", 2024)
    reason = parsed.get("reason", "")

    if route not in valid_routes:
        route = "combined_general"

    if (group, subgroup) not in valid_groups:
        group, subgroup = "all", "all"

    if year is not None:
        try:
            year = int(year)
        except Exception:
            year = 2024

        if year not in valid_years:
            year = 2024

    return {
        "route": route,
        "group": group,
        "subgroup": subgroup,
        "year": year,
        "reason": reason
    }


# ----------------------------
# Structured SQL functions
# ----------------------------

def sql_literal(value: str) -> str:
    return value.replace("'", "''")


def get_mobile_money_leaders(limit=10, group="all", subgroup="all", year=2024) -> pd.DataFrame:
    year_filter = f"AND year = {year}" if year is not None else ""

    query = f"""
    SELECT
      country,
      region,
      income_group,
      year,
      demographic_group,
      demographic_subgroup,
      mobile_money_account_pct,
      account_ownership_pct,
      owns_mobile_phone_pct
    FROM {SILVER_TABLE}
    WHERE demographic_group = '{sql_literal(group)}'
      AND demographic_subgroup = '{sql_literal(subgroup)}'
      {year_filter}
      AND mobile_money_account_pct IS NOT NULL
    ORDER BY mobile_money_account_pct DESC
    LIMIT {int(limit)}
    """

    return run_sql(query)


def get_low_account_ownership_countries(limit=10, group="all", subgroup="all", year=2024) -> pd.DataFrame:
    year_filter = f"AND year = {year}" if year is not None else ""

    query = f"""
    SELECT
      country,
      region,
      income_group,
      year,
      demographic_group,
      demographic_subgroup,
      account_ownership_pct,
      mobile_money_account_pct,
      owns_mobile_phone_pct,
      used_internet_pct
    FROM {SILVER_TABLE}
    WHERE demographic_group = '{sql_literal(group)}'
      AND demographic_subgroup = '{sql_literal(subgroup)}'
      {year_filter}
      AND account_ownership_pct IS NOT NULL
    ORDER BY account_ownership_pct ASC
    LIMIT {int(limit)}
    """

    return run_sql(query)


def get_region_summary(group="all", subgroup="all", year=2024) -> pd.DataFrame:
    year_filter = f"AND year = {year}" if year is not None else ""

    query = f"""
    SELECT
      region,
      year,
      demographic_group,
      demographic_subgroup,
      COUNT(*) AS country_count,
      ROUND(AVG(account_ownership_pct), 2) AS avg_account_ownership_pct,
      ROUND(AVG(mobile_money_account_pct), 2) AS avg_mobile_money_account_pct,
      ROUND(AVG(owns_mobile_phone_pct), 2) AS avg_mobile_phone_ownership_pct,
      ROUND(AVG(used_internet_pct), 2) AS avg_internet_usage_pct
    FROM {SILVER_TABLE}
    WHERE demographic_group = '{sql_literal(group)}'
      AND demographic_subgroup = '{sql_literal(subgroup)}'
      {year_filter}
    GROUP BY region, year, demographic_group, demographic_subgroup
    ORDER BY year DESC, avg_account_ownership_pct DESC
    """

    return run_sql(query)


def get_trend_summary(group="all", subgroup="all") -> pd.DataFrame:
    query = f"""
    SELECT
      year,
      demographic_group,
      demographic_subgroup,
      ROUND(AVG(account_ownership_pct), 2) AS avg_account_ownership_pct,
      ROUND(AVG(mobile_money_account_pct), 2) AS avg_mobile_money_account_pct,
      ROUND(AVG(owns_mobile_phone_pct), 2) AS avg_mobile_phone_ownership_pct,
      ROUND(AVG(used_internet_pct), 2) AS avg_internet_usage_pct
    FROM {SILVER_TABLE}
    WHERE demographic_group = '{sql_literal(group)}'
      AND demographic_subgroup = '{sql_literal(subgroup)}'
    GROUP BY year, demographic_group, demographic_subgroup
    ORDER BY year
    """

    return run_sql(query)


# ----------------------------
# RAG functions
# ----------------------------

def search_report(question: str, k=5) -> dict:
    return index.similarity_search(
        query_text=question,
        columns=["chunk_id", "source", "page_number", "content"],
        num_results=k
    )


def build_context_from_results(results: dict) -> str:
    columns = [col["name"] for col in results["manifest"]["columns"]]
    rows = results["result"]["data_array"]

    context_parts = []

    for row in rows:
        item = dict(zip(columns, row))
        context_parts.append(
            f"Source: {item.get('source', 'Unknown source')}\n"
            f"Page: {item.get('page_number', 'Unknown page')}\n"
            f"Content: {item.get('content', '')}"
        )

    return "\n\n---\n\n".join(context_parts)


def df_to_context(df: pd.DataFrame, max_rows=20) -> str:
    if df is None or df.empty:
        return "No structured data found."

    return df.head(max_rows).to_string(index=False)


def answer_from_report(question: str) -> str:
    report_results = search_report(question, k=5)
    report_context = build_context_from_results(report_results)

    prompt = f"""
You are an AI assistant for the World Bank Global Findex 2025 report.

Answer the user using only the report context below.

Rules:
- Use only the provided context.
- Include page numbers when using report evidence.
- Do not make up facts.
- If context is insufficient, say so clearly.

User question:
{question}

Report context:
{report_context}

Answer:
"""

    return call_llm(prompt)


def answer_with_structured_and_report(question: str, structured_df=None, structured_label="Structured data") -> str:
    structured_context = df_to_context(structured_df)
    report_results = search_report(question, k=5)
    report_context = build_context_from_results(report_results)

    prompt = f"""
You are an AI assistant for a World Bank Global Findex 2025 demo on Databricks.

Use these sources:
1. Structured Global Findex indicator data.
2. Retrieved Global Findex 2025 report context.

User question:
{question}

{structured_label}:
{structured_context}

Report context:
{report_context}

Rules:
- Use structured data for numbers, rankings, trends, and country/region comparisons.
- Use report context for explanations, causes, barriers, and interpretation.
- Include page numbers when using report evidence.
- If the user asks about an older year, use structured data for that year. Use report context only for general background unless it explicitly mentions that year.
- Do not make up facts.
- Keep the answer clear and presentation-friendly.

Answer:
"""

    return call_llm(prompt)


# ----------------------------
# Main app function
# ----------------------------

def ask_findex_ai(question: str) -> tuple[str, dict, pd.DataFrame | None]:
    route_info = classify_question(question)

    route = route_info.get("route", "combined_general")
    group = route_info.get("group", "all")
    subgroup = route_info.get("subgroup", "all")
    year = route_info.get("year", 2024)

    structured_df = None

    if year is None:
        structured_df = get_trend_summary(group=group, subgroup=subgroup)
        answer = answer_with_structured_and_report(
            question=question,
            structured_df=structured_df,
            structured_label=f"Multi-year trend summary for {group} / {subgroup}"
        )
        return answer, route_info, structured_df

    if route == "mobile_money_leaders":
        structured_df = get_mobile_money_leaders(
            limit=10,
            group=group,
            subgroup=subgroup,
            year=year
        )
        answer = answer_with_structured_and_report(
            question=question,
            structured_df=structured_df,
            structured_label=f"Top countries by mobile money ownership for {group} / {subgroup} in {year}"
        )

    elif route == "low_account_ownership":
        structured_df = get_low_account_ownership_countries(
            limit=10,
            group=group,
            subgroup=subgroup,
            year=year
        )
        answer = answer_with_structured_and_report(
            question=question,
            structured_df=structured_df,
            structured_label=f"Countries with lowest account ownership for {group} / {subgroup} in {year}"
        )

    elif route == "region_summary":
        structured_df = get_region_summary(
            group=group,
            subgroup=subgroup,
            year=year
        )
        answer = answer_with_structured_and_report(
            question=question,
            structured_df=structured_df,
            structured_label=f"Regional summary for {group} / {subgroup} in {year}"
        )

    elif route == "report_only":
        answer = answer_from_report(question)

    else:
        mobile_df = get_mobile_money_leaders(limit=5, group=group, subgroup=subgroup, year=year)
        low_df = get_low_account_ownership_countries(limit=5, group=group, subgroup=subgroup, year=year)
        region_df = get_region_summary(group=group, subgroup=subgroup, year=year)

        combined_structured_context = f"""
Selected filters:
year = {year}
group = {group}
subgroup = {subgroup}

Mobile money leaders:
{df_to_context(mobile_df, max_rows=5)}

Lowest account ownership countries:
{df_to_context(low_df, max_rows=5)}

Regional summary:
{df_to_context(region_df, max_rows=10)}
"""
        structured_df = region_df

        report_results = search_report(question, k=5)
        report_context = build_context_from_results(report_results)

        prompt = f"""
You are an AI assistant for a World Bank Global Findex 2025 demo on Databricks.

User question:
{question}

Structured context:
{combined_structured_context}

Report context:
{report_context}

Rules:
- Use the structured context for numeric insights.
- Use the report context for explanation.
- Include page numbers when using report evidence.
- Do not make up facts.
- If available context is insufficient, say so clearly.

Answer:
"""
        answer = call_llm(prompt)

    return answer, route_info, structured_df


# ----------------------------
# Streamlit UI
# ----------------------------


st.title("🌍 World Bank Global Findex AI Assistant")
st.caption("Structured indicators + Global Findex report RAG on Databricks")

with st.sidebar:
    st.subheader("Example questions")
    examples = [
        "Which countries lead in mobile money account ownership in 2024?",
        "Which countries have the lowest account ownership for women?",
        "Which regions have the lowest account ownership for poorest 40%?",
        "How has account ownership changed over time?",
        "What does the report say about barriers to financial inclusion?",
        "Explain why mobile money matters for financial inclusion."
    ]

    for ex in examples:
        if st.button(ex):
            st.session_state["pending_question"] = ex

    st.divider()

    st.write("Official World Bank data sources:")
    
    st.markdown(
        """
    - [Global Findex 2025 Data Page](https://www.worldbank.org/en/publication/globalfindex/download-data)
    - [Download Excel File](https://thedocs.worldbank.org/en/doc/be6615202d1f08a25855c8ac2d615122-0050012025/related/GlobalFindexDatabase2025.xlsx)
    - [Download Report PDF](https://openknowledge.worldbank.org/bitstreams/9288bdc5-7a9b-42de-a47c-3746fd68f22a/download)
    """
    )


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("route_info"):
            with st.expander("Router decision"):
                st.json(msg["route_info"])

        if msg.get("structured_df") is not None:
            with st.expander("Structured data used"):
                st.dataframe(msg["structured_df"])


question = st.chat_input("Ask about financial inclusion, mobile money, regions, groups, or the report...")

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching data and report context..."):
            try:
                answer, route_info, structured_df = ask_findex_ai(question)
                st.markdown(answer)

                with st.expander("Router decision"):
                    st.json(route_info)

                if structured_df is not None:
                    with st.expander("Structured data used"):
                        st.dataframe(structured_df)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "route_info": route_info,
                    "structured_df": structured_df
                })

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
