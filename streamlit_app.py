"""
Streamlit UI for the General Government GraphRAG retriever.

Wraps retrieve_gg.py's pipeline (question -> embed -> vector search -> graph
expansion -> Cypher aggregation -> Claude synthesis) behind a text box, and
renders the retrieved Divisions + Entities as an interactive pyvis subgraph.

Secrets: ANTHROPIC_API_KEY / NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD are read
from st.secrets first (Streamlit Community Cloud), falling back to the local
.env used by the CLI scripts for local dev. Nothing sensitive lives in the repo.

Run locally:
    streamlit run streamlit_app.py
"""
import datetime

import os

import streamlit as st
from dotenv import load_dotenv

# --- secrets: st.secrets (deployed) > .env (local dev), set before importing
# retrieve_gg so its module-level NEO4J_* constants pick up the right values.
# GRAPHRAG_ENV_PATH lets each machine point at wherever its .env actually
# lives (e.g. outside a cloud-synced folder); falls back to load_dotenv()'s
# default cwd/parent search.
load_dotenv(os.environ.get("GRAPHRAG_ENV_PATH"))


def _secret(key):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key)


for _key in ("ANTHROPIC_API_KEY", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
    _val = _secret(_key)
    if _val:
        os.environ[_key] = _val

import anthropic
from neo4j import GraphDatabase
from pyvis.network import Network

import retrieve_gg as rag

st.set_page_config(page_title="Miami-Dade GG GraphRAG", page_icon="\U0001F3DB", layout="wide")

ENTITY_COLORS = {
    "Service": "#4C9AFF",
    "Population": "#57D9A3",
    "Location": "#FFAB00",
    "Agency": "#FF7452",
    "Asset": "#998DD9",
    "Regulation": "#FF5C5C",
}
DIVISION_COLOR = "#2B3A67"

# Extra Cypher used only for the visualization (retrieve_gg's ENRICH doesn't
# carry entity keys, which we need here to join RELATES_TO edges).
# Capped per division (via collect()[0..cap]) so a division with dozens of
# extracted entities doesn't blow up the rendered graph into a hairball.
MENTIONS_WITH_KEYS = """
UNWIND $keys AS k
MATCH (v:Division {key: k})-[:MENTIONS]->(e:Entity)
WITH v, e ORDER BY e.type, e.name
WITH v, collect(e)[0..$cap] AS ents
UNWIND ents AS e
RETURN v.key AS div_key, e.key AS entity_key, e.name AS entity_name, e.type AS entity_type
"""

RELATES_AMONG_ENTITIES = """
MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
WHERE e1.key IN $keys AND e2.key IN $keys
RETURN e1.key AS source, e2.key AS target, r.type AS rel_type
"""


@st.cache_resource(show_spinner=False)
def get_driver():
    driver = GraphDatabase.driver(
        rag.NEO4J_URI, auth=(rag.NEO4J_USER, rag.NEO4J_PASSWORD),
        notifications_min_severity="OFF",
    )
    driver.verify_connectivity()
    return driver


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(rag.EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def get_anthropic_client():
    return anthropic.Anthropic()

# --- Public-demo guardrails -------------------------------------------------
# This app calls the Anthropic API on every question, so an open URL is an
# open-ended bill. Caps below bound per-visitor and total daily usage. The hard
# backstop is a spend limit set in the Anthropic Console (independent of this code).
MAX_QUERIES_PER_SESSION = 8
MAX_QUERIES_PER_DAY = 200          # shared across visitors; best-effort
MAX_QUESTION_CHARS = 300

@st.cache_resource(show_spinner=False)
def _global_usage():
    # Shared across sessions while the container is warm. Resets when the
    # Streamlit Cloud container sleeps/restarts — i.e. only after idle time.
    return {"date": datetime.date.today(), "count": 0}


def _within_quota() -> bool:
    usage = _global_usage()
    today = datetime.date.today()
    if usage["date"] != today:
        usage["date"], usage["count"] = today, 0

    st.session_state.setdefault("query_count", 0)
    if st.session_state["query_count"] >= MAX_QUERIES_PER_SESSION:
        st.warning(f"Session limit reached ({MAX_QUERIES_PER_SESSION} questions). "
                   "Reload the page to start a new session — this is a cost-capped public demo.")
        return False
    if usage["count"] >= MAX_QUERIES_PER_DAY:
        st.warning("The demo has hit its daily question cap. Please check back tomorrow.")
        return False

    usage["count"] += 1
    st.session_state["query_count"] += 1
    return True

def build_subgraph_html(result, max_entities_per_division) -> str:
    div_keys = [d["key"] for d in result["divisions"]]
    with get_driver().session() as session:
        mentions = list(session.run(MENTIONS_WITH_KEYS, keys=div_keys, cap=max_entities_per_division))
        entity_keys = sorted({m["entity_key"] for m in mentions})
        relates = list(session.run(RELATES_AMONG_ENTITIES, keys=entity_keys)) if entity_keys else []

    net = Network(height="620px", width="100%", directed=True, bgcolor="#0e1117",
                  font_color="#ffffff", cdn_resources="in_line")
    net.barnes_hut(gravity=-4000, spring_length=120)

    for d in result["divisions"]:
        fte = d["fte"] if d["fte"] is not None else "n/a"
        net.add_node(d["key"], label=d["name"], shape="box", color=DIVISION_COLOR, size=25,
                     title=f"{d['department']} | FTE {fte}")

    seen_entities = set()
    for m in mentions:
        ek = m["entity_key"]
        if ek not in seen_entities:
            seen_entities.add(ek)
            net.add_node(ek, label=m["entity_name"],
                         color=ENTITY_COLORS.get(m["entity_type"], "#cccccc"),
                         size=12, title=m["entity_type"])
        net.add_edge(m["div_key"], ek, color="rgba(255,255,255,0.25)", width=1)

    for r in relates:
        net.add_edge(r["source"], r["target"], title=r["rel_type"], color="#FF7452", width=1.5)

    return net.generate_html()


st.title("Miami-Dade General Government — GraphRAG")
st.caption("Vector search → graph expansion → exact Cypher aggregation → Claude synthesis.")

with st.sidebar:
    st.header("Retrieval settings")
    k = st.slider("Divisions (vector search top-k)", 1, 10, 5)
    entity_k = st.slider("Entity matches (0 disables)", 0, 10, 5)
    model = rag.SYNTH_MODEL
    show_graph = st.checkbox("Show subgraph visualization", value=True)
    max_entities_per_division = st.slider("Max entities shown per division", 3, 20, 8)

question = st.text_input("Question", placeholder="Who handles hiring and how many staff work on it?")
ask = st.button("Ask", type="primary", disabled=not question.strip())

if ask:
    if not rag.NEO4J_PASSWORD:
        st.error("NEO4J_PASSWORD not set. Add it to Streamlit secrets or your local .env.")
        st.stop()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY not set. Add it to Streamlit secrets or your local .env.")
        st.stop()
    if len(question) > MAX_QUESTION_CHARS:
        st.error(f"Question too long (max {MAX_QUESTION_CHARS} characters).")
        st.stop()
    if not _within_quota():
        st.stop()

    try:
        driver = get_driver()
    except Exception as e:
        st.error(f"Cannot reach Neo4j at {rag.NEO4J_URI}: {e}")
        st.stop()

    with st.spinner("Retrieving..."):
        embedder = get_embedder()
        vector = embedder.encode(question).tolist()
        with driver.session() as session:
            session.run(rag.CREATE_ENTITY_FULLTEXT)
            try:
                session.run("CALL db.awaitIndexes()")
            except Exception:
                pass
            result = rag.retrieve(session, question, vector, k, entity_k)

    if result is None:
        st.warning("No divisions retrieved. Has the graph been embedded (embed_gg_divisions.py)?")
        st.stop()

    context = rag.build_context(question, result)

    with st.spinner("Synthesizing answer..."):
        client = get_anthropic_client()
        resp = client.messages.create(
            model=model, max_tokens=rag.MAX_TOKENS, thinking={"type": "adaptive"},
            system=rag.SYSTEM_PROMPT, messages=[{"role": "user", "content": context}],
        )
        answer = "".join(b.text for b in resp.content if b.type == "text")

    st.subheader("Answer")
    st.markdown(answer.strip())

    st.subheader("Sources")
    for i, d in enumerate(result["divisions"], 1):
        tag = f"relevance {d['score']:.3f}" if d["score"] is not None else "entity match"
        st.markdown(f"**[{i}] {d['name']}** — {d['department']} ({tag})")

    if show_graph:
        st.subheader("Retrieved subgraph")
        st.caption("Dark boxes = Divisions. Colored circles = extracted Entities (by type). "
                   "Faint edges = MENTIONS, orange edges = RELATES_TO (hover an orange edge for its type).")
        st.components.v1.html(build_subgraph_html(result, max_entities_per_division), height=640, scrolling=True)

    with st.expander("Retrieval context sent to Claude"):
        st.text(context)
