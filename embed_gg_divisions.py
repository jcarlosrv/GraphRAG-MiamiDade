"""
Phase 3 (step 1) — semantic embeddings + vector index over the GG graph.

For each Division loaded by load_gg_graph.py, build an embed-text from its
description + services + comments, encode it to a dense vector with a LOCAL
sentence-transformers model (no API key, runs on CPU), and store the vector as
a node property:

    (Division).text_embedding : LIST<FLOAT>   (384 dims, all-MiniLM-L6-v2)

A native Neo4j vector index over that property then answers semantic
retrieval — "which divisions deal with X" — by cosine similarity, the vector
leg of the GraphRAG pipeline. The graph leg (MENTIONS / RELATES_TO traversal,
FTE aggregation) runs on top of the divisions this returns.

Why local embeddings: Anthropic/Claude has no embeddings endpoint (generation
only), the corpus is tiny (35 divisions), and a pilot shouldn't take on a
second API dependency. Swap MODEL_NAME for a hosted model later if needed — the
index dimension is derived from the model, so it always matches.

Resumable + idempotent: each Division is flagged `embedded = true` on success
and skipped on re-run; the index is created IF NOT EXISTS.

Run:
    # ensure Neo4j is running (see PROJECT_PLAN.md runbook), then:
    python embed_gg_divisions.py                 # embed all not-yet-done divisions
    python embed_gg_divisions.py --force         # re-embed everything
    python embed_gg_divisions.py --query "cybersecurity and data protection"
                                                 # semantic retrieval smoke test (top 5)
"""

import os
import sys
import argparse

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(r"C:\Users\jcarl\Quant16\Others\GraphRAG\.env")

MODEL_NAME = "all-MiniLM-L6-v2"   # local sentence-transformers model (384 dims)
VECTOR_PROP = "text_embedding"    # leaves room for a separate financial_vector later
INDEX_NAME = "division_text_embeddings"
SIMILARITY = "cosine"

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    print("ERROR: NEO4J_PASSWORD not set. Copy .env.example to .env and fill it in.", file=sys.stderr)
    sys.exit(2)


def _as_lines(val) -> list:
    """Division.services / .comments may be a list (Phase 1) or a string. Normalize to lines."""
    if val is None:
        return []
    if isinstance(val, str):
        return [val] if val.strip() else []
    return [str(v) for v in val if str(v).strip()]


def build_embed_text(div: dict) -> str:
    """Same source fields Phase 2 extracted from — description + services + comments."""
    parts = [f"DEPARTMENT: {div['dept']}", f"DIVISION: {div['name']}", ""]
    if div.get("description"):
        parts += ["DESCRIPTION:", div["description"], ""]
    services = _as_lines(div.get("services"))
    if services:
        parts += ["SERVICES:"] + [f"- {s}" for s in services] + [""]
    comments = _as_lines(div.get("comments"))
    if comments:
        parts += ["COMMENTS:"] + [f"- {c}" for c in comments] + [""]
    return "\n".join(parts).strip()


# ----------------------------------------------------------------------------
# Neo4j
# ----------------------------------------------------------------------------

# Dimension is interpolated from the model so the index can never mismatch the
# vectors it stores (a dimension typo is the classic vector-index bug).
CREATE_INDEX_TMPL = """
CREATE VECTOR INDEX {name} IF NOT EXISTS
FOR (d:Division) ON (d.{prop})
OPTIONS {{indexConfig: {{
  `vector.dimensions`: {dims},
  `vector.similarity_function`: '{sim}'
}}}}
"""

SET_VECTOR = """
MATCH (d:Division {key: $key})
CALL db.create.setNodeVectorProperty(d, $prop, $vector)
SET d.embedded = true
"""

QUERY_TOPK = """
CALL db.index.vector.queryNodes($index, $k, $vector) YIELD node, score
MATCH (dept:Department)-[:HAS_DIVISION]->(node)
RETURN node.name AS division, dept.name AS department, score
ORDER BY score DESC
"""


def fetch_divisions(session, force: bool) -> list:
    if force:
        session.run(f"MATCH (v:Division) REMOVE v.embedded, v.{VECTOR_PROP}")
    rows = session.run(
        """
        MATCH (d:Department)-[:HAS_DIVISION]->(v:Division)
        WHERE coalesce(v.embedded, false) = false
        RETURN v.key AS key, v.name AS name, d.name AS dept,
               v.description AS description, v.services AS services,
               v.comments AS comments
        ORDER BY v.key
        """
    )
    return [dict(r) for r in rows]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def connect():
    # notifications_min_severity="OFF" suppresses benign server notices (e.g.
    # "null value eliminated in set function") so runs print cleanly.
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
                                  notifications_min_severity="OFF")
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"ERROR: cannot reach Neo4j at {NEO4J_URI} ({e}).\n"
              "  Start it per the PROJECT_PLAN.md runbook, then re-run.", file=sys.stderr)
        sys.exit(3)
    return driver


def load_model():
    # Imported lazily so --help / arg errors don't pay the torch import cost.
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model '{MODEL_NAME}' (first run downloads it)...")
    model = SentenceTransformer(MODEL_NAME)
    # Method was renamed in newer sentence-transformers; support both.
    get_dims = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dims = get_dims()
    print(f"  ready — {dims} dimensions.\n")
    return model, dims


def run_query(driver, model, text: str, k: int) -> int:
    vector = model.encode(text).tolist()
    with driver.session() as session:
        rows = list(session.run(QUERY_TOPK, index=INDEX_NAME, k=k, vector=vector))
    if not rows:
        print("No results. Have you embedded the divisions yet? (run without --query first)")
        return 1
    print(f'Top {len(rows)} divisions for: "{text}"\n')
    for r in rows:
        print(f"  {r['score']:.3f}  {r['division'][:40]:42} [{r['department'][:30]}]")
    return 0


def embed_all(driver, model, dims: int) -> int:
    with driver.session() as session:
        session.run(CREATE_INDEX_TMPL.format(
            name=INDEX_NAME, prop=VECTOR_PROP, dims=dims, sim=SIMILARITY))

        divisions = fetch_divisions(session, force=False)  # force handled by caller
        total = len(divisions)
        if total == 0:
            print("Nothing to embed — all divisions already flagged embedded. "
                  "(Use --force to redo.)")
            return 0

        print(f"Embedding {total} division(s)...\n")
        texts = [build_embed_text(d) for d in divisions]
        vectors = model.encode(texts, show_progress_bar=False)  # batch encode

        for i, (div, vec) in enumerate(zip(divisions, vectors)):
            session.run(SET_VECTOR, key=div["key"], prop=VECTOR_PROP, vector=vec.tolist())
            print(f"  {div['name'][:45]:47} ({len(texts[i])} chars)")

        done = session.run(
            "MATCH (v:Division) WHERE v.embedded RETURN count(v) AS n").single()["n"]

    print(f"\n=== Phase 3 (embeddings) ===")
    print(f"  Embedded this run:        {total}")
    print(f"  Total divisions embedded: {done}")
    print(f"  Vector index:             {INDEX_NAME} ({dims}-d, {SIMILARITY})")
    print(f'\nTry it:  python embed_gg_divisions.py --query "your topic here"')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Clear embedded flags + vectors and re-embed from scratch.")
    ap.add_argument("--query", metavar="TEXT",
                    help="Run a semantic retrieval smoke test instead of embedding.")
    ap.add_argument("--k", type=int, default=5, help="Top-k for --query (default 5).")
    args = ap.parse_args()

    driver = connect()
    model, dims = load_model()

    try:
        if args.query:
            return run_query(driver, model, args.query, args.k)
        if args.force:
            with driver.session() as s:
                s.run(f"MATCH (v:Division) REMOVE v.embedded, v.{VECTOR_PROP}")
            print("Cleared existing embeddings (--force).\n")
        return embed_all(driver, model, dims)
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
