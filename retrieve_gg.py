"""
Phase 3 (step 2) — GraphRAG retrieval + answer synthesis over the GG graph.

The pipeline, end to end:

    question
      │  embed (local all-MiniLM-L6-v2, same model as embed_gg_divisions.py)
      ├──────────────────────────────┐
      ▼                              ▼
    vector search                 entity full-text search (Lucene, lexical)
    top-k Divisions (semantic)    top entities by name/alias match
      │                              │  expand via MENTIONS (exact)
      └───────────┬──────────────────┘
                   ▼  merge + dedup entry-point Divisions (NOT for counting)
    graph expansion (structural traversal, exact)
      │
    enrich each hit: Sector/Department, services, entities (MENTIONS)
    quantitative join: FTE sums via Cypher with coalesce + coverage counts
      │  assemble grounded context
      ▼
    Claude  →  answer that cites divisions and uses the EXACT figures

Design decisions (see PROJECT_PLAN.md "RESUME HERE"):
- Vector + entity search both find entry points only. Exact numbers come from
  Cypher aggregation over the merged retrieved set (and its department
  rollup), never from the LLM doing arithmetic in its head and never from a
  fuzzy score threshold.
- Entity search is lexical (Lucene full-text on Entity.name/aliases), not a
  second embedding. Entities were extracted from the same text divisions are
  embedded from, so a second vector index over entities would mostly
  duplicate that signal; a named regulation/agency/asset ("Florida Statutes",
  "the Board") is instead matched exactly via the same MENTIONS edges Phase 2
  already built, which catches divisions whose own semantic embedding doesn't
  rank them highly but which cite the exact same entity.
- FTE has 6 NULLs (ToO gaps). Every sum uses coalesce(...,0) AND reports a
  coverage count, so the answer can be honest about what's missing.

Run:
    # Neo4j running + embeddings present (embed_gg_divisions.py) + ANTHROPIC_API_KEY
    python retrieve_gg.py "who handles hiring and how many staff work on it?"
    python retrieve_gg.py "cybersecurity" --k 4
    python retrieve_gg.py "who cites Florida Statutes?" --entity-k 5
    python retrieve_gg.py "procurement" --context-only   # show retrieval, skip the LLM
"""

import os
import re
import sys
import argparse

import anthropic
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(os.environ.get("GRAPHRAG_ENV_PATH"))

EMBED_MODEL = "all-MiniLM-L6-v2"      # MUST match embed_gg_divisions.py
INDEX_NAME = "division_text_embeddings"
ENTITY_INDEX_NAME = "entity_fulltext"
# Two-part filter, empirically calibrated against this ~500-entity corpus (see
# lessons.md): a REAL name match (e.g. "Implementing Order 2-5") scores well
# above the rest with a sharp gap (17.6 vs 8.9 for the runner-up). A query with
# no true entity in the corpus (e.g. "Florida Statutes", which isn't actually
# extracted anywhere) instead produces a flat, gently-decaying tail of generic
# word-overlap noise, all clustered near the same score (4.45 -> 3.94 -> 3.54).
# ENTITY_MIN_SCORE rejects that whole tail outright; ENTITY_SCORE_FLOOR then
# keeps only hits close to a genuine top score. Retune both if the corpus
# grows enough to shift Lucene's term-frequency statistics.
ENTITY_MIN_SCORE = 5.0
ENTITY_SCORE_FLOOR = 0.5
SYNTH_MODEL = "claude-sonnet-4-6"       # synthesis model (swap to claude-sonnet-4-6 to match Phase 2)
MAX_TOKENS = 2000
DESC_CHARS = 600                      # truncate each division description in the context

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

SYSTEM_PROMPT = """\
You answer questions about Miami-Dade County's General Government using ONLY the \
context provided below. The context was retrieved from a knowledge graph: a set of \
relevant Divisions (semantic search), each with its Department, services, and extracted \
entities, plus EXACT pre-computed figures.

Rules:
- Use ONLY the provided context. If it doesn't contain the answer, say so plainly. \
Never invent divisions, numbers, or facts.
- For any quantity (FTE/headcount, counts), use the EXACT figures given in the \
"COMPUTED FIGURES" block. Do NOT add up numbers yourself — the figures there are \
authoritative and already account for missing values.
- When figures note a coverage gap (e.g. "FTE known for 5 of 6 divisions"), say so — \
do not present an undercount as a complete total.
- Cite the specific Division(s) by name that support each claim.
- Be concise and direct. Lead with the answer."""


# ----------------------------------------------------------------------------
# Cypher
# ----------------------------------------------------------------------------

VECTOR_SEARCH = """
CALL db.index.vector.queryNodes($index, $k, $vector) YIELD node, score
RETURN node.key AS key, node.name AS name, score
"""

# Lexical entry point, separate from the Division vector index. Entities were
# extracted from the same text divisions are embedded from, so this isn't a
# second embedding — it's an exact-ish name/alias match over the entities
# Phase 2 already normalized, for questions that name a regulation/agency/
# asset the semantic embedding alone might not surface strongly.
CREATE_ENTITY_FULLTEXT = """
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name, e.aliases]
OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}
"""

ENTITY_SEARCH = """
CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
RETURN node.key AS key, node.name AS name, node.type AS type, score
ORDER BY score DESC
LIMIT $k
"""

ENTITY_EXPAND = """
UNWIND $entity_keys AS ek
MATCH (v:Division)-[:MENTIONS]->(e:Entity {key: ek})
RETURN DISTINCT v.key AS div_key, e.key AS entity_key
"""

ENRICH = """
UNWIND $keys AS k
MATCH (sector:Sector)-[:HAS_DEPARTMENT]->(dept:Department)-[:HAS_DIVISION]->(v:Division {key: k})
OPTIONAL MATCH (v)-[:MENTIONS]->(e:Entity)
WITH v, dept, sector, collect(DISTINCT {name: e.name, type: e.type}) AS entities
RETURN v.key AS key, v.name AS name, dept.name AS department, sector.name AS sector,
       v.description AS description, v.services AS services,
       v.fte_fy25_26 AS fte, v.fte_fy24_25 AS fte_prev,
       v.num_services AS num_services, entities
"""

# Quantitative join — exact, over the retrieved set. coalesce makes missing=0;
# count(v.fte_fy25_26) reports how many of the hits actually have an FTE value.
# Growth is computed ONLY over divisions with a real prior-year baseline
# (fte_fy24_25 > 0). prior=0 in a post-reorg budget means "new division this
# year", not "grew from zero" — folding those in produces nonsense ratios.
# new_divs counts those new-this-year divisions, reported separately.
AGG_TOTALS = """
UNWIND $keys AS k
MATCH (v:Division {key: k})
RETURN count(v)                          AS divisions,
       sum(coalesce(v.fte_fy25_26, 0))   AS total_fte,
       count(v.fte_fy25_26)              AS fte_known,
       sum(CASE WHEN v.fte_fy24_25 > 0 AND v.fte_fy25_26 IS NOT NULL
                THEN v.fte_fy25_26 END)  AS cur_cmp,
       sum(CASE WHEN v.fte_fy24_25 > 0 AND v.fte_fy25_26 IS NOT NULL
                THEN v.fte_fy24_25 END)  AS prev_cmp,
       count(CASE WHEN v.fte_fy24_25 > 0 AND v.fte_fy25_26 IS NOT NULL
                  THEN 1 END)            AS both_known,
       count(CASE WHEN coalesce(v.fte_fy24_25, 0) = 0 AND coalesce(v.fte_fy25_26, 0) > 0
                  THEN 1 END)            AS new_divs
"""

AGG_BY_DEPT = """
UNWIND $keys AS k
MATCH (dept:Department)-[:HAS_DIVISION]->(v:Division {key: k})
RETURN dept.name                         AS department,
       count(v)                          AS hit_divisions,
       sum(coalesce(v.fte_fy25_26, 0))   AS fte,
       count(v.fte_fy25_26)              AS fte_known
ORDER BY fte DESC
"""


# ----------------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------------

def _lines(val):
    if val is None:
        return []
    return [val] if isinstance(val, str) else [str(v) for v in val if str(v).strip()]


_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


def escape_lucene(text: str) -> str:
    """Escape Lucene query-syntax characters so a raw question can't break/inject a query."""
    return _LUCENE_SPECIAL.sub(r"\\\1", text)


def search_entities(session, question: str, entity_k: int) -> list:
    """Lexical entry point: top entities whose name/alias overlaps the question,
    kept within ENTITY_SCORE_FLOOR of the top score so unrelated stray-word
    matches don't drag in noise when nothing really matches."""
    if entity_k <= 0:
        return []
    q = escape_lucene(question).strip()
    if not q:
        return []
    hits = list(session.run(ENTITY_SEARCH, index=ENTITY_INDEX_NAME, q=q, k=entity_k))
    if not hits:
        return []
    top = hits[0]["score"]
    if top < ENTITY_MIN_SCORE:
        return []  # top hit itself looks like generic word overlap, not a real match
    return [h for h in hits if h["score"] >= ENTITY_SCORE_FLOOR * top]


def retrieve(session, question, vector, k, entity_k):
    vec_hits = list(session.run(VECTOR_SEARCH, index=INDEX_NAME, k=k, vector=vector))
    vector_score = {h["key"]: h["score"] for h in vec_hits}
    vector_keys = [h["key"] for h in vec_hits]

    entity_hits = search_entities(session, question, entity_k)
    entity_by_key = {e["key"]: e for e in entity_hits}
    div_entity_matches = {}
    if entity_hits:
        for row in session.run(ENTITY_EXPAND, entity_keys=list(entity_by_key)):
            div_entity_matches.setdefault(row["div_key"], []).append(entity_by_key[row["entity_key"]])

    # Preserve vector-search ranking first, then append entity-only additions.
    keys = list(vector_keys)
    for dk in div_entity_matches:
        if dk not in keys:
            keys.append(dk)
    if not keys:
        return None

    enriched = {r["key"]: dict(r) for r in session.run(ENRICH, keys=keys)}
    divisions = []
    for key in keys:
        d = enriched[key]
        d["score"] = vector_score.get(key)          # None => entity-only hit
        d["entity_matches"] = div_entity_matches.get(key, [])
        divisions.append(d)

    totals = dict(session.run(AGG_TOTALS, keys=keys).single())
    by_dept = [dict(r) for r in session.run(AGG_BY_DEPT, keys=keys)]
    return {"divisions": divisions, "totals": totals, "by_dept": by_dept}


def build_context(question, r) -> str:
    out = [f"QUESTION: {question}", "", "RETRIEVED DIVISIONS (most relevant first):", ""]
    for i, d in enumerate(r["divisions"], 1):
        if d["score"] is not None:
            out.append(f"[{i}] {d['name']}  (semantic relevance {d['score']:.3f})")
        else:
            out.append(f"[{i}] {d['name']}  (entity match only — not a top semantic hit)")
        ems = d.get("entity_matches") or []
        if ems:
            ent_str = ", ".join(f"{e['name']} [{e['type']}]" for e in ems)
            out.append(f"    Matched via entity search: {ent_str}")
        out.append(f"    Sector: {d['sector']}  |  Department: {d['department']}")
        fte = d["fte"] if d["fte"] is not None else "not on record"
        out.append(f"    FTE (FY25-26): {fte}   Services listed: {d['num_services']}")
        desc = (d.get("description") or "").strip()
        if desc:
            out.append(f"    Description: {desc[:DESC_CHARS]}{'...' if len(desc) > DESC_CHARS else ''}")
        services = _lines(d.get("services"))
        if services:
            out.append("    Services: " + "; ".join(services[:8]))
        ents = [e for e in d.get("entities", []) if e.get("name")]
        if ents:
            by_t = {}
            for e in ents:
                by_t.setdefault(e.get("type") or "Other", []).append(e["name"])
            ent_str = " | ".join(f"{t}: {', '.join(names[:5])}"
                                 for t, names in sorted(by_t.items(), key=lambda kv: str(kv[0])))
            out.append(f"    Entities — {ent_str}")
        out.append("")

    t = r["totals"]
    out.append("COMPUTED FIGURES (exact, from the graph — use these verbatim):")
    out.append(f"  Retrieved divisions: {t['divisions']}")
    out.append(f"  Total FTE (FY25-26): {t['total_fte']}  "
               f"(FTE known for {t['fte_known']} of {t['divisions']} divisions)")
    if t["both_known"] and t["prev_cmp"]:
        growth = 100.0 * (t["cur_cmp"] - t["prev_cmp"]) / t["prev_cmp"]
        out.append(f"  FTE change, like-for-like over {t['both_known']} division(s) with a "
                   f"prior-year baseline: {t['prev_cmp']} -> {t['cur_cmp']} ({growth:+.1f}%)")
    else:
        out.append("  FTE change FY24-25 -> FY25-26: not computable "
                   "(no retrieved division has a prior-year baseline)")
    if t["new_divs"]:
        out.append(f"  New in FY25-26 (no prior-year FTE, excluded from change above): "
                   f"{t['new_divs']} division(s)")
    out.append("  By department (within retrieved set):")
    for d in r["by_dept"]:
        out.append(f"    - {d['department']}: {d['hit_divisions']} division(s), "
                   f"FTE {d['fte']} (known for {d['fte_known']})")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", help="Natural-language question about General Government.")
    ap.add_argument("--k", type=int, default=5, help="Entry-point divisions to retrieve via vector search (default 5).")
    ap.add_argument("--entity-k", type=int, default=5,
                    help="Top entities to match lexically and expand via MENTIONS (default 5, 0 disables).")
    ap.add_argument("--context-only", action="store_true",
                    help="Print the retrieved/assembled context and skip the LLM call.")
    ap.add_argument("--model", default=SYNTH_MODEL, help=f"Synthesis model (default {SYNTH_MODEL}).")
    args = ap.parse_args()

    if not NEO4J_PASSWORD:
        print("ERROR: NEO4J_PASSWORD not set. Copy .env.example to .env and fill it in.", file=sys.stderr)
        return 2

    # Windows consoles default to cp1252; model output and arrows need UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # notifications_min_severity="OFF": the aggregation queries legitimately
    # produce "null value eliminated in set function" notices (the coalesce/
    # coverage logic working as intended) — suppress that console noise.
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
                                  notifications_min_severity="OFF")
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"ERROR: cannot reach Neo4j at {NEO4J_URI} ({e}).\n"
              "  Start it per the PROJECT_PLAN.md runbook, then re-run.", file=sys.stderr)
        return 3

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    vector = model.encode(args.question).tolist()

    with driver.session() as session:
        session.run(CREATE_ENTITY_FULLTEXT)
        try:  # only matters the first time the index is created; cheap no-op after
            session.run("CALL db.awaitIndexes()")
        except Exception:
            pass
        r = retrieve(session, args.question, vector, args.k, args.entity_k)
    driver.close()

    if r is None:
        print("No divisions retrieved. Have you run embed_gg_divisions.py?", file=sys.stderr)
        return 1

    context = build_context(args.question, r)

    if args.context_only:
        print(context)
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set (needed for synthesis).\n"
              "  Use --context-only to see retrieval without the LLM.", file=sys.stderr)
        return 2

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=args.model,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )
    answer = "".join(b.text for b in resp.content if b.type == "text")

    print("=" * 70)
    print(f"Q: {args.question}\n")
    print(answer.strip())
    print("\n" + "-" * 70)
    print("Sources (retrieved divisions):")
    for i, d in enumerate(r["divisions"], 1):
        tag = f"relevance {d['score']:.3f}" if d["score"] is not None else "entity match"
        print(f"  [{i}] {d['name']}  ({d['department']})  {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
