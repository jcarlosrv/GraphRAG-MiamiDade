"""
Additive sync: write Division.budget_fy24_25/fy25_26 and corrected
Division.fte_fy24_25/fy25_26 (bug-fixed name matching, see build_gg_workbook.py
FTE_NAME_ALIASES/FTE_MERGES) plus Department.unattributed_budget_fy24_25/fy25_26
from a freshly-rebuilt General_Government.xlsx onto the EXISTING graph.

Deliberately NOT a full load_gg_graph.py rebuild: that script DETACH DELETEs the
whole graph, which would destroy Phase 2 (Entity/MENTIONS/RELATES_TO, ~35 LLM
calls) and Phase 3 (embeddings) work for no reason — this only needs to touch
Division/Department properties. Only MATCH + SET, never DETACH DELETE.

One division's own name changed between old and new xlsx (a PDF kerning
artifact fix: "POLICY , TRAINING AND COMPLIANCE" -> "POLICY, TRAINING AND
COMPLIANCE"), so its Division.key changes too. Matching purely by key would
silently miss it and (if MERGEd) create a duplicate node. Instead, match
existing divisions by department + normalized name, then update key/name too
when they've changed.

Run:
    python build_gg_workbook.py   # regenerate the xlsx first
    python add_gg_budget.py
"""

import os
import re
import sys
import math
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(os.environ.get("GRAPHRAG_ENV_PATH"))

XLSX = "General_Government.xlsx"
SECTOR = "General Government"
SEP = " > "

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    print("ERROR: NEO4J_PASSWORD not set. Copy .env.example to .env and fill it in.", file=sys.stderr)
    sys.exit(2)


def norm(s: str) -> str:
    """Same canonical form as build_gg_workbook.py's norm(clean(x))."""
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(\w) +-(\w)", r"\1-\2", s)
    s = re.sub(r"\s+([,.])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip().lower().rstrip(".")


def _int(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return int(x)


def _str(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x)


def main() -> int:
    dep_df = pd.read_excel(XLSX, sheet_name="Departments")
    div_df = pd.read_excel(XLSX, sheet_name="Divisions")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    with driver.session() as s:
        existing = {}
        for r in s.run("""
            MATCH (d:Department)-[:HAS_DIVISION]->(v:Division)
            RETURN d.key AS dept_key, v.key AS key, v.name AS name
        """):
            existing.setdefault(r["dept_key"], {})[norm(r["name"])] = (r["key"], r["name"])

        renamed, updated, missing = 0, 0, []
        for _, row in div_df.iterrows():
            dept = _str(row["department"]).strip()
            name = _str(row["division"]).strip()
            dept_key = f"{SECTOR}{SEP}{dept}"
            new_key = f"{SECTOR}{SEP}{dept}{SEP}{name}"

            match = existing.get(dept_key, {}).get(norm(name))
            if not match:
                missing.append(new_key)
                continue
            old_key, old_name = match

            s.run("""
                MATCH (v:Division {key: $old_key})
                SET v.key = $new_key,
                    v.name = $name,
                    v.fte_fy24_25 = $fte24,
                    v.fte_fy25_26 = $fte25,
                    v.budget_fy24_25 = $b24,
                    v.budget_fy25_26 = $b25
            """,
                  old_key=old_key, new_key=new_key, name=name,
                  fte24=_int(row["fte_fy24_25"]), fte25=_int(row["fte_fy25_26"]),
                  b24=_int(row["budget_fy24_25"]), b25=_int(row["budget_fy25_26"]))
            updated += 1
            if old_key != new_key:
                renamed += 1
                print(f"  renamed: {old_name!r} -> {name!r}")

        dept_updated = 0
        for _, row in dep_df.iterrows():
            dept = _str(row["department"]).strip()
            dept_key = f"{SECTOR}{SEP}{dept}"
            s.run("""
                MATCH (d:Department {key: $key})
                SET d.unattributed_budget_fy24_25 = $u24,
                    d.unattributed_budget_fy25_26 = $u25
            """, key=dept_key, u24=_int(row["unattributed_budget_fy24_25"]),
                  u25=_int(row["unattributed_budget_fy25_26"]))
            dept_updated += 1

        counts = s.run("""
            CALL () { MATCH (v:Division) WHERE v.fte_fy25_26 IS NULL RETURN count(v) AS fte_null }
            CALL () { MATCH (v:Division) WHERE v.budget_fy25_26 IS NULL RETURN count(v) AS budget_null }
            CALL () { MATCH (v:Division) RETURN count(v) AS total }
            RETURN fte_null, budget_null, total
        """).single()

    driver.close()
    print(f"\nDivisions updated: {updated} (renamed: {renamed})")
    print(f"Departments updated: {dept_updated}")
    if missing:
        print(f"WARNING: {len(missing)} division(s) in the xlsx had no existing graph match "
              f"(not created, needs review): {missing}")
    print(f"\nRemaining NULLs: fte_fy25_26={counts['fte_null']}, "
          f"budget_fy25_26={counts['budget_null']}  (of {counts['total']} divisions)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
