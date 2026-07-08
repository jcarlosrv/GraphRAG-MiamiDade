"""
Build a NEW workbook from the General_Government narrative PDFs.

Replaces the lossy distilled Program-Descriptions xlsx for this sector with data
extracted directly from the source budget PDFs (Miami-Dade FY2025-26 template).

Output: General_Government.xlsx with two sheets
  - Departments : 1 row / department  (name, description, total FTE, area, source)
  - Divisions   : 1 row / division    (description, services, comments, FTE counts)

Pure regex/rule-based parsing (no LLM). The PDFs share a consistent template:
  Page 1            -> department title + mission/description
  TABLE OF ORG.     -> each division + FY24-25/FY25-26 FTE counts + dept total FTE
  DIVISION: <name>  -> division description + bulleted services/operations
  DIVISION COMMENTS -> per-division budget/staffing change notes (not universal)
"""

import re
import glob
import os
import sys
import pandas as pd
from pypdf import PdfReader

PDF_DIR = "Department Narratives/General_Government"
OUT_XLSX = "General_Government.xlsx"
STRATEGIC_AREA = "General Government"
FISCAL_YEAR = "FY 2025-26"

BULLET = "•"

# Hand-verified corrections for name mismatches between TABLE OF ORGANIZATION
# (source of FTE) and the DIVISION: narrative blocks (source of the division's
# canonical name) / Expenditure By Program (source of budget). Each of these is
# a real inconsistency in the source PDF itself, not a parsing bug — verified
# against the FY25-26 Adopted Budget PDFs on 2026-07-03:
#   - management-and-budget.pdf: "BOND ADMINSTRATION" is a typo for "BOND
#     ADMINISTRATION" in TABLE OF ORGANIZATION; "PROGRAM MANAGEMENT DIVISION"
#     there is the same unit the narrative calls "PROGRAM MANAGEMENT
#     ADMINISTRATION".
#   - strategic-procurement.pdf: TABLE OF ORGANIZATION lists "OFFICE OF THE
#     DIRECTOR" (3/3 FTE) and "ADMINISTRATION" (18/21 FTE) as two org-table
#     rows, but the narrative combines them into one DIVISION: "OFFICE OF THE
#     DIRECTOR AND ADMINISTRATION". The Expenditure By Program table's own
#     position count for that division (21 -> 24) independently confirms the
#     sum (3+18=21, 3+21=24) is correct, not a guess.
# Guarded by "only fill in if the target key is otherwise missing", so this can
# never override a real, independently-matching entry in another department.
FTE_NAME_ALIASES = {
    "bond adminstration": "bond administration",
    "program management division": "program management administration",
}
FTE_MERGES = {
    "office of the director and administration": ["office of the director", "administration"],
}


def norm(s: str) -> str:
    """Canonical form for matching division names across sections."""
    return re.sub(r"\s+", " ", s).strip().lower().rstrip(".")


def clean(s: str) -> str:
    """Collapse whitespace/newlines, fix PDF kerning artifacts ('multi -year', 'POLICY , TRAINING')."""
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    # PDF splits hyphenated words as 'multi -year' (space before hyphen, none after).
    # Repair to 'multi-year'. A real separator ' - ' (spaces both sides) is left intact.
    s = re.sub(r"(\w) +-(\w)", r"\1-\2", s)
    # Same kerning artifact before a comma/period, e.g. 'POLICY , TRAINING' -> 'POLICY, TRAINING'.
    s = re.sub(r"\s+([,.])", r"\1", s)
    return s


def read_full(reader: PdfReader) -> str:
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    # Normalize en/em dash to a plain hyphen BEFORE any regex matching — some
    # PDFs use "HUMAN RESOURCES – SHARED SERVICES" (en dash) in one table and
    # a plain hyphen in another, which otherwise breaks name-matching between
    # tables (and can desync the regex scan for entries right after it).
    return text.replace("–", "-").replace("—", "-")


def parse_department(reader: PdfReader, full: str, source: str) -> dict:
    # Title: first non-empty line on page 1 after the FY header.
    p1 = [l.strip() for l in (reader.pages[0].extract_text() or "").splitlines() if l.strip()]
    title = ""
    for i, line in enumerate(p1):
        if line.startswith("FY ") and "Adopted Budget" in line:
            title = p1[i + 1] if i + 1 < len(p1) else ""
            break
    if not title and p1:
        title = p1[1] if len(p1) > 1 else p1[0]

    # Description: page-1 text after the title, up to TABLE OF ORGANIZATION.
    page1 = reader.pages[0].extract_text() or ""
    desc = page1
    if title and title in desc:
        desc = desc.split(title, 1)[1]
    desc = re.split(r"TABLE OF ORGANIZATION", desc)[0]
    desc = clean(desc)

    m = re.search(r"total number of full-time equivalent positions is\s+([\d,]+)", full, re.I)
    total_fte = int(m.group(1).replace(",", "")) if m else None

    return {
        "department": title,
        "description": desc,
        "total_fte": total_fte,
        "strategic_area": STRATEGIC_AREA,
        "fiscal_year": FISCAL_YEAR,
        "source_pdf": source,
    }


def parse_org_fte(full: str) -> dict:
    """Map normalized division name -> (fy_prev, fy_curr) from the Table of Organization."""
    too = re.search(r"TABLE OF ORGANIZATION(.*?)(?:DIVISION:|Strategic Plan Objectives)", full, re.S)
    if not too:
        return {}
    block = too.group(1)
    fte = {}
    # Each entry: <NAME (caps, maybe leading/trailing spaces)>\n<blurb...>
    #             FY 24-25 FY 25-26\n<prev> <curr>
    for m in re.finditer(
        r"\n[ \t]*([A-Z0-9][A-Z0-9 ,&/'\-]{2,}?)[ \t]*\n"
        r"(.*?)FY ?24-?25\s+FY ?25-?26[ \t]*\n?[ \t]*([\d,]+)\s+([\d,]+)",
        block, re.S,
    ):
        name = m.group(1).strip()
        # drop the stray total-FTE sentence / a stray 'FY ..' line captured as a name
        name = re.sub(r"The FY.*", "", name).strip()
        if name and not name.upper().startswith("FY "):
            fte[norm(name)] = (
                int(m.group(3).replace(",", "")),
                int(m.group(4).replace(",", "")),
            )

    for wrong, right in FTE_NAME_ALIASES.items():
        if wrong in fte and right not in fte:
            fte[right] = fte[wrong]
    for target, parts in FTE_MERGES.items():
        if target not in fte and all(p in fte for p in parts):
            fte[target] = (sum(fte[p][0] for p in parts), sum(fte[p][1] for p in parts))

    return fte


BUDGET_ENTRY_RE = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9 ,&/'\.\-\n]*?)[ \t]*\n?[ \t]*"
    r"([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)(?=\s*\n|\s*$)"
)


def parse_org_budget(full: str) -> dict:
    """Map normalized division name -> (budget_fy24_25, budget_fy25_26) in real
    dollars, parsed from the 'Expenditure By Program' table (dollars in
    thousands in the source, scaled up here). Sums multiple line items when the
    same division is split across more than one Strategic Area line (seen in
    management-and-budget.pdf's 'Grants Coordination', which appears once
    under Health and Society and once under General Government)."""
    m = re.search(
        r"Expenditure By Program\s+FY ?24-?25\s+FY ?25-?26\s+FY ?24-?25\s+FY ?25-?26\s*\n"
        r"(.*?)Total Operating Expenditures",
        full, re.S,
    )
    if not m:
        return {}
    block = re.sub(r"[ \t]*Strategic Area:[^\n]*\n", "", m.group(1))

    budget = {}
    for em in BUDGET_ENTRY_RE.finditer(block):
        name = clean(em.group(1))
        if not name:
            continue
        b24 = int(em.group(2).replace(",", "")) * 1000
        b25 = int(em.group(3).replace(",", "")) * 1000
        key = norm(name)
        prev24, prev25 = budget.get(key, (0, 0))
        budget[key] = (prev24 + b24, prev25 + b25)
    return budget


def parse_divisions(full: str, department: str, source: str, fte_map: dict, budget_map: dict) -> list:
    rows = []
    blocks = re.findall(r"DIVISION:\s*(.+?)\n(.*?)(?=\nDIVISION:|\Z)", full, re.S)
    for raw_name, body in blocks:
        name = clean(raw_name)

        # Split off comments (everything after DIVISION COMMENTS, may be absent).
        # The last division of each PDF runs to end-of-doc, so cut the comments
        # region at the first financial/end-matter marker to keep tables out.
        comments = ""
        if "DIVISION COMMENTS" in body:
            body, comment_part = body.split("DIVISION COMMENTS", 1)
            comments = re.split(
                r"FINANCIAL SUMMARY|EXPENDITURE BY|REVENUE BY|Total Funding"
                r"|FUNDED POSITIONS|Strategic Plan Objectives"
                r"|SELECTED ITEM HIGHLIGHTS|Line-Item Highlights"
                r"|OPERATING EXPENDITURES|ADDITIONAL OPERATING|Position Summary",
                comment_part, maxsplit=1,
            )[0]

        # Cut the injected performance-measures table out of the services region.
        services_region = re.split(r"Strategic Plan Objectives", body)[0]

        # Description = text before the first bullet; services = the bullets.
        if BULLET in services_region:
            desc_part, bullets_part = services_region.split(BULLET, 1)
            bullets_part = BULLET + bullets_part
        else:
            desc_part, bullets_part = services_region, ""

        division_description = clean(desc_part)
        services = [clean(b) for b in bullets_part.split(BULLET) if clean(b)]
        comment_items = [clean(b) for b in comments.split(BULLET) if clean(b)]

        prev, curr = fte_map.get(norm(name), (None, None))
        b24, b25 = budget_map.get(norm(name), (None, None))
        rows.append({
            "department": department,
            "division": name,
            "division_description": division_description,
            "services": " | ".join(services),
            "num_services": len(services),
            "comments": " | ".join(comment_items),
            "fte_fy24_25": prev,
            "fte_fy25_26": curr,
            "budget_fy24_25": b24,
            "budget_fy25_26": b25,
            "source_pdf": source,
        })
    return rows


def main() -> int:
    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}", file=sys.stderr)
        return 1

    dept_rows, div_rows = [], []
    for path in pdfs:
        source = os.path.basename(path)
        reader = PdfReader(path)
        full = read_full(reader)

        dept = parse_department(reader, full, source)
        fte_map = parse_org_fte(full)
        budget_map = parse_org_budget(full)
        divs = parse_divisions(full, dept["department"], source, fte_map, budget_map)
        dept["num_divisions"] = len(divs)

        # Budget-table line items that didn't match any parsed division (e.g.
        # commission-on-ethics-and-public-trust.pdf's "Commission on Ethics
        # and Public Trust" line, which is department-level overhead, not a
        # division) are kept — not silently dropped — as an explicit
        # unattributed total on the Department, same spirit as reporting FTE
        # coverage gaps rather than hiding them.
        matched_keys = {norm(d["division"]) for d in divs}
        leftover = {k: v for k, v in budget_map.items() if k not in matched_keys}
        dept["unattributed_budget_fy24_25"] = sum(v[0] for v in leftover.values()) if leftover else None
        dept["unattributed_budget_fy25_26"] = sum(v[1] for v in leftover.values()) if leftover else None

        dept_rows.append(dept)
        div_rows.extend(divs)
        matched = sum(1 for d in divs if d["fte_fy25_26"] is not None)
        budget_matched = sum(1 for d in divs if d["budget_fy25_26"] is not None)
        print(f"  {source:48} dept='{dept['department']}'  "
              f"divisions={len(divs)}  fte_matched={matched}/{len(divs)}  "
              f"budget_matched={budget_matched}/{len(divs)}  total_fte={dept['total_fte']}")
        if leftover:
            for k, (b24, b25) in leftover.items():
                print(f"    ! unattributed budget line: {k!r}  ${b24:,} -> ${b25:,}")

    dept_df = pd.DataFrame(dept_rows, columns=[
        "department", "description", "total_fte", "num_divisions",
        "strategic_area", "fiscal_year",
        "unattributed_budget_fy24_25", "unattributed_budget_fy25_26", "source_pdf"])
    div_df = pd.DataFrame(div_rows, columns=[
        "department", "division", "division_description", "services",
        "num_services", "comments", "fte_fy24_25", "fte_fy25_26",
        "budget_fy24_25", "budget_fy25_26", "source_pdf"])

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        dept_df.to_excel(xw, sheet_name="Departments", index=False)
        div_df.to_excel(xw, sheet_name="Divisions", index=False)

    print(f"\nWrote {OUT_XLSX}: {len(dept_df)} departments, {len(div_df)} divisions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
