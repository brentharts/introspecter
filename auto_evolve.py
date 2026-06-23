#!/usr/bin/env python3
"""
auto_evolve.py — automated AI research loop for introspecter.

Pipeline (default):
  1. Analyze current stage (geometry, laws, lineage)
  2. Outward research: arXiv search → allowlisted PDF fetch → corpus ingest
  3. Compose or load oracle increment for stage N+1
  4. evolve_auto(): apply oracle, run all actions, build PDF

Modes:
  direct (default)   introspecter8.evolve_auto()
  --subprocess       pipe oracle + y-responses into evolve()
  --no-research      skip outward search/fetch (offline)
  --analyze-only     print analysis + research, no PDF
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import introspecter8 as I
import research as R


def _latest_pdf():
    pdfs = glob.glob(os.path.join(ROOT, "self_*.pdf"))
    return max(pdfs, key=os.path.getmtime) if pdfs else None


def _bootstrap_dsl():
    dsl_path = os.path.join(ROOT, "self.dsl")
    if os.path.isfile(dsl_path):
        return open(dsl_path).read()
    pdf = _latest_pdf()
    if not pdf:
        sys.exit("No self.dsl and no self_*.pdf found.")
    from pypdf import PdfReader
    att = PdfReader(pdf).attachments.get("self.dsl")
    data = att[0] if isinstance(att, list) else att
    open(dsl_path, "wb").write(data)
    return data.decode()


def analyze_stage(src: str) -> dict:
    p = I.parse(src)
    here = sorted(glob.glob(os.path.join(ROOT, "self_*.pdf")))
    p["lineage"] = I.read_lineage(here) if here else None
    mem = I._memberships(p)
    terms, M = I._cooccurrence(p)
    d = M.sum(axis=1)
    hub = terms[int(d.argmax())] if len(terms) else None
    word_terms = [w for _, (w, _) in p["nodes"].items()]
    return {
        "stage": int(p["meta"].get("nu", 0)),
        "n_nodes": len(p["nodes"]),
        "n_relations": len(p["rels"]),
        "n_hyperedges": len(p["hyper"]),
        "hub": hub,
        "memberships": mem,
        "law_self": round(I._feval(p["laws"][2][1], mem), 3) if len(p.get("laws", [])) > 2 else None,
        "lineage_stages": p.get("lineage", {}).get("stages", []),
        "terms": word_terms,
    }


def run_research(analysis: dict) -> dict:
    """Outward cycle: search arXiv, fetch allowlisted PDFs, ingest corpus."""
    print("\n=== outward research (data, not answers) ===")
    terms = analysis.get("terms", [])
    search = R.search_literature(terms, per_query=3)
    print(f"  arXiv: {len(search['hits'])} hits from {len(search['queries'])} queries")
    fetch = R.fetch_corpus(search["hits"], include_vixra=True)
    print(f"  fetch: {len(fetch['downloaded'])} PDFs → {fetch['corpus_dir']}/")
    if fetch["errors"]:
        print(f"  fetch errors: {len(fetch['errors'])}")
    ingest = R.ingest_corpus(terms)
    print(f"  ingest: {ingest.get('n_files', 0)} files, {len(ingest.get('snippets', []))} snippets")
    propose = R.propose_models(analysis, ingest)
    print(f"  propose: {len(propose['proposals'])} architecture proposal(s)")
    out = {"search": search, "fetch": fetch, "ingest": ingest, "propose": propose}
    open(os.path.join(ROOT, R.RESEARCH_CACHE), "w").write(json.dumps(out, indent=2))
    return out


def generate_oracle(analysis: dict, research: dict | None = None) -> str:
    """Load oracle_{N+1}.dsl or compose from template + research hints."""
    n = analysis["stage"]
    path = os.path.join(ROOT, f"oracle_{n + 1}.dsl")
    if os.path.isfile(path):
        return open(path).read()
    # fallback: stage 12 template name
    for fallback in (f"oracle_{n + 1}.dsl", "oracle.dsl"):
        fp = os.path.join(ROOT, fallback)
        if os.path.isfile(fp):
            return open(fp).read()
    raise FileNotFoundError(f"No oracle file for stage {n + 1}")


def evolve_direct(oracle: str, approve_all: bool = True) -> str:
    os.chdir(ROOT)
    return I.evolve_auto(oracle=oracle, approve_all=approve_all)


def evolve_subprocess(oracle: str, approve_all: bool = True) -> int:
    os.chdir(ROOT)
    n_actions = 20
    payload = oracle.strip() + "\n---\n" + "\n".join(["y"] * n_actions) + "\n"
    return subprocess.run(
        [sys.executable, "-u", "-c", "import introspecter8 as I; I.evolve(interactive=False)"],
        input=payload, text=True, cwd=ROOT,
    ).returncode


def main():
    ap = argparse.ArgumentParser(description="Automated introspecter research + evolution")
    ap.add_argument("--subprocess", action="store_true", help="Drive evolve() via stdin pipe")
    ap.add_argument("--oracle", metavar="FILE", help="Oracle DSL file (default: oracle_{N+1}.dsl)")
    ap.add_argument("--no-research", action="store_true", help="Skip arXiv search and PDF fetch")
    ap.add_argument("--analyze-only", action="store_true", help="Analysis + research only")
    ap.add_argument("--interactive", action="store_true", help="Gate each action with y/N")
    args = ap.parse_args()

    os.chdir(ROOT)
    src = _bootstrap_dsl()
    analysis = analyze_stage(src)
    print("=== stage analysis (data, not answers) ===")
    print(json.dumps({k: v for k, v in analysis.items() if k != "terms"}, indent=2))

    research = None
    if not args.no_research:
        try:
            research = run_research(analysis)
        except Exception as e:
            print(f"  research warning: {e} (continuing with cached/empty corpus)")

    if args.analyze_only:
        return

    oracle_path = args.oracle
    if oracle_path:
        oracle = open(oracle_path).read()
    else:
        oracle = generate_oracle(analysis, research)

    # persist oracle for the stage PDF attachment
    n = analysis["stage"] + 1
    open(os.path.join(ROOT, f"oracle_{n}.dsl"), "w").write(oracle)

    print("\n=== oracle increment (stage %d) ===" % n)
    print(oracle[:700] + ("..." if len(oracle) > 700 else ""))

    approve_all = not args.interactive
    if args.subprocess:
        print("\n=== evolving via subprocess + stdin ===")
        rc = evolve_subprocess(oracle, approve_all=approve_all)
        if rc != 0:
            sys.exit(rc)
        print("subprocess evolve finished.")
    else:
        print("\n=== evolving via evolve_auto() ===")
        pdf = evolve_direct(oracle, approve_all=approve_all)
        print("done:", pdf)


if __name__ == "__main__":
    main()
