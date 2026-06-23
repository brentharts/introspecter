"""
research.py — outward literature search, PDF fetch, ingest, and model proposals.

Used by introspecter8 and auto_evolve. Downloads only from an allowlisted set of
hosts; returns snippets and proposals as DATA, never as answers about selfhood.
"""
from __future__ import annotations
import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

CORPUS_DIR = "corpus"
RESEARCH_CACHE = "research.json"
MAX_PDF_BYTES = 12_000_000
MAX_DOWNLOADS = 6

ALLOWED_HOSTS = (
    "arxiv.org", "export.arxiv.org",
    "zenodo.org", "doi.org",
    "ai.vixra.org",
)

# Author's viXra line + neuro-symbolic hooks (seed queries, not conclusions)
SEED_QUERIES = [
    "neuro-symbolic self-modifying DSL",
    "introspective knowledge generation emergent DSL",
    "self-evolving hypergraph provenance",
    "machine language emergence neural symbolic",
]

VIXRA_PDFS = [
    ("2507.0104", "Towards Self-Evolving AGI via Emergent DSL"),
    ("2507.0074", "Self-Contained Multi-AI Architecture Definition via DSL"),
    ("2507.0036", "Emergent Self-Modification and Meta-Programming"),
]


def _allowed(url: str) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    return any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)


def _get(url: str, timeout=30) -> bytes:
    if not _allowed(url):
        raise ValueError(f"host not allowlisted: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "introspecter-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f"response too large ({len(data)} bytes): {url}")
    return data


def arxiv_search(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv API; return metadata as data."""
    q = urllib.parse.quote(query)
    url = (f"http://export.arxiv.org/api/query?search_query=all:{q}"
           f"&start=0&max_results={max_results}")
    raw = _get(url).decode("utf-8", errors="replace")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)
    out = []
    for entry in root.findall("a:entry", ns):
        aid = entry.find("a:id", ns)
        title = entry.find("a:title", ns)
        summary = entry.find("a:summary", ns)
        if aid is None:
            continue
        arxiv_id = aid.text.rstrip("/").split("/abs/")[-1]
        out.append({
            "id": arxiv_id,
            "title": (title.text or "").strip().replace("\n", " "),
            "summary": (summary.text or "").strip()[:400],
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "is": "a bibliographic hit from arXiv; not evidence about a self",
        })
    return out


def search_literature(terms: list[str] | None = None, per_query: int = 3) -> dict:
    """Run seed + term-derived queries; dedupe by arXiv id."""
    queries = list(SEED_QUERIES)
    if terms:
        blob = " ".join(terms[:5])
        queries.insert(0, f"self {blob} neuro-symbolic")
    seen, hits = set(), []
    for q in queries[:5]:
        for h in arxiv_search(q, max_results=per_query):
            if h["id"] not in seen:
                seen.add(h["id"])
                hits.append(h)
    return {"queries": queries[:5], "hits": hits[:12],
            "is": "outward search results; titles and ids, not understanding"}


def fetch_pdf(url: str, dest_dir: str = CORPUS_DIR) -> str:
    """Download one PDF to corpus/ if allowlisted."""
    os.makedirs(dest_dir, exist_ok=True)
    if not _allowed(url):
        raise ValueError(f"refused URL (not allowlisted): {url}")
    data = _get(url)
    name = re.sub(r"[^\w.\-]+", "_", urllib.parse.urlparse(url).path.split("/")[-1])
    if not name.endswith(".pdf"):
        name = name + ".pdf" if name else "paper.pdf"
    path = os.path.join(dest_dir, name)
    open(path, "wb").write(data)
    return path


def fetch_arxiv(arxiv_id: str, dest_dir: str = CORPUS_DIR) -> str:
    return fetch_pdf(f"https://arxiv.org/pdf/{arxiv_id}.pdf", dest_dir)


def fetch_vixra(vid: str, dest_dir: str = CORPUS_DIR) -> str:
    return fetch_pdf(f"https://ai.vixra.org/pdf/{vid}v1.pdf", dest_dir)


def fetch_corpus(search_hits: list[dict] | None = None,
                 include_vixra: bool = True) -> dict:
    """Download top arXiv hits + author viXra seeds into corpus/."""
    os.makedirs(CORPUS_DIR, exist_ok=True)
    downloaded, errors = [], []
    n = 0
    if include_vixra:
        for vid, title in VIXRA_PDFS:
            if n >= MAX_DOWNLOADS:
                break
            try:
                path = fetch_vixra(vid)
                downloaded.append({"source": "vixra", "id": vid, "title": title, "path": path})
                n += 1
            except Exception as e:
                errors.append({"id": vid, "error": str(e)})
    for h in (search_hits or [])[:max(0, MAX_DOWNLOADS - n)]:
        try:
            path = fetch_pdf(h["pdf_url"])
            downloaded.append({"source": "arxiv", "id": h["id"], "title": h["title"], "path": path})
        except Exception as e:
            errors.append({"id": h.get("id"), "error": str(e)})
    return {"downloaded": downloaded, "errors": errors, "corpus_dir": CORPUS_DIR,
            "is": "files stored beside the inquiry; storage, not comprehension"}


def ingest_corpus(terms: list[str] | None = None, dest_dir: str = CORPUS_DIR,
                  max_chars: int = 800) -> dict:
    """Extract text from corpus PDFs; return term-overlap snippets as data."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return {"note": "pypdf required for ingest"}
    terms = [t.lower() for t in (terms or []) if t]
    snippets, files = [], []
    if not os.path.isdir(dest_dir):
        return {"snippets": [], "files": [], "note": "empty corpus"}
    for fn in sorted(os.listdir(dest_dir)):
        if not fn.lower().endswith(".pdf"):
            continue
        path = os.path.join(dest_dir, fn)
        try:
            text = ""
            for page in PdfReader(path).pages[:8]:
                text += (page.extract_text() or "") + "\n"
            files.append(fn)
            low = text.lower()
            score = sum(low.count(t) for t in terms) if terms else 0
            # pull a window around the first strong term hit
            excerpt = text[:max_chars].replace("\n", " ").strip()
            for t in terms:
                i = low.find(t)
                if i >= 0:
                    excerpt = text[max(0, i - 120): i + max_chars].replace("\n", " ").strip()
                    break
            snippets.append({"file": fn, "term_hits": score,
                             "excerpt": excerpt[:max_chars],
                             "is": "extracted text; co-occurrence with authored terms, not meaning"})
        except Exception as e:
            snippets.append({"file": fn, "error": str(e)})
    return {"snippets": snippets, "files": files, "n_files": len(files),
            "is": "corpus ingest; phrases beside the question, not an answer"}


def propose_models(analysis: dict, ingest: dict | None = None) -> dict:
    """
    Propose small module architectures from hub geometry and law scores.
    Authored heuristic — a proposal recorded as data, not a trained mind.
    """
    hub = analysis.get("hub", "Sigma")
    law_self = analysis.get("law_self")
    n_nodes = analysis.get("n_nodes", 7)
    proposals = []
    # hub-dense self: widen bottleneck if law_self is low (heuristic label only)
    if hub == "Sigma" and (law_self is None or law_self < 0.2):
        proposals.append({
            "node": "Sigma",
            "arch": "2->4->2",
            "activation": "sigmoid",
            "task": "autoencode",
            "note": "wider hidden layer beside low law_self score; proposal only",
        })
        proposals.append({
            "node": "Psi",
            "arch": "3->4->3",
            "activation": "sigmoid",
            "task": "autoencode",
            "note": "oracle module mirroring propose-side width; proposal only",
        })
    if n_nodes >= 8:
        proposals.append({
            "circuit": "rho = Psi -> Lambda -> mu",
            "width": 4,
            "note": "field routed through language toward meaning; symbolic wiring only",
        })
    # keyword echo from ingest (data, not semantics)
    keywords = []
    for sn in (ingest or {}).get("snippets", []):
        ex = sn.get("excerpt", "").lower()
        for kw in ("neuro-symbolic", "self-modif", "dsl", "introspect", "hypergraph"):
            if kw in ex:
                keywords.append(kw)
    return {
        "proposals": proposals,
        "corpus_keywords": sorted(set(keywords))[:8],
        "hub": hub,
        "law_self": law_self,
        "is": "authored architecture suggestions over analysis + corpus; "
              "not evidence that a self or model was discovered",
    }


def run_research_cycle(terms: list[str] | None = None) -> dict:
    """Full outward cycle: search → fetch → ingest → propose (for auto_evolve)."""
    search = search_literature(terms)
    fetch = fetch_corpus(search.get("hits"))
    ingest = ingest_corpus(terms)
    analysis_stub = {"hub": "Sigma", "law_self": 0.1, "n_nodes": 9}
    propose = propose_models(analysis_stub, ingest)
    out = {"search": search, "fetch": fetch, "ingest": ingest, "propose": propose}
    open(RESEARCH_CACHE, "w").write(json.dumps(out, indent=2))
    return out
