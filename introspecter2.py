"""
introspecter2.py — the next model's engine (v2).

Extends a parent PDF by DOI (full look-back), and adds over the question-only
engine: a hypergraph (edges over many nodes), fuzzy(Greek)+precise node pairs,
and gated ACTIONS whose output is recorded as DATA beside a question, never as
an answer to it. Self-contained: rides inside each PDF as an embedded stream.
"""
from __future__ import annotations
import ast, base64, glob, os, re, subprocess
import numpy as np

PARENT_DOI = "10.5281/zenodo.20707850"   # self_3.pdf

VAR_TEX = {"Omega":r"\Omega","Sigma":r"\Sigma","Lambda":r"\Lambda","mu":r"\mu",
           "Phi":r"\Phi","Tau":r"\Tau","Delta":r"\Delta","Psi":r"\Psi","rho":r"\rho"}
GLYPH = {"Omega":"Ω","Sigma":"Σ","Lambda":"Λ","mu":"μ","Phi":"Φ","Tau":"Τ",
         "Delta":"Δ","Psi":"Ψ","rho":"ρ"}
# operators inherited from the parent series (now established):
OP_TEX = {"->":r"\longrightarrow","|=":r"\models","(x)":r"\otimes",
          "?=":r"\stackrel{?}{=}","-<":r"\nprec","~>":r"\rightsquigarrow",
          "*>":r"\Rrightarrow","!>":r"\dashrightarrow"}
OP_Q = {"->":"Does {a} give rise to {b} — and what precedes {a}?",
        "|=":"On what does {a} entailing {b} rest?",
        "(x)":"When {a} is carried by {b}, what survives and what is lost?",
        "?=":"Is {a} the same as {b}, or is {b} supplied by the reader?",
        "-<":"If {a} does not reduce to {b}, where is the remainder?",
        "~>":"Does {a} haunt {b} — present as an absence?",
        "*>":"What residue remains because {a} leaves a trace upon {b}?",
        "!>":"What collapses when {a} shatters the structure of {b}?"}
OPS = list(OP_TEX)
N_MAX = 16


def _txt(t):
    t = t.lstrip("@")
    return (r"\partial " if t != t else "") + VAR_TEX.get(t, r"\mathrm{%s}" % t)

def _term_tex(t):
    return (r"\partial " + VAR_TEX.get(t[1:], r"\mathrm{%s}"%t[1:])
            if t.startswith("@") else VAR_TEX.get(t, r"\mathrm{%s}"%t))

def _glyph(t):
    s = t.lstrip("@"); g = GLYPH.get(s, s)
    return ("∂"+g) if t.startswith("@") else g


# ---- actions: defined procedures; their output is DATA, not an answer ----
def act_coherence(nodes, model):
    words = [set(p.lower().split()) for _, _, p in nodes if p]
    if len(words) < 2:
        return {"coherence": None}
    inter, union = set.intersection(*words), set.union(*words)
    return {"coherence": round(len(inter)/max(1, len(union)), 3),
            "shared": sorted(inter)}

def act_neural(nodes, model):
    rng = np.random.default_rng(len(nodes))
    x = np.array([len(p) for _, _, p in nodes], float)
    y = float(1/(1+np.exp(-(rng.standard_normal((1, len(x))) @ x)[0])))
    return {"neural_response": round(y, 3),
            "note": "untrained net; data, not an answer"}

ACTIONS = {"coherence": act_coherence, "neural": act_neural}


# ---- @dslai: a node's spec compiles into a small TRAINABLE module (numpy) ----
# Mirrors the torch @dslai of viXra:2507.0074; kept dependency-light so the
# engine stays self-contained inside the PDF. A module computes; it does not
# comprehend. Its outputs and losses are recorded as DATA.
def _sig(z): return 1/(1+np.exp(-z))

def compile_module(spec, seed=0):
    d = spec["dims"]
    if len(d) < 2: d = [d[0], d[0]]
    i, h, o = d[0], (d[1] if len(d) >= 3 else d[0]), d[-1]
    rng = np.random.default_rng(seed)
    return {"W1": rng.standard_normal((h, i))*0.5, "b1": np.zeros(h),
            "W2": rng.standard_normal((o, h))*0.5, "b2": np.zeros(o),
            "in": i, "out": o, "hidden": h}

def forward(m, X):
    h = _sig(X @ m["W1"].T + m["b1"])
    return _sig(h @ m["W2"].T + m["b2"]), h

def train(m, X, Y, epochs=300, lr=0.5):
    n = len(X); loss = 0.0
    for _ in range(epochs):
        o, h = forward(m, X); loss = float(np.mean((o-Y)**2))
        do = 2*(o-Y)/n * o*(1-o)
        dh = (do @ m["W2"]) * h*(1-h)
        m["W2"] -= lr*(do.T @ h); m["b2"] -= lr*do.sum(0)
        m["W1"] -= lr*(dh.T @ X); m["b1"] -= lr*dh.sum(0)
    return loss

def act_train(nodes, p):
    """Compile a node's module and fit it to a defined toy objective."""
    for s, _, _ in nodes:
        spec = p.get("modules", {}).get(s)
        if spec:
            m = compile_module(spec); rng = np.random.default_rng(1)
            X = rng.random((64, m["in"]))
            Y = X[:, :m["out"]] if m["out"] <= m["in"] else rng.random((64, m["out"]))
            l0 = float(np.mean((forward(m, X)[0]-Y)**2))
            lf = train(m, X, Y, epochs=400, lr=0.6)
            return {"node": s, "arch": spec["arch"], "task":
                    "autoencode" if m["out"] == m["in"] else "fit",
                    "mse_start": round(l0, 4), "mse_final": round(lf, 4),
                    "is": "an optimization result over a defined toy objective; "
                          "not the self learning what it is"}
    return {"note": "no module-bearing node in this edge"}

def act_infer(nodes, p):
    for s, _, _ in nodes:
        spec = p.get("modules", {}).get(s)
        if spec:
            m = compile_module(spec)
            y, _ = forward(m, np.ones((1, m["in"])))
            return {"node": s, "arch": spec["arch"],
                    "output": [round(v, 3) for v in y[0]],
                    "is": "an untrained module's forward pass; data, not an answer"}
    return {"note": "no module-bearing node in this edge"}

ACTIONS.update({"train": act_train, "infer": act_infer})


# ---- parsing (protect comments AND strings before splitting on ';') ----
def _protect(src):
    store = {}
    def grab(m):
        k = "\x00%d\x00" % len(store); store[k] = m.group(0); return k
    src = re.sub(r"```.*?```", grab, src, flags=re.S)   # code blocks
    src = re.sub(r"/\*.*?\*/", grab, src, flags=re.S)   # comments
    src = re.sub(r'"[^"]*"', grab, src)                 # quoted strings
    return src, store

def parse(src):
    protected, store = _protect(src)
    meta, nodes, rels, hyper, code, modules = {"nu": "0"}, {}, [], [], [], {}
    for k, v in store.items():
        if v.startswith("```"):
            code.append(re.sub(r"```(?:python)?\n?|```", "", v, flags=re.S).strip("\n"))
    last = None
    for chunk in protected.split(";"):
        cm = re.search(r"\x00\d+\x00", chunk)
        gloss = ""
        if cm and store.get(cm.group(0), "").startswith("/*") and last is not None:
            g = store[cm.group(0)][2:-2].strip()
            if rels and last == "rel" and not rels[-1][3]:
                rels[-1][3] = g
        stmt = re.sub(r"\s+", "", re.sub(r"\x00\d+\x00",
                lambda m: "" if store.get(m.group(0),"").startswith(("/*","```")) else m.group(0),
                chunk))
        if not stmt:
            continue
        if stmt.startswith("{"):                         # hyperedge
            mm = re.match(r"\{([^}]*)\}\?(\x00\d+\x00)(?:!([A-Za-z]+))?", stmt)
            if mm:
                syms = [x for x in re.split(r"[,\s]+", mm.group(1)) if x]
                q = store[mm.group(2)][1:-1]
                hyper.append((syms, q, mm.group(3))); last = "hyper"
            continue
        if stmt.startswith("module"):                   # @dslai: node -> trainable net
            body = stmt[len("module"):]
            mp = re.match(r"([A-Za-z]+):([0-9\->]+):([A-Za-z]+)", body)
            if mp:
                dims = [int(d) for d in re.split(r"->", mp.group(2)) if d]
                modules[mp.group(1)] = {"dims": dims, "act": mp.group(3),
                                        "arch": "->".join(map(str, dims))}
            last = "decl"; continue
        op = next((o for o in OPS if o in stmt), None)
        if op and ":" not in stmt:                       # binary relation
            a, b = stmt.split(op, 1); rels.append([a, op, b, ""]); last = "rel"
        else:                                            # decl / node / meta
            parts = stmt.split(":")
            key = parts[0]
            if key in ("nu", "parent"):
                meta[key] = parts[1]
            elif len(parts) >= 2:
                precise = ""
                if len(parts) >= 3 and parts[2] in store:
                    precise = store[parts[2]][1:-1]
                nodes[key] = (parts[1], precise)
            last = "decl"
    return {"meta": meta, "nodes": nodes, "rels": rels, "hyper": hyper,
            "code": code, "modules": modules}


def to_questions(p):
    qs = [OP_Q[op].format(a=_glyph(a), b=_glyph(b)) for a, op, b, _ in p["rels"]]
    qs += [q for _, q, _ in p["hyper"]]
    return qs


# ---- introspective metadata ----
def _funcs(path):
    return [n.name for n in ast.parse(open(path).read()).body
            if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]

def title(p):
    ws = [w.capitalize() for _, (w, _) in p["nodes"].items()]
    return ", ".join(ws[:-1]) + ", and " + ws[-1] + ": A Self-Evolving Hypergraph" if ws else "Inquiry"

def abstract(p, engine=None):
    n = p["meta"].get("nu", "0")
    mods = p.get("modules", {})
    mtext = (f" Of these, {len(mods)} compile via @dslai into small trainable "
             f"modules ({', '.join(mods)}), making the hypergraph partly "
             f"differentiable; a module computes, it does not comprehend." if mods else "")
    return (f"Stage {n} of a self-evolving hypergraph, extended from its parent "
            f"(doi:{p['meta'].get('parent', PARENT_DOI)}) so the inquiry can look "
            f"fully back. It declares {len(p['nodes'])} nodes, each pairing a "
            f"suggestive Greek glyph with a precise description.{mtext} It relates "
            f"them by {len(p['rels'])} binary relations and {len(p['hyper'])} "
            f"hyperedges, and may trigger gated actions whose output is recorded as "
            f"data, never as an answer. The engine, introspected from its own source, "
            f"exposes {', '.join(_funcs(engine or __file__))}. It poses questions and "
            f"enacts computations beside them; it makes no claim to resolve what a "
            f"self is.")


# ---- gated action run; records results keyed by question ----
def run_actions(p, approve, results):
    for syms, q, action in p["hyper"]:
        if not action:
            continue
        nodes = [(s, *p["nodes"].get(s, (s, ""))) for s in syms]
        print(f"\n[action '{action}'] for: \"{q}\"")
        if approve(q, action):
            results[q] = ACTIONS[action]([(s, w, pr) for s, w, pr in nodes], p)
            print("  RECORDED (data, not an answer):", results[q])
        else:
            print("  declined; the question stands as a question.")


# ---- LaTeX / PDF ----
_SP = [("\\",r"\textbackslash{}"),("&",r"\&"),("%",r"\%"),("$",r"\$"),
       ("#",r"\#"),("_",r"\_"),("{",r"\{"),("}",r"\}"),("~",r"\textasciitilde{}"),
       ("^",r"\textasciicircum{}")]
def _esc(s):
    for a, b in _SP: s = s.replace(a, b)
    return s

EXTRACTOR = '''import sys, glob, os
sys.path.insert(0, ".")
from pypdf import PdfReader
pdf = max(glob.glob("*.pdf"), key=os.path.getmtime)
for name, data in PdfReader(pdf).attachments.items():
    open(name, "wb").write(data[0] if isinstance(data, list) else data)
import importlib, introspecter2 as I; importlib.reload(I); I.evolve()
'''
def _bootstrap():
    b = base64.b64encode(EXTRACTOR.encode()).decode()
    return "import base64\nexec(base64.b64decode(\n" + \
        "\n".join('  "%s"' % b[i:i+58] for i in range(0, len(b), 58)) + "))"

def build_tex(p, results):
    rows = [rf"{_term_tex(a)} \; {OP_TEX[op]} \; {_term_tex(b)}" for a, op, b, _ in p["rels"]]
    math = "\\begin{align*}\n" + r" \\[3pt] ".join(rows) + "\n\\end{align*}"
    mods = p.get("modules", {})
    terms = "\n".join(rf"\item[$ {VAR_TEX.get(s,s)} $] {_esc(w)}" +
                      (rf" --- \emph{{{_esc(pr)}}}" if pr else " [inherited]") +
                      (rf" \quad $\rightarrow$ \texttt{{module {mods[s]['arch']} "
                       rf"({mods[s]['act']}), trainable}}" if s in mods else "")
                      for s, (w, pr) in p["nodes"].items())
    hyper = []
    for syms, q, action in p["hyper"]:
        glyphs = ",\\,".join(VAR_TEX.get(s, s) for s in syms)
        line = rf"\item $\{{{glyphs}\}}$ \quad \textbf{{?}} {_esc(q)}"
        if action: line += rf" \quad[\texttt{{{action}}}]"
        if q in results:
            line += rf"\\ \hspace*{{1em}}$\hookrightarrow$ \emph{{recorded data (not an answer):}} \texttt{{{_esc(str(results[q]))}}}"
        hyper.append(line)
    qs = "\n".join(rf"\item {_esc(x)}" for x in to_questions(p))
    tex = TEX
    for tok, val in [("@@TITLE@@", _esc(title(p))), ("@@N@@", p["meta"].get("nu","0")),
                     ("@@PARENT@@", _esc(p["meta"].get("parent", PARENT_DOI))),
                     ("@@ABSTRACT@@", _esc(abstract(p))), ("@@MATH@@", math),
                     ("@@TERMS@@", terms), ("@@HYPER@@", "\n".join(hyper) or r"\item (none yet)"),
                     ("@@QUESTIONS@@", qs), ("@@BOOT@@", _bootstrap())]:
        tex = tex.replace(tok, val)
    return tex

def build_pdf(src=None, results=None):
    src = src or open("self.dsl").read()
    p = parse(src); results = results or {}
    n = p["meta"].get("nu", "0"); base = f"self_{n}"
    open(base + ".tex", "w").write(build_tex(p, results))
    for _ in range(2):
        subprocess.run(["xelatex", "-interaction=nonstopmode", base + ".tex"],
                       capture_output=True)
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter(); w.append(PdfReader(base + ".pdf"))
    w.add_attachment("introspecter2.py", open(__file__, "rb").read())
    w.add_attachment("self.dsl", src.encode())
    with open(base + ".pdf", "wb") as f: w.write(f)
    return base + ".pdf"

def evolve():
    src = open("self.dsl").read(); p = parse(src)
    for c in p["code"]:
        print("\n[carried code]\n" + c)
        if input("re-activate into the engine? [y/N]: ").strip().lower() == "y":
            try: exec(c, globals())
            except Exception as e: print("skip:", e)
    results = {}
    run_actions(p, lambda q, a: input(f"run action '{a}'? [y/N]: ").strip().lower()=="y", results)
    print("\n" + abstract(p))
    print("\nEdit self.dsl to extend the hypergraph, then run build_pdf().")

TEX = r"""\documentclass[11pt]{article}
\usepackage{fontspec}\setmonofont{DejaVu Sans Mono}[Scale=0.8]
\usepackage{amsmath,amssymb}\usepackage[margin=1in]{geometry}
\usepackage{fancyvrb}\usepackage[hidelinks]{hyperref}\usepackage{enumitem}
\usepackage{newunicodechar}
\newunicodechar{μ}{\ensuremath{\mu}}
\newunicodechar{∂}{\ensuremath{\partial}}
\newunicodechar{Τ}{\ensuremath{\mathrm{T}}}
\newunicodechar{ρ}{\ensuremath{\rho}}
\newunicodechar{Ψ}{\ensuremath{\Psi}}
\setlist{nosep,leftmargin=1.4em}
\newcommand{\Tau}{\mathrm{T}}
\title{@@TITLE@@}\author{Brent Hartshorn\\ \texttt{brenthartshorn@proton.me}}
\date{Stage @@N@@ \quad (parent: doi:@@PARENT@@)}
\begin{document}\maketitle
\begin{abstract}@@ABSTRACT@@\end{abstract}
\section*{Symbolic core (inherited relations)}@@MATH@@
\section*{Nodes (suggestive glyph + precise description)}
\begin{description}@@TERMS@@\end{description}
\section*{Hypergraph (questions, and the data actions returned)}
\begin{itemize}@@HYPER@@\end{itemize}
\section*{Questions this stage poses}\begin{itemize}@@QUESTIONS@@\end{itemize}
\section*{Lineage and provenance}
This model extends its parent self-evolving inquiry, doi:@@PARENT@@, ingesting
its relations so the series can look fully back \cite{p3,agi,meta,arch}. Actions
record data beside questions; the document poses and enacts, and makes no claim
to resolve what a self is.
\section*{Appendix: self-extraction}
This PDF carries its engine (\texttt{introspecter2.py}) and DSL
(\texttt{self.dsl}) as embedded streams. Requires \texttt{pip install pypdf
numpy} and a LaTeX install. To produce the next stage, save and run:
\begin{Verbatim}[fontsize=\small,frame=single]
@@BOOT@@
\end{Verbatim}
\begin{thebibliography}{9}
\bibitem{p3} B.~Hartshorn, \emph{Origin, Self, Language, Meaning, Other, Time,
and Difference: A Self-Evolving Inquiry} (self\_3). doi:10.5281/zenodo.20707850
\bibitem{agi} B.~Hartshorn, \emph{Towards Self-Evolving AGI via Emergent DSL}.
viXra:2507.0104
\bibitem{meta} B.~Hartshorn, \emph{Emergent Self-Modification and
Meta-Programming in Dynamic Systems}. viXra:2507.0036
\bibitem{arch} B.~Hartshorn, \emph{Self-Contained Multi-AI Architecture
Definition via DSL}. viXra:2507.0074
\end{thebibliography}\end{document}
"""

if __name__ == "__main__":
    print("built:", build_pdf())
