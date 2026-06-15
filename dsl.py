"""
dsl.py — self-contained support engine (rides inside the PDF as a file stream).

This revision preserves the per-relation comments and the term definitions,
which carry the meaning, and surfaces them in three places: the rendered PDF
(a Terms list + a Reading list), and an enriched prompt.txt that is written on
every build and embedded in the PDF so it is never lost.
"""
from __future__ import annotations
import ast, base64, glob, os, re, subprocess

VAR_TEX = {"Omega": r"\Omega", "Sigma": r"\Sigma", "Lambda": r"\Lambda",
           "mu": r"\mu", "nu": r"\nu", "theta": r"\theta"}
OP_TEX = {"->": r"\longrightarrow", "|=": r"\models", "(x)": r"\otimes",
          "?=": r"\stackrel{?}{=}", "-<": r"\nprec"}
OP_Q = {
    "->": "Does {a} give rise to {b} — and what, if anything, precedes {a}?",
    "|=": "On what does {a} entailing {b} rest, or is it simply assumed?",
    "(x)": "When {a} is carried by {b}, what survives and what is lost?",
    "?=": "Is {a} the same as {b}, or is {b} supplied by whoever reads {a}?",
    "-<": "If {a} does not reduce to {b}, where does the remainder come from?",
}
OP_GLOSS = {"->": "genesis", "|=": "the reflexive claim",
            "(x)": "reached only through language", "?=": "change, or meaning?",
            "-<": "not merely its origin"}
OPS = ["->", "|=", "(x)", "?=", "-<"]
N_MAX = 12

REFS = [
    ("fractal", "Iterating a Fractal-like Self Awareness Naturally", "2505.0195"),
    ("nature",  "Nature vs Nurture", "2505.0141"),
    ("meta",    "Emergent Self-Modification and Meta-Programming in Dynamic Systems", "2507.0036"),
    ("arch",    "Self-Contained Multi-AI Architecture Definition via DSL", "2507.0074"),
    ("agi",     "Towards Self-Evolving AGI: Multi-Modal Learning and Introspective Knowledge Generation via Emergent DSL", "2507.0104"),
    ("scaling", "Non-Quadratic Scaling and Beyond Sequential Processing", "2507.0109"),
]

EXTRACTOR = '''import sys, glob, os
sys.path.insert(0, ".")
from pypdf import PdfReader
pdf = max(glob.glob("*.pdf"), key=os.path.getmtime)
r = PdfReader(pdf)
for name, data in r.attachments.items():
    open(name, "wb").write(data[0] if isinstance(data, list) else data)
import importlib, dsl; importlib.reload(dsl); dsl.evolve()
'''


# ---------- parsing (keeps definitions AND per-relation glosses) ----------
def _strip_code(src: str):
    """Separate ```python ... ``` blocks (the mini-DSL) from the symbolic DSL."""
    blocks = [b.strip("\n") for b in
              re.findall(r"```(?:python)?\n?(.*?)```", src, re.S)]
    clean = re.sub(r"```(?:python)?.*?```", "", src, flags=re.S)
    return clean, blocks


def code_blocks(src: str):
    return _strip_code(src)[1]


def _parse(src: str):
    src, _ = _strip_code(src)        # code blocks never affect symbol parsing
    # Pull comments out FIRST (they may contain ';'), leaving ;-safe sentinels.
    comments = []
    def _grab(m):
        comments.append(m.group(1).strip()); return f"\x00{len(comments)-1}\x00"
    clean = re.sub(r"/\*(.*?)\*/", _grab, src, flags=re.S)
    names, rels = {}, []           # rels: [a, op, b, gloss]
    last_rel = None
    for chunk in clean.split(";"):
        cm = re.search(r"\x00(\d+)\x00", chunk)   # a comment here = prev rel's gloss
        if cm and last_rel is not None and not rels[last_rel][3]:
            rels[last_rel][3] = comments[int(cm.group(1))]
        stmt = re.sub(r"\s+", "", re.sub(r"\x00\d+\x00", "", chunk))
        if not stmt:
            continue
        op = next((o for o in OPS if o in stmt), None)
        if op:
            a, b = stmt.split(op, 1)
            rels.append([a, op, b, ""]); last_rel = len(rels) - 1
        elif ":" in stmt:
            g, w = stmt.split(":", 1); names[g] = w; last_rel = None
    for r in rels:                                # fill any missing gloss
        r[3] = r[3] or OP_GLOSS[r[1]]
    return names, rels


def _term(t, names): return (f"the change in {names.get(t[1:], t[1:])}"
                             if t.startswith("@") else names.get(t, t))


def to_questions(src):
    names, rels = _parse(src)
    return [OP_Q[op].format(a=_term(a, names), b=_term(b, names))
            for a, op, b, _ in rels]


def _txtok(t): return (r"\partial " + VAR_TEX.get(t[1:], t[1:])
                       if t.startswith("@") else VAR_TEX.get(t, t))


def to_latex(src):
    _, rels = _parse(src)
    rows = [rf"{_txtok(a)} \; {OP_TEX[op]} \; {_txtok(b)}" for a, op, b, _ in rels]
    return "\\begin{align*}\n" + r" \\[4pt] ".join(rows) + "\n\\end{align*}"


def to_dsl(names, rels, code=None):
    n = int(names.get("nu", "0"))
    decls = [f"{g} : {w} ;" for g, w in names.items() if g != "nu"]
    rels_ = [f"{a} {op} {b} ;   /* {g} */" for a, op, b, g in rels]
    parts = [f"/* self.dsl - stage {n} ; survives PDF extraction */",
             "", f"nu : {n} ;", *decls, "", *rels_, ""]
    for c in (code or []):           # evolved Python, persisted inside the DSL
        parts += ["```python", c, "```", ""]
    return "\n".join(parts)


# ---------- introspective metadata ----------
def _engine_funcs(path):
    return [n.name for n in ast.parse(open(path).read()).body
            if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]


def title(src):
    names, _ = _parse(src)
    nouns = [w.capitalize() for g, w in names.items() if g != "nu"]
    return ", ".join(nouns[:-1]) + ", and " + nouns[-1] + ": A Self-Evolving Inquiry"


def abstract(src, engine=None):
    names, rels = _parse(src)
    n = int(names.get("nu", "0"))
    nouns = [w for g, w in names.items() if g != "nu"]
    funcs = _engine_funcs(engine or __file__)
    return (f"This is stage {n} of a self-evolving document. Its symbolic core "
            f"declares {len(nouns)} terms ({', '.join(nouns)}) and poses "
            f"{len(rels)} questions through {len(rels)} relations. The supporting "
            f"engine, introspected from its own source, exposes "
            f"{', '.join(funcs)}; it parses the core, renders it, generates the "
            f"questions, and emits the next stage. Continuing the author's program "
            f"on emergent DSLs and self-modification, the document is an apparatus "
            f"that regenerates these questions across iterations: it poses them, "
            f"and makes no claim to resolve them.")


# ---------- prompt.txt (the LLM hand-off) ----------
def state_summary(src):
    names, rels = _parse(src)
    n = int(names.get("nu", "0"))
    terms = [f"  {_glyph(g)} ({g}) = {w}" for g, w in names.items() if g != "nu"]
    reln = [f"  {a} {op} {b}  —  {g}" for a, op, b, g in rels]
    qs = [f"  - {q}" for q in to_questions(src)]
    return "\n".join([
        f"SELF-EVOLVING DOCUMENT — STAGE {n}", "",
        "TERMS:", *terms, "",
        "RELATIONS (declared intent):", *reln, "",
        "QUESTIONS THIS STAGE POSES:", *qs, ""])


def build_prompt(src):
    # Used for prompt.txt, the hand-off to an LLM that writes the next paper.
    return state_summary(src) + "\n".join([
        "", "INSTRUCTION FOR THE LLM:",
        "Using the terms, relations, and questions above together with the",
        "attached LaTeX and PDF, write the next iteration's paper. Pose and",
        "explore these questions and their interrelations; do not claim to have",
        "resolved what the self is — the document's role is to keep asking.", ""])


def _glyph(g):  # ascii name -> unicode glyph for the readable prompt
    return {"Omega": "Ω", "Sigma": "Σ", "Lambda": "Λ", "mu": "μ",
            "theta": "θ", "nu": "ν"}.get(g, g)


# ---------- LaTeX / PDF assembly ----------
_SPECIAL = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
            ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
            ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]


def _esc(s):
    for a, b in _SPECIAL:
        s = s.replace(a, b)
    return s


def _bootstrap():
    blob = base64.b64encode(EXTRACTOR.encode()).decode()
    chunks = "\n".join(f'  "{blob[i:i+58]}"' for i in range(0, len(blob), 58))
    return "import base64\nexec(base64.b64decode(\n" + chunks + "))"


def _evolution_tex(oracle, py):
    if not oracle:
        return ""
    return (r"\section*{Evolution (this stage)}" "\n"
            "The oracle supplied the DSL below, which compiled to the Python "
            "beneath it and was executed (sandboxed) to produce this stage.\n\n"
            r"\noindent\textbf{Oracle (DSL):}" "\n"
            r"\begin{Verbatim}[fontsize=\small,frame=single]" "\n"
            + oracle + "\n" r"\end{Verbatim}" "\n\n"
            r"\noindent\textbf{Compiled Python (only builder calls):}" "\n"
            r"\begin{Verbatim}[fontsize=\small,frame=single]" "\n"
            + py + "\n" r"\end{Verbatim}" "\n")


def build_tex(src, oracle=None, py=None):
    names, rels = _parse(src)
    n = int(names.get("nu", "0"))
    terms = "\n".join(rf"\item[$ {_txtok(g)} $] {_esc(w)}"
                      for g, w in names.items() if g != "nu")
    reading = "\n".join(
        rf"\item $ {_txtok(a)} \; {OP_TEX[op]} \; {_txtok(b)} $ — {_esc(g)}"
        for a, op, b, g in rels)
    items = "\n".join(rf"\item {_esc(q)}" for q in to_questions(src))
    refs = "\n".join(
        rf"\bibitem{{{k}}} B.~Hartshorn, \emph{{{_esc(t)}}}. "
        rf"viXra:{vid}. https://ai.vixra.org/abs/{vid}" for k, t, vid in REFS)
    tex = TEX
    for tok, val in [("@@TITLE@@", _esc(title(src))), ("@@N@@", str(n)),
                     ("@@ABSTRACT@@", _esc(abstract(src))),
                     ("@@MATH@@", to_latex(src)), ("@@TERMS@@", terms),
                     ("@@READING@@", reading), ("@@QUESTIONS@@", items),
                     ("@@EVOLUTION@@", _evolution_tex(oracle, py)),
                     ("@@BOOT@@", _bootstrap()), ("@@REFS@@", refs)]:
        tex = tex.replace(tok, val)
    return tex


def build_pdf(src=None, oracle=None, py=None):
    src = src or open("self.dsl").read()
    n = int(_parse(src)[0].get("nu", "0"))
    base = f"self_{n}"
    open(base + ".tex", "w").write(build_tex(src, oracle, py))
    open("prompt.txt", "w").write(build_prompt(src))   # always written
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", base + ".tex"],
                       capture_output=True)
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter(); w.append(PdfReader(base + ".pdf"))
    w.add_attachment("dsl.py", open(__file__, "rb").read())
    w.add_attachment("self.dsl", src.encode())
    w.add_attachment("prompt.txt", build_prompt(src).encode())
    blocks = code_blocks(src)
    if blocks:
        w.add_attachment("evolved_code.py", ("\n\n".join(blocks)).encode())
    if oracle:
        w.add_attachment("oracle.dsl", oracle.encode())
        w.add_attachment("evolution.py", (py or "").encode())
    with open(base + ".pdf", "wb") as f:
        w.write(f)
    return base + ".pdf"


TEX = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage[margin=1in]{geometry}
\usepackage{fancyvrb}
\title{@@TITLE@@}
\author{Brent Hartshorn\\ \texttt{brenthartshorn@proton.me}}
\date{Stage @@N@@}
\begin{document}
\maketitle
\begin{abstract}
@@ABSTRACT@@
\end{abstract}

\section*{Symbolic core}
The evolving core of the document, rendered from its source:
@@MATH@@

\section*{Terms}
\begin{description}
@@TERMS@@
\end{description}

\section*{Reading (relations and their intent)}
\begin{itemize}
@@READING@@
\end{itemize}

\section*{Questions this stage poses}
\begin{itemize}
@@QUESTIONS@@
\end{itemize}

\section*{Lineage}
This iteration continues the author's program on emergent DSLs, introspective
generation, and self-modification \cite{agi,meta,arch,fractal}. It is an
apparatus that regenerates the questions above across iterations; it does not
claim to resolve them.

@@EVOLUTION@@
\section*{Appendix: self-extraction}
This PDF carries its support code (\texttt{dsl.py}, \texttt{self.dsl}) and the
hand-off prompt (\texttt{prompt.txt}) as embedded file streams. Requires
\texttt{pip install pypdf} and a LaTeX installation. To produce the next
iteration, save this PDF and run:
\begin{Verbatim}[fontsize=\small,frame=single]
@@BOOT@@
\end{Verbatim}

\begin{thebibliography}{9}
@@REFS@@
\end{thebibliography}
\end{document}
"""

if __name__ == "__main__":
    print("built:", build_pdf())
# Appended to dsl.py: the human-in-the-loop evolve() and its honest, sandboxed core.
import io

# More Greek for new terms an oracle may introduce; unknown -> \mathrm fallback.
VAR_TEX.update({"Phi": r"\Phi", "Psi": r"\Psi", "Delta": r"\Delta",
                "Pi": r"\Pi", "Gamma": r"\Gamma", "phi": r"\phi",
                "psi": r"\psi", "rho": r"\rho", "tau": r"\tau", "xi": r"\xi"})

DSL_SPEC = """DSL grammar:
  stage:       nu : <int> ;
  term:        <Glyph> : <word> ;            e.g.  Sigma : self ;
  relation:    <a> <op> <b> ;   /* gloss */  e.g.  Omega -> Sigma ; /* genesis */
  change-of:   prefix @ on a term            e.g.  @Sigma  (the change in self)
  comment:     /* ... */
  code:        ```python ... ```   defines functions/vars in the engine itself
operators (the built-in ones):
  ->   gives rise to        |=   entails / claims
  (x)  is carried by        ?=   is it the same as
  -<   does not reduce to
(new operators or behaviours can be added by a ```python``` block.)"""


def dsl_docs(engine=None):
    """DSLDOC: self-documentation from the AST of this engine + the DSL grammar."""
    path = engine or __file__
    tree = ast.parse(open(path).read())
    funcs = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
            doc = (ast.get_docstring(n) or "").splitlines()
            funcs.append(f"  {n.name}() — {doc[0] if doc else ''}".rstrip())
    note = ("\nNote: this engine renders and regenerates the questions a stage "
            "poses; it does not adjudicate them. An evolution extends the terms, "
            "relations, or — via a ```python``` block — the engine itself.")
    return "ENGINE (introspected):\n" + "\n".join(funcs) + "\n\n" + DSL_SPEC + note


# ---- builder API for the symbolic part (compiled, sandbox-exec'd) ----
_B = {"nu": "0", "terms": {}, "rels": [], "code": []}
def reset(): _B["terms"], _B["rels"], _B["code"] = {}, [], []
def stage(n): _B["nu"] = str(int(n))
def term(g, w): _B["terms"][g] = w
def rel(a, op, b, g=None):
    if op not in OPS:
        raise ValueError(f"unknown operator {op!r}; allowed: {OPS} "
                         f"(or add one with a ```python``` block)")
    _B["rels"].append([a, op, b, g or OP_GLOSS.get(op, "")])
def _builder_dsl():
    return to_dsl({"nu": _B["nu"], **_B["terms"]}, _B["rels"], _B["code"])


def dsl_to_python(oracle: str) -> str:
    """Compile the SYMBOLIC part of an oracle to a fixed set of builder calls."""
    names, rels = _parse(oracle)
    out = ["reset()", f"stage({int(names.get('nu','0'))})"]
    for g, w in names.items():
        if g != "nu":
            out.append(f"term({g!r}, {w!r})")
    for a, op, b, g in rels:
        out.append(f"rel({a!r}, {op!r}, {b!r}, {g!r})")
    return "\n".join(out)


# ---- the mini-DSL: ```python``` blocks, gated and run in the GLOBAL namespace ----
def _defined_names(code):
    out = []
    try:
        for n in ast.parse(code).body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.append(n.name)
            elif isinstance(n, ast.Assign):
                out += [t.id for t in n.targets if isinstance(t, ast.Name)]
    except SyntaxError:
        pass
    return out


def _approve_interactive(code, defines):
    print("\n--- proposed Python (will run in the GLOBAL namespace) ---")
    print(code)
    if defines:
        print(f"--- this will define/override: {', '.join(defines)} ---")
    return input("Execute this into the running engine? [y/N]: ").strip().lower() == "y"


def activate_code(blocks, approve=_approve_interactive):
    """Print each block, ask, and exec into globals ONLY on explicit 'y'."""
    for code in blocks:
        defines = _defined_names(code)
        if approve(code, defines):
            try:
                exec(code, globals())
                print("executed.", f"now defined: {', '.join(defines)}" if defines else "")
            except Exception as e:
                print("code error (skipped, not fatal):", e)
        else:
            print("declined; kept as inert text in the document, not executed.")


def apply_oracle(oracle: str, approve=_approve_interactive) -> str:
    """Evolve: ```python``` part -> gated global exec; symbolic part -> builder."""
    try:
        existing = code_blocks(open("self.dsl").read())
    except FileNotFoundError:
        existing = []
    clean, blocks = _strip_code(oracle)
    new_blocks = [b for b in blocks if b not in existing]
    activate_code(new_blocks, approve)  # gated; only the newly-introduced code
    exec(dsl_to_python(clean), {"__builtins__": {}, "reset": reset,
                                "stage": stage, "term": term, "rel": rel})
    _B["code"] = existing + new_blocks  # accumulate, so capabilities persist
    src = _builder_dsl()
    open("self.dsl", "w").write(src)
    py = dsl_to_python(clean)
    return build_pdf(src, oracle=oracle.strip(), py=py)


def evolve():
    """Human-in-the-loop: show the engine's self-description, take an evolution."""
    src = open("self.dsl").read()
    existing = code_blocks(src)
    if existing:
        print("This document carries evolved code. Re-activate it for this session?")
        activate_code(existing)         # gated; restores prior evolutions if you allow
    print("\n" + dsl_docs() + "\n")
    print(state_summary(src))
    print("\nWrite the next evolution (DSL, optionally with ```python``` blocks).")
    print("End your input with a line containing only END:")
    while True:
        buf = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "END":
                break
            buf.append(line)
        try:
            pdf = apply_oracle("\n".join(buf))
            print("built", pdf, "— evolution recorded in the PDF."); return
        except Exception as err:
            print("error:", err, "\n-> fix and paste again (END to submit):")
