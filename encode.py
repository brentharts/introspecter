#!/usr/bin/env python3
"""
encode.py -- render introspecter8.py as a compact system of mathematical equations,
while storing a byte-exact compressed copy that decode.py rebuilds the file from.

TWO THINGS LIVE IN engine.tex:
  * VISIBLE: each function as a guarded equation -- if/elif/else as a \\begin{cases},
    for as forall with an underbraced body, assignments as <-, returns as |->.
    Leaves are dictionary-compressed and truncated, so the equations stay small.
  * INVISIBLE: one comment line per block, '%ILINE<i>:<condensed>', carrying the
    lossless compression (words->symbols, keywords->glyphs, newlines/indents->glyph
    ranges). The blocks tile the source exactly; decode.py reads these comments and
    the appendix dictionary and rebuilds introspecter8.py byte-for-byte.
Nothing is executed.
"""
import ast, re, keyword, numpy as np
from collections import Counter

ENGINE = "introspecter8.py"
SRC = open(ENGINE, encoding="utf-8").read()
N_DICT = 90

KW = {"def":"ƒ","return":"↦","if":"⊃","elif":"⊆","else":"⊂","for":"∀","while":"↻",
      "in":"∈","not":"¬","and":"∧","or":"∨","is":"≡","None":"⊘","True":"⊤","False":"⊥",
      "lambda":"λ"}
NL0, NLN = 0x2580, 0x2600
ID0, IDN = 0x2500, 0x2580
is_nl = lambda c: NL0 <= ord(c) < NLN
is_id = lambda c: ID0 <= ord(c) < IDN
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_w = np.tanh(np.random.default_rng(0).standard_normal(12))
PAYLOAD = bytes(int(round((x*0.5+0.5)*127)) & 0x7F for x in _w)
def nl_glyph(i): return chr(NL0 + (PAYLOAD[i] if i < len(PAYLOAD) else (i & 0x7F)))
def id_glyph(k): return chr(ID0 + (k % (IDN-ID0)))


def build_dict():
    cnt = Counter(WORD.findall(SRC))
    cand = [(w, c) for w, c in cnt.items()
            if len(w) >= 2 and not keyword.iskeyword(w) and w not in KW]
    cand.sort(key=lambda wc: (-wc[1]*(len(wc[0])-1), wc[0]))
    pool = [chr(c) for c in range(0x0400, 0x0460) if chr(c) not in SRC]
    return {w: pool[i] for i, (w, _) in enumerate(cand[:min(N_DICT, len(pool))])}

DICT = build_dict()
SUB = {**KW, **DICT}
SYM2WORD = {sym: word for word, sym in SUB.items()}


def check_symbols():
    s = set(SRC)
    for word, sym in SUB.items():
        assert sym not in s, "collision: %s->%s" % (word, sym)
    assert not any(is_nl(c) or is_id(c) for c in SRC)

def word_substitute(src):
    return WORD.sub(lambda m: SUB.get(m.group(0), m.group(0)), src)

_nlc = [0]
def ws_encode(txt):
    parts = []
    for line in txt.split("\n"):
        i = 0
        while i < len(line) and line[i] == " ": i += 1
        q, r = divmod(i, 4)
        parts.append("".join(id_glyph(j) for j in range(q)) + " " * r + line[i:])
    out = parts[0]
    for p in parts[1:]:
        out += nl_glyph(_nlc[0]); _nlc[0] += 1; out += p
    return out

def decode_line(line):
    return "".join("\n" if is_nl(c) else "    " if is_id(c) else SYM2WORD.get(c, c)
                   for c in line)

def partition(elines):
    tree = ast.parse(SRC); chunks = []; cur = 1
    for n in tree.body:
        chunks.append((cur, n.end_lineno, n)); cur = n.end_lineno + 1
    if cur <= len(elines):
        s, e, n = chunks[-1]; chunks[-1] = (s, len(elines), n)
    return chunks


# ============ math rendering (the visible, compressed equations) ============
def esc(s):
    for a, b in [("\\", r"{\textbackslash}"), ("{", r"\{"), ("}", r"\}"), ("$", r"\$"),
                 ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("^", r"\textasciicircum{}"),
                 ("~", r"\textasciitilde{}")]:
        s = s.replace(a, b)
    return s

def comp(text):
    return word_substitute(text)

def W(node_or_text, maxlen=54):
    t = node_or_text if isinstance(node_or_text, str) else ast.unparse(node_or_text)
    t = " ".join(t.split())
    c = comp(t)
    if len(c) > maxlen:
        c = c[:maxlen-1] + "…"
    return r"\w{%s}" % esc(c)

def body_src(stmts, maxlen=64):
    return W("; ".join(ast.unparse(s) for s in stmts), maxlen)


def collect_cases(stmts):
    """Return list of (rhs, cond_or_None) if stmts are a pure if/return dispatch."""
    rows = []
    def walk(seq):
        for s in seq:
            if isinstance(s, ast.Return):
                rows.append((W(s.value) if s.value else r"\bot", None))
            elif isinstance(s, ast.If) and len(s.body) == 1 and isinstance(s.body[0], ast.Return):
                rows.append((W(s.body[0].value) if s.body[0].value else r"\bot", W(s.test)))
                if s.orelse:
                    if not walk(s.orelse): return False
            else:
                return False
        return True
    return rows if walk(stmts) else None


def render_stmt(s):
    if isinstance(s, ast.Assign):
        return r"%s \leftarrow %s" % (W(", ".join(ast.unparse(t) for t in s.targets)), W(s.value))
    if isinstance(s, ast.AnnAssign):
        return r"%s \leftarrow %s" % (W(s.target), W(s.value) if s.value else r"\cdot")
    if isinstance(s, ast.AugAssign):
        op = {ast.Add:"+",ast.Sub:"-",ast.Mult:r"\cdot"}.get(type(s.op), r"\circ")
        return r"%s \mathrel{%s}\!= %s" % (W(s.target), op, W(s.value))
    if isinstance(s, ast.Return):
        return r"\mapsto %s" % (W(s.value) if s.value else "")
    if isinstance(s, ast.For):
        return r"\forall\,%s\!\in\!%s\ \underbrace{%s}_{}" % (W(s.target,18), W(s.iter,24), body_src(s.body))
    if isinstance(s, ast.While):
        return r"\circlearrowright %s\ \underbrace{%s}_{}" % (W(s.test,18), body_src(s.body))
    if isinstance(s, ast.If):
        out = r"\overbrace{%s}^{\,\supset %s}" % (body_src(s.body), W(s.test,28))
        if s.orelse:
            out += r"\,\underbrace{%s}_{\subset}" % body_src(s.orelse)
        return out
    if isinstance(s, ast.With):
        items = ", ".join(ast.unparse(i) for i in s.items)
        return r"\vdash %s\ \underbrace{%s}_{}" % (W(items,24), body_src(s.body))
    if isinstance(s, ast.Try):
        return r"\rotatebox{90}{$\hookrightarrow$}\,\underbrace{%s}_{}" % body_src(s.body)
    if isinstance(s, (ast.Import, ast.ImportFrom)):
        return W(s, 60)
    if isinstance(s, ast.FunctionDef):
        return r"\mathnormal{f}\,%s\ \underbrace{%s}_{}" % (W(s.name,16), body_src(s.body))
    if isinstance(s, ast.Raise):
        return r"\Uparrow %s" % (W(s.exc) if s.exc else "")
    if isinstance(s, ast.Expr):
        return W(s.value)
    if isinstance(s, ast.Pass): return r"\square"
    if isinstance(s, ast.Break): return r"\boxtimes"
    if isinstance(s, ast.Continue): return r"\boxdot"
    return W(s)


def render_function(fn):
    name = fn.name.replace("_", r"\_")
    args = esc(", ".join(a.arg for a in fn.args.args))
    head = r"\mathsf{%s}(\mathit{%s})" % (name, args)
    # split leading simple assignments
    lets, rest = [], list(fn.body)
    while rest and isinstance(rest[0], ast.Assign) and all(isinstance(t, ast.Name) for t in rest[0].targets):
        a = rest.pop(0); lets.append((W(", ".join(ast.unparse(t) for t in a.targets)), W(a.value)))
    rows = collect_cases(rest)
    if rows is not None and rows:
        cases = r"\\[1pt]".join((r"%s & \text{if } %s" % (r, c) if c else r"%s & \text{otherwise}" % r)
                                for r, c in rows)
        body = r"\begin{cases}%s\end{cases}" % cases
        where = (r"\;\text{where}\ " + r",\ ".join("%s = %s" % (t, v) for t, v in lets)) if lets else ""
        return r"%s = %s%s" % (head, body, where)
    # otherwise: aligned sequence, one statement per line
    seq = [render_stmt(s) for s in fn.body]
    aligned = r"\\".join("&" + r for r in seq if r)
    return r"%s =\begin{aligned}[t]\\[-1.2\baselineskip]%s\end{aligned}" % (head, aligned)


def render_module(node):
    if isinstance(node, ast.Assign) and node.targets:
        return r"%s \leftarrow %s" % (W(", ".join(ast.unparse(t) for t in node.targets), 22), W(node.value, 60))
    return W(node, 78)


def il_block(i, line):
    return f"%ILINE{i}:{line}"


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage{fontspec}
\newfontfamily{\eng}{DejaVu Sans Mono}[Scale=0.78]
\usepackage{amsmath,amssymb}
\usepackage[margin=0.75in]{geometry}
\usepackage{booktabs,longtable,array}
\usepackage{graphicx}
\usepackage{tikz}\usetikzlibrary{positioning,arrows.meta,fit,backgrounds}
\usepackage{xcolor}\usepackage[hidelinks]{hyperref}
\newcommand{\g}[1]{{\eng #1}}
\newcommand{\w}[1]{\text{\eng #1}}
\allowdisplaybreaks
\setlength{\parindent}{0pt}\setlength{\parskip}{1pt}
\setlength{\abovedisplayskip}{2pt}\setlength{\belowdisplayskip}{2pt}
\setlength{\abovedisplayshortskip}{1pt}\setlength{\belowdisplayshortskip}{1pt}
\title{\textbf{introspecter8, as a system of equations}\\[2pt]
\large The whole engine compressed to guarded equations --- losslessly decodable}
\author{}\date{}
"""


def kw_legend():
    cells = [(sym, kw) for kw, sym in KW.items()]
    body = "".join(" & ".join(r"\g{%s} & \texttt{%s}" % (s, esc(k)) for s, k in cells[a:a+4]) + r" \\" + "\n"
                   for a in range(0, len(cells), 4))
    return r"\begin{center}\small\begin{tabular}{" + "cl"*4 + r"}\toprule" + "\n" + body + r"\bottomrule\end{tabular}\end{center}"

def dict_appendix():
    items = sorted(DICT.items(), key=lambda kv: kv[1])
    per = 5
    body = "".join(" & ".join(r"\g{%s} & \verb|%s|" % (sym, w) for w, sym in items[a:a+per]) + r" \\" + "\n"
                   for a in range(0, len(items), per))
    return (r"\section*{Appendix: word dictionary (read by the decoder)}" + "\n"
            r"The %d most space-saving words each map to one symbol, substituted as whole words "
            r"(so they shrink the equations and the strings alike). \texttt{decode.py} reads this "
            r"very table to expand them." % len(items) + "\n"
            r"\begin{center}\scriptsize\begin{tabular}{" + "ll"*per + r"}\toprule" + "\n"
            + body + r"\bottomrule\end{tabular}\end{center}")

def architecture():
    return (r"""\begin{center}\begin{tikzpicture}[font=\footnotesize,node distance=4.5mm,
  box/.style={draw,rounded corners,fill=blue!7,minimum height=6.5mm,inner sep=2.5pt},
  ar/.style={-{Stealth[length=2mm]}}]
\node[box](parse){parse};\node[box,right=of parse](runs){run\_actions};
\node[box,right=of runs](tex){build\_tex};\node[box,right=of tex](pdf){build\_pdf};
\node[box,right=of pdf](evo){evolve};
\draw[ar](parse)--(runs);\draw[ar](runs)--(tex);\draw[ar](tex)--(pdf);\draw[ar](pdf)--(evo);
\draw[ar](evo.south) to[out=-90,in=-90](parse.south);
\node[below=7mm of tex,box,fill=gray!8,align=center](act){16 action procedures
$\mapsto$ coherence, neural, train, infer, provenance, circuit,\\
geometry, jacobian, filtration, spectrum, oracle, search, fetch, ingest, propose, law};
\draw[ar](runs.south)--(act.north);\end{tikzpicture}\end{center}""")

def dispatch_eq(chunks):
    pairs = []
    for _, _, node in chunks:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ACTIONS" \
           and isinstance(node.value, ast.Dict):
            pairs += [(k.value, v.id) for k, v in zip(node.value.keys, node.value.values)
                      if isinstance(k, ast.Constant) and isinstance(v, ast.Name)]
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            c = node.value
            if isinstance(c.func, ast.Attribute) and c.func.attr == "update" \
               and getattr(c.func.value, "id", "") == "ACTIONS" and c.args and isinstance(c.args[0], ast.Dict):
                pairs += [(k.value, v.id) for k, v in zip(c.args[0].keys, c.args[0].values)
                          if isinstance(k, ast.Constant) and isinstance(v, ast.Name)]
    rows = r"\\".join(r"\mathtt{%s} & \mapsto & \mathsf{%s}" % (esc(k), v.replace("_", r"\_")) for k, v in pairs)
    return (r"\[\mathrm{evolve}=\mathrm{build\_pdf}\circ\mathrm{build\_tex}\circ\mathrm{run\_actions}"
            r"\circ\mathrm{parse},\qquad \mathrm{ACTIONS}:\ \begin{array}{lcl}" + rows + r"\end{array}\]")


def main():
    check_symbols()
    full = word_substitute(SRC)
    elines = full.splitlines(keepends=True)
    chunks = partition(elines)
    enc = [ws_encode("".join(elines[c[0]-1:c[1]])) for c in chunks]

    out = [PREAMBLE, r"\begin{document}\maketitle"]
    out.append(r"Every function below is a guarded equation; the engine's exact bytes ride in "
               r"invisible \texttt{\%ILINE} comments that \texttt{decode.py} reads to rebuild "
               r"\texttt{" + esc(ENGINE) + r"} verbatim. Frequent words are compressed to the "
               r"symbols in the appendix, keywords to math glyphs ($\forall,\lambda,\mapsto,"
               r"\dots$), and newlines/indents to glyph ranges. Nothing here is executed.")

    out.append(r"\section*{Symbol legend}"); out.append(kw_legend())
    out.append(r"\section*{The engine as a map}"); out.append(architecture()); out.append(dispatch_eq(chunks))

    out.append(r"\section*{The engine as equations}")
    for i, c in enumerate(chunks):
        node = c[2]
        out.append(il_block(i, enc[i]))            # invisible lossless store
        try:
            tex = render_function(node) if isinstance(node, ast.FunctionDef) else render_module(node)
        except Exception:
            tex = W(node, 80)
        out.append(r"\[\scriptstyle %s\]" % tex)

    out.append(dict_appendix())
    out.append(r"\end{document}")
    open("engine.tex", "w", encoding="utf-8").write("\n".join(out))

    rec = "".join(decode_line(e) for e in enc)
    assert rec == SRC, "SELF-CHECK FAILED"
    print("wrote engine.tex (%d blocks, %d functions, dict=%d) -- byte-exact ✓"
          % (len(chunks), sum(isinstance(c[2], ast.FunctionDef) for c in chunks), len(DICT)))


if __name__ == "__main__":
    main()
