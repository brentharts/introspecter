"""
introspecter5.py — engine (v5): symbolic weights — trained networks written as visible glyphs.

Extends a parent PDF by DOI (full look-back), and adds over the question-only
engine: a hypergraph (edges over many nodes), fuzzy(Greek)+precise node pairs,
and gated ACTIONS whose output is recorded as DATA beside a question, never as
an answer to it. Self-contained: rides inside each PDF as an embedded stream.
"""
from __future__ import annotations
import ast, base64, glob, os, re, subprocess
import numpy as np

PARENT_DOI = "10.5281/zenodo.20719465"   # self_8.pdf

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
    """Compile a node's module (or reconstruct it from saved glyphs), fit it to a
    toy objective, and write the trained weights to the appendix as glyphs."""
    for s, _, _ in nodes:
        spec = p.get("modules", {}).get(s)
        if spec:
            rng = np.random.default_rng(1)
            saved = p.get("weights", {}).get(s)
            reconstructed = bool(saved)
            m = _dec_net(*saved) if saved else compile_module(spec)
            X = rng.random((64, m["in"]))
            Y = X[:, :m["out"]] if m["out"] <= m["in"] else rng.random((64, m["out"]))
            l0 = float(np.mean((forward(m, X)[0] - Y)**2))
            lf = train(m, X, Y, epochs=400, lr=0.6)
            enc = _enc_net(m)
            err = float(np.abs(_net_flat(m) - _net_flat(_dec_net(*enc))).max())
            if len(enc[5]) // 2 <= 32:           # keep it small; tiny nets only
                TRAINED[s] = enc
            return {"node": s, "arch": spec["arch"], "task":
                    "autoencode" if m["out"] == m["in"] else "fit",
                    "mse_start": round(l0, 4), "mse_final": round(lf, 4),
                    "weights_from_symbols": reconstructed,
                    "params": len(enc[5]) // 2,
                    "symbolic_roundtrip_err": round(err, 6),
                    "is": "an optimization result over a defined toy objective, with the "
                          "trained weights written to the appendix as glyphs; not the "
                          "self learning what it is"}
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


# ---- symbolic weights: a trained network, written as VISIBLE glyph fractions ----
# A weight in [0,1] is stored as a fraction h/c of an Egyptian-hieroglyph numerator
# over a cuneiform-sign denominator (after the user's symnet idea). The runtime
# reconstructs the network by decoding the glyphs from the appendix; nothing is a
# binary blob. The appendix looks like esoteric mathematics; it is a quantized
# weight matrix --- the symbols ARE the numbers, not meaning.
HIERO, CUNEI, WGRID = 0x13000, 0x12000, 512
TRAINED = {}     # node -> (i,h,o,wmin,wmax,glyphs) : freshly trained, to persist

def _w_components(val):
    val = max(0.0, min(1.0, float(val))); best = (0, 1); err = 9.9
    for c in range(1, WGRID):
        h = round(val * c)
        if 0 <= h < WGRID:
            e = abs(val - h / c)
            if e < err: err = e; best = (h, c)
            if e == 0: break
    return best

def _net_flat(m):
    return np.concatenate([m["W1"].ravel(), m["b1"], m["W2"].ravel(), m["b2"]])

def _enc_net(m):
    flat = _net_flat(m); wmin, wmax = float(flat.min()), float(flat.max())
    rng = (wmax - wmin) or 1.0
    glyphs = "".join(chr(HIERO + hc[0]) + chr(CUNEI + hc[1])
                     for hc in (_w_components((v - wmin) / rng) for v in flat))
    return m["in"], m["hidden"], m["out"], round(wmin, 6), round(wmax, 6), glyphs

def _dec_net(i, h, o, wmin, wmax, glyphs):
    vals = [(ord(glyphs[k]) - HIERO) / (ord(glyphs[k + 1]) - CUNEI)
            for k in range(0, len(glyphs), 2)]
    a = np.array(vals) * (wmax - wmin) + wmin
    n1, n2, n3 = h * i, h, o * h
    return {"W1": a[:n1].reshape(h, i), "b1": a[n1:n1 + n2],
            "W2": a[n1 + n2:n1 + n2 + n3].reshape(o, h), "b2": a[n1 + n2 + n3:],
            "in": i, "out": o, "hidden": h}


# ---- self-reading: open the lineage PDFs, extract their DSL, build a corpus ----
# Drawn from the "self-research" idea of viXra:2507.0104, kept honest and safe:
# the document searches its OWN published lineage and reports where each term
# entered and what was said about it. This is provenance retrieval recorded as
# data; it is not self-understanding, and it downloads nothing.
def read_lineage(pdf_paths):
    from pypdf import PdfReader
    corpus = {"terms": {}, "relations": [], "stages": []}
    for path in pdf_paths:                       # in stage order, ancestor-first
        try:
            att = PdfReader(path).attachments["self.dsl"]
            dsl = (att[0] if isinstance(att, list) else att).decode()
        except Exception:
            continue
        p = parse(dsl); stage = p["meta"].get("nu", "?")
        corpus["stages"].append(stage)
        for sym, (word, precise) in p["nodes"].items():
            t = corpus["terms"].setdefault(
                sym, {"word": word, "first": stage, "precise": "", "stages": []})
            t["word"] = word
            if precise: t["precise"] = precise
            if stage not in t["stages"]: t["stages"].append(stage)
        for a, op, b, gloss in p["rels"]:
            corpus["relations"].append({"stage": stage, "a": a.lstrip("@"),
                                        "op": op, "b": b.lstrip("@"), "gloss": gloss})
    return corpus

def lineage_search(corpus, query):
    q = query.lower(); hits = []
    for r in corpus["relations"]:
        blob = f"{r['a']} {r['b']} {r['gloss']}".lower()
        if q in blob:
            hits.append((r["stage"], f"{r['a']} {r['op']} {r['b']}  /* {r['gloss']} */"))
    return hits

def act_provenance(nodes, p):
    """Report, as data, where each node's term entered the lineage."""
    corpus = p.get("lineage")
    if not corpus: return {"note": "no lineage loaded"}
    out = {}
    for s, _, _ in nodes:
        t = corpus["terms"].get(s)
        if t:
            rels = [r for r in corpus["relations"] if s in (r["a"], r["b"])]
            out[s] = {"word": t["word"], "entered": "stage " + str(t["first"]),
                      "seen_in": t["stages"], "n_relations": len(rels)}
    return {"provenance": out,
            "is": "a citation of the lineage; where terms entered, not what they mean"}

ACTIONS.update({"provenance": act_provenance})


# ---- circuits: an equation that compiles into MANY networks, one per symbol,
# wired into a composite. e.g.  circuit Sigma = Phi -> Lambda -> mu : 5
# creates a trainable net for each of Phi, Lambda, mu and chains them. The
# numbers it returns are data; the symbols' meanings do not thereby combine.
def _square_net(w, seed):
    return compile_module({"dims": [w, w, w], "act": "sigmoid"}, seed)

def act_circuit(nodes, p):
    circs = p.get("circuits", [])
    if not circs: return {"note": "no circuit equation defined"}
    rng = np.random.default_rng(0); out = {}
    for c in circs:
        w = c["width"]; X = rng.random((48, w)); nets = {}
        for i, s in enumerate(c["chain"]):                 # one trained net per symbol
            m = _square_net(w, i); loss = train(m, X, X, epochs=200, lr=0.5)
            nets[s] = (m, round(loss, 4))
        x = np.ones((1, w))                                # compose along the chain
        for s in c["chain"]: x, _ = forward(nets[s][0], x)
        out[c["result"]] = {
            "wiring": " -> ".join(c["chain"]),
            "networks": {s: {"arch": f"{w}->{w}->{w}", "trained_mse": nets[s][1]}
                         for s in c["chain"]},
            "n_networks": len(c["chain"]),
            "composite_output_norm": round(float(np.linalg.norm(x)), 3)}
    return {"circuit": out,
            "is": "several trained networks wired by the equation; the numbers are "
                  "data, not the symbols' meanings combining"}

ACTIONS.update({"circuit": act_circuit})


# ---- fuzzy logic: connectives, axioms, and laws (large equations) ----
# Each term has a membership mu in [0,1] taken from the relation geometry
# (normalized degree centrality, the same M as the manifold). A law evaluates to
# a number in [0,1] -- the degree to which an authored postulate holds under
# authored memberships -- recorded as data, never as a verdict about a self.
FUZZY_TEX = {"&": r"\otimes", "|": r"\oplus", "(-)": r"\ominus",
             "=>": r"\Rightarrow", "<=>": r"\Longleftrightarrow",
             "<=": r"\Longleftarrow", "=": r"="}

def _ftok(s):
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == "\x00":
            j = s.index("\x00", i + 1); toks.append(("str", s[i:j + 1])); i = j + 1; continue
        hit = False
        for op in ("<=>", "=>", "<=", "(-)"):
            if s.startswith(op, i): toks.append(("op", op)); i += len(op); hit = True; break
        if hit: continue
        if c in "&|~()=:": toks.append(("op", c)); i += 1
        elif c.isalnum() or c == "_":
            j = i
            while j < n and (s[j].isalnum() or s[j] == "_"): j += 1
            w = s[i:j]; toks.append(("kw" if w in ("over", "under") else "term", w)); i = j
        else: i += 1

    return toks

class _FP:                                            # recursive-descent parser
    def __init__(self, toks, store): self.t, self.store, self.i = toks, store, 0
    def _peek(self): return self.t[self.i] if self.i < len(self.t) else (None, None)
    def _eat(self): tok = self.t[self.i]; self.i += 1; return tok
    def expr(self):
        a = self._orx(); k, v = self._peek()
        if k == "op" and v in ("<=>", "=>", "<=", "="):
            self._eat(); return ("rel", v, a, self._orx())
        return a
    def _orx(self):
        a = self._andx()
        while self._peek() == ("op", "|") or self._peek() == ("op", "(-)"):
            _, v = self._eat(); a = ("bin", v, a, self._andx())
        return a
    def _andx(self):
        a = self._un()
        while self._peek() == ("op", "&"):
            self._eat(); a = ("bin", "&", a, self._un())
        return a
    def _un(self):
        if self._peek() == ("op", "~"): self._eat(); return ("not", self._un())
        return self._atom()
    def _atom(self):
        k, v = self._eat()
        if k == "kw":                                 # over( E : "label" ) / under(...)
            self._eat(); inner = self.expr(); self._eat()        # ( ... :
            lbl = self.store.get(self._eat()[1], '""')[1:-1]; self._eat()  # str )
            return (v, inner, lbl)
        if k == "op" and v == "(":
            inner = self.expr(); self._eat(); return ("group", inner)
        return ("term", v)

def _parse_fuzzy(exprstr, store):
    return _FP(_ftok(exprstr), store).expr()

def _feval(node, mem):
    t = node[0]
    if t == "term": return mem.get(node[1], 0.5)
    if t in ("group", "over", "under"): return _feval(node[1], mem)
    if t == "not": return 1.0 - _feval(node[1], mem)
    a = _feval(node[2], mem); b = _feval(node[3], mem); op = node[1]
    if op == "&":  return min(a, b)
    if op == "|":  return max(a, b)
    if op == "(-)": return max(0.0, a - b)
    if op == "=>": return min(1.0, 1.0 - a + b)        # Lukasiewicz
    if op == "<=": return min(1.0, 1.0 - b + a)
    if op in ("<=>", "="): return 1.0 - abs(a - b)
    return 0.0

def _txesc(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(a, b)
    return s

def _frender(node):
    t = node[0]
    if t == "term": return VAR_TEX.get(node[1], r"\mathrm{%s}" % node[1])
    if t == "group": return "(" + _frender(node[1]) + ")"
    if t == "over":  return r"\overbrace{%s}^{\text{%s}}" % (_frender(node[1]), _txesc(node[2]))
    if t == "under": return r"\underbrace{%s}_{\text{%s}}" % (_frender(node[1]), _txesc(node[2]))
    if t == "not":   return r"\neg " + _frender(node[1])
    return _frender(node[2]) + " " + FUZZY_TEX[node[1]] + " " + _frender(node[3])

def _memberships(p):
    deg = {}
    def bump(a, b):
        a, b = a.lstrip("@"), b.lstrip("@")
        if a != b: deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
    lin = p.get("lineage")
    if lin:
        for r in lin["relations"]: bump(r["a"], r["b"])
    for a, op, b, _ in p["rels"]: bump(a, b)
    for syms, q, act in p["hyper"]:
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)): bump(syms[i], syms[j])
    if not deg: return {}
    mx = max(deg.values())
    return {k: round(v / mx, 2) for k, v in deg.items()}

def act_law(nodes, p):
    laws = p.get("laws", [])
    if not laws: return {"note": "no law defined"}
    mem = _memberships(p)
    return {"law": {name: round(_feval(ast, mem), 3) for name, ast in laws},
            "memberships": mem,
            "is": "the degree to which each authored law holds under degree-derived "
                  "memberships; numbers in [0,1], not truths about a self"}

ACTIONS.update({"law": act_law})


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
    meta, nodes, rels, hyper, code, modules, circuits = {"nu": "0"}, {}, [], [], [], {}, []
    axioms, laws = [], []
    weights = {}
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
        if stmt.startswith("circuit"):                   # equation wiring many nets
            cm = re.match(r"circuit([A-Za-z]+)=([A-Za-z>\-]+):(\d+)", stmt)
            if cm:
                chain = [x for x in re.split(r"->", cm.group(2)) if x]
                circuits.append({"result": cm.group(1), "chain": chain,
                                 "width": int(cm.group(3))})
            last = "decl"; continue
        if stmt.startswith("weights"):                  # symbolic network weights
            parts = stmt[len("weights"):].split(":")
            if len(parts) >= 4:
                try:
                    d = [int(x) for x in parts[1].split(",")]
                    mn, mx = [float(x) for x in parts[2].split(",")]
                    weights[parts[0]] = (d[0], d[1], d[2], mn, mx, parts[3])
                except Exception: pass
            last = "decl"; continue
        if stmt.startswith("fuzzy"):                     # membership source
            meta["fuzzy"] = stmt.split(":", 1)[1] if ":" in stmt else "degree"
            last = "decl"; continue
        if stmt.startswith("axiom"):                     # authored postulate
            body = stmt[len("axiom"):]
            if ":" in body:
                name, e = body.split(":", 1)
                try: axioms.append((name, _parse_fuzzy(e, store)))
                except Exception: pass
            last = "decl"; continue
        if stmt.startswith("law"):                       # large fuzzy equation
            body = stmt[len("law"):]
            if ":" in body:
                name, e = body.split(":", 1)
                try: laws.append((name, _parse_fuzzy(e, store)))
                except Exception: pass
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
            "code": code, "modules": modules, "circuits": circuits,
            "axioms": axioms, "laws": laws, "weights": weights}


def to_questions(p):
    qs = [OP_Q[op].format(a=_glyph(a), b=_glyph(b)) for a, op, b, _ in p["rels"]]
    qs += [q for _, q, _ in p["hyper"]]
    return qs


# ---- introspective metadata ----
def _funcs(path):
    return [n.name for n in ast.parse(open(path).read()).body
            if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]

# the title is generated from the engine's PUBLIC function names; rename a
# function (or hide it behind a leading underscore) and the title mutates.
TITLE_PRETTY = {
    "read_lineage": "Reading the Lineage", "lineage_search": "Searching the Lineage",
    "compile_module": "Compiling Modules", "to_questions": "Posing Questions",
    "run_actions": "Enacting", "build_pdf": "Self-Printing", "evolve": "Evolving",
    "act_provenance": "Provenance", "act_coherence": "Coherence",
    "act_train": "Training a Self", "act_neural": "A Neural Response",
    "act_infer": "Inference", "abstract": "Abstracting", "parse": "Parsing",
    "forward": "A Forward Pass", "train": "Gradient Descent", "title": "Naming Itself",
    "build_tex": "Typesetting", "act_circuit": "Wiring Networks", "act_law": "Weighing the Laws",
}
def _pretty_fn(n):
    if n in TITLE_PRETTY: return TITLE_PRETTY[n]
    parts = n.split("_")
    if parts and parts[0] in ("act", "to", "is", "do", "get"): parts = parts[1:]
    return " ".join(w.capitalize() for w in parts) or n.capitalize()

def title(p, engine=None):
    import random
    pretty, seen = [], set()
    for f in _funcs(engine or __file__):
        pp = _pretty_fn(f)
        if pp not in seen: seen.add(pp); pretty.append(pp)
    random.Random(int(p["meta"].get("nu", "0"))).shuffle(pretty)
    head = pretty[:3] or ["Inquiry"]
    sub = pretty[3:5] or pretty[:1]
    h = (", ".join(head[:-1]) + ", and " + head[-1]) if len(head) > 1 else head[0]
    s = (" and ".join(sub)) if sub else "A Self-Evolving Inquiry"
    return f"{h}: {s}"

def abstract(p, engine=None):
    n = p["meta"].get("nu", "0")
    mods = p.get("modules", {})
    mtext = (f" Of these, {len(mods)} compile via @dslai into small trainable "
             f"modules ({', '.join(mods)}), making the hypergraph partly "
             f"differentiable; a module computes, it does not comprehend." if mods else "")
    lin = p.get("lineage")
    ltext = (f" It also reads its own published lineage --- opening "
             f"{len(lin['stages'])} ancestor stages and tracing where each term "
             f"entered --- and records that provenance as data, not as "
             f"self-understanding." if lin and lin.get("stages") else "")
    circ = p.get("circuits", [])
    ctext = (f" A circuit equation compiles {sum(len(c['chain']) for c in circ)} "
             f"trainable networks --- one bound to each symbol --- and wires them into "
             f"{len(circ)} composite(s); the equation builds the networks, and the "
             f"numbers they return are data, not the symbols' meanings combining." if circ else "")
    nax, nlaw = len(p.get("axioms", [])), len(p.get("laws", []))
    ftext = (f" It posits {nax} fuzzy axioms and {nlaw} large laws over the terms, "
             f"with memberships taken from the relation geometry; each law evaluates "
             f"to a number in [0,1] recorded as data --- the degree to which an "
             f"authored postulate holds, not a truth about a self." if nlaw else "")
    return (f"Stage {n} of a self-evolving hypergraph, extended from its parent "
            f"(doi:{p['meta'].get('parent', PARENT_DOI)}) so the inquiry can look "
            f"fully back. It declares {len(p['nodes'])} nodes, each pairing a "
            f"suggestive Greek glyph with a precise description.{mtext}{ltext}{ctext}{ftext} It "
            f"relates them by {len(p['rels'])} binary relations and {len(p['hyper'])} "
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
cand = None
for f in glob.glob("*.pdf"):
    try: a = PdfReader(f).attachments
    except Exception: continue
    if "introspecter5.py" in a and (cand is None or os.path.getmtime(f) > os.path.getmtime(cand)):
        cand = f
pdf = cand or max(glob.glob("*.pdf"), key=os.path.getmtime)
for name, data in PdfReader(pdf).attachments.items():
    open(name, "wb").write(data[0] if isinstance(data, list) else data)
import importlib, introspecter5 as I; importlib.reload(I); I.evolve()
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
    lin = p.get("lineage")
    if lin and lin["stages"]:
        traced = ", ".join(rf"$ {VAR_TEX.get(s,s)} $ ({t['word']}, from stage {t['first']})"
                           for s, t in list(lin["terms"].items())[:5])
        lineage_blk = (rf"This stage read {len(lin['stages'])} ancestor stages "
                       rf"(\,{', '.join(str(s) for s in lin['stages'])}\,) from their "
                       rf"embedded sources, tracing {len(lin['terms'])} terms and "
                       rf"{len(lin['relations'])} relations across the lineage. "
                       rf"For example: {traced}. This is provenance retrieval --- "
                       rf"where terms entered and what was said --- recorded as data, "
                       rf"not a claim to understand the lineage.")
    else:
        lineage_blk = r"(no ancestor PDFs were present to read.)"
    circ = p.get("circuits", [])
    if circ:
        eqs = []
        for c in circ:
            chain = r" \longrightarrow ".join(VAR_TEX.get(s, s) for s in c["chain"])
            eqs.append(rf"{VAR_TEX.get(c['result'], c['result'])} \;=\; {chain} "
                       rf"\qquad (\text{{width }} {c['width']})")
        body = r"\\[4pt] ".join(eqs)
        circuit_blk = (r"\section*{Circuit (a new equation that wires several networks)}"
                       r"Beyond the inherited relations, this stage adds an equation that "
                       r"binds a small trainable network to \emph{each} symbol and composes "
                       r"them along the arrow:"
                       r"\begin{equation*}" + body + r"\end{equation*}"
                       + (rf"Each such equation compiles to {sum(len(c['chain']) for c in circ)} "
                          r"trainable networks (one per symbol), wired into a composite. "
                          r"Building the networks is what the equation does; the training "
                          r"losses and outputs the composite returns are recorded as data, "
                          r"not the symbols' meanings combining."))
    else:
        circuit_blk = ""
    tex = TEX
    for tok, val in [("@@TITLE@@", _esc(title(p))), ("@@N@@", p["meta"].get("nu","0")),
                     ("@@PARENT@@", _esc(p["meta"].get("parent", PARENT_DOI))),
                     ("@@ABSTRACT@@", _esc(abstract(p))), ("@@MATH@@", math),
                     ("@@TERMS@@", terms), ("@@HYPER@@", "\n".join(hyper) or r"\item (none yet)"),
                     ("@@CIRCUIT@@", circuit_blk),
                     ("@@LAWS@@", _laws_block(p)),
                     ("@@WEIGHTS@@", _weights_block(p)),
                     ("@@LINEAGE@@", lineage_blk),
                     ("@@MANIFOLD@@", _manifold_block()),
                     ("@@QUESTIONS@@", qs), ("@@BOOT@@", _bootstrap())]:
        tex = tex.replace(tok, val)
    return tex

def _laws_block(p):
    ax, laws = p.get("axioms", []), p.get("laws", [])
    if not ax and not laws: return ""
    mem = _memberships(p)
    out = [r"\section*{Axioms and laws (fuzzy logic over the terms)}"]
    if mem:
        ms = r",\;\; ".join(rf"\mu({VAR_TEX.get(k, k)}){{=}}{v:.2f}"
                            for k, v in sorted(mem.items(), key=lambda kv: -kv[1]))
        out.append(r"Memberships are taken from the relation geometry, "
                   r"$\mu(\sigma)=d(\sigma)/\max_\tau d(\tau)$:"
                   rf"\begin{{equation*}}{ms}\end{{equation*}}")
    if ax:
        items = "".join(rf"\item \textsc{{{n}}}: $\,{_frender(a)}\,$" for n, a in ax)
        out.append(r"\noindent\textbf{Axioms} (authored postulates, not verified "
                   rf"truths):\begin{{itemize}}{items}\end{{itemize}}")
    if laws:
        out.append(r"\noindent\textbf{Laws.} Each large equation brings several "
                   r"terms together; the braces draw the reading. Each evaluates to "
                   r"a number in $[0,1]$ --- recorded as data, not as a verdict.")
        for n, a in laws:
            out.append(rf"\begin{{equation*}}{_frender(a)}\end{{equation*}}")
        ev = r",\;\; ".join(rf"\textsc{{{n}}}\rightsquigarrow {round(_feval(a, mem), 3):.3f}"
                            for n, a in laws)
        out.append(r"Under the memberships above, the laws evaluate to"
                   rf"\begin{{equation*}}{ev}\end{{equation*}}"
                   r"\noindent degrees to which authored postulates hold under "
                   r"authored memberships --- data, not truths about a self.")
    return "\n".join(out)

def _weights_block(p):
    W = p.get("weights", {})
    if not W: return ""
    out = [r"\section*{Symbolic weights (the trained network, written as glyphs)}",
           r"Each trained weight is stored as a fraction $h/c$ --- an Egyptian "
           r"hieroglyph over a cuneiform sign --- normalized to the recorded range. "
           r"The runtime reconstructs the network by decoding these glyphs; no binary "
           r"blob is attached. It looks like esoteric mathematics; it is a quantized "
           r"weight matrix, and the symbols are the numbers, not meaning."]
    for node, (i, h, o, mn, mx, glyphs) in W.items():
        fr = r"\;".join(r"\frac{\glyph{%s}}{\cunei{%s}}" % (glyphs[k], glyphs[k + 1])
                        for k in range(0, len(glyphs), 2))
        out.append(rf"\medskip\noindent ${VAR_TEX.get(node, node)}$-module "
                   rf"(${i}{{\to}}{h}{{\to}}{o}$), {len(glyphs)//2} weights in "
                   rf"range $[{mn},\,{mx}]$:")
        out.append(r"\[ " + fr + r" \]")
    return "\n".join(out)

def _manifold_block():
    import json
    if not os.path.isfile("manifold.png"):
        return ""
    method = "UMAP"
    try: method = json.load(open("manifold.json")).get("method", "UMAP")
    except Exception: pass
    return (r"\section*{Manifold (the geometry of the relations)}" "\n"
            r"\begin{figure}[H]\centering"
            r"\includegraphics[width=0.92\textwidth]{manifold.png}" "\n"
            r"\caption{A projection of the relation co-occurrence across the lineage. "
            r"Left: how often terms share a relation (\(\Sigma\) is densest). Right: a "
            + method + r" projection of the same geometry. This is a map of the authored "
            r"structure --- where the terms sit relative to one another --- not a "
            r"discovery of hidden meaning. Computed by an external prototype "
            r"(\texttt{umap\_manifold.py}); the engine only embeds it.}\end{figure}")

def build_pdf(src=None, results=None, lineage=None):
    src = src or open("self.dsl").read()
    # write any freshly trained weights back into the source as visible glyph lines
    for node, (i, h, o, mn, mx, glyphs) in TRAINED.items():
        src = re.sub(r"(?m)^\s*weights %s\s*:.*$" % node, "", src)
        src = src.rstrip() + "\n\nweights %s : %d,%d,%d : %s,%s : %s ;\n" % (
            node, i, h, o, mn, mx, glyphs)
    p = parse(src); results = results or {}; p["lineage"] = lineage
    n = p["meta"].get("nu", "0"); base = f"self_{n}"
    open(base + ".tex", "w").write(build_tex(p, results))
    for _ in range(2):
        subprocess.run(["xelatex", "-interaction=nonstopmode", base + ".tex"],
                       capture_output=True)
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter(); w.append(PdfReader(base + ".pdf"))
    w.add_attachment("introspecter5.py", open(__file__, "rb").read())
    w.add_attachment("self.dsl", src.encode())
    for extra in ("manifold.json", "manifold.png"):
        if os.path.isfile(extra):
            w.add_attachment(extra, open(extra, "rb").read())
    with open(base + ".pdf", "wb") as f: w.write(f)
    return base + ".pdf"

def evolve():
    import glob
    src = open("self.dsl").read(); p = parse(src)
    # read the local lineage (ancestor self_*.pdf), if present
    here = sorted(g for g in glob.glob("self_*.pdf"))
    p["lineage"] = read_lineage(here) if here else None
    if p["lineage"] and p["lineage"]["stages"]:
        print(f"read lineage: stages {p['lineage']['stages']}, "
              f"{len(p['lineage']['terms'])} terms traced")
    for c in p["code"]:
        print("\n[carried code]\n" + c)
        if input("re-activate into the engine? [y/N]: ").strip().lower() == "y":
            try: exec(c, globals())
            except Exception as e: print("skip:", e)
    results = {}
    run_actions(p, lambda q, a: input(f"run action '{a}'? [y/N]: ").strip().lower()=="y", results)
    print("\n" + abstract(p))
    print("\nEdit self.dsl to extend the hypergraph, then run build_pdf(lineage=...).")

TEX = r"""\documentclass[11pt]{article}
\usepackage{fontspec}\setmonofont{DejaVu Sans Mono}[Scale=0.8]
\usepackage{amsmath,amssymb}\usepackage[margin=1in]{geometry}
\usepackage{fancyvrb}\usepackage[hidelinks]{hyperref}\usepackage{enumitem}
\usepackage{graphicx}\usepackage{float}
\usepackage{newunicodechar}
\newunicodechar{μ}{\ensuremath{\mu}}
\newunicodechar{∂}{\ensuremath{\partial}}
\newunicodechar{Τ}{\ensuremath{\mathrm{T}}}
\newunicodechar{ρ}{\ensuremath{\rho}}
\newunicodechar{Ψ}{\ensuremath{\Psi}}
\setlist{nosep,leftmargin=1.4em}
\newcommand{\Tau}{\mathrm{T}}
\IfFontExistsTF{Noto Sans Egyptian Hieroglyphs}{\newfontfamily{\egyptian}{Noto Sans Egyptian Hieroglyphs}}{\newcommand{\egyptian}{}}
\IfFontExistsTF{Noto Sans Cuneiform}{\newfontfamily{\cuneiform}{Noto Sans Cuneiform}}{\newcommand{\cuneiform}{}}
\newcommand{\glyph}[1]{\text{\egyptian #1}}
\newcommand{\cunei}[1]{\text{\cuneiform #1}}
\title{@@TITLE@@}\author{Brent Hartshorn\\ \texttt{brenthartshorn@proton.me}}
\date{Stage @@N@@ \quad (parent: doi:@@PARENT@@)}
\begin{document}\maketitle
\begin{abstract}@@ABSTRACT@@\end{abstract}
\section*{Symbolic core (inherited relations)}@@MATH@@
@@CIRCUIT@@
\section*{Nodes (suggestive glyph + precise description)}
\begin{description}@@TERMS@@\end{description}
\section*{Hypergraph (questions, and the data actions returned)}
\begin{itemize}@@HYPER@@\end{itemize}
@@LAWS@@
\section*{Questions this stage poses}\begin{itemize}@@QUESTIONS@@\end{itemize}
\section*{Lineage read (this stage searched its ancestors)}@@LINEAGE@@
@@MANIFOLD@@
@@WEIGHTS@@
\section*{Lineage and provenance}
This model extends its parent self-evolving inquiry, doi:@@PARENT@@, ingesting
its relations so the series can look fully back \cite{p6,paper1,paper2,p3}. The
companion papers set out the method and its discipline \cite{paper1,paper2}.
Actions record data beside questions; the document poses and enacts, and makes
no claim to resolve what a self is.
\section*{Appendix: self-extraction}
This PDF carries its engine (\texttt{introspecter5.py}) and DSL
(\texttt{self.dsl}) as embedded streams. Requires \texttt{pip install pypdf
numpy} and a LaTeX install. To produce the next stage, save and run:
\begin{Verbatim}[fontsize=\small,frame=single]
@@BOOT@@
\end{Verbatim}
\begin{thebibliography}{9}
\bibitem{p6} B.~Hartshorn, \emph{Origin, Self, Language, Meaning, Other, Time,
and Difference: A Self-Evolving Hypergraph} (self\_6, parent).
doi:10.5281/zenodo.20710258
\bibitem{paper1} B.~Hartshorn, \emph{Staging the Question of Self: A
self-contained, self-modifying document that mixes poetry, philosophy, and
code}. doi:10.5281/zenodo.20709754
\bibitem{paper2} B.~Hartshorn, \emph{Enacting the Question of Self: Provenance,
a hypergraph, and a differentiable route that records computation as data}.
doi:10.5281/zenodo.20710444
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
