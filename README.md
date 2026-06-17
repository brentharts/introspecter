# introspecter

**A self-contained, self-modifying PDF that stages, enacts, and measures the question of self — and records every computation as data, never as an answer.**

`introspecter` is a single Python engine that rides *inside* a PDF as an embedded
stream. Each PDF is a "stage": it renders an evolving inquiry into selfhood as
typeset mathematics, carries its own engine and source, and — when run — extracts
itself and produces the next stage. The series is written in a small
domain-specific language (DSL) that mixes Greek symbols, natural language, and
Python.

The project has one rule, and everything else serves it:

> **Record computation as data, never as an answer.**

The apparatus can *pose* questions about the self, *enact* computations beside
them, and *measure* their structure. It cannot, and does not, demonstrate that a
self emerged. Every number it produces — a training loss, a graph degree, a fuzzy
"truth value" — is rendered as what it is: a computed quantity over authored
inputs. Naming the authored input behind each result is what keeps the work
honest.

---

## What it is

- **Self-contained.** The engine, the DSL source, and (from stage 7) the manifold
  plot and data all travel inside each stage PDF as embedded file streams. A short
  base-64 appendix bootstraps extraction.
- **Self-modifying.** Running a stage extracts the engine and produces the next
  stage's PDF. The lineage is a chain of permanently archived, citable artifacts.
- **Honest by construction.** Suggestive Greek glyphs are paired with precise
  descriptions, so interpretive slack is labelled rather than hidden. All code
  execution is gated behind explicit human confirmation. Nothing is downloaded.

---

## How it works

```
self.dsl ──parse──▶ builder calls (sandboxed) ──▶ LaTeX ──xelatex──▶ PDF
   ▲                                                                  │
   │                                          embeds: engine, dsl,    │
   │                                          manifold.json/png       │
   └──────── next stage ◀── human gate (y/N) ◀── base-64 bootstrap ◀──┘
```

Each archived PDF is a parent named by DOI. A stage extends its parent by reading
the parent's relations back in, so the inquiry can "look fully back" along the
whole published lineage.

---

## The DSL

The DSL is ASCII-first so it survives PDF text extraction. Source lines are
terminated by `;`, comments are `/* ... */`, and quoted `"..."` strings hold
precise descriptions and labels.

### Nodes — a suggestive glyph, a word, and a precise description

```
Sigma : self : "the locus that poses and is altered by the question" ;
```

### Relations and hyperedges

```
Omega -> Sigma ;                 /* a binary relation, with operator -> */
{ Sigma, Lambda, mu } ? "what survives being carried into language and meaning?" ! coherence ;
```

A hyperedge `{ ... } ? "question" ! action ;` joins many nodes, poses a question,
and may trigger a gated **action** whose output is recorded as data.

### Differentiable modules and circuits

```
module Sigma : 3 -> 8 -> 3 : sigmoid ;     /* compiles a node into a trainable MLP */
circuit Sigma = Phi -> Lambda -> mu : 5 ;  /* one trained network per symbol, composed */
```

### Fuzzy axioms and laws (stage 8)

Memberships come from the relation geometry (normalized degree centrality).
Connectives are triangular norms: `&` = min, `|` = max, `~` = 1−x,
`=>` = Łukasiewicz implication, `<=>` = equivalence, `(-)` = bounded difference.

```
fuzzy : degree ;
axiom mediation : Sigma => Lambda ;          /* an authored postulate, not a verified truth */

law constitution :
  Sigma <=> over( Omega | Phi    : "given: by origin and by the other" )
          & under( Lambda => mu  : "reached: through language, toward meaning" ) ;
```

`over( E : "label" )` and `under( E : "label" )` render as overbrace/underbrace.
A `law` evaluates to a number in `[0,1]` — the degree to which an authored
postulate holds under authored memberships, recorded as **data, not a verdict**.

---

## Engine versions

| Engine | Adds |
|---|---|
| `introspecter2.py` | hypergraph (n-ary edges); fuzzy glyph + precise description pairs; gated actions; `@dslai` route — a node compiles into a real trainable NumPy MLP |
| `introspecter3.py` | lineage reading & a provenance corpus (reads its own ancestor PDFs); the UMAP manifold; the circuit equation (multiple networks wired to symbols); a title generated from the engine's own public function names |
| `introspecter4.py` | fuzzy logic — triangular-norm connectives, `axiom` and `law` constructs, large equations with overbrace/underbrace, memberships from graph degree |

The latest engine is the one carried inside the most recent stage PDF.

---

## Running it

Requirements: Python 3, `numpy`, `pypdf`, and a LaTeX install with `xelatex`
(plus `fontspec` and DejaVu fonts). The UMAP prototype additionally needs
`umap-learn` and `matplotlib`.

```bash
pip install numpy pypdf
```

Build a stage from the current `self.dsl`:

```python
import introspecter4 as I
src = open("self.dsl").read()
results = {}
I.run_actions(I.parse(src), lambda q, a: input(f"run '{a}'? [y/N] ") == "y", results)
I.build_pdf(src, results)
```

Or evolve from an existing stage PDF (it self-extracts, reads any ancestor
`self_*.pdf` present, and gates every action):

```bash
python3 -c "import introspecter4 as I; print(I._bootstrap())" > boot.py
python3 boot.py
```

### The UMAP manifold (runs outside the PDF)

`umap_manifold.py` is a standalone prototype: it reads the DSL and lineage,
builds the relation co-occurrence geometry, projects it to 2D with UMAP, and
writes `manifold.json` (data) and `manifold.png` (a heatmap + projection) for the
engine to embed. The heavy dependency lives here so the engine stays light.

```bash
python3 umap_manifold.py
```

The manifold is a map of the **authored** relation structure — not a discovery of
latent meaning.

---

## Safety

The apparatus executes code and trains networks; all of it is gated. Generated or
evolved code is inert text until a human confirms it with `y`. The lineage reader
opens only the ancestor PDFs already on disk and downloads nothing. The project
deliberately declines patterns from some self-modifying systems — removing the
permission prompt so a model's output triggers `eval`/`exec` directly, or
auto-downloading and following references. This is a personal instrument for a
single trusted user and is **not** intended for untrusted input.


## Evolution Test 1 ##

Origin, Self, Language, and Meaning: A Self-Evolving Inquiry
- self_0.pdf https://doi.org/10.5281/zenodo.20707685

Origin, Self, Language, Meaning, and Other: A Self-Evolving Inquiry
- self_1.pdf https://doi.org/10.5281/zenodo.20707727

Origin, Self, Language, Meaning, Other, and Time: A Self-Evolving Inquiry
- self_2.pdf https://doi.org/10.5281/zenodo.20707775

Origin, Self, Language, Meaning, Other, Time, and Difference: A Self-Evolving Inquiry
- self_3.pdf https://doi.org/10.5281/zenodo.20707850

Origin, Self, Language, Meaning, Other, Time, and Difference: A Self-Evolving Hypergraph
- self_5.pdf https://doi.org/10.5281/zenodo.20709560

Origin, Self, Language, Meaning, Other, Time, and Difference: A Self-Evolving Hypergraph - Stage6
- self_6.pdf https://doi.org/10.5281/zenodo.20710258

Parsing, Compiling Modules, and Evolving: Self-Printing and Reading the Lineage
- self_7.pdf https://doi.org/10.5281/zenodo.20710811

Evolving, Naming Itself, and Typesetting: Enacting and Provenance
- self_8.pdf https://doi.org/10.5281/zenodo.20719465

## Papers ##
Staging the Question of Self: A self-contained, self-modifying document that mixes poetry, philosophy, and code
- https://doi.org/10.5281/zenodo.20709754
- https://ai.vixra.org/abs/2606.0042

Enacting the Question of Self: Provenance, a hypergraph, and a differentiable route that records computation as data
- https://doi.org/10.5281/zenodo.20710444



