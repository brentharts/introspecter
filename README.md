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

### Symbolic weights (stage 9)

A trained network's weights are written into the appendix as **visible glyphs**,
not a binary blob: each weight is a fraction `h/c` of an Egyptian hieroglyph
(numerator) over a cuneiform sign (denominator), normalized to a recorded range.
The runtime reconstructs the network by decoding the glyphs, and the engine writes
the learned weights back into the document's own embedded source so the next stage
can decode and continue. Networks are kept tiny (e.g. `2 -> 2 -> 2`, 12 weights)
so the symbolic block stays a line or two. It looks like esoteric mathematics; it
is a quantized weight matrix — the symbols are the numbers, not meaning.

```
weights Sigma : 2,2,2 : -1.199,1.627 : 𓀙𒄔𓀁𒀁𓄁𒄳 … ;   /* h/c fractions, ~4e-5 quantization */
```

---

## Engine versions

| Engine | Adds |
|---|---|
| `introspecter2.py` | hypergraph (n-ary edges); fuzzy glyph + precise description pairs; gated actions; `@dslai` route — a node compiles into a real trainable NumPy MLP |
| `introspecter3.py` | lineage reading & a provenance corpus (reads its own ancestor PDFs); the UMAP manifold; the circuit equation (multiple networks wired to symbols); a title generated from the engine's own public function names |
| `introspecter4.py` | fuzzy logic — triangular-norm connectives, `axiom` and `law` constructs, large equations with overbrace/underbrace, memberships from graph degree |
| `introspecter5.py` | symbolic weights — trained networks written into the appendix as visible hieroglyph/cuneiform fractions, reconstructed by the runtime and persisted into the document's own source |

The latest engine is the one carried inside the most recent stage PDF.

---

## Running it

Requirements: Python 3, `numpy`, `pypdf`, and a LaTeX install with `xelatex`
(plus `fontspec` and DejaVu fonts). The UMAP prototype additionally needs
`umap-learn` and `matplotlib`.

```bash
pip install numpy pypdf
# fonts: Noto Sans Egyptian Hieroglyphs + Cuneiform (for symbolic weights)
#   Debian/Ubuntu: sudo apt-get install fonts-noto-core
```

Build a stage from the current `self.dsl`:

```python
import introspecter5 as I
src = open("self.dsl").read()
results = {}
I.run_actions(I.parse(src), lambda q, a: input(f"run '{a}'? [y/N] ") == "y", results)
I.build_pdf(src, results)
```

Or evolve from an existing stage PDF (it self-extracts, reads any ancestor
`self_*.pdf` present, and gates every action):

```bash
python3 -c "import introspecter5 as I; print(I._bootstrap())" > boot.py
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

---

## The published lineage

Stages (self-extracting PDFs, each extends the previous):

| Stage | Title | DOI |
|---|---|---|
| self_0 | Origin, Self, Language, and Meaning | [10.5281/zenodo.20707685](https://doi.org/10.5281/zenodo.20707685) |
| self_1 | …and Other | [10.5281/zenodo.20707727](https://doi.org/10.5281/zenodo.20707727) |
| self_2 | …and Time | [10.5281/zenodo.20707775](https://doi.org/10.5281/zenodo.20707775) |
| self_3 | …and Difference | [10.5281/zenodo.20707850](https://doi.org/10.5281/zenodo.20707850) |
| self_6 | …A Self-Evolving Hypergraph | [10.5281/zenodo.20710258](https://doi.org/10.5281/zenodo.20710258) |
| self_7 | Parsing, Compiling Modules, and Evolving | [10.5281/zenodo.20710811](https://doi.org/10.5281/zenodo.20710811) |
| self_8 | Evolving, Naming Itself, and Typesetting | [10.5281/zenodo.20719465](https://doi.org/10.5281/zenodo.20719465) |
| self_9 | Compiling Modules, Evolving, and Provenance: A Forward Pass and Parsing (symbolic weights) | [10.5281/zenodo.20777747](https://doi.org/10.5281/zenodo.20777747) |

## Papers

| # | Title | DOI |
|---|---|---|
| 1 | **Staging the Question of Self** — a self-contained, self-modifying document that mixes poetry, philosophy, and code | [10.5281/zenodo.20709754](https://doi.org/10.5281/zenodo.20709754) |
| 2 | **Enacting the Question of Self** — provenance, a hypergraph, and a differentiable route that records computation as data | [10.5281/zenodo.20710444](https://doi.org/10.5281/zenodo.20710444) |
| 3 | **The Question of Self** — Matrix Inscriptions and Faux-Calculus | [zenodo.20814778](https://doi.org/10.5281/zenodo.20814778) |

The three papers move staging → enacting → measuring. Paper 3 formalizes
provenance as a filtration (birth times along a monotone chain), the relations as
a weighted-graph Laplacian (with *self* as the hub), the circuit as function
composition (whose Jacobian is a matrix product), and the fuzzy laws as
triangular-norm expressions — observing that each measured quantity is a
deterministic function of authored data.

---

## Repository layout

```
introspecter2.py     engine v2 (hypergraph, modules, gated actions)
introspecter3.py     engine v3 (lineage, manifold, circuits, self-naming title)
introspecter4.py     engine v4 (fuzzy axioms and laws)
introspecter5.py     engine v5 (symbolic weights)              ← current
umap_manifold.py     standalone UMAP prototype (geometry capture)
self.dsl             the current stage seed
self_*.pdf           the published lineage (self-extracting stages)
```

> Rendering the symbolic weights needs the Noto Sans Egyptian Hieroglyphs and Noto
> Sans Cuneiform fonts (`fonts-noto-core`). The engine guards the font load with
> `\IfFontExistsTF`, so it degrades gracefully if they are absent.

---

## Citation

If you reference this work, please cite the relevant stage or paper by its DOI
above. Author: Brent Hartshorn (`brenthartshorn@proton.me`).

---

*The apparatus poses questions and enacts computations beside them. It makes no
claim to resolve what a self is.*
