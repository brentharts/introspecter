"""
umap_manifold.py — standalone prototype (heavy deps live HERE, outside the PDF).

It reads the DSL and its lineage, builds the relation co-occurrence geometry,
projects it to 2D with UMAP, and captures BOTH the data (manifold.json) and a
plot (manifold.png) for the engine to embed. The picture is a map of the
authored relation structure, not a discovery about selfhood.
"""
import sys, json, glob
sys.path.insert(0, ".")
import numpy as np
import introspecter4 as I


def geometry(dsl_path="self.dsl"):
    p = I.parse(open(dsl_path).read())
    anc = sorted(glob.glob("self_*.pdf"))
    corpus = I.read_lineage(anc) if anc else {"terms": {}, "relations": []}
    nodes = list(dict.fromkeys(list(p["nodes"]) + list(corpus["terms"])))
    idx = {n: i for i, n in enumerate(nodes)}
    M = np.zeros((len(nodes), len(nodes)))
    def bump(a, b):
        if a in idx and b in idx and a != b:
            M[idx[a], idx[b]] += 1; M[idx[b], idx[a]] += 1
    for r in corpus["relations"]: bump(r["a"], r["b"])
    for a, op, b, _ in p["rels"]: bump(a.lstrip("@"), b.lstrip("@"))
    for syms, q, act in p["hyper"]:
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)): bump(syms[i], syms[j])
    return nodes, [I._glyph(n) for n in nodes], M


def embed(M):
    n = M.shape[0]
    try:
        import umap
        red = umap.UMAP(n_components=2, n_neighbors=min(n - 1, 5),
                        min_dist=0.3, random_state=42, metric="cosine")
        return np.asarray(red.fit_transform(M + 1e-6)), "UMAP"
    except Exception as e:
        print("UMAP unavailable, SVD fallback:", e)
        Mc = M - M.mean(0); U, S, _ = np.linalg.svd(Mc)
        return U[:, :2] * S[:2], "SVD"


def main():
    nodes, glyphs, M = geometry()
    coords, method = embed(M)
    json.dump({"labels": nodes, "glyphs": glyphs, "coords": coords.tolist(),
               "heatmap": M.tolist(), "method": method}, open("manifold.json", "w"))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    im = ax[0].imshow(M, cmap="magma")
    ax[0].set_xticks(range(len(glyphs))); ax[0].set_xticklabels(glyphs, fontsize=12)
    ax[0].set_yticks(range(len(glyphs))); ax[0].set_yticklabels(glyphs, fontsize=12)
    ax[0].set_title("relation co-occurrence (the geometry)", fontsize=10)
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    ax[1].scatter(coords[:, 0], coords[:, 1], s=12, color="#333")
    for i, g in enumerate(glyphs):
        ax[1].annotate(g, (coords[i, 0], coords[i, 1]), fontsize=14,
                       ha="center", va="center")
    ax[1].set_title("%s projection of the same" % method, fontsize=10)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    plt.tight_layout(); plt.savefig("manifold.png", dpi=130)
    print("wrote manifold.json + manifold.png  (%s, %d nodes)" % (method, len(nodes)))


if __name__ == "__main__":
    main()
