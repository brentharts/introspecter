#!/usr/bin/env python3
"""Non-interactive build of self_10.pdf from self.dsl and ancestor PDFs."""
import glob
import introspecter6 as I

src = open("self.dsl").read()
p = I.parse(src)
here = sorted(glob.glob("self_*.pdf"))
p["lineage"] = I.read_lineage(here) if here else None
results = {}
I.run_actions(p, lambda q, a: True, results)
print(I.abstract(p))
print("building PDF...")
print("built:", I.build_pdf(src, results, p.get("lineage")))
