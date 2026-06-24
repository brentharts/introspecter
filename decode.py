#!/usr/bin/env python3
"""
decode.py -- rebuild introspecter8.py from engine.tex, byte-for-byte.

It reads (a) the appendix word-dictionary table and (b) the invisible %ILINE
comment lines that carry the compressed source, maps every glyph back -- newline
range -> '\n', indent range -> four spaces, keyword/word symbols -> their text --
and concatenates the blocks in order. Source text only; nothing is executed.
"""
import re, sys

KW = {"def":"ƒ","return":"↦","if":"⊃","elif":"⊆","else":"⊂","for":"∀","while":"↻",
      "in":"∈","not":"¬","and":"∧","or":"∨","is":"≡","None":"⊘","True":"⊤","False":"⊥",
      "lambda":"λ"}
NL0, NLN = 0x2580, 0x2600
ID0, IDN = 0x2500, 0x2580
is_nl = lambda c: NL0 <= ord(c) < NLN
is_id = lambda c: ID0 <= ord(c) < IDN

ILINE = re.compile(r"^%ILINE(\d+):(.*)$", re.M)
DICTROW = re.compile(r"\\g\{(.)\}\s*&\s*\\verb\|([^|]*)\|")


def load_sym2word(tex):
    s2w = {sym: kw for kw, sym in KW.items()}          # keyword glyphs (fixed)
    for sym, word in DICTROW.findall(tex):             # word dictionary (from appendix)
        s2w[sym] = word
    return s2w


def recreate(tex):
    s2w = load_sym2word(tex)
    blocks = sorted(((int(i), body) for i, body in ILINE.findall(tex)), key=lambda b: b[0])
    out = []
    for _, body in blocks:
        out.append("".join("\n" if is_nl(c) else "    " if is_id(c) else s2w.get(c, c)
                           for c in body))
    return "".join(out)


def read_payload(tex, n=12):
    blocks = sorted(((int(i), body) for i, body in ILINE.findall(tex)), key=lambda b: b[0])
    vals = []
    for _, body in blocks:
        for c in body:
            if is_nl(c):
                vals.append(ord(c) - NL0)
                if len(vals) >= n:
                    return bytes(vals)
    return bytes(vals)


if __name__ == "__main__":
    tex_path = sys.argv[1] if len(sys.argv) > 1 else "engine.tex"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "introspecter8_from_tex.py"
    tex = open(tex_path, encoding="utf-8").read()
    src = recreate(tex)
    open(out_path, "w", encoding="utf-8").write(src)
    print("recreated %d bytes -> %s" % (len(src.encode()), out_path))
    print("newline-channel payload (NN weight-bytes):", list(read_payload(tex)))
