import os
import subprocess
import re
import numpy as np

# ==========================================
# 1. UNICODE CODEC CONFIGURATION
# ==========================================
HIERO_START = 0x13000
CUNEI_START = 0x12000
GRID_SIZE = 1024

def float_to_glyphs(val):
    """
    Clamps a float to [0, 1] and finds the closest fraction Hiero / Cunei.
    Returns the LaTeX fraction string.
    """
    val = max(0.0, min(1.0, float(val)))
    
    best_h, best_c = 0, 1
    min_err = float('inf')
    
    for c in range(1, GRID_SIZE):
        h = int(round(val * c))
        if 0 <= h < GRID_SIZE:
            err = abs(val - (h / c))
            if err < min_err:
                min_err = err
                best_h, best_c = h, c
            if min_err == 0:
                break
                
    hiero_char = chr(HIERO_START + best_h)
    cunei_char = chr(CUNEI_START + best_c)
    
    # FIX: Ensure math mode handles text fonts properly by keeping \glyph and \cunei safe
    return "$\\frac{\\glyph{%s}}{\\cunei{%s}}$" % (hiero_char, cunei_char)

def glyphs_to_float(latex_str):
    """
    Parses a LaTeX fraction back into a float.
    """
    pattern = r"\\frac\{\\glyph\{([^}]+)\}\}\{\\cunei\{([^}]+)\}\}"
    match = re.search(pattern, latex_str)
    if not match:
        print("PARSE ERROR", latex_str)
        return 0.0
    
    hiero_char = match.group(1)
    cunei_char = match.group(2)
    
    h_val = ord(hiero_char) - HIERO_START
    c_val = ord(cunei_char) - CUNEI_START
    
    return h_val / c_val

# ==========================================
# 2. MODEL SKETCH
# ==========================================
np.random.seed(42)
layer1 = np.random.uniform(-1, 1, (3, 4))
layer2 = np.random.uniform(-1, 1, (4, 2))

l1_min, l1_max = layer1.min(), layer1.max()
l2_min, l2_max = layer2.min(), layer2.max()

l1_norm = (layer1 - l1_min) / (l1_max - l1_min)
l2_norm = (layer2 - l2_min) / (l2_max - l2_min)

# ==========================================
# 3. GENERATE LATEX DOCUMENT
# ==========================================
def generate_latex_table(matrix, name):
    rows, cols = matrix.shape
    tex = f"\\subsection*{{Layer: {name}}}\n"
    tex += "\\begin{tabular}{|%s} \n \\hline" % ('c|' * cols)
    
    encoded_matrix = []
    for r in range(rows):
        row_tex = []
        encoded_row = []
        for c in range(cols):
            glyph_tex = float_to_glyphs(matrix[r, c])
            row_tex.append(glyph_tex)
            encoded_row.append(glyph_tex)
        tex += " & ".join(row_tex) + " \\\\\n\\hline\n"
        encoded_matrix.append(encoded_row)
        
    tex += "\\end{tabular}\n\\vspace{0.5cm}\n"
    return tex, encoded_matrix

tex_l1, encoded_l1 = generate_latex_table(l1_norm, "network-layer-1")
tex_l2, encoded_l2 = generate_latex_table(l2_norm, "network-layer-2")

# FIX: Added \text{...} wrappers to commands so math engines let fontspec render unicode
latex_document = f"""\\documentclass{{article}}
\\usepackage{{fontspec}}
\\usepackage{{amsmath}}
\\usepackage{{geometry}}
\\geometry{{a4paper, margin=1in}}

% Set up fonts for both systems
\\newfontfamily{{\\egyptian}}{{Noto Sans Egyptian Hieroglyphs}}
\\newfontfamily{{\\cuneiform}}{{Noto Sans Cuneiform}}

\\newcommand{{\\glyph}}[1]{{\\text{{\\egyptian #1}}}}
\\newcommand{{\\cunei}}[1]{{\\text{{\\cuneiform #1}}}}

\\begin{{document}}

\\title{{Esoteric Lossy Neural Network Weights Storage}}
\\author{{AI Prototype}}
\\date{{\\today}}
\\maketitle

\\section*{{Encoded Network Weights}}
This document contains the serialized floating-point weight matrices of a 3-layer neural network, represented via mathematical fractions of ancient scripts.

{tex_l1}
{tex_l2}

\\end{{document}}
"""

# Print template to terminal
print("================ GENERATED LATEX TEMPLATE ================")
print(latex_document)
print("==========================================================\n")

# ==========================================
# 4. WRITE FILE & COMPILE VIA LUALATEX
# ==========================================
tex_path = "/tmp/test.tex"
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_document)

print(f"LaTeX file saved to {tex_path}")
print("Compiling with lualatex...")

subprocess.check_call(["lualatex", "-output-directory=/tmp", tex_path])
print("Success! PDF generated successfully at /tmp/test.pdf")

# ==========================================
# 5. DECODE BACK TO FLOATS (Verification)
# ==========================================
print("\n================ VERIFICATION & DECODING ================")

def decode_matrix(encoded_matrix, v_min, v_max):
    rows = len(encoded_matrix)
    cols = len(encoded_matrix[0])
    decoded = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            norm_val = glyphs_to_float(encoded_matrix[r][c])
            decoded[r, c] = norm_val * (v_max - v_min) + v_min
    return decoded

decoded_l1 = decode_matrix(encoded_l1, l1_min, l1_max)

print("Original Layer 1 Weights:")
print(layer1)
print("\nDecoded Layer 1 Weights (After Lossy Roundtrip):")
print(decoded_l1)
print(r"Mean Absolute Quantization Error: ", np.mean(np.abs(layer1 - decoded_l1)))