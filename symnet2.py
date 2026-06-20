import os
import subprocess
import re
import numpy as np

# ==========================================
# 1. UNICODE & MATHEMATICAL CONFIGURATION
# ==========================================
HIERO_START = 0x13000
CUNEI_START = 0x12000
GRID_SIZE = 1024

def is_prime(n):
    """Helper to check primality for injecting special operators like \\oint."""
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def float_to_components(val):
    """Converts a float into the optimal 10-bit Hiero and Cunei integer indices."""
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
    return best_h, best_c

# ==========================================
# 2. CREATIVE EQUATION GENERATION ENGINE
# ==========================================
def build_esoteric_equation(matrix_row):
    """
    Transforms a row of normalized weights into a profound-looking 
    LaTeX display math statement based on internal numerical properties.
    """
    eq_elements = []
    cols = len(matrix_row)
    
    for c in range(cols):
        val = matrix_row[c]
        h_idx, c_idx = float_to_components(val)
        
        hiero_char = chr(HIERO_START + h_idx)
        cunei_char = chr(CUNEI_START + c_idx)
        
        # Core storage block
        #frac_tex = f"\\frac{{\\glyph{{{hiero_char}}}}{{\\cunei{{{cunei_char}}}}}"
        frac_tex = "\\frac{\\glyph{%s}}{\\cunei{%s}}" % (hiero_char, cunei_char)

        # Rule 1: Structural wrappers based on absolute value magnitude
        if val > 0.75:
            frac_tex = f"\\overbrace{{{frac_tex}}}^{{\\alpha}}"
        elif val < 0.25:
            frac_tex = f"\\underbrace{{{frac_tex}}}_{{\\beta}}"
            
        # Rule 2: Advanced calculus operators prefixed if components are prime numbers
        if is_prime(h_idx):
            frac_tex = f"\\oint {frac_tex}"
        elif is_prime(c_idx):
            frac_tex = f"\\sum_{{\\lambda=0}}^{{\\infty}} {frac_tex}"
            
        eq_elements.append(frac_tex)
        
        # Rule 3: Set operators or relations between sequential terms
        if c < cols - 1:
            if c == cols - 2:
                # The second-to-last transition creates an equality/statement relation
                rel_hash = (h_idx + c_idx) % 4
                if rel_hash == 0:    op = " \\equiv "
                elif rel_hash == 1:  op = " \\propto "
                elif rel_hash == 2:  op = " \\approx "
                else:                op = " \\neq "
            else:
                # Standard algebraic operators tracking the delta direction of weights
                next_val = matrix_row[c + 1]
                op = " + " if next_val >= val else " - "
                
            eq_elements.append(op)
            
    # Wrap in LaTeX display math format
    return "\\[ " + "".join(eq_elements) + " \\]"

# ==========================================
# 3. CHASSIS AND MODEL SETUP
# ==========================================
np.random.seed(42)
layer1 = np.random.uniform(-1, 1, (3, 4))
layer2 = np.random.uniform(-1, 1, (4, 2))

l1_min, l1_max = layer1.min(), layer1.max()
l2_min, l2_max = layer2.min(), layer2.max()

l1_norm = (layer1 - l1_min) / (l1_max - l1_min)
l2_norm = (layer2 - l2_min) / (l2_max - l2_min)

# Generate the visual math sections
latex_l1_statements = []
for row in l1_norm:
    latex_l1_statements.append(build_esoteric_equation(row))

latex_l2_statements = []
for row in l2_norm:
    latex_l2_statements.append(build_esoteric_equation(row))

# Construct final LuaLaTeX Document Template
latex_document = f"""\\documentclass{{article}}
\\usepackage{{fontspec}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{geometry}}
\\geometry{{a4paper, margin=1in}}

\\newfontfamily{{\\egyptian}}{{Noto Sans Egyptian Hieroglyphs}}
\\newfontfamily{{\\cuneiform}}{{Noto Sans Cuneiform}}

\\newcommand{{\\glyph}}[1]{{\\text{{\\egyptian #1}}}}
\\newcommand{{\\cunei}}[1]{{\\text{{\\cuneiform #1}}}}

\\begin{{document}}

\\title{{Esoteric Lossy Neural Network Weights Storage \\\\ \\large Pseudo-Mathematical Statement Serialization}}
\\author{{AI Prototype}}
\\date{{\\today}}
\\maketitle

\\section*{{Statement-Encoded Layers}}
The floating point values of the neural networks are encoded below. The mathematical context symbols are derived contextually from the local weight metrics.

\\subsection*{{Layer: network-layer-1}}
{"\n".join(latex_l1_statements)}

\\subsection*{{Layer: network-layer-2}}
{"\n".join(latex_l2_statements)}

\\end{{document}}
"""

print("================ GENERATED LATEX TEMPLATE ================")
print(latex_document)
print("==========================================================\n")

# ==========================================
# 4. EXPORT & LIVE COMPILATION
# ==========================================
tex_path = "/tmp/test.tex"
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_document)

print(f"LaTeX file saved to {tex_path}")
print("Compiling with lualatex...")

subprocess.check_call(["lualatex", "-output-directory=/tmp", tex_path])
print("Success! PDF generated successfully at /tmp/test.pdf")

# ==========================================
# 5. NOISE-ISOLATING DECODER ENGINE
# ==========================================
print("\n================ VERIFICATION & DECODING ================")

def decode_esoteric_layer(statement_list, cols, v_min, v_max):
    """
    Parses complex LaTeX math strings, strips away structural 
    noise, and extracts pure floating-point representations.
    """
    rows = len(statement_list)
    decoded_matrix = np.zeros((rows, cols))
    
    # Non-greedy expression capturing our core glyph components cleanly out of anything
    pattern = r"\\frac\{\\glyph\{([^}]+)\}\}\{\\cunei\{([^}]+)\}\}"
    
    for r in range(rows):
        matches = re.findall(pattern, statement_list[r])
        for c in range(cols):
            hiero_char, cunei_char = matches[c]
            
            h_val = ord(hiero_char) - HIERO_START
            c_val = ord(cunei_char) - CUNEI_START
            
            norm_val = h_val / c_val
            # Reverse min-max normalization back into raw weight ranges
            decoded_matrix[r, c] = norm_val * (v_max - v_min) + v_min
            
    return decoded_matrix

decoded_l1 = decode_esoteric_layer(latex_l1_statements, 4, l1_min, l1_max)

print("Original Layer 1 Weights:")
print(layer1)
print("\nDecoded Layer 1 Weights (Extracted right out of the faux math expressions):")
print(decoded_l1)
print(f"\nMean Absolute Quantization Error: {np.mean(np.abs(layer1 - decoded_l1)):.6e}")