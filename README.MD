# 2D Frame Analyzer – Direct Stiffness Method (DSM-Viz)

A glass-box teaching app for the **direct stiffness method** for 2D frames. Students see every matrix the method produces, from the element stiffness matrix to the support reactions, instead of only the final answer. Uniformly distributed loads (UDL) are handled explicitly: the app shows the fixed-end forces and how they are added to the load vector and later subtracted from the member forces.

**Live app:** https://2d-frame-analysis.streamlit.app/

---

## What students see: ten steps

| Step | Screen | What it shows |
|---|---|---|
| 0 | Problem Setup | Frame drawing, node and element tables, material and section |
| 1 | DOF Numbering | Global degree-of-freedom numbers; free and fixed DOFs |
| 2 | Local [k] | 6×6 element stiffness matrix in local axes |
| 3 | Transform [T] | Transformation matrix, element angle α, cos α and sin α |
| 4 | Global [K]ₑ | Element stiffness in global axes, [T]ᵀ[k][T], and an assembly map showing which rows and columns of [K] each element fills and where each load lands in {F} |
| 5 | Assembly [K] | Global stiffness matrix (12×12 for the portal frame) and load vector {F}, with the UDL fixed-end forces |
| 6 | Partition & BCs | Reduced system [K_ff]{U_f} = {F_f} |
| 7 | Solve {U} | Nodal displacements and the condition number of [K_ff] |
| 8 | Member Forces | {f'} = [k][T]{u} − {FEF}, side by side, plus bending-moment and shear-force diagrams |
| 9 | Reactions | Support reactions and an equilibrium check |

Step 8 also has a **"Why do we subtract the FEF?"** panel. It explains the fixed-end-force correction in four pictures (lock the ends, move the load to the nodes, solve, add the locked state back) with live numbers.

## Features

- Frames with point loads at nodes and UDLs on members
- Five ready-made examples: portal frame (wind + UDL), two-span continuous beam, cantilever with tip load, cantilever with UDL, pitched roof frame
- Editable nodes, elements, supports, loads, E, A and I
- Colour-coded matrices: free DOFs highlighted, non-zero loads marked
- Exact bending-moment and shear-force diagrams, including the parabolic moment under a UDL
- Runs in a web browser; nothing to install for students
- Key results highlighted: largest displacement and rotation (Step 7), a table of the largest positive and negative bending moment in each member including span maxima (Step 8), and the largest reactions (Step 9)
- Click any step number to jump straight to it; hover over a number to see the step name
- The Run DSM Analysis button stays visible at the bottom of the sidebar, with clear messages while the analysis runs and a warning when inputs have changed

## Sign convention

| Quantity | Positive when |
|---|---|
| Axial force N | tension |
| Bending moment M | sagging (tension on the local −y face) |
| Shear force V | V = dM/dx |
| UDL w | acting in the local +y direction (so a downward load on a left-to-right beam is negative, for example w = −10 kN/m) |
| End moments in force vectors | counter-clockwise |

Bending-moment diagrams are drawn on the tension side of each member.

## Checked against hand calculations

| Example | App result | Hand result |
|---|---|---|
| Cantilever, L = 5 m, w = −15 kN/m | M at support = −187.5 kN·m, V = +75.0 kN | wL²/2 = 187.5, wL = 75 |
| Cantilever, L = 4 m, tip load 80 kN | M at support = −320.0 kN·m, V = +80.0 kN | PL = 320, P = 80 |
| Two-span continuous beam | Support moment −78.125 kN·m | Three-moment equation: −78.125 |
| Portal frame, beam E2 | End moments −6.46 and −38.43 kN·m; peak sagging 23.97 kN·m at x = 2.47 m | Peak below wL²/8 = 45; shear difference V_i − V_j = 60 kN = wL |
| Propped cantilever (explainer panel), w = −10, L = 6 | M at fixed end = 45.0 kN·m; reactions 37.5 and 22.5 kN | wL²/8 = 45; 5wL/8 = 37.5; 3wL/8 = 22.5 |

All five examples also pass the global equilibrium check in Step 9 (ΣFx, ΣFy and ΣM below 10⁻¹²).

## Run it on your own computer

```bash
git clone https://github.com/mdaipayan/2D-Frame-Analysis-Ver-1.0.git
cd 2D-Frame-Analysis-Ver-1.0
pip install -r requirements.txt
streamlit run supplement_app2_anonymized_fixed.py
```

The app opens in your browser at http://localhost:8501.

Tested with Python 3.13, Streamlit 1.67, NumPy 2.5, pandas 3.0 and Matplotlib 3.11. The minimum versions are in `requirements.txt`.

## Files

| File | Purpose |
|---|---|
| `supplement_app2_anonymized_fixed.py` | The whole app: solver, plots and user interface |
| `requirements.txt` | Python packages needed |
| `LICENSE` | GNU General Public License v3.0 |

## Using it in class

1. Solve a small frame by hand first.
2. Load the same frame in the app and go through Steps 0 to 9, comparing each matrix with your own working.
3. On Step 8, check that you subtracted the fixed-end forces. This is the step where most mistakes happen.
4. Change one input (for example w or I) and see which matrices change.

## Limitations

- 2D frames only (no 3D, no grillages)
- Linear elastic, small displacements (no P-Δ or nonlinear analysis)
- Loads: nodal point loads and member UDLs only (no point loads within a member, no temperature or settlement loads)
- One E, A and I for all members

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
