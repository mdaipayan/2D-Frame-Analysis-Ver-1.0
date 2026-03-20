"""
╔══════════════════════════════════════════════════════════════════╗
║   2D Frame Analyzer — Direct Stiffness Method (DSM)              ║
║   Glass-Box Educational Tool for B.Tech / M.Tech Students        ║
║   Author  : Educational Structural Engineering Lab               ║
║   Deploy  : streamlit run dsm_2d_frame.py                        ║
║   Requires: streamlit, numpy, pandas, matplotlib                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="2D Frame Analyzer – DSM",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background:#f8fafc; }
  .stApp { background:#f8fafc; }
  section[data-testid="stSidebar"] { background:#eef2f7; border-right:1px solid #cbd5e1; }
  h1,h2,h3,h4,h5,h6 { color:#1e40af !important; font-family: 'Courier New', monospace; }
  .stDataFrame { font-family: 'Courier New', monospace; font-size:12px; }
  .step-card {
    background:#ffffff; border:1px solid #2563eb;
    border-radius:8px; padding:18px; margin-bottom:16px;
    box-shadow:0 1px 6px rgba(37,99,235,0.08);
  }
  .formula-box {
    background:#f1f5f9; border-left:4px solid #2563eb;
    padding:12px 16px; border-radius:4px;
    font-family:'Courier New',monospace; font-size:13px; color:#1e293b;
    white-space:pre-wrap; margin:10px 0;
  }
  .badge-safe   { background:#dcfce7; color:#166534; padding:3px 10px; border-radius:12px; font-size:12px; }
  .badge-warn   { background:#fef9c3; color:#854d0e; padding:3px 10px; border-radius:12px; font-size:12px; }
  .badge-fail   { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:12px; font-size:12px; }
  .step-title { font-size:18px; font-weight:700; color:#1e40af; font-family:'Courier New',monospace; }
  .step-subtitle { font-size:13px; color:#475569; margin-bottom:8px; }
  .ref-tag { font-size:11px; color:#64748b; font-style:italic; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PRESET PROBLEMS
# ══════════════════════════════════════════════════════════════════════════════
PRESETS = {
    "Portal Frame (Wind + UDL on Beam)": {
        "desc": "Fixed-base single-bay portal frame — horizontal wind load at left column top + 10 kN/m gravity UDL on beam",
        "nodes": [(0,0),(0,4),(6,4),(6,0)],
        "elements": [(0,1),(1,2),(2,3)],
        "fixed_dofs": {0:[0,1,2], 3:[0,1,2]},
        "E": 200e6, "A": 0.01, "I": 1e-4,
        "nodal_loads": {1:{0:20.0}},
        "udl_loads":   {1: -10.0},          # ← 10 kN/m downward on the beam
        "labels": ["Left Col","Beam","Right Col"],
    },
    "Two-Span Continuous Beam (UDL)": {
        "desc": "Continuous beam — 20 kN/m UDL on full first span + concentrated load on second span midpoint",
        "nodes": [(0,0),(2.5,0),(5,0),(7.5,0),(10,0)],
        "elements": [(0,1),(1,2),(2,3),(3,4)],
        "fixed_dofs": {0:[0,1], 2:[1], 4:[1]},
        "E": 200e6, "A": 0.02, "I": 2e-4,
        "nodal_loads": {3:{1:-100.0}},
        "udl_loads":   {0: -20.0, 1: -20.0},  # ← 20 kN/m on entire first span (E1+E2)
        "labels": ["Span 1a","Span 1b","Span 2a","Span 2b"],
    },
    "Cantilever Beam (Tip Load)": {
        "desc": "Fixed-free cantilever — vertical tip load",
        "nodes": [(0,0),(4,0)],
        "elements": [(0,1)],
        "fixed_dofs": {0:[0,1,2]},
        "E": 200e6, "A": 0.01, "I": 1e-4,
        "nodal_loads": {1:{1:-80.0}},
        "udl_loads":   {},
        "labels": ["Cantilever"],
    },
    "Cantilever Beam (UDL Only)": {
        "desc": "Fixed-free cantilever — 15 kN/m downward UDL only (no tip load), classic DSM UDL demo",
        "nodes": [(0,0),(5,0)],
        "elements": [(0,1)],
        "fixed_dofs": {0:[0,1,2]},
        "E": 200e6, "A": 0.01, "I": 1e-4,
        "nodal_loads": {},
        "udl_loads":   {0: -15.0},           # ← 15 kN/m downward on entire beam
        "labels": ["Cantilever"],
    },
    "Pitched Roof Frame": {
        "desc": "Symmetric gabled frame — vertical snow load on both rafters",
        "nodes": [(0,0),(0,4),(4,6),(8,4),(8,0)],
        "elements": [(0,1),(1,2),(2,3),(3,4)],
        "fixed_dofs": {0:[0,1,2], 4:[0,1,2]},
        "E": 200e6, "A": 0.015, "I": 1.5e-4,
        "nodal_loads": {1:{1:-20.0}, 2:{1:-40.0}, 3:{1:-20.0}},
        "udl_loads":   {},
        "labels": ["Left Col","Left Rafter","Right Rafter","Right Col"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  DSM SOLVER — with UDL Fixed-End Force support
# ══════════════════════════════════════════════════════════════════════════════
def run_dsm(nodes, elements, fixed_dofs, nodal_loads, E, A, I, udl_loads=None):
    """
    Parameters
    ----------
    nodes        : list of (x, y) tuples
    elements     : list of (ni, nj) node-index pairs
    fixed_dofs   : dict {node_id: [local_dof_indices]}  — 0=u,1=v,2=θ
    nodal_loads  : dict {node_id: {local_dof: value}}  — kN or kN·m
    E, A, I      : material / section (kN/m², m², m⁴)
    udl_loads    : dict {element_index: w}  — w in kN/m in local transverse (y)
                   direction.  Negative = pointing in local −y direction
                   (downward for horizontal members, i.e. gravity loads).

    UDL Implementation
    ------------------
    For each element carrying a UDL w, the Fixed-End Forces (FEF) in the
    element's LOCAL coordinate system are:

        FEF_local = [ 0,  wL/2,  wL²/12,  0,  wL/2,  −wL²/12 ]
                      axial  shear  moment  axial  shear   moment
                      (i-end)                     (j-end)

    These are transformed to global and added to the global load vector {F}.
    After solving, member end forces are corrected:

        f_local = [k_local] · [T] · {u_global_elem} − FEF_local

    Returns a dict with all intermediate matrices for each step.
    """
    if udl_loads is None:
        udl_loads = {}

    nn = len(nodes)
    ne = len(elements)
    nDOF = nn * 3  # 3 DOFs per node: [u, v, θ]

    # ── Step 1: DOF numbering ────────────────────────────────
    dof_map = {i: [3*i, 3*i+1, 3*i+2] for i in range(nn)}
    fixed_global = []
    for nid, ldofs in fixed_dofs.items():
        for ld in ldofs:
            fixed_global.append(dof_map[nid][ld])
    fixed_global = sorted(set(fixed_global))
    free_global  = [d for d in range(nDOF) if d not in fixed_global]

    # ── Step 2 & 3: Local k and Transformation T per element ─
    elem_data = []
    for idx, (ni, nj) in enumerate(elements):
        xi, yi = nodes[ni]
        xj, yj = nodes[nj]
        L  = max(math.hypot(xj-xi, yj-yi), 1e-9)
        al = math.atan2(yj-yi, xj-xi)        # angle w.r.t. x-axis
        c  = math.cos(al)
        s  = math.sin(al)

        # 6×6 local stiffness
        a  = E*A/L
        b  = 12*E*I/L**3
        cc = 6*E*I/L**2
        d  = 4*E*I/L
        e  = 2*E*I/L
        k_loc = np.array([
            [ a,  0,  0, -a,  0,  0],
            [ 0,  b, cc,  0, -b, cc],
            [ 0, cc,  d,  0,-cc,  e],
            [-a,  0,  0,  a,  0,  0],
            [ 0, -b,-cc,  0,  b,-cc],
            [ 0, cc,  e,  0,-cc,  d],
        ])

        # 6×6 transformation matrix
        lam = np.array([[c, s, 0],[-s, c, 0],[0, 0, 1]])
        T6  = np.zeros((6,6))
        T6[:3,:3] = lam
        T6[3:,3:] = lam

        # Global element stiffness
        k_glob = T6.T @ k_loc @ T6

        # Global DOF indices for this element
        gdofs = dof_map[ni] + dof_map[nj]

        elem_data.append({
            "ni": ni, "nj": nj, "L": L, "alpha_deg": math.degrees(al),
            "c": c, "s": s,
            "k_loc": k_loc, "T6": T6, "k_glob": k_glob,
            "gdofs": gdofs,
        })

    # ── Step 5: Assemble global K and F ─────────────────────
    K = np.zeros((nDOF, nDOF))
    F = np.zeros(nDOF)

    for ed in elem_data:
        gdofs = ed["gdofs"]
        for r, gr in enumerate(gdofs):
            for c2, gc in enumerate(gdofs):
                K[gr, gc] += ed["k_glob"][r, c2]

    # Nodal point loads
    for nid, ldmap in nodal_loads.items():
        for ld, val in ldmap.items():
            F[dof_map[nid][ld]] += val

    # ── UDL: compute FEF and add equivalent nodal loads to F ─
    # FEF_local = [0, wL/2, wL²/12, 0, wL/2, -wL²/12]
    # Equivalent joint loads (global) += T6^T @ FEF_local
    fef_locals = {}   # store per element for member-force recovery in Step 8
    fef_summaries = {}  # human-readable summary for Step 8 display
    for idx, ed_item in enumerate(elem_data):
        w = udl_loads.get(idx, 0.0)
        if abs(w) < 1e-12:
            fef_locals[idx] = np.zeros(6)
            continue
        L = ed_item["L"]
        fef = np.array([
            0.0,
            w * L / 2.0,
            w * L**2 / 12.0,
            0.0,
            w * L / 2.0,
           -w * L**2 / 12.0,
        ])
        fef_locals[idx] = fef
        # Transform FEF to global and scatter into F
        f_global = ed_item["T6"].T @ fef
        for r, gr in enumerate(ed_item["gdofs"]):
            F[gr] += f_global[r]
        fef_summaries[idx] = {
            "w": w, "L": L,
            "Vi": w*L/2, "Mi": w*L**2/12,
            "Vj": w*L/2, "Mj": -w*L**2/12,
            "fef_local": fef.copy(),
            "fef_global": f_global.copy(),
        }

    # ── Step 6: Partition ─────────────────────────────────────
    ff = free_global
    Kff = K[np.ix_(ff, ff)]
    Ff  = F[ff]

    # ── Step 7: Solve ─────────────────────────────────────────
    cond_num = np.linalg.cond(Kff)
    if cond_num > 1e10:
        st.warning(
            f"⚠️ Warning: Stiffness matrix is highly ill-conditioned "
            f"(κ ≈ {cond_num:.2e}). Check for disconnected elements or "
            f"extreme differences in element stiffness."
        )
    try:
        Uf = np.linalg.solve(Kff, Ff)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Singular stiffness matrix — structure is a mechanism. "
            "Check boundary conditions (insufficient supports)."
        )

    U = np.zeros(nDOF)
    for i, gi in enumerate(ff):
        U[gi] = Uf[i]

    # ── Step 8: Member end forces (local) ────────────────────
    # f_local = [k_local]·[T]·{u_global} − FEF_local
    member_results = []
    for idx, ed_item in enumerate(elem_data):
        u_el      = U[ed_item["gdofs"]]              # global displacements
        u_loc     = ed_item["T6"] @ u_el             # local displacements
        f_loc_el  = ed_item["k_loc"] @ u_loc         # elastic restoring forces
        f_loc     = f_loc_el - fef_locals[idx]       # subtract FEF → true member forces
        member_results.append({
            "u_global":   u_el,
            "u_local":    u_loc,
            "f_local_elastic": f_loc_el,             # before FEF correction (for teaching)
            "fef_local":  fef_locals[idx],           # fixed-end forces (for teaching)
            "f_local":    f_loc,                     # corrected local end forces
            "N_i": -f_loc[0], "V_i": -f_loc[1], "M_i": -f_loc[2],
            "N_j": -f_loc[3], "V_j": -f_loc[4], "M_j": -f_loc[5],
        })

    # ── Step 9: Reactions ─────────────────────────────────────
    # Reactions use the ORIGINAL F (which already includes UDL equiv. loads)
    R_full = K @ U - F
    reactions = {gi: R_full[gi] for gi in fixed_global}

    # Global moment equilibrium about origin
    sigma_M = 0.0
    for nid in fixed_dofs:
        x, y = nodes[nid]
        Rx = R_full[dof_map[nid][0]]
        Ry = R_full[dof_map[nid][1]]
        RM = R_full[dof_map[nid][2]]
        sigma_M += RM + Ry * x - Rx * y

    # Moment from nodal point loads
    for nid, ldmap in nodal_loads.items():
        x, y = nodes[nid]
        Fx = ldmap.get(0, 0.0); Fy = ldmap.get(1, 0.0); M = ldmap.get(2, 0.0)
        sigma_M += M + Fy * x - Fx * y

    # Moment from UDL resultants
    for idx, ed_item in enumerate(elem_data):
        w = udl_loads.get(idx, 0.0)
        if abs(w) < 1e-12:
            continue
        ni, nj = ed_item["ni"], ed_item["nj"]
        xi, yi = nodes[ni]; xj, yj = nodes[nj]
        L = ed_item["L"]
        # Resultant of UDL acts at midpoint of element, perpendicular to member axis
        mx, my = (xi + xj) / 2, (yi + yj) / 2
        # UDL resultant in local y → global components
        al = math.atan2(yj - yi, xj - xi)
        perp_x = -math.sin(al); perp_y = math.cos(al)
        Ry_udl = w * L * perp_y
        Rx_udl = w * L * perp_x
        sigma_M += Ry_udl * mx - Rx_udl * my

    eq_check = {
        "ΣFx": (sum(R_full[dof_map[n][0]] for n in fixed_dofs)
                + sum(v for nid, ldm in nodal_loads.items()
                      for ld, v in ldm.items() if ld == 0)
                + sum(
                    udl_loads.get(i, 0.0) * ed_item["L"] * (-math.sin(math.atan2(
                        nodes[ed_item["nj"]][1] - nodes[ed_item["ni"]][1],
                        nodes[ed_item["nj"]][0] - nodes[ed_item["ni"]][0])))
                    for i, ed_item in enumerate(elem_data)
                )),
        "ΣFy": (sum(R_full[dof_map[n][1]] for n in fixed_dofs)
                + sum(v for nid, ldm in nodal_loads.items()
                      for ld, v in ldm.items() if ld == 1)
                + sum(
                    udl_loads.get(i, 0.0) * ed_item["L"] * math.cos(math.atan2(
                        nodes[ed_item["nj"]][1] - nodes[ed_item["ni"]][1],
                        nodes[ed_item["nj"]][0] - nodes[ed_item["ni"]][0]))
                    for i, ed_item in enumerate(elem_data)
                )),
        "ΣM":  sigma_M,
    }

    return {
        "nn": nn, "ne": ne, "nDOF": nDOF,
        "dof_map": dof_map,
        "fixed_global": fixed_global,
        "free_global":  free_global,
        "elem_data":    elem_data,
        "K": K, "F": F,
        "Kff": Kff, "Ff": Ff,
        "U": U,
        "member_results":  member_results,
        "reactions":       reactions,
        "eq_check":        eq_check,
        "udl_loads":       dict(udl_loads),   # pass-through for display
        "fef_summaries":   fef_summaries,     # detailed FEF breakdowns
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MATRIX DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fmt_val(v, precision=4):
    if abs(v) < 1e-10:
        return "0"
    if abs(v) > 1e5 or (abs(v) < 0.001 and v != 0):
        return f"{v:.3e}"
    return f"{v:.{precision}f}"

def show_matrix(M, row_labels=None, col_labels=None, caption="",
                highlight_rows=None, highlight_cols=None):
    n, m = M.shape
    data = {(col_labels[j] if col_labels else f"c{j}"): [fmt_val(M[i,j]) for i in range(n)]
            for j in range(m)}
    df = pd.DataFrame(data, index=row_labels if row_labels else [f"r{i}" for i in range(n)])

    def styler(s):
        styles = pd.DataFrame("", index=s.index, columns=s.columns)
        if highlight_rows:
            for r in highlight_rows:
                if r < len(styles.index):
                    styles.iloc[r] = "background-color:#dcfce7; color:#166534"
        if highlight_cols:
            for c in highlight_cols:
                if c < len(styles.columns):
                    styles.iloc[:, c] = "background-color:#dbeafe; color:#1e40af"
        return styles

    styled = df.style.apply(styler, axis=None)
    st.dataframe(styled, width="stretch", height=min(35*n+38, 500))
    if caption:
        st.caption(caption)

def show_vector(v, labels=None, caption="", highlight=None):
    n = len(v)
    idx = labels if labels else [f"d{i}" for i in range(n)]
    data = {"Value": [fmt_val(x) for x in v]}
    df = pd.DataFrame(data, index=idx)

    def styler(s):
        styles = pd.DataFrame("", index=s.index, columns=s.columns)
        if highlight:
            for h in highlight:
                if h < len(styles.index):
                    styles.iloc[h] = "background-color:#fef9c3; color:#854d0e"
        return styles

    styled = df.style.apply(styler, axis=None)
    st.dataframe(styled, width="stretch", height=min(35*n+38, 400))
    if caption:
        st.caption(caption)


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME VISUALIZATION (Matplotlib) — with UDL arrow bands
# ══════════════════════════════════════════════════════════════════════════════
def classify_support(fixed_ldofs):
    if len(fixed_ldofs) >= 3: return "fixed"
    if 0 in fixed_ldofs and 1 in fixed_ldofs: return "pin"
    return "roller"

def draw_frame(nodes, elements, fixed_dofs, nodal_loads,
               U=None, dof_map=None, reactions=None,
               scale=0.3, show_dofs=False, show_loads=True,
               show_deformed=False, show_reactions=False,
               elem_labels=None, node_labels=True,
               udl_loads=None):
    """Draw the 2D frame with supports, loads, UDL arrows, deformed shape, reactions."""
    if udl_loads is None:
        udl_loads = {}

    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#f8fafc")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#334155")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cbd5e1")

    xs = [n[0] for n in nodes]; ys = [n[1] for n in nodes]
    span = max(max(xs)-min(xs), max(ys)-min(ys), 1.0)
    arrowsc = span * 0.08

    # ── Draw elements ──────────────────────────────────────────
    colors = ["#2563eb","#d97706","#7c3aed","#059669","#dc2626"]
    for idx, (ni, nj) in enumerate(elements):
        xi, yi = nodes[ni]; xj, yj = nodes[nj]
        col = colors[idx % len(colors)]
        ax.plot([xi, xj], [yi, yj], color=col, lw=3, solid_capstyle="round", zorder=3)
        if elem_labels:
            mx, my = (xi+xj)/2, (yi+yj)/2
            ax.text(mx, my, elem_labels[idx], color=col, fontsize=8,
                    ha="center", va="bottom", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=col, lw=0.8))

    # ── Draw UDL arrows along elements ────────────────────────
    if show_loads:
        for idx, (ni, nj) in enumerate(elements):
            w = udl_loads.get(idx, 0.0)
            if abs(w) < 1e-12:
                continue
            xi, yi = nodes[ni]; xj, yj = nodes[nj]
            L = math.hypot(xj - xi, yj - yi)
            if L < 1e-9:
                continue
            al = math.atan2(yj - yi, xj - xi)
            perp = (-math.sin(al), math.cos(al))   # local +y in global

            # Arrow length proportional to span; direction follows sign of w
            arrow_len = span * 0.055 * (1 if w > 0 else -1)
            n_arrows  = max(4, int(L / (span * 0.12)))
            t_vals    = np.linspace(0.05, 0.95, n_arrows)

            udl_color = "#f59e0b"

            # Baseline (load application line)
            bx0 = xi + 0.05*(xj-xi) - arrow_len*perp[0]
            by0 = yi + 0.05*(yj-yi) - arrow_len*perp[1]
            bx1 = xi + 0.95*(xj-xi) - arrow_len*perp[0]
            by1 = yi + 0.95*(yj-yi) - arrow_len*perp[1]
            ax.plot([bx0, bx1], [by0, by1], color=udl_color, lw=2.0, zorder=5)

            # Arrow shafts
            for t in t_vals:
                px = xi + t*(xj-xi)
                py = yi + t*(yj-yi)
                start_x = px - arrow_len * perp[0]
                start_y = py - arrow_len * perp[1]
                ax.annotate(
                    "", xy=(px, py), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle="-|>", color=udl_color, lw=1.4),
                    zorder=5,
                )

            # UDL label near midpoint
            mx = (xi + xj) / 2 - arrow_len * perp[0] * 1.6
            my = (yi + yj) / 2 - arrow_len * perp[1] * 1.6
            ax.text(mx, my, f"w={w:+.1f}\nkN/m", color=udl_color,
                    fontsize=8, ha="center", va="center",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#fffbeb",
                              ec=udl_color, lw=0.9),
                    zorder=6)

    # ── Draw deformed shape ────────────────────────────────────
    if show_deformed and U is not None and dof_map is not None:
        disp_max = max(abs(U)) if max(abs(U)) > 1e-12 else 1
        sc = scale * span / disp_max
        for ni, nj in elements:
            xi, yi = nodes[ni]; xj, yj = nodes[nj]
            ui = U[dof_map[ni][:2]]; uj = U[dof_map[nj][:2]]
            ax.plot([xi+sc*ui[0], xj+sc*uj[0]],
                    [yi+sc*ui[1], yj+sc*uj[1]],
                    color="#dc2626", lw=2, ls="--", zorder=4, alpha=0.85)

    # ── Draw nodes ────────────────────────────────────────────
    for i, (x, y) in enumerate(nodes):
        ax.scatter(x, y, s=60, color="#1e293b", zorder=6)
        if node_labels:
            ax.text(x, y+span*0.025, f"N{i+1}", color="#1e293b",
                    fontsize=8, ha="center", va="bottom",
                    fontfamily="monospace", fontweight="bold", zorder=7)

    # ── Draw supports ─────────────────────────────────────────
    for nid, ldofs in fixed_dofs.items():
        x, y = nodes[nid]
        stype = classify_support(ldofs)
        if stype == "fixed":
            ax.barh(y, -span*0.04, height=span*0.12, left=x, color="#dc2626", alpha=0.35, zorder=2)
            for dy in np.linspace(-span*0.06, span*0.06, 5):
                ax.plot([x-span*0.04, x-span*0.06],[y+dy, y+dy+span*0.015],
                        color="#dc2626", lw=1, alpha=0.7)
        elif stype == "pin":
            tri = plt.Polygon([[x,y],[x-span*0.03,y-span*0.06],[x+span*0.03,y-span*0.06]],
                              closed=True, color="#16a34a", alpha=0.5, zorder=2)
            ax.add_patch(tri)
            ax.plot([x-span*0.05, x+span*0.05],[y-span*0.065, y-span*0.065],
                    color="#16a34a", lw=2)
        else:
            tri = plt.Polygon([[x,y],[x-span*0.03,y-span*0.05],[x+span*0.03,y-span*0.05]],
                              closed=True, color="#2563eb", alpha=0.35, zorder=2)
            ax.add_patch(tri)
            circle = plt.Circle((x, y-span*0.07), span*0.022, color="#2563eb", alpha=0.4, zorder=2)
            ax.add_patch(circle)

    # ── Draw point loads ──────────────────────────────────────
    if show_loads:
        for nid, ldmap in nodal_loads.items():
            x, y = nodes[nid]
            for ld, val in ldmap.items():
                if ld == 0:
                    dx = arrowsc * (1 if val > 0 else -1)
                    ax.annotate("", xy=(x, y), xytext=(x-dx, y),
                                arrowprops=dict(arrowstyle="->", color="#d97706", lw=2))
                    ax.text(x-dx/2, y+arrowsc*0.3, f"{val:.0f} kN",
                            color="#d97706", fontsize=8, ha="center", fontfamily="monospace")
                elif ld == 1:
                    dy = arrowsc * (1 if val > 0 else -1)
                    ax.annotate("", xy=(x, y), xytext=(x, y-dy),
                                arrowprops=dict(arrowstyle="->", color="#d97706", lw=2))
                    ax.text(x+arrowsc*0.4, y-dy/2, f"{val:.0f} kN",
                            color="#d97706", fontsize=8, ha="left", fontfamily="monospace")

    # ── Draw DOF arrows (for step 1) ──────────────────────────
    if show_dofs and dof_map is not None:
        colors_dof = ["#1d4ed8","#15803d","#7c3aed"]
        labels_dof = ["u","v","θ"]
        for nid, gdofs in dof_map.items():
            x, y = nodes[nid]
            offsets = [(arrowsc*1.1,0),(0,arrowsc*1.1),(arrowsc*0.6,arrowsc*0.6)]
            for d, (dx, dy) in enumerate(offsets):
                col = colors_dof[d]
                ax.annotate("", xy=(x+dx,y+dy), xytext=(x,y),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=0.7))
                ax.text(x+dx*1.1, y+dy*1.1, f"d{gdofs[d]}\n({labels_dof[d]})",
                        color=col, fontsize=7, ha="center", fontfamily="monospace")

    # ── Draw reactions ────────────────────────────────────────
    if show_reactions and reactions is not None and dof_map is not None:
        for nid, ldofs in fixed_dofs.items():
            x, y = nodes[nid]
            for ld in ldofs:
                gi  = dof_map[nid][ld]
                val = reactions.get(gi, 0)
                if abs(val) < 1e-4: continue
                if ld == 0:
                    dx = arrowsc * (1 if val > 0 else -1)
                    ax.annotate("", xy=(x+dx,y), xytext=(x,y),
                                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))
                    ax.text(x+dx, y-arrowsc*0.4, f"Rx={val:.1f}",
                            color="#dc2626", fontsize=7, ha="center", fontfamily="monospace")
                elif ld == 1:
                    dy = arrowsc * (1 if val > 0 else -1)
                    ax.annotate("", xy=(x,y+dy), xytext=(x,y),
                                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))
                    ax.text(x-arrowsc*0.6, y+dy, f"Ry={val:.1f}",
                            color="#dc2626", fontsize=7, ha="right", fontfamily="monospace")

    ax.set_aspect('equal', adjustable='box')
    ax.relim(); ax.autoscale_view()
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    max_span = max(xmax-xmin, ymax-ymin, 1.0)
    margin   = max_span * 0.15
    ax.set_xlim(xmin-margin, xmax+margin); ax.set_ylim(ymin-margin, ymax+margin)
    ax.grid(True, color="#e2e8f0", lw=0.5, alpha=0.8)
    ax.set_xlabel("X (m)", color="#475569", fontfamily="monospace")
    ax.set_ylabel("Y (m)", color="#475569", fontfamily="monospace")
    plt.tight_layout()
    return fig


def draw_bmd_sfd(nodes, elements, member_results, elem_labels=None, member_loads=None):
    """
    Draw Bending Moment and Shear Force diagrams.
    Supports UDL superposition via member_loads = {element_index: w_kN_per_m}.

    The end forces (M_i, M_j, V_i, V_j) stored in member_results have already
    been corrected for FEF, so they represent the TRUE end values.
    The UDL parabolic / linear shapes are then added between those endpoints
    to produce the exact continuous diagram.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#f8fafc")
    titles  = ["Bending Moment Diagram (kN·m)", "Shear Force Diagram (kN)"]
    colors  = ["#7c3aed","#0891b2","#dc2626","#16a34a","#d97706"]

    if member_loads is None:
        member_loads = {}

    for ax_idx, ax in enumerate(axes):
        ax.set_facecolor("#ffffff")
        ax.tick_params(colors="#334155")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")
        ax.set_title(titles[ax_idx], color="#1e40af", fontfamily="monospace", pad=10)
        ax.grid(True, color="#e2e8f0", lw=0.5)

        xs   = [n[0] for n in nodes]; ys = [n[1] for n in nodes]
        span = max(max(xs)-min(xs), max(ys)-min(ys), 1.0)

        for idx, ((ni, nj), mr) in enumerate(zip(elements, member_results)):
            xi, yi = nodes[ni]; xj, yj = nodes[nj]
            L = math.hypot(xj-xi, yj-yi)
            if L < 1e-9:
                continue
            alpha = math.atan2(yj-yi, xj-xi)
            perp  = (-math.sin(alpha), math.cos(alpha))

            n_pts   = 60
            t_vals  = np.linspace(0, 1, n_pts + 1)
            x_local = t_vals * L
            w       = member_loads.get(idx, 0.0)

            col = colors[idx % len(colors)]

            if ax_idx == 0:  # ── BMD ──────────────────────────
                Mi = mr["M_i"]; Mj = mr["M_j"]
                ts_mult = -1   # sagging = below beam (tension-side convention)

                # Linear interpolation between corrected end moments
                # + parabolic component from UDL:  M(x) = Mi(1-x/L) + Mj(x/L) + w·x(L−x)/2
                vals = Mi * (1 - t_vals) + Mj * t_vals
                vals += (w * x_local * (L - x_local)) / 2.0
                vals *= ts_mult

                local_max = max(np.max(np.abs(vals)), 1e-9)
                scale     = span * 0.15 / local_max

                # ── Annotate peak value if UDL present ──────────
                if abs(w) > 1e-9:
                    peak_idx = np.argmax(np.abs(vals))
                    px_g = xi + t_vals[peak_idx]*(xj-xi) + scale*vals[peak_idx]*perp[0]
                    py_g = yi + t_vals[peak_idx]*(yj-yi) + scale*vals[peak_idx]*perp[1]
                    ax.text(px_g, py_g, f"{vals[peak_idx]/ts_mult:.1f}",
                            color=col, fontsize=7, ha="center", fontfamily="monospace",
                            fontweight="bold")

            else:             # ── SFD ──────────────────────────
                Vi = mr["V_i"]; Vj = mr["V_j"]
                ts_mult = 1

                # V(x) = Vi − w·x  (linear drop due to UDL)
                if abs(w) > 1e-9:
                    vals = Vi - w * x_local
                else:
                    vals = Vi * (1 - t_vals) + Vj * t_vals

                local_max = max(np.max(np.abs(vals)), 1e-9)
                scale     = span * 0.15 / local_max

            pts_x = xi + t_vals*(xj-xi) + scale*vals*perp[0]
            pts_y = yi + t_vals*(yj-yi) + scale*vals*perp[1]
            bx    = xi + t_vals*(xj-xi)
            by    = yi + t_vals*(yj-yi)

            # Skeleton
            ax.plot([xi,xj],[yi,yj], color="#94a3b8", lw=1)
            # Fill
            ax.fill(np.concatenate([bx, pts_x[::-1]]),
                    np.concatenate([by, pts_y[::-1]]),
                    color=col, alpha=0.22)
            # Outline
            ax.plot(pts_x, pts_y, color=col, lw=2)

            # End labels
            val0 = vals[0]  / ts_mult if ax_idx == 0 else vals[0]
            val1 = vals[-1] / ts_mult if ax_idx == 0 else vals[-1]
            ax.text(xi+scale*vals[0]*perp[0]*1.15, yi+scale*vals[0]*perp[1]*1.15,
                    f"{val0:.1f}", color=col, fontsize=7, ha="center", fontfamily="monospace")
            ax.text(xj+scale*vals[-1]*perp[0]*1.15, yj+scale*vals[-1]*perp[1]*1.15,
                    f"{val1:.1f}", color=col, fontsize=7, ha="center", fontfamily="monospace")

        ax.set_aspect('equal', adjustable='box')
        ax.relim(); ax.autoscale_view()
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        max_span = max(xmax-xmin, ymax-ymin, 1.0); margin = max_span * 0.15
        ax.set_xlim(xmin-margin, xmax+margin); ax.set_ylim(ymin-margin, ymax+margin)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "step"   not in st.session_state: st.session_state.step   = 0
if "preset" not in st.session_state: st.session_state.preset = list(PRESETS.keys())[0]
if "result" not in st.session_state: st.session_state.result = None


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏗️ 2D DSM Frame Analyzer")
    st.caption("Glass-Box Educational Tool for B.Tech / M.Tech")
    st.divider()

    st.markdown("### 📌 Preset Examples")
    chosen = st.selectbox("Load Preset:", list(PRESETS.keys()),
                          index=list(PRESETS.keys()).index(st.session_state.preset))
    if chosen != st.session_state.preset:
        st.session_state.preset = chosen
        st.session_state.result = None
        st.session_state.step   = 0

    P = PRESETS[chosen]
    st.info(P["desc"])

    st.divider()
    st.markdown("### ⚙️ Material & Section")
    E_val = st.number_input("E — Elastic Modulus (kN/m²)", value=float(P["E"]),
                            format="%.2e", help="Young's modulus")
    A_val = st.number_input("A — Cross-Section Area (m²)", value=float(P["A"]),
                            format="%.4f")
    I_val = st.number_input("I — Second Moment (m⁴)",      value=float(P["I"]),
                            format="%.6f")

    st.divider()
    st.markdown("### 📐 Geometry")
    st.caption("Node coordinates (x, y) in metres")
    node_df = pd.DataFrame(
        [{"Node": f"N{i+1}", "x (m)": x, "y (m)": y}
         for i,(x,y) in enumerate(P["nodes"])]
    )
    node_df = st.data_editor(node_df, num_rows="dynamic", width="stretch",
                              key=f"nodes_{chosen}")

    st.caption("Elements (node index pairs, 1-based)")
    elem_df = pd.DataFrame(
        [{"Elem": f"E{i+1}", "Node i": ni+1, "Node j": nj+1, "Label": lbl}
         for i, ((ni,nj), lbl) in enumerate(zip(P["elements"], P["labels"]))]
    )
    elem_df = st.data_editor(elem_df, num_rows="dynamic", width="stretch",
                              key=f"elems_{chosen}")

    st.divider()
    st.markdown("### 🔩 Boundary Conditions")
    st.caption("Check DOFs to fix (u=horiz, v=vert, θ=rotation)")
    fixed_dofs_ui = {}
    for i in range(len(node_df)):
        def_check = P["fixed_dofs"].get(i, [])
        cols_bc   = st.columns(3)
        checks    = []
        with cols_bc[0]: checks.append(0 if st.checkbox(f"N{i+1} u", value=0 in def_check, key=f"u_{i}_{chosen}") else None)
        with cols_bc[1]: checks.append(1 if st.checkbox(f"N{i+1} v", value=1 in def_check, key=f"v_{i}_{chosen}") else None)
        with cols_bc[2]: checks.append(2 if st.checkbox(f"N{i+1} θ", value=2 in def_check, key=f"t_{i}_{chosen}") else None)
        fixed = [c for c in checks if c is not None]
        if fixed:
            fixed_dofs_ui[i] = fixed

    st.divider()
    st.markdown("### ➡️ Nodal Point Loads")
    st.caption("kN (force) or kN·m (moment). Node numbers are 1-based.")
    load_rows = []
    for nid, ldmap in P["nodal_loads"].items():
        for ld, val in ldmap.items():
            load_rows.append({"Node": nid+1, "DOF (0=u,1=v,2=θ)": ld,
                              "Value (kN or kN·m)": val})
    load_df = pd.DataFrame(load_rows if load_rows else
                           [{"Node":1,"DOF (0=u,1=v,2=θ)":1,"Value (kN or kN·m)":0.0}])
    load_df = st.data_editor(load_df, num_rows="dynamic", width="stretch",
                              key=f"loads_{chosen}")

    # ── NEW: UDL Loads ────────────────────────────────────────
    st.divider()
    st.markdown("### 〰️ Element UDL Loads")
    st.caption(
        "Uniformly Distributed Load (kN/m) along each element, "
        "in the **local transverse (y) direction** perpendicular to the member axis.\n\n"
        "• **Negative w** → local −y direction (gravity for horizontal beams)\n"
        "• **Positive w** → local +y direction (upward / wind suction)\n\n"
        "_Element numbers are 1-based (E1, E2 …)_"
    )
    udl_preset = P.get("udl_loads", {})
    udl_rows   = [{"Element": f"E{ei+1}", "w (kN/m)": w}
                  for ei, w in udl_preset.items()]
    udl_df     = pd.DataFrame(udl_rows if udl_rows else
                              [{"Element": "E1", "w (kN/m)": 0.0}])
    udl_df     = st.data_editor(udl_df, num_rows="dynamic", width="stretch",
                                key=f"udl_{chosen}")

    st.divider()
    run_btn = st.button("🚀 Run DSM Analysis", type="primary", width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
#  PARSE UI INPUTS
# ══════════════════════════════════════════════════════════════════════════════
def _is_valid_row(r, keys):
    for k in keys:
        v = r.get(k)
        if v is None or str(v).strip() in ("","None"):
            return False
    return True

nodes_parsed = [
    (float(r["x (m)"]), float(r["y (m)"]))
    for _, r in node_df.iterrows()
    if _is_valid_row(r, ["x (m)","y (m)"])
]
elems_parsed = [
    (int(r["Node i"])-1, int(r["Node j"])-1)
    for _, r in elem_df.iterrows()
    if _is_valid_row(r, ["Node i","Node j"])
]
elem_labels_p = [
    str(r["Label"]) if _is_valid_row(r, ["Label"]) else f"E{i+1}"
    for i, (_, r) in enumerate(elem_df.iterrows())
    if _is_valid_row(r, ["Node i","Node j"])
]

nodal_loads_parsed = {}
for _, r in load_df.iterrows():
    if not _is_valid_row(r, ["Node","DOF (0=u,1=v,2=θ)","Value (kN or kN·m)"]): continue
    nid = int(r["Node"])-1; ld = int(r["DOF (0=u,1=v,2=θ)"]); val = float(r["Value (kN or kN·m)"])
    if abs(val) > 1e-10:
        nodal_loads_parsed.setdefault(nid, {})[ld] = val

# ── Parse UDL table ───────────────────────────────────────────
udl_loads_parsed = {}
for _, r in udl_df.iterrows():
    if not _is_valid_row(r, ["Element","w (kN/m)"]): continue
    elem_str = str(r["Element"]).strip().upper()
    if not elem_str.startswith("E"): continue
    try:
        ei  = int(elem_str[1:]) - 1
        w   = float(r["w (kN/m)"])
        if abs(w) > 1e-12:
            udl_loads_parsed[ei] = w
    except (ValueError, IndexError):
        continue


if run_btn:
    if len(nodes_parsed) < 2:
        st.error("🚨 Need at least 2 nodes.")
    elif len(elems_parsed) < 1:
        st.error("🚨 Need at least 1 element.")
    elif not fixed_dofs_ui:
        st.error("🚨 No boundary conditions defined — structure is a mechanism.")
    else:
        max_node = len(nodes_parsed) - 1
        bad = [(i, ni, nj) for i,(ni,nj) in enumerate(elems_parsed)
               if ni > max_node or nj > max_node or ni < 0 or nj < 0]
        if bad:
            for i, ni, nj in bad:
                st.error(f"🚨 Element E{i+1} references node {max(ni,nj)+1} but only "
                         f"{len(nodes_parsed)} nodes exist.")
            st.stop()

        # Warn about out-of-range UDL element indices
        oob_udl = [ei for ei in udl_loads_parsed if ei >= len(elems_parsed) or ei < 0]
        for ei in oob_udl:
            st.warning(f"⚠️ UDL entry E{ei+1} has no matching element — ignored.")
            del udl_loads_parsed[ei]

        with st.spinner("Assembling stiffness matrices (including UDL Fixed-End Forces)…"):
            try:
                st.session_state.result = run_dsm(
                    nodes_parsed, elems_parsed, fixed_dofs_ui,
                    nodal_loads_parsed, E_val, A_val, I_val,
                    udl_loads=udl_loads_parsed,
                )
            except ValueError as err:
                st.error(f"🚨 {err}"); st.stop()
        st.session_state.step = 0
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='text-align:center; font-family:Courier New; color:#1e40af; letter-spacing:2px;'>
  🏗️ 2D FRAME ANALYZER — DIRECT STIFFNESS METHOD
</h1>
<p style='text-align:center; color:#475569; font-family:Courier New; font-size:14px;'>
  Glass-Box Educational Tool &nbsp;|&nbsp; B.Tech / M.Tech Structural Analysis &nbsp;|&nbsp;
  Supports Point Loads &amp; UDL
</p>
""", unsafe_allow_html=True)
st.divider()

res = st.session_state.result

if res is None:
    fig_welcome = draw_frame(
        nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed,
        elem_labels=elem_labels_p, node_labels=True,
        udl_loads=udl_loads_parsed)
    st.pyplot(fig_welcome, width="stretch")
    plt.close(fig_welcome)
    st.caption("Select a preset in the sidebar and click **🚀 Run DSM Analysis** to begin")

    with st.expander("📖 What is the Direct Stiffness Method?", expanded=False):
        st.markdown("""
The <b>Direct Stiffness Method (DSM)</b> is the fundamental algorithm behind every commercial
structural analysis software (STAAD, ETABS, SAP2000, OpenSees…).
<br><br>
This app makes DSM a <b>glass box</b> — every intermediate matrix, every equation,
every transformation is shown step by step so you can trace exactly what the computer does.
<br><br>
<b>DSM in 10 Steps:</b>
<ol style='font-family:Courier New; font-size:13px; color:#1e293b; line-height:1.9;'>
  <li>Problem Setup — geometry, properties, UDL</li>
  <li>DOF Numbering — 3 DOFs per node</li>
  <li>Local Stiffness [k] — 6×6 per element</li>
  <li>Transformation [T] — rotate to global</li>
  <li>Global Element [K]ₑ = [T]ᵀ[k][T]</li>
  <li>Assembly — build nDOF × nDOF [K] and {F} (incl. UDL Fixed-End Forces)</li>
  <li>Partition & Apply BCs — extract [Kff]</li>
  <li>Solve — {Uf} = [Kff]⁻¹ {Ff}</li>
  <li>Member End Forces — {f'} = [k][T]{u} − {FEF}</li>
  <li>Reactions & Equilibrium Check</li>
</ol>
""", unsafe_allow_html=True)

    with st.expander("📚 UDL Fixed-End Force Theory — click to expand", expanded=False):
        st.markdown("""
<div class='formula-box'>
UDL on a member: w (kN/m) in the local transverse y-direction.
Negative w → gravity (downward for horizontal beams).

Fixed-End Forces (FEF) in LOCAL coordinates
(reactions assuming ALL joint displacements = 0):

       Node i                Node j
  ┌─────────────────────────────────┐
  │  FEF_local = [ 0,  wL/2,  wL²/12,  0,  wL/2,  -wL²/12 ]
  │               axial  Vi    Mi      axial  Vj    Mj
  └─────────────────────────────────┘

Equivalent Joint Loads (added to global {F}):
  {F} += [T]ᵀ · {FEF_local}

Member end force recovery (Step 8):
  {f_local} = [k_local]·[T]·{u_global_elem} − {FEF_local}

Bending moment at distance x from i (exact):
  M(x) = Mi·(1 − x/L) + Mj·(x/L) + w·x·(L−x)/2

Shear force at distance x from i (exact):
  V(x) = Vi − w·x
</div>
""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP NAVIGATOR
# ══════════════════════════════════════════════════════════════════════════════
STEP_NAMES = [
    "0 · Problem Setup",
    "1 · DOF Numbering",
    "2 · Local [k]",
    "3 · Transform [T]",
    "4 · Global [K]ₑ",
    "5 · Assembly [K]",
    "6 · Partition & BCs",
    "7 · Solve {U}",
    "8 · Member Forces",
    "9 · Reactions",
]

tab_cols = st.columns(len(STEP_NAMES))
for i, (col, name) in enumerate(zip(tab_cols, STEP_NAMES)):
    with col:
        btn_type = "primary" if i == st.session_state.step else "secondary"
        if st.button(name.split("·")[0].strip(), key=f"tab_{i}",
                     width="stretch", type=btn_type):
            st.session_state.step = i

st.progress((st.session_state.step) / (len(STEP_NAMES)-1))

nav_c1, nav_mid, nav_c2 = st.columns([1, 6, 1])
with nav_c1:
    if st.button("◀ Prev", width="stretch") and st.session_state.step > 0:
        st.session_state.step -= 1; st.rerun()
with nav_mid:
    st.markdown(f"<h3 style='text-align:center;margin:0;'>{STEP_NAMES[st.session_state.step]}</h3>",
                unsafe_allow_html=True)
with nav_c2:
    if st.button("Next ▶", width="stretch") and st.session_state.step < len(STEP_NAMES)-1:
        st.session_state.step += 1; st.rerun()

st.divider()

step = st.session_state.step
ed   = res["elem_data"]
ne   = res["ne"]
nn   = res["nn"]
udl_r = res.get("udl_loads", {})          # UDL dict from solver result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Step 0: Problem Setup ─────────────────────────────────────────────────────
if step == 0:
    col_v, col_t = st.columns([1.4, 1])
    with col_v:
        fig0 = draw_frame(nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed,
                          elem_labels=elem_labels_p, udl_loads=udl_r)
        st.pyplot(fig0, width="stretch")
        plt.close(fig0)

    with col_t:
        st.markdown("<div class='step-card'>", unsafe_allow_html=True)
        st.markdown("<span class='step-title'>Problem Setup</span>", unsafe_allow_html=True)
        st.markdown("<span class='step-subtitle'>Geometry, section, material, loads and UDL</span>",
                    unsafe_allow_html=True)

        st.markdown("**📍 Nodes**")
        st.dataframe(pd.DataFrame([{"ID":f"N{i+1}","x (m)":f"{x:.3f}","y (m)":f"{y:.3f}"}
                                    for i,(x,y) in enumerate(nodes_parsed)]),
                     width="stretch", hide_index=True)

        st.markdown("**🔗 Elements**")
        elem_rows = []
        for i, (e, lbl) in enumerate(zip(ed, elem_labels_p)):
            w_tag = f"w={udl_r.get(i,0.0):+.1f} kN/m" if udl_r.get(i,0.0) != 0.0 else "—"
            elem_rows.append({"ID":f"E{i+1}","Label":lbl,"Ni":f"N{e['ni']+1}",
                              "Nj":f"N{e['nj']+1}","L(m)":f"{e['L']:.3f}",
                              "α(°)":f"{e['alpha_deg']:.2f}","UDL":w_tag})
        st.dataframe(pd.DataFrame(elem_rows), width="stretch", hide_index=True)

        st.markdown("**⚙️ Material / Section**")
        st.dataframe(pd.DataFrame([{"E (kN/m²)":f"{E_val:.3e}","A (m²)":f"{A_val:.4f}",
                                    "I (m⁴)":f"{I_val:.6f}"}]),
                     width="stretch", hide_index=True)

        st.markdown("**➡️ Nodal Loads**")
        dof_names = {0:"Fx (kN)",1:"Fy (kN)",2:"M (kN·m)"}
        lrows = [{"Node":f"N{nid+1}","Load":dof_names[ld],"Value":f"{val:.1f}"}
                 for nid,ldmap in nodal_loads_parsed.items()
                 for ld,val in ldmap.items()]
        st.dataframe(pd.DataFrame(lrows if lrows else [{"Note":"No nodal loads"}]),
                     width="stretch", hide_index=True)

        if udl_r:
            st.markdown("**〰️ UDL Summary**")
            udl_table = [{"Element":f"E{ei+1}","w (kN/m)":f"{w:+.3f}",
                          "Direction":"Local −y (↓)" if w < 0 else "Local +y (↑)"}
                         for ei,w in udl_r.items()]
            st.dataframe(pd.DataFrame(udl_table), width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ── Step 1: DOF Numbering ─────────────────────────────────────────────────────
elif step == 1:
    col_v, col_t = st.columns([1.4, 1])
    with col_v:
        fig1 = draw_frame(nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed,
                          U=res["U"], dof_map=res["dof_map"],
                          show_dofs=True, show_loads=False, udl_loads=udl_r)
        st.pyplot(fig1, width="stretch")
        plt.close(fig1)

    with col_t:
        st.markdown("<div class='step-card'>", unsafe_allow_html=True)
        st.markdown("<span class='step-title'>Step 1 — DOF Numbering</span>", unsafe_allow_html=True)
        st.markdown("<span class='step-subtitle'>Each node gets 3 global DOFs: u (horiz), v (vert), θ (rotation)</span>",
                    unsafe_allow_html=True)
        with st.expander("📐 Theory", expanded=False):
            st.markdown("""
<div class='formula-box'>
Node i  →  Global DOFs: [ 3i ,  3i+1 ,  3i+2 ]
                              u     v      θ
Total DOFs = 3 × nNodes
</div>""", unsafe_allow_html=True)

        dof_rows = []
        for i in range(nn):
            dm = res["dof_map"][i]
            f_tags = ["FREE" if g in res["free_global"] else "FIXED" for g in dm]
            dof_rows.append({"Node":f"N{i+1}","u→d":dm[0],"v→d":dm[1],"θ→d":dm[2],
                             "u BC":f_tags[0],"v BC":f_tags[1],"θ BC":f_tags[2]})
        df_dofs = pd.DataFrame(dof_rows)
        def color_bc(val):
            if val=="FIXED": return "background-color:#fee2e2;color:#991b1b"
            if val=="FREE":  return "background-color:#dcfce7;color:#166534"
            return ""
        st.dataframe(df_dofs.style.map(color_bc, subset=["u BC","v BC","θ BC"]),
                     width="stretch", hide_index=True)
        st.markdown(f"- Total DOFs: **{res['nDOF']}** | Fixed: **{len(res['fixed_global'])}** "
                    f"→ {res['fixed_global']} | Free: **{len(res['free_global'])}** "
                    f"→ {res['free_global']}")
        st.markdown("</div>", unsafe_allow_html=True)


# ── Step 2: Local Stiffness ────────────────────────────────────────────────────
elif step == 2:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 2 — Local Stiffness Matrix [k] (6×6)</span>", unsafe_allow_html=True)
    st.markdown("<span class='step-subtitle'>Euler-Bernoulli beam-column element in its own local coordinate system</span>", unsafe_allow_html=True)
    with st.expander("📐 Theory", expanded=False):
        st.markdown("""
<div class='formula-box'>
Local DOF order: [u_i  v_i  θ_i  |  u_j  v_j  θ_j]
EA/L    = Axial stiffness
12EI/L³ = Transverse stiffness
6EI/L²  = Bending-shear coupling
4EI/L   = Same-end moment
2EI/L   = Far-end carry-over moment
</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    elem_sel = st.selectbox("Select Element:", [f"E{i+1} — {lbl}" for i,lbl in enumerate(elem_labels_p)])
    ei = int(elem_sel.split("E")[1].split(" ")[0])-1
    e  = ed[ei]

    a = E_val*A_val/e["L"]; b = 12*E_val*I_val/e["L"]**3
    c = 6*E_val*I_val/e["L"]**2; d = 4*E_val*I_val/e["L"]
    ev= 2*E_val*I_val/e["L"]
    st.markdown(f"""
<div class='formula-box'>
  EA/L    = {a:.4e} kN/m    |  12EI/L³ = {b:.4e} kN/m
  6EI/L²  = {c:.4e} kN     |  4EI/L   = {d:.4e} kN·m
  2EI/L   = {ev:.4e} kN·m
</div>""", unsafe_allow_html=True)

    dof_lbls = [f"u{e['ni']+1}",f"v{e['ni']+1}",f"θ{e['ni']+1}",
                f"u{e['nj']+1}",f"v{e['nj']+1}",f"θ{e['nj']+1}"]
    show_matrix(e["k_loc"], row_labels=dof_lbls, col_labels=dof_lbls,
                caption=f"[k] for E{ei+1} (kN/m units)")


# ── Step 3: Transformation Matrix ─────────────────────────────────────────────
elif step == 3:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 3 — Transformation Matrix [T] (6×6)</span>", unsafe_allow_html=True)
    st.markdown("<span class='step-subtitle'>Rotates local DOFs to global x-y coordinate system</span>", unsafe_allow_html=True)
    with st.expander("📐 Theory", expanded=False):
        st.markdown("""
<div class='formula-box'>
c = cos(α),  s = sin(α),  α = angle from global +X

[λ] = ⌈ c   s   0 ⌉       [T] = ⌈ [λ]   0  ⌉
      │-s   c   0 │              ⌊  0   [λ] ⌋
      ⌊ 0   0   1 ⌋

{u_local} = [T] · {u_global_element}
</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    elem_sel = st.selectbox("Select Element:", [f"E{i+1} — {lbl}" for i,lbl in enumerate(elem_labels_p)])
    ei = int(elem_sel.split("E")[1].split(" ")[0])-1
    e  = ed[ei]
    st.markdown(f"**E{ei+1}: α = {e['alpha_deg']:.3f}°  →  c = {e['c']:.4f},  s = {e['s']:.4f}**")

    c1, c2 = st.columns(2)
    with c1:
        dof_lbls = [f"u{e['ni']+1}",f"v{e['ni']+1}",f"θ{e['ni']+1}",
                    f"u{e['nj']+1}",f"v{e['nj']+1}",f"θ{e['nj']+1}"]
        show_matrix(e["T6"], row_labels=dof_lbls, col_labels=dof_lbls, caption="[T] — 6×6")
    with c2:
        fig3 = draw_frame(nodes_parsed, elems_parsed, fixed_dofs_ui, {}, elem_labels=elem_labels_p)
        ni2, nj2 = nodes_parsed[e["ni"]], nodes_parsed[e["nj"]]
        fig3.axes[0].annotate(f"α={e['alpha_deg']:.1f}°",
            xy=((ni2[0]+nj2[0])/2,(ni2[1]+nj2[1])/2),
            color="#7c3aed", fontsize=10, fontfamily="monospace",
            bbox=dict(boxstyle="round", fc="#ffffff", ec="#7c3aed"))
        st.pyplot(fig3, width="stretch")
        plt.close(fig3)


# ── Step 4: Global Element Stiffness ──────────────────────────────────────────
elif step == 4:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 4 — Global Element Stiffness [K]ₑ = [T]ᵀ [k] [T]</span>", unsafe_allow_html=True)
    with st.expander("📐 Theory", expanded=False):
        st.markdown("""
<div class='formula-box'>
[K]ₑ = [T]ᵀ · [k] · [T]   (6×6 in global DOF coords, ready to scatter into [K])
</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    elem_sel = st.selectbox("Select Element:", [f"E{i+1} — {lbl}" for i,lbl in enumerate(elem_labels_p)])
    ei = int(elem_sel.split("E")[1].split(" ")[0])-1
    e  = ed[ei]
    dof_lbls = [f"d{g}" for g in e["gdofs"]]
    show_matrix(e["k_glob"], row_labels=dof_lbls, col_labels=dof_lbls,
                caption=f"[K]ₑ for E{ei+1} mapped to global DOFs {e['gdofs']}")
    dm_rows = [{"Local DOF":f"u{e['ni']+1}/v{e['ni']+1}/θ{e['ni']+1}/u{e['nj']+1}/v{e['nj']+1}/θ{e['nj']+1}".split("/")[li],
                "Global Index":g} for li,g in enumerate(e["gdofs"])]
    st.dataframe(pd.DataFrame(dm_rows), width="stretch", hide_index=True)


# ── Step 5: Assembly ──────────────────────────────────────────────────────────
elif step == 5:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 5 — Global Assembly [K] and {F}</span>", unsafe_allow_html=True)
    st.markdown("<span class='step-subtitle'>All element matrices + UDL Fixed-End Forces scattered into the global system</span>",
                unsafe_allow_html=True)
    with st.expander("📐 Theory — including UDL", expanded=False):
        st.markdown("""
<div class='formula-box'>
[K] = Σ scatter_add([K]ₑ)         (element stiffnesses — unchanged by UDL)

{F} = {F_nodal}  +  Σ [T]ᵀ · {FEF_local}
      ↑ point loads   ↑ UDL equivalent joint loads

FEF_local = [ 0,  wL/2,  wL²/12,  0,  wL/2,  -wL²/12 ]
</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Show FEF breakdown if any UDL present
    fef_s = res.get("fef_summaries", {})
    if fef_s:
        st.markdown("#### 〰️ UDL Fixed-End Force Details")
        fef_rows = []
        for ei, fs in fef_s.items():
            fef_rows.append({
                "Element": f"E{ei+1}",
                "w (kN/m)": f"{fs['w']:+.3f}",
                "L (m)": f"{fs['L']:.3f}",
                "Vi = wL/2": f"{fs['Vi']:.4f}",
                "Mi = wL²/12": f"{fs['Mi']:.4f}",
                "Vj = wL/2": f"{fs['Vj']:.4f}",
                "Mj = −wL²/12": f"{fs['Mj']:.4f}",
            })
        st.dataframe(pd.DataFrame(fef_rows), width="stretch", hide_index=True)
        st.caption("These FEF values were transformed to global coords and added to {F} before solving.")
        st.divider()

    nDOF = res["nDOF"]
    dof_lbls = [f"d{i}" for i in range(nDOF)]
    col_k, col_f = st.columns([3, 1])
    with col_k:
        show_matrix(res["K"], row_labels=dof_lbls, col_labels=dof_lbls,
                    caption=f"Global [K] — {nDOF}×{nDOF} (kN/m)",
                    highlight_rows=res["free_global"], highlight_cols=res["free_global"])
    with col_f:
        show_vector(res["F"], labels=dof_lbls,
                    caption="{F} — combined point loads + UDL equiv. loads (kN / kN·m)",
                    highlight=[i for i,v in enumerate(res["F"]) if abs(v) > 1e-10])

    st.info(f"ℹ️ Highlighted rows/columns = **Free DOFs** {res['free_global']}. "
            f"Fixed DOFs {res['fixed_global']} eliminated in the next step.")


# ── Step 6: Partitioning ──────────────────────────────────────────────────────
elif step == 6:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 6 — Partition & Apply Boundary Conditions</span>", unsafe_allow_html=True)
    st.markdown("<span class='step-subtitle'>Delete fixed DOF rows/columns → reduced system [Kff]{Uf} = {Ff}</span>",
                unsafe_allow_html=True)
    with st.expander("📐 Theory", expanded=False):
        st.markdown("""
<div class='formula-box'>
  ┌ Kff  Kfc ┐ ┌ Uf ┐   ┌ Ff ┐      ({Ff} already includes UDL FEF contributions)
  │          │ │    │ = │    │
  └ Kcf  Kcc ┘ └ Uc ┘   └ Fc ┘

  Since Uc = 0:  [Kff] {Uf} = {Ff}
</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    ff = res["free_global"]; nf = len(ff)
    dof_lbls_f = [f"d{g}" for g in ff]
    col_kff, col_ff = st.columns([3, 1])
    with col_kff:
        show_matrix(res["Kff"], row_labels=dof_lbls_f, col_labels=dof_lbls_f,
                    caption=f"[Kff] — Reduced stiffness {nf}×{nf}")
    with col_ff:
        show_vector(res["Ff"], labels=dof_lbls_f, caption="{Ff} — Reduced loads (incl. UDL)")

    st.markdown(f"- **Full system:** {res['nDOF']}×{res['nDOF']}  →  "
                f"**Reduced system:** {nf}×{nf}")


# ── Step 7: Solve ─────────────────────────────────────────────────────────────
elif step == 7:
    col_v, col_t = st.columns([1.4, 1])
    with col_v:
        fig7 = draw_frame(nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed,
                          U=res["U"], dof_map=res["dof_map"],
                          show_deformed=True, show_loads=True, udl_loads=udl_r)
        st.pyplot(fig7, width="stretch")
        plt.close(fig7)
        st.caption("— Deformed shape (dashed red, exaggerated scale)")

    with col_t:
        st.markdown("<div class='step-card'>", unsafe_allow_html=True)
        st.markdown("<span class='step-title'>Step 7 — Solve {Uf} = [Kff]⁻¹ {Ff}</span>", unsafe_allow_html=True)
        st.markdown("<span class='step-subtitle'>Gaussian elimination → nodal displacements (UDL effects included via FEF in {Ff})</span>",
                    unsafe_allow_html=True)

        nDOF = res["nDOF"]
        U_tagged = []
        for gi in range(nDOF):
            tag  = "FIXED=0" if gi in res["fixed_global"] else "SOLVED"
            dn   = ["u","v","θ"][gi%3]; nid = gi//3
            U_tagged.append({"DOF":f"d{gi}","Node":f"N{nid+1}","Type":dn,
                             "Value":fmt_val(res["U"][gi],6),"Status":tag})
        df_U = pd.DataFrame(U_tagged)
        def col_status(val):
            return ("background-color:#dcfce7;color:#166534" if val=="SOLVED"
                    else "background-color:#f1f5f9;color:#475569")
        st.dataframe(df_U.style.map(col_status, subset=["Status"]),
                     width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ── Step 8: Member End Forces ─────────────────────────────────────────────────
elif step == 8:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 8 — Member End Forces {f'} = [k][T]{u} − {FEF}</span>",
                unsafe_allow_html=True)
    st.markdown(
        "<span class='step-subtitle'>"
        "Recover N, V, M at each member end in LOCAL axes. "
        "<b>FEF is subtracted</b> to obtain the true member forces when UDL is present."
        "</span>",
        unsafe_allow_html=True,
    )
    with st.expander("📐 Theory — UDL correction", expanded=False):
        st.markdown("""
<div class='formula-box'>
Without UDL:  {f_local} = [k_local] · {u_local}
With    UDL:  {f_local} = [k_local] · {u_local} − {FEF_local}
                                                    ↑ corrects for the distributed load
                                                      not captured by end displacements alone

This ensures M(0) = M_i and M(L) = M_j match the parabolic BMD exactly.
</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    force_rows = []
    for i, (mr, e) in enumerate(zip(res["member_results"], ed)):
        w_tag = f"w={udl_r.get(i,0.0):+.2f}" if udl_r.get(i,0.0) != 0.0 else "—"
        force_rows.append({
            "Elem":f"E{i+1}","Label":elem_labels_p[i],"UDL":w_tag,
            "N_i (kN)":f"{mr['N_i']:.3f}","V_i (kN)":f"{mr['V_i']:.3f}",
            "M_i (kN·m)":f"{mr['M_i']:.3f}",
            "N_j (kN)":f"{mr['N_j']:.3f}","V_j (kN)":f"{mr['V_j']:.3f}",
            "M_j (kN·m)":f"{mr['M_j']:.3f}",
        })
    st.dataframe(pd.DataFrame(force_rows), width="stretch", hide_index=True)

    # Detail view — show elastic, FEF, and corrected vectors side-by-side
    st.divider()
    elem_sel = st.selectbox("Detail view:", [f"E{i+1} — {lbl}" for i,lbl in enumerate(elem_labels_p)])
    ei = int(elem_sel.split("E")[1].split(" ")[0])-1
    mr = res["member_results"][ei]; e = ed[ei]
    loc_lbls = [f"u'i",f"v'i",f"θ'i",f"u'j",f"v'j",f"θ'j"]
    gdof_lbls = [f"d{g}" for g in e["gdofs"]]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**{u_local} = [T]·{u_global}**")
        show_vector(mr["u_local"], labels=loc_lbls, caption="Local displacements (m/rad)")
    with c2:
        st.markdown("**{f_elastic} = [k]·{u_local}**")
        show_vector(mr["f_local_elastic"], labels=loc_lbls, caption="Before FEF correction")
    with c3:
        fef_l = mr["fef_local"]
        w_ei  = udl_r.get(ei, 0.0)
        header = f"**{{FEF_local}}** (w={w_ei:+.2f} kN/m)" if abs(w_ei) > 1e-9 else "**{FEF_local}** = 0 (no UDL)"
        st.markdown(header)
        show_vector(fef_l, labels=loc_lbls, caption="Fixed-end forces (subtracted)")

    st.info(f"✅ **{f'E{ei+1}'}** corrected end forces = f_elastic − FEF  "
            f"→  M_i = {mr['M_i']:.3f}, V_i = {mr['V_i']:.3f}, "
            f"M_j = {mr['M_j']:.3f}, V_j = {mr['V_j']:.3f} kN/kN·m")

    st.divider()
    fig_bmd = draw_bmd_sfd(nodes_parsed, elems_parsed, res["member_results"],
                           elem_labels_p, member_loads=udl_r)
    st.pyplot(fig_bmd, width="stretch")
    plt.close(fig_bmd)

    if udl_r:
        st.caption(
            "BMD uses exact formula: M(x) = Mᵢ(1−x/L) + Mⱼ(x/L) + w·x(L−x)/2  "
            "— the parabolic humps arise from the UDL term."
        )


# ── Step 9: Reactions ─────────────────────────────────────────────────────────
elif step == 9:
    col_v, col_t = st.columns([1.4, 1])
    with col_v:
        fig9 = draw_frame(nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed,
                          U=res["U"], dof_map=res["dof_map"],
                          show_deformed=True, show_reactions=True,
                          reactions=res["reactions"], udl_loads=udl_r)
        st.pyplot(fig9, width="stretch")
        plt.close(fig9)

    with col_t:
        st.markdown("<div class='step-card'>", unsafe_allow_html=True)
        st.markdown("<span class='step-title'>Step 9 — Support Reactions & Equilibrium</span>",
                    unsafe_allow_html=True)
        st.markdown("<span class='step-subtitle'>{R} = [K]{U} − {F}  at fixed DOFs</span>",
                    unsafe_allow_html=True)
        with st.expander("📐 Theory", expanded=False):
            st.markdown("""
<div class='formula-box'>
  {R_fixed} = [K]{U} − {F}   at fixed DOF rows
  ({F} already includes the UDL equivalent joint loads,
   so reactions balance BOTH point loads AND distributed loads.)

  Equilibrium check (must equal zero):
    ΣFx = reactions_x + point_loads_x + UDL_resultants_x = 0
    ΣFy = reactions_y + point_loads_y + UDL_resultants_y = 0
    ΣM  = all moment contributions about origin          = 0
</div>""", unsafe_allow_html=True)

        rx_rows = []
        dof_names_map = {0:"Rx (kN)",1:"Ry (kN)",2:"RM (kN·m)"}
        for gi, val in sorted(res["reactions"].items()):
            nid = gi//3; ld = gi%3
            rx_rows.append({"Node":f"N{nid+1}","DOF":f"d{gi}",
                            "Type":dof_names_map[ld],"Reaction":f"{val:.4f}"})
        st.dataframe(pd.DataFrame(rx_rows), width="stretch", hide_index=True)

        st.markdown("**⚖️ Global Equilibrium Check**")
        eq  = res["eq_check"]; tol = 1e-3
        for label, val in eq.items():
            passed = abs(val) < tol
            badge  = "✅" if passed else "❌"
            color  = "#166534" if passed else "#991b1b"
            st.markdown(
                f"<span style='color:{color}; font-family:Courier New; font-size:14px;'>"
                f"{badge} {label} = {val:.6f} kN  "
                f"({'OK' if passed else 'WARN — check model'})</span>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("**📋 Load vs. Reaction Summary**")
        total_Fx = sum(v for nid,ldm in nodal_loads_parsed.items()
                       for ld,v in ldm.items() if ld==0)
        total_Fy = sum(v for nid,ldm in nodal_loads_parsed.items()
                       for ld,v in ldm.items() if ld==1)
        # Add UDL resultants to "applied load" totals for the summary table
        for idx, e_item in enumerate(ed):
            w = udl_r.get(idx, 0.0)
            if abs(w) < 1e-12: continue
            L  = e_item["L"]
            al = math.atan2(nodes[e_item["nj"]][1]-nodes[e_item["ni"]][1],
                            nodes[e_item["nj"]][0]-nodes[e_item["ni"]][0])
            total_Fx += w * L * (-math.sin(al))
            total_Fy += w * L * ( math.cos(al))

        rx_Fx = sum(v for gi,v in res["reactions"].items() if gi%3==0)
        rx_Fy = sum(v for gi,v in res["reactions"].items() if gi%3==1)
        st.markdown(f"""
| Component | Applied (incl. UDL) | Reactions | Net |
|-----------|---------------------|-----------|-----|
| Horiz Fx  | {total_Fx:.3f} kN  | {rx_Fx:.3f} kN | {total_Fx+rx_Fx:.6f} |
| Vert  Fy  | {total_Fy:.3f} kN  | {rx_Fy:.3f} kN | {total_Fy+rx_Fy:.6f} |
""")
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
ref_c1, ref_c2, ref_c3 = st.columns(3)
with ref_c1:
    st.markdown("""
**📚 References**
- Kassimali A. — *Matrix Analysis of Structures* (Ch. 3–6)
- McGuire, Gallagher, Ziemian — *Matrix Structural Analysis*
- Bhavikatti S.S. — *Matrix Methods of Structural Analysis*
""")
with ref_c2:
    st.markdown("""
**🧮 DSM + UDL Formula Card**
- [k] = 6×6 Euler-Bernoulli stiffness
- [T] = block-diagonal rotation matrix
- [K]ₑ = [T]ᵀ[k][T] (global element)
- FEF_local = [0, wL/2, wL²/12, 0, wL/2, −wL²/12]
- {F} += [T]ᵀ · {FEF_local}
- {f_local} = [k][T]{u} − {FEF_local}
""")
with ref_c3:
    st.markdown("**🚀 Deploy This App**")
    st.code("pip install streamlit numpy pandas matplotlib\nstreamlit run dsm_2d_frame.py",
            language="bash")
    st.markdown("Or push to GitHub and connect at  \n[share.streamlit.io](https://share.streamlit.io)")
