"""
╔══════════════════════════════════════════════════════════════════╗
║   2D Frame Analyzer — Direct Stiffness Method (DSM) First        ║
║   Glass-Box Educational Tool for B.Tech / M.Tech Students        ║
║   Author  : Educational Structural Engineering Lab               ║
║   Deploy  : streamlit run app.py                                  ║
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
  /* Light engineering theme */
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
    "Portal Frame (Wind + Gravity)": {
        "desc": "Fixed-base single-bay portal frame — horizontal wind load + vertical gravity on beam",
        "nodes": [(0,0),(0,4),(6,4),(6,0)],
        "elements": [(0,1),(1,2),(2,3)],
        "fixed_dofs": {0:[0,1,2], 3:[0,1,2]},
        "E": 200e6, "A": 0.01, "I": 1e-4,
        "nodal_loads": {1:{0:20.0}},
        "member_loads": {1: 10.0},
        "labels": ["Left Col","Beam","Right Col"],
    },
    "Two-Span Continuous Beam": {
        "desc": "Continuous beam on pin + 2 rollers — uniform gravity load on both spans",
        "nodes": [(0,0),(2.5,0),(5,0),(7.5,0),(10,0)],
        "elements": [(0,1),(1,2),(2,3),(3,4)],
        "fixed_dofs": {0:[0,1], 2:[1], 4:[1]},
        "E": 200e6, "A": 0.02, "I": 2e-4,
        "nodal_loads": {},
        "member_loads": {0: 20.0, 1: 20.0, 2: 20.0, 3: 20.0},
        "labels": ["Span 1a","Span 1b","Span 2a","Span 2b"],
    },
    "Cantilever Beam (Tip Load)": {
        "desc": "Fixed-free cantilever — vertical tip load",
        "nodes": [(0,0),(4,0)],
        "elements": [(0,1)],
        "fixed_dofs": {0:[0,1,2]},
        "E": 200e6, "A": 0.01, "I": 1e-4,
        "nodal_loads": {1:{1:-80.0}},
        "member_loads": {},
        "labels": ["Cantilever"],
    },
    "Pitched Roof Frame": {
        "desc": "Symmetric gabled frame — vertical snow load on both rafters",
        "nodes": [(0,0),(0,4),(4,6),(8,4),(8,0)],
        "elements": [(0,1),(1,2),(2,3),(3,4)],
        "fixed_dofs": {0:[0,1,2], 4:[0,1,2]},
        "E": 200e6, "A": 0.015, "I": 1.5e-4,
        "nodal_loads": {2:{1:-20.0}},
        "member_loads": {1: 12.0, 2: 12.0},
        "labels": ["Left Col","Left Rafter","Right Rafter","Right Col"],
    },
    "Plane Frame (Custom UDL + Point Load)": {
        "desc": "Professional plane-frame example with partial-span UDL and member point load",
        "nodes": [(0,0),(0,4),(5,4),(9,4),(9,0)],
        "elements": [(0,1),(1,2),(2,3),(3,4)],
        "fixed_dofs": {0:[0,1,2], 4:[0,1,2]},
        "E": 200e6, "A": 0.012, "I": 1.2e-4,
        "nodal_loads": {1:{0:12.0}},
        "member_loads": [
            {"element": 1, "type": "UDL", "w": 18.0, "a": 1.0, "b": 4.0},
            {"element": 2, "type": "Point", "P": 35.0, "x": 2.0},
        ],
        "labels": ["Left Column","Beam A","Beam B","Right Column"],
    },
}



# ══════════════════════════════════════════════════════════════════════════════
#  MEMBER LOAD HELPERS — plane-frame equivalent nodal loads
# ══════════════════════════════════════════════════════════════════════════════
def _clean_load_type(load_type):
    text = str(load_type or "UDL").strip().lower()
    if text.startswith("point") or text in {"pl", "p"}:
        return "Point"
    return "UDL"


def normalize_member_loads(member_loads, elem_lengths=None):
    """Return a list of standard member-load specs.

    Supports the legacy `{element_id: w_down}` dict as full-span UDLs and the
    newer list-of-dicts format with custom UDL spans and point-load positions.
    Positive load values act downward in the member's local transverse direction.
    """
    specs = []
    if member_loads is None:
        return specs
    if isinstance(member_loads, dict):
        for eid, w in member_loads.items():
            if abs(float(w)) > 1e-12:
                L = elem_lengths[eid] if elem_lengths and eid < len(elem_lengths) else None
                specs.append({"element": int(eid), "type": "UDL", "w": float(w),
                              "a": 0.0, "b": L, "P": 0.0, "x": 0.0})
        return specs
    for row in member_loads:
        if not row:
            continue
        eid = int(row.get("element", row.get("Element", 0)))
        load_type = _clean_load_type(row.get("type", row.get("Type", "UDL")))
        L = elem_lengths[eid] if elem_lengths and 0 <= eid < len(elem_lengths) else None
        if load_type == "Point":
            P = float(row.get("P", row.get("Load", row.get("load", 0.0))))
            x = float(row.get("x", row.get("Position", row.get("position", 0.0))))
            if abs(P) > 1e-12:
                specs.append({"element": eid, "type": "Point", "P": P, "x": x,
                              "a": x, "b": x, "w": 0.0})
        else:
            w = float(row.get("w", row.get("Load", row.get("load", 0.0))))
            a = float(row.get("a", row.get("Start", row.get("start", 0.0))))
            raw_b = row.get("b", row.get("End", row.get("end", L)))
            b = L if raw_b is None or str(raw_b).strip() == "" else float(raw_b)
            if abs(w) > 1e-12:
                specs.append({"element": eid, "type": "UDL", "w": w,
                              "a": a, "b": b, "P": 0.0, "x": 0.0})
    return specs


def _beam_shape_functions(x, L):
    xi = x / L
    return np.array([
        1 - 3*xi**2 + 2*xi**3,
        L * (xi - 2*xi**2 + xi**3),
        3*xi**2 - 2*xi**3,
        L * (-xi**2 + xi**3),
    ])


def equivalent_member_load_vector(load_specs, L):
    """Consistent local nodal load vector for transverse UDL/point loads."""
    p_local = np.zeros(6)
    applied = []
    for spec in load_specs:
        ltype = _clean_load_type(spec.get("type"))
        if ltype == "UDL":
            w = float(spec.get("w", 0.0))
            a = max(0.0, min(L, float(spec.get("a", 0.0))))
            b = max(0.0, min(L, float(spec.get("b", L))))
            if b < a:
                a, b = b, a
            if b - a <= 1e-12 or abs(w) <= 1e-12:
                continue
            pts, weights = np.polynomial.legendre.leggauss(8)
            xs = 0.5*(b-a)*pts + 0.5*(a+b)
            ws = 0.5*(b-a)*weights
            q = -w  # positive UI value = downward local transverse load
            for x, wt in zip(xs, ws):
                N1, N2, N3, N4 = _beam_shape_functions(float(x), L)
                p_local[[1, 2, 4, 5]] += wt * q * np.array([N1, N2, N3, N4])
            applied.append({"type": "UDL", "w": w, "a": a, "b": b})
        else:
            P = float(spec.get("P", 0.0))
            x = max(0.0, min(L, float(spec.get("x", 0.0))))
            if abs(P) <= 1e-12:
                continue
            N1, N2, N3, N4 = _beam_shape_functions(x, L)
            p_local[[1, 2, 4, 5]] += -P * np.array([N1, N2, N3, N4])
            applied.append({"type": "Point", "P": P, "x": x})
    return p_local, applied

# ══════════════════════════════════════════════════════════════════════════════
#  DSM SOLVER — returns full step-by-step data
# ══════════════════════════════════════════════════════════════════════════════
def run_dsm(nodes, elements, fixed_dofs, nodal_loads, member_loads, E, A, I):
    """
    Parameters
    ----------
    nodes        : list of (x, y) tuples
    elements     : list of (ni, nj) node-index pairs
    fixed_dofs   : dict {node_id: [local_dof_indices]}  — 0=u,1=v,2=θ
    nodal_loads  : dict {node_id: {local_dof: value}}  — kN or kN·m
    member_loads : list/dict — UDL or point member loads in local transverse direction
    E, A, I      : material / section (kN/m² , m², m⁴)

    Returns a dict with all intermediate matrices for each step.
    """
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

        # 6×6 transformation matrix (block-diagonal, 2×[c s 0; -s c 0; 0 0 1])
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
            "p_local": np.zeros(6),
            "p_global": np.zeros(6),
            "member_load_specs": [],
        })

    # ── Step 5: Assemble global K and F ─────────────────────
    K = np.zeros((nDOF, nDOF))
    F = np.zeros(nDOF)
    F_nodal = np.zeros(nDOF)
    F_member = np.zeros(nDOF)

    elem_lengths = [ed["L"] for ed in elem_data]
    normalized_loads = normalize_member_loads(member_loads, elem_lengths)
    loads_by_element = {i: [] for i in range(ne)}
    for spec in normalized_loads:
        if 0 <= spec["element"] < ne:
            loads_by_element[spec["element"]].append(spec)

    for idx, ed in enumerate(elem_data):
        gdofs = ed["gdofs"]
        p_local, applied_specs = equivalent_member_load_vector(loads_by_element.get(idx, []), ed["L"])
        if applied_specs:
            ed["p_local"] = p_local
            ed["p_global"] = ed["T6"].T @ p_local
            ed["member_load_specs"] = applied_specs
        for r, gr in enumerate(gdofs):
            F_member[gr] += ed["p_global"][r]
            for c2, gc in enumerate(gdofs):
                K[gr, gc] += ed["k_glob"][r, c2]

    for nid, ldmap in nodal_loads.items():
        for ld, val in ldmap.items():
            F_nodal[dof_map[nid][ld]] += val

    F = F_nodal + F_member

    # ── Step 6: Partition ─────────────────────────────────────
    ff = free_global
    Kff = K[np.ix_(ff, ff)]
    Ff  = F[ff]

    # ── Step 7: Solve ─────────────────────────────────────────
    cond_num = np.linalg.cond(Kff)
    if cond_num > 1e10:
        st.warning(f"⚠️ Warning: The stiffness matrix is highly ill-conditioned (κ ≈ {cond_num:.2e}). Results may be plagued by round-off errors. Check for disconnected elements or extreme differences in element stiffness.")
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
    member_results = []
    for ed in elem_data:
        u_el  = U[ed["gdofs"]]                  # global displacements
        u_loc = ed["T6"] @ u_el                 # in local coords
        f_loc = ed["k_loc"] @ u_loc - ed["p_local"]  # local member end forces incl. fixed-end load effects
        member_results.append({
            "u_global": u_el,
            "u_local":  u_loc,
            "f_local":  f_loc,
            "p_local":  ed["p_local"],
            "member_load_specs": ed["member_load_specs"],
            # Force member exerts ON joint = negative of internal force vector
            "N_i": -f_loc[0], "V_i": -f_loc[1], "M_i": -f_loc[2],
            "N_j": -f_loc[3], "V_j": -f_loc[4], "M_j": -f_loc[5],
        })

    # ── Step 9: Reactions ─────────────────────────────────────
    R_full = K @ U - F
    reactions = {gi: R_full[gi] for gi in fixed_global}

    # True global force/moment equilibrium about origin. Equivalent member
    # loads are already in F_member, so they can be checked with nodal loads.
    sigma_M = 0.0
    for nid in fixed_dofs:
        x, y = nodes[nid]
        Rx = R_full[dof_map[nid][0]]
        Ry = R_full[dof_map[nid][1]]
        RM = R_full[dof_map[nid][2]]
        sigma_M += RM + Ry * x - Rx * y
    for nid, (x, y) in enumerate(nodes):
        Fx = F[dof_map[nid][0]]
        Fy = F[dof_map[nid][1]]
        M = F[dof_map[nid][2]]
        sigma_M += M + Fy * x - Fx * y

    eq_check = {
        "ΣFx": sum(R_full[dof_map[n][0]] for n in fixed_dofs) + sum(F[0::3]),
        "ΣFy": sum(R_full[dof_map[n][1]] for n in fixed_dofs) + sum(F[1::3]),
        "ΣM":  sigma_M,
    }

    return {
        "nn": nn, "ne": ne, "nDOF": nDOF,
        "dof_map": dof_map,
        "fixed_global": fixed_global,
        "free_global":  free_global,
        "elem_data":    elem_data,
        "K": K, "F": F, "F_nodal": F_nodal, "F_member": F_member,
        "Kff": Kff, "Ff": Ff,
        "U": U,
        "member_results": member_results,
        "member_load_specs": normalized_loads,
        "reactions": reactions,
        "eq_check": eq_check,
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

def show_matrix(M, row_labels=None, col_labels=None, caption="", highlight_rows=None, highlight_cols=None):
    """Render a numpy matrix as a styled pandas DataFrame."""
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
    """Render a vector as a 1-column DataFrame."""
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
#  FRAME VISUALIZATION (Matplotlib)
# ══════════════════════════════════════════════════════════════════════════════
SUPPORT_SYMBOLS = {
    "fixed":  "#dc2626",
    "pin":    "#16a34a",
    "roller": "#2563eb",
}

def classify_support(fixed_ldofs):
    if len(fixed_ldofs) >= 3: return "fixed"
    if 0 in fixed_ldofs and 1 in fixed_ldofs: return "pin"
    return "roller"

def draw_frame(nodes, elements, fixed_dofs, nodal_loads, member_loads=None,
               U=None, dof_map=None, reactions=None,
               scale=0.3, show_dofs=False, show_loads=True,
               show_deformed=False, show_reactions=False,
               elem_labels=None, node_labels=True):
    """Draw the 2D frame with supports, loads, deformed shape, reactions."""
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#f8fafc")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#334155")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cbd5e1")

    xs = [n[0] for n in nodes]; ys = [n[1] for n in nodes]
    span = max(max(xs)-min(xs), max(ys)-min(ys), 1.0)
    arrowsc = span * 0.08   # arrow scale

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
                ax.plot([x-span*0.04, x-span*0.06], [y+dy, y+dy+span*0.015],
                        color="#dc2626", lw=1, alpha=0.7)
        elif stype == "pin":
            tri = plt.Polygon([[x, y],[x-span*0.03, y-span*0.06],[x+span*0.03, y-span*0.06]],
                              closed=True, color="#16a34a", alpha=0.5, zorder=2)
            ax.add_patch(tri)
            ax.plot([x-span*0.05, x+span*0.05],[y-span*0.065, y-span*0.065],
                    color="#16a34a", lw=2)
        else:
            tri = plt.Polygon([[x, y],[x-span*0.03, y-span*0.05],[x+span*0.03, y-span*0.05]],
                              closed=True, color="#2563eb", alpha=0.35, zorder=2)
            ax.add_patch(tri)
            circle = plt.Circle((x, y-span*0.07), span*0.022, color="#2563eb", alpha=0.4, zorder=2)
            ax.add_patch(circle)

    # ── Draw loads ────────────────────────────────────────────
    if show_loads:
        lengths = [max(math.hypot(nodes[nj][0]-nodes[ni][0], nodes[nj][1]-nodes[ni][1]), 1e-9)
                   for ni, nj in elements]
        specs_by_element = {i: [] for i in range(len(elements))}
        for spec in normalize_member_loads(member_loads, lengths):
            if 0 <= spec["element"] < len(elements):
                specs_by_element[spec["element"]].append(spec)
        for idx, (ni, nj) in enumerate(elements):
            xi, yi = nodes[ni]; xj, yj = nodes[nj]
            L = lengths[idx]
            alpha = math.atan2(yj-yi, xj-xi)
            # Downward local transverse load direction for positive loads.
            nx, ny = math.sin(alpha), -math.cos(alpha)
            for spec in specs_by_element.get(idx, []):
                if spec["type"] == "UDL":
                    w = spec["w"]
                    a, b = spec["a"], spec["b"]
                    n_arrows = max(2, min(8, int((b-a) / max(L, 1e-9) * 8) + 1))
                    for k in range(1, n_arrows + 1):
                        xloc = a + (b-a) * k / (n_arrows + 1)
                        t = xloc / L
                        x = xi + t*(xj-xi); y = yi + t*(yj-yi)
                        ax.annotate("", xy=(x, y), xytext=(x - nx*arrowsc*0.75, y - ny*arrowsc*0.75),
                                    arrowprops=dict(arrowstyle="->", color="#ea580c", lw=1.4, alpha=0.8))
                    tm = ((a+b)/2) / L
                    ax.text(xi + tm*(xj-xi) + nx*arrowsc*0.35, yi + tm*(yj-yi) + ny*arrowsc*0.35,
                            f"w={w:.2g} kN/m", color="#ea580c", fontsize=8,
                            ha="center", fontfamily="monospace",
                            bbox=dict(boxstyle="round,pad=0.2", fc="#fff7ed", ec="#fdba74", lw=0.8))
                else:
                    P, xloc = spec["P"], spec["x"]
                    t = xloc / L
                    x = xi + t*(xj-xi); y = yi + t*(yj-yi)
                    ax.annotate("", xy=(x, y), xytext=(x - nx*arrowsc, y - ny*arrowsc),
                                arrowprops=dict(arrowstyle="-|>", color="#b45309", lw=2.2))
                    ax.text(x + nx*arrowsc*0.45, y + ny*arrowsc*0.45, f"P={P:.2g} kN",
                            color="#b45309", fontsize=8, ha="center", fontfamily="monospace",
                            bbox=dict(boxstyle="round,pad=0.2", fc="#fffbeb", ec="#f59e0b", lw=0.8))

        for nid, ldmap in nodal_loads.items():
            x, y = nodes[nid]
            for ld, val in ldmap.items():
                if ld == 0:  # Fx
                    dx = arrowsc * (1 if val > 0 else -1)
                    ax.annotate("", xy=(x, y), xytext=(x - dx, y),
                                arrowprops=dict(arrowstyle="->", color="#d97706", lw=2))
                    ax.text(x - dx/2, y + arrowsc*0.3, f"{val:.0f} kN",
                            color="#d97706", fontsize=8, ha="center", fontfamily="monospace")
                elif ld == 1:  # Fy
                    dy = arrowsc * (1 if val > 0 else -1)
                    ax.annotate("", xy=(x, y), xytext=(x, y - dy),
                                arrowprops=dict(arrowstyle="->", color="#d97706", lw=2))
                    ax.text(x + arrowsc*0.4, y - dy/2, f"{val:.0f} kN",
                            color="#d97706", fontsize=8, ha="left", fontfamily="monospace")

    # ── Draw DOF arrows (for step 1) ──────────────────────────
    if show_dofs and dof_map is not None:
        colors_dof = ["#1d4ed8","#15803d","#7c3aed"]
        labels_dof = ["u","v","θ"]
        for nid, gdofs in dof_map.items():
            x, y = nodes[nid]
            offsets = [(arrowsc*1.1, 0),(0, arrowsc*1.1),(arrowsc*0.6, arrowsc*0.6)]
            for d, (dx, dy) in enumerate(offsets):
                col = colors_dof[d]
                ax.annotate("", xy=(x+dx, y+dy), xytext=(x, y),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=0.7))
                ax.text(x+dx*1.1, y+dy*1.1, f"d{gdofs[d]}\n({labels_dof[d]})",
                        color=col, fontsize=7, ha="center", fontfamily="monospace")

    # ── Draw reactions ────────────────────────────────────────
    if show_reactions and reactions is not None and dof_map is not None:
        for nid, ldofs in fixed_dofs.items():
            x, y = nodes[nid]
            for ld in ldofs:
                gi = dof_map[nid][ld]
                val = reactions.get(gi, 0)
                if abs(val) < 1e-4: continue
                if ld == 0:
                    dx = arrowsc * (1 if val > 0 else -1)
                    ax.annotate("", xy=(x+dx, y), xytext=(x, y),
                                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))
                    ax.text(x+dx, y-arrowsc*0.4, f"Rx={val:.1f}",
                            color="#dc2626", fontsize=7, ha="center", fontfamily="monospace")
                elif ld == 1:
                    dy = arrowsc * (1 if val > 0 else -1)
                    ax.annotate("", xy=(x, y+dy), xytext=(x, y),
                                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))
                    ax.text(x-arrowsc*0.6, y+dy, f"Ry={val:.1f}",
                            color="#dc2626", fontsize=7, ha="right", fontfamily="monospace")

    margin = span * 0.2
    ax.set_xlim(min(xs)-margin, max(xs)+margin)
    ax.set_ylim(min(ys)-margin, max(ys)+margin)
    ax.set_aspect("equal")
    ax.grid(True, color="#e2e8f0", lw=0.5, alpha=0.8)
    ax.set_xlabel("X (m)", color="#475569", fontfamily="monospace")
    ax.set_ylabel("Y (m)", color="#475569", fontfamily="monospace")
    plt.tight_layout()
    return fig


def _member_diagram_values(x, L, mr, load_specs):
    """Sample local shear/moment from left-end forces and member loads."""
    V = mr["V_i"]
    M = mr["M_i"] + mr["V_i"] * x
    for spec in load_specs:
        if spec["type"] == "UDL":
            w, a, b = spec["w"], spec["a"], spec["b"]
            covered = max(0.0, min(x, b) - a)
            if covered > 0:
                V -= w * covered
                M -= w * covered * (x - (a + covered/2))
        else:
            P, xp = spec["P"], spec["x"]
            if x >= xp:
                V -= P
                M -= P * (x - xp)
    return V, M


def _local_deflection_at(x, L, u_local):
    xi = x / L
    Nu_i = 1 - xi
    Nu_j = xi
    N1, N2, N3, N4 = _beam_shape_functions(x, L)
    u = Nu_i*u_local[0] + Nu_j*u_local[3]
    v = N1*u_local[1] + N2*u_local[2] + N3*u_local[4] + N4*u_local[5]
    return u, v


def draw_bmd_sfd(nodes, elements, member_results, elem_labels=None, member_loads=None):
    """Draw improved bending-moment and shear-force diagrams with point/partial UDL loads."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#f8fafc")
    titles = ["Bending Moment Diagram (kN·m)", "Shear Force Diagram (kN)"]
    colors = ["#7c3aed", "#0891b2"]
    lengths = [max(math.hypot(nodes[nj][0]-nodes[ni][0], nodes[nj][1]-nodes[ni][1]), 1e-9)
               for ni, nj in elements]
    specs_by_element = {i: [] for i in range(len(elements))}
    for spec in normalize_member_loads(member_loads, lengths):
        if 0 <= spec["element"] < len(elements):
            specs_by_element[spec["element"]].append(spec)

    xs = [n[0] for n in nodes]; ys = [n[1] for n in nodes]
    span = max(max(xs)-min(xs), max(ys)-min(ys), 1.0)

    for ax_idx, ax in enumerate(axes):
        ax.set_facecolor("#ffffff")
        ax.tick_params(colors="#334155")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")
        ax.set_title(titles[ax_idx], color="#1e40af", fontfamily="monospace", pad=10)
        ax.grid(True, color="#e2e8f0", lw=0.5)

        for ni, nj in elements:
            ax.plot([nodes[ni][0], nodes[nj][0]], [nodes[ni][1], nodes[nj][1]],
                    color="#94a3b8", lw=1, zorder=1)

        for idx, ((ni, nj), mr) in enumerate(zip(elements, member_results)):
            xi, yi = nodes[ni]; xj, yj = nodes[nj]
            L = lengths[idx]
            alpha = math.atan2(yj-yi, xj-xi)
            perp = (-math.sin(alpha), math.cos(alpha))
            load_specs = specs_by_element.get(idx, mr.get("member_load_specs", []))
            sample_x = sorted(set([0.0, L] +
                                  [k*L/80 for k in range(81)] +
                                  [s["a"] for s in load_specs if s["type"] == "UDL"] +
                                  [s["b"] for s in load_specs if s["type"] == "UDL"] +
                                  [s["x"] for s in load_specs if s["type"] == "Point"]))
            values = []
            for xloc in sample_x:
                V, M = _member_diagram_values(xloc, L, mr, load_specs)
                values.append(M if ax_idx == 0 else V)
            scale = span * 0.16 / max(max(abs(v) for v in values), 1)
            pts_x, pts_y, base_x, base_y = [], [], [], []
            for xloc, val in zip(sample_x, values):
                t = xloc / L
                bx = xi + t*(xj-xi); by = yi + t*(yj-yi)
                base_x.append(bx); base_y.append(by)
                pts_x.append(bx + scale*val*perp[0]); pts_y.append(by + scale*val*perp[1])

            col = colors[ax_idx]
            ax.fill(base_x + pts_x[::-1], base_y + pts_y[::-1], color=col, alpha=0.22)
            ax.plot(pts_x, pts_y, color=col, lw=2)
            for xloc, val in [(sample_x[0], values[0]), (sample_x[-1], values[-1]),
                              (sample_x[int(np.argmax(np.abs(values)))], values[int(np.argmax(np.abs(values)))])]:
                t = xloc / L
                bx = xi + t*(xj-xi); by = yi + t*(yj-yi)
                ax.text(bx + scale*val*perp[0]*1.12, by + scale*val*perp[1]*1.12,
                        f"{val:.1f}", color=col, fontsize=7, ha="center", fontfamily="monospace")
            if elem_labels:
                ax.text((xi+xj)/2, (yi+yj)/2, elem_labels[idx], color="#475569",
                        fontsize=7, ha="center", va="bottom", fontfamily="monospace")

        ax.set_aspect("equal")
        margin = span * 0.25
        ax.set_xlim(min(xs)-margin, max(xs)+margin)
        ax.set_ylim(min(ys)-margin, max(ys)+margin)

    plt.tight_layout()
    return fig


def draw_deflection_diagram(nodes, elements, member_results, scale=0.3, elem_labels=None):
    """Draw a smooth plane-frame deflection diagram using beam shape functions."""
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#f8fafc")
    ax.set_facecolor("#ffffff")
    xs = [n[0] for n in nodes]; ys = [n[1] for n in nodes]
    span = max(max(xs)-min(xs), max(ys)-min(ys), 1.0)
    max_disp = max(max(abs(mr["u_local"][i]) for i in [0, 1, 3, 4]) for mr in member_results) or 1.0
    sc = scale * span / max(max_disp, 1e-12)
    for idx, ((ni, nj), mr) in enumerate(zip(elements, member_results)):
        xi, yi = nodes[ni]; xj, yj = nodes[nj]
        L = max(math.hypot(xj-xi, yj-yi), 1e-9)
        alpha = math.atan2(yj-yi, xj-xi)
        c, s = math.cos(alpha), math.sin(alpha)
        base_x = [xi + t*(xj-xi) for t in np.linspace(0, 1, 30)]
        base_y = [yi + t*(yj-yi) for t in np.linspace(0, 1, 30)]
        ax.plot(base_x, base_y, color="#94a3b8", lw=1.2)
        dxs, dys = [], []
        for xloc in np.linspace(0, L, 50):
            u, v = _local_deflection_at(float(xloc), L, mr["u_local"])
            gx = xi + (xloc/L)*(xj-xi) + sc*(c*u - s*v)
            gy = yi + (xloc/L)*(yj-yi) + sc*(s*u + c*v)
            dxs.append(gx); dys.append(gy)
        ax.plot(dxs, dys, color="#dc2626", lw=2.4)
        if elem_labels:
            ax.text((xi+xj)/2, (yi+yj)/2, elem_labels[idx], color="#475569",
                    fontsize=8, ha="center", fontfamily="monospace")
    ax.scatter(xs, ys, s=45, color="#1e293b", zorder=3)
    ax.set_title("Deflection Diagram (smooth, exaggerated)", color="#1e40af", fontfamily="monospace")
    ax.grid(True, color="#e2e8f0", lw=0.5)
    ax.set_aspect("equal")
    margin = span * 0.25
    ax.set_xlim(min(xs)-margin, max(xs)+margin)
    ax.set_ylim(min(ys)-margin, max(ys)+margin)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "step" not in st.session_state:
    st.session_state.step = 0
if "preset" not in st.session_state:
    st.session_state.preset = list(PRESETS.keys())[0]
if "result" not in st.session_state:
    st.session_state.result = None


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
    st.markdown("### 🧭 Analysis Mode")
    st.success("Plane Frame (2D beam-column): axial, shear, bending, rotations, reactions, SFD/BMD, and deflection diagrams")

    st.divider()
    st.markdown("### ⚙️ Material & Section")
    E_val = st.number_input("E — Elastic Modulus (kN/m²)", value=float(P["E"]),
                            format="%.2e", help="Young's modulus")
    A_val = st.number_input("A — Cross-Section Area (m²)",  value=float(P["A"]),
                            format="%.4f")
    I_val = st.number_input("I — Second Moment (m⁴)",       value=float(P["I"]),
                            format="%.6f")

    st.divider()
    st.markdown("### 📐 Geometry")
    st.caption("Node coordinates (x, y) in metres")
    node_df = pd.DataFrame(
        [{"Node": f"N{i+1}", "x (m)": x, "y (m)": y} for i,(x,y) in enumerate(P["nodes"])]
    )
    node_df = st.data_editor(node_df, num_rows="dynamic", width="stretch",
                              key=f"nodes_{chosen}")

    st.caption("Elements (node index pairs, 1-based)")
    elem_df = pd.DataFrame(
        [{"Elem": f"E{i+1}", "Node i": ni+1, "Node j": nj+1, "Label": lbl}
         for i, ((ni, nj), lbl) in enumerate(zip(P["elements"], P["labels"]))]
    )
    elem_df = st.data_editor(elem_df, num_rows="dynamic", width="stretch",
                              key=f"elems_{chosen}")

    st.divider()
    st.markdown("### 🔩 Boundary Conditions")
    st.caption("Check DOFs to fix (u=horiz, v=vert, θ=rotation)")
    fixed_dofs_ui = {}
    for i in range(len(node_df)):
        def_check = P["fixed_dofs"].get(i, [])
        cols_bc = st.columns(3)
        checks = []
        with cols_bc[0]: checks.append(0 if st.checkbox(f"N{i+1} u",  value=0 in def_check, key=f"u_{i}_{chosen}") else None)
        with cols_bc[1]: checks.append(1 if st.checkbox(f"N{i+1} v",  value=1 in def_check, key=f"v_{i}_{chosen}") else None)
        with cols_bc[2]: checks.append(2 if st.checkbox(f"N{i+1} θ",  value=2 in def_check, key=f"t_{i}_{chosen}") else None)
        fixed = [c for c in checks if c is not None]
        if fixed:
            fixed_dofs_ui[i] = fixed

    st.divider()
    st.markdown("### ➡️ Nodal Loads")
    st.caption("kN (force) or kN·m (moment). Node numbers are 1-based. Positive = +ve axis direction.")
    load_rows = []
    for nid, ldmap in P["nodal_loads"].items():
        for ld, val in ldmap.items():
            load_rows.append({"Node": nid+1, "DOF (0=u,1=v,2=θ)": ld, "Value (kN or kN·m)": val})
    load_df = pd.DataFrame(load_rows if load_rows else
                           [{"Node": 1, "DOF (0=u,1=v,2=θ)": 1, "Value (kN or kN·m)": 0.0}])
    load_df = st.data_editor(load_df, num_rows="dynamic", width="stretch",
                              key=f"loads_{chosen}")

    st.divider()
    st.markdown("### ⏬ Standard Member Loads")
    st.caption("Plane-frame member loads in local transverse direction. UDL supports custom start/end spans; point loads use position x from node i. Positive = downward.")
    preset_lengths = [math.hypot(P["nodes"][nj][0]-P["nodes"][ni][0], P["nodes"][nj][1]-P["nodes"][ni][1])
                      for ni, nj in P["elements"]]
    member_rows = []
    for spec in normalize_member_loads(P.get("member_loads", []), preset_lengths):
        if spec["type"] == "Point":
            member_rows.append({"Element": f"E{spec['element']+1}", "Type": "Point", "Load ↓": spec["P"],
                                "Start x": "", "End x": "", "Position x": spec["x"]})
        else:
            member_rows.append({"Element": f"E{spec['element']+1}", "Type": "UDL", "Load ↓": spec["w"],
                                "Start x": spec["a"], "End x": spec["b"], "Position x": ""})
    load_spec_df = pd.DataFrame(member_rows if member_rows else [
        {"Element": "E1", "Type": "UDL", "Load ↓": 0.0, "Start x": 0.0, "End x": "", "Position x": ""}
    ])
    load_spec_df = st.data_editor(
        load_spec_df, num_rows="dynamic", width="stretch", key=f"member_loads_{chosen}",
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["UDL", "Point"]),
            "Load ↓": st.column_config.NumberColumn("Load ↓", help="UDL in kN/m or point load in kN"),
        },
    )

    st.divider()
    st.markdown("### ✅ Quality Controls")
    st.caption("Model checks include invalid connectivity, duplicate members, zero-length elements, property positivity, and matrix conditioning.")
    run_btn = st.button("🚀 Run DSM Analysis", type="primary", width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
#  PARSE UI INPUTS  (skip blank / None rows from data_editor)
# ══════════════════════════════════════════════════════════════════════════════
def _is_valid_row(r, keys):
    """Return True only if all required keys have non-None, non-empty values."""
    for k in keys:
        v = r.get(k)
        if v is None or pd.isna(v) or str(v).strip() in ("", "None", "nan"):
            return False
    return True


def _optional_float(value, default=0.0):
    if value is None or pd.isna(value) or str(value).strip() in ("", "None", "nan"):
        return default
    return float(value)

nodes_parsed = [
    (float(r["x (m)"]), float(r["y (m)"]))
    for _, r in node_df.iterrows()
    if _is_valid_row(r, ["x (m)", "y (m)"])
]
elems_parsed = [
    (int(r["Node i"]) - 1, int(r["Node j"]) - 1)
    for _, r in elem_df.iterrows()
    if _is_valid_row(r, ["Node i", "Node j"])
]
elem_labels_p = [
    str(r["Label"]) if _is_valid_row(r, ["Label"]) else f"E{i+1}"
    for i, (_, r) in enumerate(elem_df.iterrows())
    if _is_valid_row(r, ["Node i", "Node j"])
]

nodal_loads_parsed = {}
for _, r in load_df.iterrows():
    if not _is_valid_row(r, ["Node", "DOF (0=u,1=v,2=θ)", "Value (kN or kN·m)"]):
        continue
    nid = int(r["Node"]) - 1; ld = int(r["DOF (0=u,1=v,2=θ)"]); val = float(r["Value (kN or kN·m)"])
    if abs(val) > 1e-10:
        nodal_loads_parsed.setdefault(nid, {})[ld] = val

member_loads_parsed = []
for _, r in load_spec_df.iterrows():
    if not _is_valid_row(r, ["Element", "Type", "Load ↓"]):
        continue
    raw_elem = str(r["Element"]).strip().upper().replace("E", "")
    if not raw_elem:
        continue
    eid = int(float(raw_elem)) - 1
    load_type = _clean_load_type(r["Type"])
    load_value = float(r["Load ↓"])
    if abs(load_value) <= 1e-10:
        continue
    if load_type == "Point":
        x = _optional_float(r.get("Position x"), 0.0)
        member_loads_parsed.append({"element": eid, "type": "Point", "P": load_value, "x": x})
    else:
        a = _optional_float(r.get("Start x"), 0.0)
        raw_b = r.get("End x")
        member_loads_parsed.append({"element": eid, "type": "UDL", "w": load_value,
                                    "a": a, "b": None if raw_b is None or pd.isna(raw_b) or str(raw_b).strip() == "" else float(raw_b)})

if run_btn:
    if len(nodes_parsed) < 2:
        st.error("🚨 Need at least 2 nodes.")
    elif len(elems_parsed) < 1:
        st.error("🚨 Need at least 1 element.")
    elif E_val <= 0 or A_val <= 0 or I_val <= 0:
        st.error("🚨 Material and section properties must be positive: E > 0, A > 0, I > 0.")
    elif not fixed_dofs_ui:
        st.error("🚨 No boundary conditions defined — structure is a mechanism.")
    else:
        max_node = len(nodes_parsed) - 1
        bad = [(i, ni, nj) for i, (ni, nj) in enumerate(elems_parsed)
               if ni > max_node or nj > max_node or ni < 0 or nj < 0]
        zero_len = []
        duplicate = []
        seen = set()
        for i, (ni, nj) in enumerate(elems_parsed):
            if 0 <= ni <= max_node and 0 <= nj <= max_node:
                if ni == nj or math.hypot(nodes_parsed[nj][0]-nodes_parsed[ni][0], nodes_parsed[nj][1]-nodes_parsed[ni][1]) < 1e-9:
                    zero_len.append(i)
                key = tuple(sorted((ni, nj)))
                if key in seen:
                    duplicate.append(i)
                seen.add(key)
        bad_udl = []
        bad_span = []
        elem_lengths = [math.hypot(nodes_parsed[nj][0]-nodes_parsed[ni][0], nodes_parsed[nj][1]-nodes_parsed[ni][1])
                        for ni, nj in elems_parsed]
        for spec in member_loads_parsed:
            eid = spec["element"]
            if eid < 0 or eid >= len(elems_parsed):
                bad_udl.append(eid)
                continue
            L = elem_lengths[eid]
            if spec["type"] == "UDL":
                b = L if spec.get("b") is None else spec["b"]
                if spec["a"] < 0 or b > L or b <= spec["a"]:
                    bad_span.append(f"E{eid+1} UDL span must satisfy 0 ≤ start < end ≤ {L:.3f} m")
            elif spec["x"] < 0 or spec["x"] > L:
                bad_span.append(f"E{eid+1} point-load position must satisfy 0 ≤ x ≤ {L:.3f} m")
        bad_loads = [(nid, ld) for nid, ldm in nodal_loads_parsed.items() for ld in ldm if nid < 0 or nid > max_node or ld not in (0, 1, 2)]
        if bad:
            for i, ni, nj in bad:
                st.error(f"🚨 Element E{i+1} references node {max(ni,nj)+1} but only "
                         f"{len(nodes_parsed)} nodes exist (N1–N{len(nodes_parsed)}).")
            st.stop()
        if zero_len:
            st.error("🚨 Zero-length elements detected: " + ", ".join(f"E{i+1}" for i in zero_len))
            st.stop()
        if duplicate:
            st.warning("⚠️ Duplicate connectivity detected for: " + ", ".join(f"E{i+1}" for i in duplicate))
        if bad_udl:
            st.error("🚨 Member load references invalid element(s): " + ", ".join(f"E{i+1}" for i in bad_udl))
            st.stop()
        if bad_span:
            for msg in bad_span:
                st.error(f"🚨 {msg}")
            st.stop()
        if bad_loads:
            st.error("🚨 Nodal load references invalid node or DOF. Valid DOFs are 0=u, 1=v, 2=θ.")
            st.stop()

        with st.spinner("Assembling stiffness matrices and solving…"):
            try:
                st.session_state.result = run_dsm(
                    nodes_parsed, elems_parsed, fixed_dofs_ui,
                    nodal_loads_parsed, member_loads_parsed, E_val, A_val, I_val)
            except ValueError as err:
                st.error(f"🚨 {err}")
                st.stop()
        st.session_state.step = 0
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='text-align:center; font-family:Courier New; color:#1e40af; letter-spacing:2px;'>
  🏗️ 2D / PLANE FRAME ANALYZER — DIRECT STIFFNESS METHOD
</h1>
<p style='text-align:center; color:#475569; font-family:Courier New; font-size:14px;'>
  Professional Plane-Frame Analysis &nbsp;|&nbsp; DSM Glass-Box Educational Tool
</p>
""", unsafe_allow_html=True)
st.divider()

res = st.session_state.result

if res is None:
    # ── Welcome screen ─────────────────────────────────────────
    fig_welcome = draw_frame(
        nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed,
        member_loads=member_loads_parsed, elem_labels=elem_labels_p, node_labels=True)
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
  <li>Problem Setup — geometry, properties</li>
  <li>DOF Numbering — 3 DOFs per node</li>
  <li>Local Stiffness [k] — 6×6 per element</li>
  <li>Transformation [T] — rotate to global</li>
  <li>Global Element [K]ₑ = [T]ᵀ[k][T]</li>
  <li>Assembly — build nDOF × nDOF [K] and {F}</li>
  <li>Partition & Apply BCs — extract [Kff]</li>
  <li>Solve — {Uf} = [Kff]⁻¹ {Ff}</li>
  <li>Member End Forces — {f'} = [k][T]{u}</li>
  <li>Reactions & Equilibrium Check</li>
</ol>
""", unsafe_allow_html=True)

    with st.expander("📚 DSM Equation Reference — click to expand", expanded=False):
        st.markdown("""
<div class='formula-box'>
Global stiffness assembly:   [K] = Σ [T]ᵀ [k] [T]   (element by element)

Partitioned system:          [Kff]{Uf} = {Ff}  −  [Kfc]{Uc}

Local stiffness (Euler-Bernoulli beam-column, 6×6):

       ┌  EA/L    0       0      -EA/L   0       0    ┐
       │   0    12EI/L³  6EI/L²   0   -12EI/L³  6EI/L² │
[k] = │   0     6EI/L²  4EI/L    0    -6EI/L²  2EI/L  │
       │ -EA/L   0       0       EA/L   0       0    │
       │   0   -12EI/L³ -6EI/L²  0    12EI/L³ -6EI/L² │
       └   0     6EI/L²  2EI/L   0    -6EI/L²  4EI/L  ┘

Transformation (2D, angle α from x-axis):
       ┌  c   s   0   0   0   0 ┐        c = cos(α)
[T] = │ -s   c   0   0   0   0 │        s = sin(α)
       │  0   0   1   0   0   0 │
       │  0   0   0   c   s   0 │
       │  0   0   0  -s   c   0 │
       └  0   0   0   0   0   1 ┘

Member end forces (local):   {f'} = [k][T]{u_global}
Reactions:                   {R}  = [K]{U} − {F}
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

# Tab bar
tab_cols = st.columns(len(STEP_NAMES))
for i, (col, name) in enumerate(zip(tab_cols, STEP_NAMES)):
    with col:
        btn_type = "primary" if i == st.session_state.step else "secondary"
        if st.button(name.split("·")[0].strip(), key=f"tab_{i}",
                     width="stretch", type=btn_type):
            st.session_state.step = i

st.progress((st.session_state.step) / (len(STEP_NAMES)-1))

# Prev / Next
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

# ══════════════════════════════════════════════════════════════════════════════
#  STEP CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Step 0: Problem Setup ─────────────────────────────────────────────────────
if step == 0:
    col_v, col_t = st.columns([1.4, 1])
    with col_v:
        fig0 = draw_frame(nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed,
                          member_loads=member_loads_parsed, elem_labels=elem_labels_p)
        st.pyplot(fig0, width="stretch")

    with col_t:
        st.markdown("<div class='step-card'>", unsafe_allow_html=True)
        st.markdown("<span class='step-title'>Problem Setup</span>", unsafe_allow_html=True)
        st.markdown("<span class='step-subtitle'>Define geometry, section, material, loads</span>",
                    unsafe_allow_html=True)

        st.markdown("**📍 Nodes**")
        node_rows = [{"ID": f"N{i+1}", "x (m)": f"{x:.3f}", "y (m)": f"{y:.3f}"}
                     for i,(x,y) in enumerate(nodes_parsed)]
        st.dataframe(pd.DataFrame(node_rows), width="stretch", hide_index=True)

        st.markdown("**🔗 Elements**")
        elem_rows = []
        for i, (e, lbl) in enumerate(zip(ed, elem_labels_p)):
            elem_rows.append({"ID":f"E{i+1}", "Label":lbl, "Node i":f"N{e['ni']+1}",
                              "Node j":f"N{e['nj']+1}", "L (m)":f"{e['L']:.3f}",
                              "α (°)":f"{e['alpha_deg']:.2f}"})
        st.dataframe(pd.DataFrame(elem_rows), width="stretch", hide_index=True)

        st.markdown("**⚙️ Material / Section (same for all elements)**")
        st.dataframe(pd.DataFrame([{"E (kN/m²)": f"{E_val:.3e}", "A (m²)": f"{A_val:.4f}",
                                    "I (m⁴)": f"{I_val:.6f}"}]),
                     width="stretch", hide_index=True)

        st.markdown("**➡️ Applied Loads**")
        dof_names = {0:"Fx (kN)", 1:"Fy (kN)", 2:"M (kN·m)"}
        lrows = [{"Source":"Node", "ID":f"N{nid+1}", "Load":dof_names[ld], "Value":f"{val:.1f}"}
                 for nid, ldmap in nodal_loads_parsed.items()
                 for ld, val in ldmap.items()]
        for spec in normalize_member_loads(member_loads_parsed, [e["L"] for e in ed]):
            if spec["type"] == "Point":
                lrows.append({"Source":"Member Point", "ID":f"E{spec['element']+1}",
                              "Load":f"P↓ at x={spec['x']:.3f} m", "Value":f"{spec['P']:.2f} kN"})
            else:
                lrows.append({"Source":"Member UDL", "ID":f"E{spec['element']+1}",
                              "Load":f"w↓ from {spec['a']:.3f}–{spec['b']:.3f} m", "Value":f"{spec['w']:.2f} kN/m"})
        st.dataframe(pd.DataFrame(lrows if lrows else [{"Note":"No applied loads"}]),
                     width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ── Step 1: DOF Numbering ─────────────────────────────────────────────────────
elif step == 1:
    col_v, col_t = st.columns([1.4, 1])
    with col_v:
        fig1 = draw_frame(nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed,
                          member_loads=member_loads_parsed, U=res["U"], dof_map=res["dof_map"], show_dofs=True, show_loads=False)
        st.pyplot(fig1, width="stretch")

    with col_t:
        st.markdown("<div class='step-card'>", unsafe_allow_html=True)
        st.markdown("<span class='step-title'>Step 1 — DOF Numbering</span>",
                    unsafe_allow_html=True)
        st.markdown("<span class='step-subtitle'>Each node gets 3 global DOFs: <b>u</b> (horiz), <b>v</b> (vert), <b>θ</b> (rotation)</span>",
                    unsafe_allow_html=True)
        with st.expander("📐 Theory & Formula", expanded=False):
            st.markdown("""
<div class='formula-box'>
Node i  →  Global DOFs: [ 3i ,  3i+1 ,  3i+2 ]
                              u     v      θ
Total DOFs = 3 × nNodes
</div>
""", unsafe_allow_html=True)

        dof_rows = []
        for i in range(nn):
            dm = res["dof_map"][i]
            f_tags = []
            for li, gi in enumerate(dm):
                tag = "FREE" if gi in res["free_global"] else "FIXED"
                f_tags.append(tag)
            dof_rows.append({
                "Node": f"N{i+1}",
                "u → d": dm[0], "v → d": dm[1], "θ → d": dm[2],
                "u BC": f_tags[0], "v BC": f_tags[1], "θ BC": f_tags[2],
            })
        df_dofs = pd.DataFrame(dof_rows)

        def color_bc(val):
            if val == "FIXED": return "background-color:#fee2e2;color:#991b1b"
            if val == "FREE":  return "background-color:#dcfce7;color:#166534"
            return ""
        styled_dofs = df_dofs.style.map(color_bc, subset=["u BC","v BC","θ BC"])
        st.dataframe(styled_dofs, width="stretch", hide_index=True)

        st.markdown(f"""
**Summary:**
- Total DOFs: **{res['nDOF']}**
- Fixed DOFs: **{len(res['fixed_global'])}** → {res['fixed_global']}
- Free DOFs:  **{len(res['free_global'])}**  → {res['free_global']}
""")
        st.markdown("</div>", unsafe_allow_html=True)


# ── Step 2: Local Stiffness ────────────────────────────────────────────────────
elif step == 2:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 2 — Local Stiffness Matrix [k] (6×6)</span>",
                unsafe_allow_html=True)
    st.markdown("<span class='step-subtitle'>Euler-Bernoulli beam-column element in its own local coordinate system</span>",
                unsafe_allow_html=True)
    with st.expander("📐 Theory & Formula", expanded=False):
        st.markdown("""
<div class='formula-box'>
Local DOF order: [u_i  v_i  θ_i  |  u_j  v_j  θ_j]
                  axial shear rot. | axial shear rot.
                  ←─── Node i ───► | ←─── Node j ───►

EA/L    =  Axial stiffness
12EI/L³ =  Transverse (bending) stiffness
6EI/L²  =  Bending-shear coupling
4EI/L   =  End moment (same end)
2EI/L   =  End moment (far end, carry-over)
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    elem_sel = st.selectbox("Select Element:", [f"E{i+1} — {lbl}" for i,(lbl) in
                                                 enumerate(elem_labels_p)])
    ei = int(elem_sel.split("E")[1].split(" ")[0]) - 1
    e  = ed[ei]

    st.markdown(f"**E{ei+1} ({elem_labels_p[ei]}) — L = {e['L']:.3f} m, α = {e['alpha_deg']:.2f}°**")

    a  = E_val*A_val/e['L']
    b  = 12*E_val*I_val/e['L']**3
    c  = 6*E_val*I_val/e['L']**2
    d  = 4*E_val*I_val/e['L']
    ev = 2*E_val*I_val/e['L']

    st.markdown(f"""
<div class='formula-box'>
  EA/L  = {E_val:.2e} × {A_val:.4f} / {e['L']:.3f}  = {a:.4e} kN/m
  12EI/L³ = 12×{E_val:.2e}×{I_val:.6f}/{e['L']:.3f}³ = {b:.4e} kN/m
  6EI/L²  = {c:.4e} kN
  4EI/L   = {d:.4e} kN·m
  2EI/L   = {ev:.4e} kN·m
</div>
""", unsafe_allow_html=True)

    dof_lbls = [f"u{e['ni']+1}", f"v{e['ni']+1}", f"θ{e['ni']+1}", f"u{e['nj']+1}", f"v{e['nj']+1}", f"θ{e['nj']+1}"]
    show_matrix(e["k_loc"], row_labels=dof_lbls, col_labels=dof_lbls,
                caption=f"[k] for E{ei+1} (kN/m units; diagonal = stiffness coefficients)")


# ── Step 3: Transformation Matrix ─────────────────────────────────────────────
elif step == 3:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 3 — Transformation Matrix [T] (6×6)</span>",
                unsafe_allow_html=True)
    st.markdown("<span class='step-subtitle'>Rotates local DOFs to global x-y coordinate system</span>",
                unsafe_allow_html=True)
    with st.expander("📐 Theory & Formula", expanded=False):
        st.markdown("""
<div class='formula-box'>
  Direction cosines:   c = cos(α),  s = sin(α)
  α = angle of member axis measured from global +X axis

  [λ] = ⌈ c   s   0 ⌉       [T] = block-diagonal:  ⌈ [λ]   [0] ⌉
        │-s   c   0 │                                ⌊ [0]   [λ] ⌋
        ⌊ 0   0   1 ⌋

  Local coords = [T] × {u_global}       →   {u_loc} = [T]{u}
  Global forces from local = [T]ᵀ × {f_local}
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    elem_sel = st.selectbox("Select Element:", [f"E{i+1} — {lbl}" for i, lbl in
                                                 enumerate(elem_labels_p)])
    ei = int(elem_sel.split("E")[1].split(" ")[0]) - 1
    e  = ed[ei]
    st.markdown(f"**E{ei+1}: α = {e['alpha_deg']:.3f}°  →  c = {e['c']:.4f},  s = {e['s']:.4f}**")

    c1, c2 = st.columns(2)
    with c1:
        dof_lbls = [f"u{e['ni']+1}", f"v{e['ni']+1}", f"θ{e['ni']+1}", f"u{e['nj']+1}", f"v{e['nj']+1}", f"θ{e['nj']+1}"]
        show_matrix(e["T6"], row_labels=dof_lbls, col_labels=dof_lbls,
                    caption="[T] — 6×6 transformation matrix")
    with c2:
        fig3 = draw_frame(nodes_parsed, elems_parsed, fixed_dofs_ui, {},
                          elem_labels=elem_labels_p)
        # Annotate angle
        ni, nj = nodes_parsed[e["ni"]], nodes_parsed[e["nj"]]
        fig3.axes[0].annotate(
            f"α={e['alpha_deg']:.1f}°",
            xy=((ni[0]+nj[0])/2, (ni[1]+nj[1])/2),
            color="#7c3aed", fontsize=10, fontfamily="monospace",
            bbox=dict(boxstyle="round", fc="#ffffff", ec="#7c3aed"),
        )
        st.pyplot(fig3, width="stretch")


# ── Step 4: Global Element Stiffness ──────────────────────────────────────────
elif step == 4:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 4 — Global Element Stiffness [K]ₑ = [T]ᵀ [k] [T]</span>",
                unsafe_allow_html=True)
    st.markdown("<span class='step-subtitle'>Each element's stiffness expressed in global x-y axes, ready for assembly</span>",
                unsafe_allow_html=True)
    with st.expander("📐 Theory & Formula", expanded=False):
        st.markdown("""
<div class='formula-box'>
  [K]ₑ = [T]ᵀ · [k] · [T]     (6×6 in global DOF coordinates)

  This is the "scatter" matrix whose entries will be added to the
  corresponding rows/columns of the global stiffness matrix [K].
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    elem_sel = st.selectbox("Select Element:", [f"E{i+1} — {lbl}" for i, lbl in
                                                 enumerate(elem_labels_p)])
    ei = int(elem_sel.split("E")[1].split(" ")[0]) - 1
    e  = ed[ei]
    dof_lbls = [f"d{g}" for g in e["gdofs"]]
    show_matrix(e["k_glob"], row_labels=dof_lbls, col_labels=dof_lbls,
                caption=f"[K]ₑ for E{ei+1} mapped to global DOFs {e['gdofs']}")

    st.markdown(f"**Global DOF mapping for E{ei+1}:**")
    dm_rows = [{"Local DOF": f"u{e['ni']+1}/v{e['ni']+1}/θ{e['ni']+1}/u{e['nj']+1}/v{e['nj']+1}/θ{e['nj']+1}".split("/")[li],
                "Global Index": g}
               for li, g in enumerate(e["gdofs"])]
    st.dataframe(pd.DataFrame(dm_rows), width="stretch", hide_index=True)


# ── Step 5: Assembly ──────────────────────────────────────────────────────────
elif step == 5:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 5 — Global Assembly [K] and {F}</span>",
                unsafe_allow_html=True)
    st.markdown("<span class='step-subtitle'>All element matrices scattered into the global nDOF×nDOF system</span>",
                unsafe_allow_html=True)
    with st.expander("📐 Theory & Formula", expanded=False):
        st.markdown("""
<div class='formula-box'>
  [K] = Σ (element by element scatter-add of [K]ₑ)
  {F} = applied nodal loads vector (size nDOF)

  Scatter rule: K[row_global, col_global] += Ke[row_local, col_local]
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    nDOF = res["nDOF"]
    dof_lbls = [f"d{i}" for i in range(nDOF)]
    col_k, col_f = st.columns([3, 1])
    with col_k:
        show_matrix(res["K"], row_labels=dof_lbls, col_labels=dof_lbls,
                    caption=f"Global [K] — {nDOF}×{nDOF} (kN/m)",
                    highlight_rows=res["free_global"], highlight_cols=res["free_global"])
    with col_f:
        show_vector(res["F"], labels=dof_lbls, caption="{F} — Load vector (kN / kN·m)",
                    highlight=[i for i,v in enumerate(res["F"]) if abs(v) > 1e-10])

    st.info(f"ℹ️ Highlighted rows/columns = **Free DOFs** {res['free_global']} that will be solved. "
            f"Fixed DOFs {res['fixed_global']} are eliminated in the next step.")


# ── Step 6: Partitioning & BCs ────────────────────────────────────────────────
elif step == 6:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 6 — Partition & Apply Boundary Conditions</span>",
                unsafe_allow_html=True)
    st.markdown("<span class='step-subtitle'>Delete fixed DOF rows/columns → reduced system [Kff]{Uf} = {Ff}</span>",
                unsafe_allow_html=True)
    with st.expander("📐 Theory & Formula", expanded=False):
        st.markdown("""
<div class='formula-box'>
  Partition: DOFs split into FREE (f) and CONSTRAINED (c)

  ┌ Kff  Kfc ┐ ┌ Uf ┐   ┌ Ff ┐
  │          │ │    │ = │    │
  └ Kcf  Kcc ┘ └ Uc ┘   └ Fc ┘

  Since Uc = 0 (all fixed DOFs have zero prescribed displacement):
       [Kff] {Uf} = {Ff}   ← this is what we solve!

  Size reduction: nDOF → nFree × nFree
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    ff = res["free_global"]
    nf = len(ff)
    dof_lbls_f = [f"d{g}" for g in ff]

    col_kff, col_ff = st.columns([3, 1])
    with col_kff:
        show_matrix(res["Kff"], row_labels=dof_lbls_f, col_labels=dof_lbls_f,
                    caption=f"[Kff] — Reduced stiffness {nf}×{nf}")
    with col_ff:
        show_vector(res["Ff"], labels=dof_lbls_f, caption="{Ff} — Reduced loads")

    st.markdown(f"""
- **Full system:**   {res['nDOF']} × {res['nDOF']}  →  **Reduced system:** {nf} × {nf}
- Eliminated DOFs: {res['fixed_global']}  (all prescribed = 0 in this problem)
""")


# ── Step 7: Solve ─────────────────────────────────────────────────────────────
elif step == 7:
    col_v, col_t = st.columns([1.4, 1])
    with col_v:
        fig7 = draw_frame(nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed,
                          member_loads=member_loads_parsed, U=res["U"], dof_map=res["dof_map"],
                          show_deformed=True, show_loads=True)
        st.pyplot(fig7, width="stretch")
        st.caption("— Nodal deformed shape (dashed red, exaggerated scale)")
        fig_def = draw_deflection_diagram(nodes_parsed, elems_parsed, res["member_results"], elem_labels=elem_labels_p)
        st.pyplot(fig_def, width="stretch")
        plt.close(fig_def)
        st.caption("— Smooth member deflection diagram using plane-frame interpolation functions")

    with col_t:
        st.markdown("<div class='step-card'>", unsafe_allow_html=True)
        st.markdown("<span class='step-title'>Step 7 — Solve {Uf} = [Kff]⁻¹ {Ff}</span>",
                    unsafe_allow_html=True)
        st.markdown("<span class='step-subtitle'>Gaussian elimination with partial pivoting → nodal displacements</span>",
                    unsafe_allow_html=True)
        with st.expander("📐 Theory & Formula", expanded=False):
            st.markdown("""
<div class='formula-box'>
  [Kff] · {Uf} = {Ff}
  Solved by numpy.linalg.solve (LU factorization)
  Units: metres (m) for translations, radians (rad) for rotations
</div>
""", unsafe_allow_html=True)

        nDOF = res["nDOF"]
        dof_lbls = [f"d{i}" for i in range(nDOF)]
        U_tagged = []
        for gi in range(nDOF):
            tag = "FIXED=0" if gi in res["fixed_global"] else "SOLVED"
            dof_name = ["u","v","θ"][gi % 3]
            node_id  = gi // 3
            U_tagged.append({
                "DOF": f"d{gi}", "Node": f"N{node_id+1}", "Type": dof_name,
                "Value": fmt_val(res["U"][gi], 6),
                "Status": tag,
            })
        df_U = pd.DataFrame(U_tagged)
        def col_status(val):
            return "background-color:#dcfce7;color:#166534" if val=="SOLVED" else \
                   "background-color:#f1f5f9;color:#475569"
        st.dataframe(df_U.style.map(col_status, subset=["Status"]),
                     width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ── Step 8: Member End Forces ─────────────────────────────────────────────────
elif step == 8:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 8 — Member End Forces {f'} = [k][T]{u_global}</span>",
                unsafe_allow_html=True)
    st.markdown("<span class='step-subtitle'>Recover axial force N, shear V, moment M at each member end in LOCAL axes</span>",
                unsafe_allow_html=True)
    with st.expander("📐 Theory & Formula", expanded=False):
        st.markdown("""
<div class='formula-box'>
  {u_local} = [T] · {u_global_element}     ← transform displacements to local
  {f_local} = [k] · {u_local} − {p_fixed}  ← include equivalent UDL fixed-end effects

  Sign convention (local x = member axis, +ve left to right):
    N  — axial  (+ve = tension)
    V  — shear  (+ve = upward at i-end)
    M  — moment (+ve = sagging / counter-clockwise)
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    force_rows = []
    for i, (mr, e) in enumerate(zip(res["member_results"], ed)):
        force_rows.append({
            "Elem": f"E{i+1}", "Label": elem_labels_p[i],
            "N_i (kN)": f"{mr['N_i']:.3f}", "V_i (kN)": f"{mr['V_i']:.3f}",
            "M_i (kN·m)": f"{mr['M_i']:.3f}",
            "N_j (kN)": f"{mr['N_j']:.3f}", "V_j (kN)": f"{mr['V_j']:.3f}",
            "M_j (kN·m)": f"{mr['M_j']:.3f}",
        })
    st.dataframe(pd.DataFrame(force_rows), width="stretch", hide_index=True)

    # Local force vectors for selected element
    st.divider()
    elem_sel = st.selectbox("Detail view — Select Element:", [f"E{i+1} — {lbl}" for i, lbl in
                                                               enumerate(elem_labels_p)])
    ei = int(elem_sel.split("E")[1].split(" ")[0]) - 1
    mr = res["member_results"][ei]
    e  = ed[ei]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**{u_global} for element**")
        gdof_lbls = [f"d{g}" for g in e["gdofs"]]
        show_vector(mr["u_global"], labels=gdof_lbls, caption="Global displacements (m / rad)")
    with c2:
        st.markdown("**{u_local} = [T]·{u_global}**")
        loc_lbls = [f"u'{e['ni']+1}", f"v'{e['ni']+1}", f"θ'{e['ni']+1}", f"u'{e['nj']+1}", f"v'{e['nj']+1}", f"θ'{e['nj']+1}"]
        show_vector(mr["u_local"], labels=loc_lbls, caption="Local displacements (m / rad)")
    with c3:
        st.markdown("**{f_local} = [k]·{u_local} − {p_fixed}**")
        show_vector(mr["f_local"], labels=loc_lbls, caption="Local end forces (kN / kN·m)")

    st.divider()
    fig_bmd = draw_bmd_sfd(nodes_parsed, elems_parsed, res["member_results"], elem_labels_p, member_loads_parsed)
    st.pyplot(fig_bmd, width="stretch")


# ── Step 9: Reactions ─────────────────────────────────────────────────────────
elif step == 9:
    col_v, col_t = st.columns([1.4, 1])
    with col_v:
        fig9 = draw_frame(nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed,
                          member_loads=member_loads_parsed, U=res["U"], dof_map=res["dof_map"],
                          show_deformed=True, show_reactions=True,
                          reactions=res["reactions"])
        st.pyplot(fig9, width="stretch")

    with col_t:
        st.markdown("<div class='step-card'>", unsafe_allow_html=True)
        st.markdown("<span class='step-title'>Step 9 — Support Reactions & Equilibrium</span>",
                    unsafe_allow_html=True)
        st.markdown("<span class='step-subtitle'>{R} = [K]{U} − {F}  at fixed DOFs</span>",
                    unsafe_allow_html=True)
        with st.expander("📐 Theory & Formula", expanded=False):
            st.markdown("""
<div class='formula-box'>
  Reactions are the forces the supports must exert to
  maintain equilibrium. They are recovered as:

      {R_fixed} = [Kcf]{Uf} + [Kcc]{Uc} − {Fc}
                = [K]{U} − {F}  (at fixed DOF rows)

  Equilibrium check:
      ΣFx = 0  (sum of all horizontal reactions + loads)
      ΣFy = 0  (sum of all vertical reactions + loads)
      ΣM  = 0  (sum of all moment reactions + loads)
</div>
""", unsafe_allow_html=True)

        rx_rows = []
        dof_names_map = {0:"Rx (kN)", 1:"Ry (kN)", 2:"RM (kN·m)"}
        for gi, val in sorted(res["reactions"].items()):
            nid  = gi // 3; ld = gi % 3
            rx_rows.append({"Node": f"N{nid+1}", "DOF": f"d{gi}",
                            "Type": dof_names_map[ld], "Reaction": f"{val:.4f}"})
        st.dataframe(pd.DataFrame(rx_rows), width="stretch", hide_index=True)

        st.markdown("**⚖️ Global Equilibrium Check**")
        eq = res["eq_check"]
        tol = 1e-3
        for label, val in eq.items():
            passed = abs(val) < tol
            badge  = "✅" if passed else "❌"
            color  = "#166534" if passed else "#991b1b"
            st.markdown(
                f"<span style='color:{color}; font-family:Courier New; font-size:14px;'>"
                f"{badge} {label} = {val:.6f} kN  "
                f"({'OK — equilibrium satisfied' if passed else 'WARN — check model'})"
                f"</span>", unsafe_allow_html=True)

        # Applied load summary vs reactions
        st.divider()
        st.markdown("**📋 Load vs. Reaction Summary**")
        total_Fx = float(sum(res["F"][0::3]))
        total_Fy = float(sum(res["F"][1::3]))
        rx_Fx = sum(v for gi, v in res["reactions"].items() if gi % 3 == 0)
        rx_Fy = sum(v for gi, v in res["reactions"].items() if gi % 3 == 1)
        st.markdown(f"""
| Component | Applied Load | Reactions | Net |
|-----------|-------------|-----------|-----|
| Horizontal (Fx) | {total_Fx:.3f} kN | {rx_Fx:.3f} kN | {total_Fx+rx_Fx:.6f} |
| Vertical (Fy)   | {total_Fy:.3f} kN | {rx_Fy:.3f} kN | {total_Fy+rx_Fy:.6f} |
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
**🧮 DSM Formula Card**
- [k] = 6×6 Euler-Bernoulli stiffness
- [T] = block-diagonal rotation matrix
- [K]ₑ = [T]ᵀ[k][T] (global element)
- [K] = Σ[K]ₑ (scatter-add assembly)
""")
with ref_c3:
    st.markdown("""
**🚀 Deploy This App**
```bash
pip install streamlit numpy pandas matplotlib
streamlit run app.py
```
Or push to GitHub and connect at  
[share.streamlit.io](https://share.streamlit.io)
""")
