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
#  PRESET PROBLEMS (Updated with member loads and hinge flags)
# ══════════════════════════════════════════════════════════════════════════════
PRESETS = {
    "Portal Frame (Wind + UDL)": {
        "desc": "Fixed-base portal frame — wind nodal load + 20 kN/m gravity UDL on beam",
        "nodes": [(0,0),(0,4),(6,4),(6,0)],
        "elements": [(0,1,False,False),(1,2,False,False),(2,3,False,False)],
        "fixed_dofs": {0:[0,1,2], 3:[0,1,2]},
        "E": 200e6, "A": 0.01, "I": 1e-4,
        "nodal_loads": {1:{0:15.0}},
        "member_loads": {1: -20.0},
        "labels": ["Left Col","Beam","Right Col"],
    },
    "Frame with Internal Hinge": {
        "desc": "Portal frame with an internal hinge in the beam and UDL applied.",
        "nodes": [(0,0),(0,4),(3,4),(6,4),(6,0)],
        "elements": [(0,1,False,False),(1,2,False,False),(2,3,True,False),(3,4,False,False)],
        "fixed_dofs": {0:[0,1,2], 4:[0,1,2]},
        "E": 200e6, "A": 0.01, "I": 1e-4,
        "nodal_loads": {},
        "member_loads": {1: -30.0, 2: -30.0},
        "labels": ["Left Col","Beam L","Beam R (Hinged)","Right Col"],
    },
    "Two-Span Continuous Beam": {
        "desc": "Continuous beam on pin + 2 rollers — UDL on entire length",
        "nodes": [(0,0),(5,0),(10,0)],
        "elements": [(0,1,False,False),(1,2,False,False)],
        "fixed_dofs": {0:[0,1], 1:[1], 2:[1]},
        "E": 200e6, "A": 0.02, "I": 2e-4,
        "nodal_loads": {},
        "member_loads": {0: -25.0, 1: -25.0},
        "labels": ["Span 1","Span 2"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  DSM SOLVER — returns full step-by-step data
# ══════════════════════════════════════════════════════════════════════════════
def run_dsm(nodes, elements, fixed_dofs, nodal_loads, member_loads, E, A, I):
    nn = len(nodes)
    ne = len(elements)
    nDOF = nn * 3

    dof_map = {i: [3*i, 3*i+1, 3*i+2] for i in range(nn)}
    fixed_global = []
    for nid, ldofs in fixed_dofs.items():
        for ld in ldofs:
            fixed_global.append(dof_map[nid][ld])
    fixed_global = sorted(set(fixed_global))
    free_global  = [d for d in range(nDOF) if d not in fixed_global]

    elem_data = []
    for idx, (ni, nj, hi, hj) in enumerate(elements):
        xi, yi = nodes[ni]
        xj, yj = nodes[nj]
        L  = max(math.hypot(xj-xi, yj-yi), 1e-9)
        al = math.atan2(yj-yi, xj-xi)
        c  = math.cos(al)
        s  = math.sin(al)

        a = E*A/L
        w = member_loads.get(idx, 0.0) # Local UDL

        # Modified Stiffness & FEF for Internal Hinges
        if not hi and not hj: # Fixed-Fixed (Standard)
            b  = 12*E*I/L**3; cc = 6*E*I/L**2; d = 4*E*I/L; ev = 2*E*I/L
            k_loc = np.array([
                [ a,  0,  0, -a,  0,  0],
                [ 0,  b, cc,  0, -b, cc],
                [ 0, cc,  d,  0,-cc, ev],
                [-a,  0,  0,  a,  0,  0],
                [ 0, -b,-cc,  0,  b,-cc],
                [ 0, cc, ev,  0,-cc,  d],
            ])
            fef_loc = np.array([0, -w*L/2, -w*L**2/12, 0, -w*L/2, w*L**2/12])

        elif hi and not hj: # Pinned-Fixed
            b = 3*E*I/L**3; cc = 3*E*I/L**2; d = 3*E*I/L
            k_loc = np.array([
                [ a,  0,  0, -a,  0,  0],
                [ 0,  b,  0,  0, -b, cc],
                [ 0,  0,  0,  0,  0,  0],
                [-a,  0,  0,  a,  0,  0],
                [ 0, -b,  0,  0,  b,-cc],
                [ 0, cc,  0,  0,-cc,  d],
            ])
            fef_loc = np.array([0, -3*w*L/8, 0, 0, -5*w*L/8, w*L**2/8])

        elif hj and not hi: # Fixed-Pinned
            b = 3*E*I/L**3; cc = 3*E*I/L**2; d = 3*E*I/L
            k_loc = np.array([
                [ a,  0,  0, -a,  0,  0],
                [ 0,  b, cc,  0, -b,  0],
                [ 0, cc,  d,  0,-cc,  0],
                [-a,  0,  0,  a,  0,  0],
                [ 0, -b,-cc,  0,  b,  0],
                [ 0,  0,  0,  0,  0,  0],
            ])
            fef_loc = np.array([0, -5*w*L/8, -w*L**2/8, 0, -3*w*L/8, 0])

        else: # Pinned-Pinned (Truss Element)
            k_loc = np.array([
                [ a,  0,  0, -a,  0,  0],
                [ 0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0],
                [-a,  0,  0,  a,  0,  0],
                [ 0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0],
            ])
            fef_loc = np.array([0, -w*L/2, 0, 0, -w*L/2, 0])

        # Transformation
        lam = np.array([[c, s, 0],[-s, c, 0],[0, 0, 1]])
        T6  = np.zeros((6,6))
        T6[:3,:3] = lam; T6[3:,3:] = lam

        k_glob = T6.T @ k_loc @ T6
        ejl_glob = T6.T @ (-fef_loc)

        elem_data.append({
            "ni": ni, "nj": nj, "L": L, "alpha_deg": math.degrees(al),
            "c": c, "s": s, "hi": hi, "hj": hj,
            "k_loc": k_loc, "T6": T6, "k_glob": k_glob,
            "gdofs": dof_map[ni] + dof_map[nj],
            "fef_loc": fef_loc, "ejl_glob": ejl_glob
        })

    # Assembly
    K = np.zeros((nDOF, nDOF))
    F = np.zeros(nDOF)

    for ed in elem_data:
        gdofs = ed["gdofs"]
        for r, gr in enumerate(gdofs):
            F[gr] += ed["ejl_glob"][r] # Apply EJL
            for c2, gc in enumerate(gdofs):
                K[gr, gc] += ed["k_glob"][r, c2]

    for nid, ldmap in nodal_loads.items():
        for ld, val in ldmap.items():
            F[dof_map[nid][ld]] += val

    # Solve
    ff = free_global
    Kff = K[np.ix_(ff, ff)]
    Ff  = F[ff]
    
    cond_num = np.linalg.cond(Kff)
    if cond_num > 1e10:
        st.warning(f"⚠️ highly ill-conditioned matrix (κ ≈ {cond_num:.2e}). Check hinges and supports.")
        
    try:
        Uf = np.linalg.solve(Kff, Ff)
    except np.linalg.LinAlgError:
        raise ValueError("Singular stiffness matrix — structure is a mechanism.")

    U = np.zeros(nDOF)
    for i, gi in enumerate(ff): U[gi] = Uf[i]

    # Member Forces
    member_results = []
    for ed in elem_data:
        u_el  = U[ed["gdofs"]]
        u_loc = ed["T6"] @ u_el
        f_loc = (ed["k_loc"] @ u_loc) + ed["fef_loc"] # Superimpose FEF
        member_results.append({
            "u_global": u_el, "u_local": u_loc, "f_local": f_loc,
            "N_i": -f_loc[0], "V_i": -f_loc[1], "M_i": -f_loc[2],
            "N_j": -f_loc[3], "V_j": -f_loc[4], "M_j": -f_loc[5],
        })

    R_full = K @ U - F
    reactions = {gi: R_full[gi] for gi in fixed_global}

    eq_check = {
        "ΣFx": sum(R_full[dof_map[n][0]] for n in fixed_dofs) + sum(F[dof_map[n][0]] - sum(ed["ejl_glob"][0 if ed["ni"]==n else 3] for ed in elem_data if ed["ni"]==n or ed["nj"]==n) for n in range(nn)),
        "ΣFy": sum(R_full[dof_map[n][1]] for n in fixed_dofs) + sum(F[dof_map[n][1]] - sum(ed["ejl_glob"][1 if ed["ni"]==n else 4] for ed in elem_data if ed["ni"]==n or ed["nj"]==n) for n in range(nn)),
    }

    return {
        "nn": nn, "ne": ne, "nDOF": nDOF,
        "dof_map": dof_map, "fixed_global": fixed_global, "free_global": free_global,
        "elem_data": elem_data, "K": K, "F": F, "Kff": Kff, "Ff": Ff, "U": U,
        "member_results": member_results, "reactions": reactions, "eq_check": eq_check,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MATRIX DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fmt_val(v, precision=4):
    if abs(v) < 1e-10: return "0"
    if abs(v) > 1e5 or (abs(v) < 0.001 and v != 0): return f"{v:.3e}"
    return f"{v:.{precision}f}"

def show_matrix(M, row_labels=None, col_labels=None, caption="", highlight_rows=None, highlight_cols=None):
    n, m = M.shape
    data = {(col_labels[j] if col_labels else f"c{j}"): [fmt_val(M[i,j]) for i in range(n)] for j in range(m)}
    df = pd.DataFrame(data, index=row_labels if row_labels else [f"r{i}" for i in range(n)])
    def styler(s):
        styles = pd.DataFrame("", index=s.index, columns=s.columns)
        if highlight_rows:
            for r in highlight_rows:
                if r < len(styles.index): styles.iloc[r] = "background-color:#dcfce7; color:#166534"
        if highlight_cols:
            for c in highlight_cols:
                if c < len(styles.columns): styles.iloc[:, c] = "background-color:#dbeafe; color:#1e40af"
        return styles
    st.dataframe(df.style.apply(styler, axis=None), use_container_width=True, height=min(35*n+38, 500))
    if caption: st.caption(caption)

def show_vector(v, labels=None, caption="", highlight=None):
    n = len(v)
    idx = labels if labels else [f"d{i}" for i in range(n)]
    df = pd.DataFrame({"Value": [fmt_val(x) for x in v]}, index=idx)
    def styler(s):
        styles = pd.DataFrame("", index=s.index, columns=s.columns)
        if highlight:
            for h in highlight:
                if h < len(styles.index): styles.iloc[h] = "background-color:#fef9c3; color:#854d0e"
        return styles
    st.dataframe(df.style.apply(styler, axis=None), use_container_width=True, height=min(35*n+38, 400))
    if caption: st.caption(caption)


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def classify_support(fixed_ldofs):
    if len(fixed_ldofs) >= 3: return "fixed"
    if 0 in fixed_ldofs and 1 in fixed_ldofs: return "pin"
    return "roller"

def draw_frame(nodes, elements, fixed_dofs, nodal_loads, U=None, dof_map=None, reactions=None, scale=0.3, show_dofs=False, show_loads=True, show_deformed=False, show_reactions=False, elem_labels=None, node_labels=True):
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#f8fafc")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#334155")
    for spine in ax.spines.values(): spine.set_edgecolor("#cbd5e1")

    xs = [n[0] for n in nodes]; ys = [n[1] for n in nodes]
    span = max(max(xs)-min(xs), max(ys)-min(ys), 1.0)
    arrowsc = span * 0.08

    # Elements & Hinges
    colors = ["#2563eb","#d97706","#7c3aed","#059669","#dc2626"]
    for idx, (ni, nj, hi, hj) in enumerate(elements):
        xi, yi = nodes[ni]; xj, yj = nodes[nj]
        col = colors[idx % len(colors)]
        ax.plot([xi, xj], [yi, yj], color=col, lw=3, solid_capstyle="round", zorder=3)
        
        # Draw Hinges visually offset slightly from node
        L = max(math.hypot(xj-xi, yj-yi), 1e-9)
        if hi: ax.plot(xi + (xj-xi)*0.06, yi + (yj-yi)*0.06, 'o', color="#ffffff", mec="#1e293b", mew=1.5, markersize=7, zorder=5)
        if hj: ax.plot(xj - (xj-xi)*0.06, yj - (yj-yi)*0.06, 'o', color="#ffffff", mec="#1e293b", mew=1.5, markersize=7, zorder=5)

        if elem_labels:
            ax.text((xi+xj)/2, (yi+yj)/2, elem_labels[idx], color=col, fontsize=8, ha="center", va="bottom", fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=col, lw=0.8))

    if show_deformed and U is not None and dof_map is not None:
        disp_max = max(abs(U)) if max(abs(U)) > 1e-12 else 1
        sc = scale * span / disp_max
        for ni, nj, _, _ in elements:
            ax.plot([nodes[ni][0]+sc*U[dof_map[ni][0]], nodes[nj][0]+sc*U[dof_map[nj][0]]], [nodes[ni][1]+sc*U[dof_map[ni][1]], nodes[nj][1]+sc*U[dof_map[nj][1]]], color="#dc2626", lw=2, ls="--", zorder=4, alpha=0.85)

    for i, (x, y) in enumerate(nodes):
        ax.scatter(x, y, s=60, color="#1e293b", zorder=6)
        if node_labels: ax.text(x, y+span*0.025, f"N{i+1}", color="#1e293b", fontsize=8, ha="center", fontfamily="monospace", fontweight="bold", zorder=7)

    for nid, ldofs in fixed_dofs.items():
        x, y = nodes[nid]
        stype = classify_support(ldofs)
        if stype == "fixed":
            ax.barh(y, -span*0.04, height=span*0.12, left=x, color="#dc2626", alpha=0.35, zorder=2)
            for dy in np.linspace(-span*0.06, span*0.06, 5): ax.plot([x-span*0.04, x-span*0.06], [y+dy, y+dy+span*0.015], color="#dc2626", lw=1, alpha=0.7)
        elif stype == "pin":
            ax.add_patch(plt.Polygon([[x, y],[x-span*0.03, y-span*0.06],[x+span*0.03, y-span*0.06]], closed=True, color="#16a34a", alpha=0.5, zorder=2))
            ax.plot([x-span*0.05, x+span*0.05],[y-span*0.065, y-span*0.065], color="#16a34a", lw=2)
        else:
            ax.add_patch(plt.Polygon([[x, y],[x-span*0.03, y-span*0.05],[x+span*0.03, y-span*0.05]], closed=True, color="#2563eb", alpha=0.35, zorder=2))
            ax.add_patch(plt.Circle((x, y-span*0.07), span*0.022, color="#2563eb", alpha=0.4, zorder=2))

    if show_loads:
        for nid, ldmap in nodal_loads.items():
            x, y = nodes[nid]
            for ld, val in ldmap.items():
                if ld == 0:
                    ax.annotate("", xy=(x, y), xytext=(x - arrowsc*(1 if val>0 else -1), y), arrowprops=dict(arrowstyle="->", color="#d97706", lw=2))
                elif ld == 1:
                    ax.annotate("", xy=(x, y), xytext=(x, y - arrowsc*(1 if val>0 else -1)), arrowprops=dict(arrowstyle="->", color="#d97706", lw=2))

    if show_reactions and reactions is not None and dof_map is not None:
        for nid, ldofs in fixed_dofs.items():
            x, y = nodes[nid]
            for ld in ldofs:
                val = reactions.get(dof_map[nid][ld], 0)
                if abs(val) < 1e-4: continue
                if ld == 0: ax.annotate("", xy=(x+arrowsc*(1 if val>0 else -1), y), xytext=(x, y), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))
                elif ld == 1: ax.annotate("", xy=(x, y+arrowsc*(1 if val>0 else -1)), xytext=(x, y), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))

    ax.set_aspect('equal', adjustable='box'); ax.relim(); ax.autoscale_view()
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    margin = max(xmax - xmin, ymax - ymin, 1.0) * 0.15
    ax.set_xlim(xmin - margin, xmax + margin); ax.set_ylim(ymin - margin, ymax + margin)
    ax.grid(True, color="#e2e8f0", lw=0.5, alpha=0.8); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    plt.tight_layout()
    return fig


def draw_bmd_sfd(nodes, elements, member_results, elem_labels=None, member_loads=None):
    if member_loads is None: member_loads = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#f8fafc")
    colors = ["#7c3aed", "#0891b2"]

    for ax_idx, ax in enumerate(axes):
        ax.set_facecolor("#ffffff")
        ax.set_title(["Bending Moment Diagram (kN·m)", "Shear Force Diagram (kN)"][ax_idx], color="#1e40af", fontfamily="monospace", pad=10)
        ax.grid(True, color="#e2e8f0", lw=0.5)

        xs = [n[0] for n in nodes]; ys = [n[1] for n in nodes]
        span = max(max(xs)-min(xs), max(ys)-min(ys), 1.0)

        for ni, nj, _, _ in elements: ax.plot([nodes[ni][0], nodes[nj][0]], [nodes[ni][1], nodes[nj][1]], color="#94a3b8", lw=1, zorder=1)

        for idx, ((ni, nj, hi, hj), mr) in enumerate(zip(elements, member_results)):
            xi, yi = nodes[ni]; xj, yj = nodes[nj]
            L = math.hypot(xj-xi, yj-yi)
            if L < 1e-9: continue
            
            alpha = math.atan2(yj-yi, xj-xi)
            perp  = (-math.sin(alpha), math.cos(alpha))
            t_vals = np.linspace(0, 1, 50)
            x_local = t_vals * L
            w = member_loads.get(idx, 0.0)

            if ax_idx == 0:
                ts_mult = -1 
                vals = (mr["M_i"] * (1 - t_vals) + mr["M_j"] * t_vals) + (w * x_local * (L - x_local)) / 2.0
                vals *= ts_mult
            else:
                vals = mr["V_i"] * (1 - t_vals) + mr["V_j"] * t_vals
                if w != 0: vals = mr["V_i"] - w * x_local

            scale = span * 0.15 / max(max(np.max(np.abs(vals)), 1e-9), 1)
            pts_x = xi + t_vals*(xj - xi) + scale*vals*perp[0]
            pts_y = yi + t_vals*(yj - yi) + scale*vals*perp[1]

            col = colors[idx % len(colors)]
            ax.plot([xi, xj], [yi, yj], color="#94a3b8", lw=1)
            ax.fill(np.concatenate([xi + t_vals*(xj-xi), pts_x[::-1]]), np.concatenate([yi + t_vals*(yj-yi), pts_y[::-1]]), color=col, alpha=0.25)
            ax.plot(pts_x, pts_y, color=col, lw=2)

        ax.set_aspect('equal', adjustable='box'); ax.relim(); ax.autoscale_view()
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        margin = max(xmax - xmin, ymax - ymin, 1.0) * 0.15
        ax.set_xlim(xmin - margin, xmax + margin); ax.set_ylim(ymin - margin, ymax + margin)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  STATE & SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
if "step" not in st.session_state: st.session_state.step = 0
if "preset" not in st.session_state: st.session_state.preset = list(PRESETS.keys())[0]
if "result" not in st.session_state: st.session_state.result = None

with st.sidebar:
    st.markdown("## 🏗️ 2D DSM Frame Analyzer")
    st.caption("Glass-Box Educational Tool")
    chosen = st.selectbox("Load Preset:", list(PRESETS.keys()), index=list(PRESETS.keys()).index(st.session_state.preset))
    if chosen != st.session_state.preset:
        st.session_state.preset = chosen; st.session_state.result = None; st.session_state.step = 0
    P = PRESETS[chosen]
    
    st.markdown("### 📐 Geometry & Hinges")
    node_df = st.data_editor(pd.DataFrame([{"Node": f"N{i+1}", "x": x, "y": y} for i,(x,y) in enumerate(P["nodes"])]), num_rows="dynamic", use_container_width=True)
    
    elem_df = st.data_editor(pd.DataFrame([{"Elem": f"E{i+1}", "Node i": ni+1, "Node j": nj+1, "Hinge i": hi, "Hinge j": hj, "Label": lbl} for i, ((ni, nj, hi, hj), lbl) in enumerate(zip(P["elements"], P["labels"]))]), num_rows="dynamic", use_container_width=True)

    st.markdown("### 🔩 BCs & ➡️ Nodal Loads")
    fixed_dofs_ui = {}
    for i in range(len(node_df)):
        cols = st.columns(3); checks = []
        with cols[0]: checks.append(0 if st.checkbox(f"N{i+1} u", value=0 in P["fixed_dofs"].get(i, [])) else None)
        with cols[1]: checks.append(1 if st.checkbox(f"N{i+1} v", value=1 in P["fixed_dofs"].get(i, [])) else None)
        with cols[2]: checks.append(2 if st.checkbox(f"N{i+1} θ", value=2 in P["fixed_dofs"].get(i, [])) else None)
        if [c for c in checks if c is not None]: fixed_dofs_ui[i] = [c for c in checks if c is not None]

    load_df = st.data_editor(pd.DataFrame([{"Node": nid+1, "DOF": ld, "Value": val} for nid, ldmap in P["nodal_loads"].items() for ld, val in ldmap.items()] if P["nodal_loads"] else [{"Node": 1, "DOF": 1, "Value": 0.0}]), num_rows="dynamic", use_container_width=True)

    st.markdown("### ⏬ Distributed Loads (UDL)")
    udl_df = st.data_editor(pd.DataFrame([{"Element": f"E{eid+1}", "w (kN/m)": val} for eid, val in P.get("member_loads", {}).items()] if P.get("member_loads", {}) else [{"Element": "E1", "w (kN/m)": 0.0}]), num_rows="dynamic", use_container_width=True)
    
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# --- Parse ---
def _val(r, k): return r.get(k) is not None and str(r.get(k)).strip() not in ("", "None")
nodes_p = [(float(r["x"]), float(r["y"])) for _, r in node_df.iterrows() if _val(r, "x") and _val(r, "y")]
elems_p = [(int(r["Node i"])-1, int(r["Node j"])-1, bool(r.get("Hinge i",False)), bool(r.get("Hinge j",False))) for _, r in elem_df.iterrows() if _val(r, "Node i") and _val(r, "Node j")]
labels_p = [str(r.get("Label", f"E{i+1}")) for i, r in elem_df.iterrows() if _val(r, "Node i")]
nloads_p = {}
for _, r in load_df.iterrows():
    if _val(r, "Node"): nloads_p.setdefault(int(r["Node"])-1, {})[int(r["DOF"])] = float(r["Value"])
mloads_p = {}
for _, r in udl_df.iterrows():
    if _val(r, "Element") and abs(float(r["w (kN/m)"])) > 1e-10:
        try: mloads_p[int(str(r["Element"]).upper().replace("E", ""))-1] = float(r["w (kN/m)"])
        except ValueError: pass

if run_btn:
    if len(nodes_p) < 2 or not elems_p or not fixed_dofs_ui: st.error("🚨 Invalid model setup."); st.stop()
    st.session_state.result = run_dsm(nodes_p, elems_p, fixed_dofs_ui, nloads_p, mloads_p, P["E"], P["A"], P["I"])
    st.session_state.step = 0; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<h1 style='text-align:center; font-family:Courier New; color:#1e40af;'>🏗️ 2D FRAME ANALYZER</h1>", unsafe_allow_html=True)
st.divider()

if st.session_state.result is None:
    fig_w = draw_frame(nodes_p, elems_p, fixed_dofs_ui, nloads_p, elem_labels=labels_p)
    st.pyplot(fig_w); plt.close(fig_w); st.stop()

STEP_NAMES = ["0 · Setup", "1 · DOFs", "2 · Local [k]", "3 · Transform", "4 · Global [K]ₑ", "5 · Assembly", "6 · Partition", "7 · Solve", "8 · Forces", "9 · Reactions"]
tab_cols = st.columns(10)
for i, (col, name) in enumerate(zip(tab_cols, STEP_NAMES)):
    if col.button(name.split("·")[0], use_container_width=True, type="primary" if i==st.session_state.step else "secondary"): st.session_state.step = i

step, res = st.session_state.step, st.session_state.result
st.divider()

if step == 0:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        f0 = draw_frame(nodes_p, elems_p, fixed_dofs_ui, nloads_p, elem_labels=labels_p)
        st.pyplot(f0); plt.close(f0)
    with c2:
        st.markdown("### Setup Data")
        st.dataframe(pd.DataFrame([{"E": f"E{i+1}", "Type": "Hinged" if e[2] or e[3] else "Fixed"} for i,e in enumerate(elems_p)]), use_container_width=True)

elif step == 1:
    f1 = draw_frame(nodes_p, elems_p, fixed_dofs_ui, nloads_p, U=res["U"], dof_map=res["dof_map"], show_dofs=True)
    st.pyplot(f1); plt.close(f1)

elif step == 2:
    e_sel = st.selectbox("Element:", [f"E{i+1}" for i in range(res["ne"])])
    ei = int(e_sel.replace("E",""))-1
    st.info(f"Hinge at start: **{res['elem_data'][ei]['hi']}** | Hinge at end: **{res['elem_data'][ei]['hj']}**")
    show_matrix(res['elem_data'][ei]['k_loc'], caption="Local Stiffness Matrix")
    show_vector(res['elem_data'][ei]['fef_loc'], caption="Fixed End Forces (FEF) due to UDL")

elif step == 5:
    c1, c2 = st.columns([3,1])
    with c1: show_matrix(res["K"], caption="Global K Matrix")
    with c2: show_vector(res["F"], caption="Global Load Vector {F} (Nodal + EJL)")

elif step == 7:
    f7 = draw_frame(nodes_p, elems_p, fixed_dofs_ui, nloads_p, U=res["U"], dof_map=res["dof_map"], show_deformed=True)
    st.pyplot(f7); plt.close(f7)

elif step == 8:
    f8 = draw_bmd_sfd(nodes_p, elems_p, res["member_results"], labels_p, mloads_p)
    st.pyplot(f8); plt.close(f8)

elif step == 9:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        f9 = draw_frame(nodes_p, elems_p, fixed_dofs_ui, nloads_p, reactions=res["reactions"], dof_map=res["dof_map"], show_reactions=True)
        st.pyplot(f9); plt.close(f9)
    with c2:
        st.markdown("### Equilibrium Check")
        for k,v in res["eq_check"].items(): st.success(f"{k} = {v:.3f} kN")

st.divider()
st.code("pip install streamlit numpy pandas matplotlib\nstreamlit run dsm_2d_frame.py", language="bash")
