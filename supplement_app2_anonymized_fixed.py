"""
╔══════════════════════════════════════════════════════════════════╗
║   2D Frame Analyzer — Direct Stiffness Method (DSM)              ║
║   Glass-Box Educational Tool for B.Tech / M.Tech Students        ║
║   Author  : Educational Structural Engineering Lab               ║
║   Deploy  : streamlit run app2.py                                ║
║   Requires: streamlit, numpy, pandas, matplotlib                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

import math
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Page configuration and styling
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="2D Frame Analyzer – DSM",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  .main { background:#f8fafc; }
  .stApp { background:#f8fafc; }
  section[data-testid="stSidebar"] { background:#eef2f7; border-right:1px solid #cbd5e1; }
  h1,h2,h3,h4,h5,h6 { color:#1e40af !important; font-family:'Courier New', monospace; }
  .stDataFrame { font-family:'Courier New', monospace; font-size:12px; }
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
  .step-title { font-size:18px; font-weight:700; color:#1e40af; font-family:'Courier New',monospace; }
  .step-subtitle { font-size:13px; color:#475569; margin-bottom:8px; }
/* Run button stays visible at the bottom of the sidebar */
[data-testid="stLayoutWrapper"]:has(> .st-key-run_bar),
.st-key-run_bar {
  position: sticky; bottom: 0; z-index: 999;
  background: var(--secondary-background-color, #f0f2f6);
  padding: 0.6rem 0 0.8rem 0; margin-top: 0.4rem;
  border-top: 1px solid rgba(49, 51, 63, 0.2);
  box-shadow: 0 -6px 10px -6px rgba(0, 0, 0, 0.15);
}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Preset problems
# ══════════════════════════════════════════════════════════════════════════════
PRESETS: Dict[str, Dict[str, Any]] = {
    "Portal Frame (Wind + UDL on Beam)": {
        "desc": "Fixed-base single-bay portal frame — horizontal wind load at left column top + 10 kN/m gravity UDL on beam",
        "nodes": [(0.0, 0.0), (0.0, 4.0), (6.0, 4.0), (6.0, 0.0)],
        "elements": [(0, 1), (1, 2), (2, 3)],
        "fixed_dofs": {0: [0, 1, 2], 3: [0, 1, 2]},
        "E": 200e6,
        "A": 0.01,
        "I": 1e-4,
        "nodal_loads": {1: {0: 20.0}},
        "udl_loads": {1: -10.0},
        "labels": ["Left Col", "Beam", "Right Col"],
    },
    "Two-Span Continuous Beam (UDL)": {
        "desc": "Continuous beam — 20 kN/m UDL on full first span + concentrated load on second span midpoint",
        "nodes": [(0.0, 0.0), (2.5, 0.0), (5.0, 0.0), (7.5, 0.0), (10.0, 0.0)],
        "elements": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "fixed_dofs": {0: [0, 1], 2: [1], 4: [1]},
        "E": 200e6,
        "A": 0.02,
        "I": 2e-4,
        "nodal_loads": {3: {1: -100.0}},
        "udl_loads": {0: -20.0, 1: -20.0},
        "labels": ["Span 1a", "Span 1b", "Span 2a", "Span 2b"],
    },
    "Cantilever Beam (Tip Load)": {
        "desc": "Fixed-free cantilever — vertical tip load",
        "nodes": [(0.0, 0.0), (4.0, 0.0)],
        "elements": [(0, 1)],
        "fixed_dofs": {0: [0, 1, 2]},
        "E": 200e6,
        "A": 0.01,
        "I": 1e-4,
        "nodal_loads": {1: {1: -80.0}},
        "udl_loads": {},
        "labels": ["Cantilever"],
    },
    "Cantilever Beam (UDL Only)": {
        "desc": "Fixed-free cantilever — 15 kN/m downward UDL only (classic DSM UDL demo)",
        "nodes": [(0.0, 0.0), (5.0, 0.0)],
        "elements": [(0, 1)],
        "fixed_dofs": {0: [0, 1, 2]},
        "E": 200e6,
        "A": 0.01,
        "I": 1e-4,
        "nodal_loads": {},
        "udl_loads": {0: -15.0},
        "labels": ["Cantilever"],
    },
    "Pitched Roof Frame": {
        "desc": "Symmetric gabled frame — vertical snow load on both rafters",
        "nodes": [(0.0, 0.0), (0.0, 4.0), (4.0, 6.0), (8.0, 4.0), (8.0, 0.0)],
        "elements": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "fixed_dofs": {0: [0, 1, 2], 4: [0, 1, 2]},
        "E": 200e6,
        "A": 0.015,
        "I": 1.5e-4,
        "nodal_loads": {1: {1: -20.0}, 2: {1: -40.0}, 3: {1: -20.0}},
        "udl_loads": {},
        "labels": ["Left Col", "Left Rafter", "Right Rafter", "Right Col"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# DSM solver with UDL Fixed-End Force support
# ══════════════════════════════════════════════════════════════════════════════
def run_dsm(
    nodes: List[Tuple[float, float]],
    elements: List[Tuple[int, int]],
    fixed_dofs: Dict[int, List[int]],
    nodal_loads: Dict[int, Dict[int, float]],
    E: float,
    A: float,
    I: float,
    udl_loads: Dict[int, float] | None = None,
) -> Dict[str, Any]:
    """Run a 2D frame DSM analysis and return all intermediate arrays.

    UDL sign convention:
    w is in the local transverse y direction. Negative w acts in local -y
    direction, which is gravity for a horizontal beam running left-to-right.
    """
    if udl_loads is None:
        udl_loads = {}

    nn = len(nodes)
    n_dof = 3 * nn
    dof_map = {i: [3 * i, 3 * i + 1, 3 * i + 2] for i in range(nn)}

    fixed_global: List[int] = []
    for nid, ldofs in fixed_dofs.items():
        if nid < 0 or nid >= nn:
            raise ValueError(f"Boundary condition references missing node N{nid + 1}.")
        for ld in ldofs:
            if ld not in (0, 1, 2):
                raise ValueError(f"Invalid DOF {ld} at node N{nid + 1}. Use 0=u, 1=v, 2=θ.")
            fixed_global.append(dof_map[nid][ld])
    fixed_global = sorted(set(fixed_global))
    free_global = [d for d in range(n_dof) if d not in fixed_global]
    if not free_global:
        raise ValueError("All DOFs are fixed; no unknown displacement DOFs remain.")

    elem_data: List[Dict[str, Any]] = []
    for idx, (ni, nj) in enumerate(elements):
        if ni < 0 or nj < 0 or ni >= nn or nj >= nn:
            raise ValueError(f"Element E{idx + 1} references a node that does not exist.")
        xi, yi = nodes[ni]
        xj, yj = nodes[nj]
        L = math.hypot(xj - xi, yj - yi)
        if L < 1e-9:
            raise ValueError(f"Element E{idx + 1} has zero length. Check node coordinates.")

        alpha = math.atan2(yj - yi, xj - xi)
        c = math.cos(alpha)
        s = math.sin(alpha)

        a = E * A / L
        b = 12.0 * E * I / L**3
        cc = 6.0 * E * I / L**2
        d = 4.0 * E * I / L
        e = 2.0 * E * I / L
        k_loc = np.array(
            [
                [a, 0, 0, -a, 0, 0],
                [0, b, cc, 0, -b, cc],
                [0, cc, d, 0, -cc, e],
                [-a, 0, 0, a, 0, 0],
                [0, -b, -cc, 0, b, -cc],
                [0, cc, e, 0, -cc, d],
            ],
            dtype=float,
        )

        lam = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=float)
        T6 = np.zeros((6, 6))
        T6[:3, :3] = lam
        T6[3:, 3:] = lam
        k_glob = T6.T @ k_loc @ T6
        gdofs = dof_map[ni] + dof_map[nj]

        elem_data.append(
            {
                "ni": ni,
                "nj": nj,
                "L": L,
                "alpha_deg": math.degrees(alpha),
                "c": c,
                "s": s,
                "k_loc": k_loc,
                "T6": T6,
                "k_glob": k_glob,
                "gdofs": gdofs,
            }
        )

    K = np.zeros((n_dof, n_dof))
    F = np.zeros(n_dof)

    for ed in elem_data:
        gdofs = ed["gdofs"]
        for r, gr in enumerate(gdofs):
            for c2, gc in enumerate(gdofs):
                K[gr, gc] += ed["k_glob"][r, c2]

    for nid, ldmap in nodal_loads.items():
        if nid < 0 or nid >= nn:
            raise ValueError(f"Nodal load references missing node N{nid + 1}.")
        for ld, val in ldmap.items():
            if ld not in (0, 1, 2):
                raise ValueError(f"Invalid nodal load DOF {ld} at N{nid + 1}.")
            F[dof_map[nid][ld]] += val

    fef_locals: Dict[int, np.ndarray] = {}
    fef_summaries: Dict[int, Dict[str, Any]] = {}
    for idx, ed_item in enumerate(elem_data):
        w = float(udl_loads.get(idx, 0.0))
        if abs(w) < 1e-12:
            fef_locals[idx] = np.zeros(6)
            continue
        L = ed_item["L"]
        fef = np.array([0.0, w * L / 2.0, w * L**2 / 12.0, 0.0, w * L / 2.0, -w * L**2 / 12.0])
        fef_locals[idx] = fef
        f_global = ed_item["T6"].T @ fef
        for r, gr in enumerate(ed_item["gdofs"]):
            F[gr] += f_global[r]
        fef_summaries[idx] = {
            "w": w,
            "L": L,
            "Vi": w * L / 2.0,
            "Mi": w * L**2 / 12.0,
            "Vj": w * L / 2.0,
            "Mj": -w * L**2 / 12.0,
            "fef_local": fef.copy(),
            "fef_global": f_global.copy(),
        }

    Kff = K[np.ix_(free_global, free_global)]
    Ff = F[free_global]
    cond_num = float(np.linalg.cond(Kff))

    try:
        Uf = np.linalg.solve(Kff, Ff)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Singular stiffness matrix — structure is a mechanism. Check boundary conditions."
        ) from exc

    U = np.zeros(n_dof)
    for i, gi in enumerate(free_global):
        U[gi] = Uf[i]

    member_results: List[Dict[str, Any]] = []
    for idx, ed_item in enumerate(elem_data):
        u_el = U[ed_item["gdofs"]]
        u_loc = ed_item["T6"] @ u_el
        f_loc_el = ed_item["k_loc"] @ u_loc
        f_loc = f_loc_el - fef_locals[idx]
        member_results.append(
            {
                "u_global": u_el,
                "u_local": u_loc,
                "f_local_elastic": f_loc_el,
                "fef_local": fef_locals[idx],
                "f_local": f_loc,
                # Internal forces at the member ends (beam convention):
                # N tension +, M sagging + (tension on the local -y face), V = dM/dx.
                "N_i": -f_loc[0],
                "V_i": f_loc[1],
                "M_i": -f_loc[2],
                "N_j": f_loc[3],
                "V_j": -f_loc[4],
                "M_j": f_loc[5],
            }
        )

    R_full = K @ U - F
    reactions = {gi: R_full[gi] for gi in fixed_global}

    sigma_m = 0.0
    for nid in fixed_dofs:
        x, y = nodes[nid]
        rx = R_full[dof_map[nid][0]]
        ry = R_full[dof_map[nid][1]]
        rm = R_full[dof_map[nid][2]]
        sigma_m += rm + ry * x - rx * y

    for nid, ldmap in nodal_loads.items():
        x, y = nodes[nid]
        fx = ldmap.get(0, 0.0)
        fy = ldmap.get(1, 0.0)
        m = ldmap.get(2, 0.0)
        sigma_m += m + fy * x - fx * y

    for idx, ed_item in enumerate(elem_data):
        w = float(udl_loads.get(idx, 0.0))
        if abs(w) < 1e-12:
            continue
        ni, nj = ed_item["ni"], ed_item["nj"]
        xi, yi = nodes[ni]
        xj, yj = nodes[nj]
        L = ed_item["L"]
        mx, my = (xi + xj) / 2.0, (yi + yj) / 2.0
        alpha = math.atan2(yj - yi, xj - xi)
        fx_udl = w * L * (-math.sin(alpha))
        fy_udl = w * L * math.cos(alpha)
        sigma_m += fy_udl * mx - fx_udl * my

    eq_check = {
        "ΣFx": (
            sum(R_full[dof_map[n][0]] for n in fixed_dofs)
            + sum(v for _, ldm in nodal_loads.items() for ld, v in ldm.items() if ld == 0)
            + sum(
                float(udl_loads.get(i, 0.0))
                * ed_item["L"]
                * (-math.sin(math.atan2(nodes[ed_item["nj"]][1] - nodes[ed_item["ni"]][1], nodes[ed_item["nj"]][0] - nodes[ed_item["ni"]][0])))
                for i, ed_item in enumerate(elem_data)
            )
        ),
        "ΣFy": (
            sum(R_full[dof_map[n][1]] for n in fixed_dofs)
            + sum(v for _, ldm in nodal_loads.items() for ld, v in ldm.items() if ld == 1)
            + sum(
                float(udl_loads.get(i, 0.0))
                * ed_item["L"]
                * math.cos(math.atan2(nodes[ed_item["nj"]][1] - nodes[ed_item["ni"]][1], nodes[ed_item["nj"]][0] - nodes[ed_item["ni"]][0]))
                for i, ed_item in enumerate(elem_data)
            )
        ),
        "ΣM": sigma_m,
    }

    warnings: List[str] = []
    if cond_num > 1e10:
        warnings.append(
            f"Stiffness matrix is highly ill-conditioned (κ ≈ {cond_num:.2e}). Check for mechanisms, disconnected elements, or extreme stiffness ratios."
        )

    return {
        "nn": nn,
        "ne": len(elements),
        "nDOF": n_dof,
        "dof_map": dof_map,
        "fixed_global": fixed_global,
        "free_global": free_global,
        "elem_data": elem_data,
        "K": K,
        "F": F,
        "Kff": Kff,
        "Ff": Ff,
        "U": U,
        "member_results": member_results,
        "reactions": reactions,
        "eq_check": eq_check,
        "udl_loads": dict(udl_loads),
        "fef_summaries": fef_summaries,
        "cond_num": cond_num,
        "warnings": warnings,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Display helpers
# ══════════════════════════════════════════════════════════════════════════════
def fmt_val(v: float, precision: int = 4) -> str:
    if abs(v) < 1e-10:
        return "0"
    if abs(v) > 1e5 or abs(v) < 1e-3:
        return f"{v:.3e}"
    return f"{v:.{precision}f}"


def show_matrix(M: np.ndarray, row_labels=None, col_labels=None, caption: str = "", highlight_rows=None, highlight_cols=None) -> None:
    n, m = M.shape
    df = pd.DataFrame(
        {(col_labels[j] if col_labels else f"c{j}"): [fmt_val(M[i, j]) for i in range(n)] for j in range(m)},
        index=row_labels if row_labels else [f"r{i}" for i in range(n)],
    )

    def styler(_):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        if highlight_rows:
            for r in highlight_rows:
                if 0 <= r < len(styles.index):
                    styles.iloc[r] = "background-color:#dcfce7; color:#166534"
        if highlight_cols:
            for c in highlight_cols:
                if 0 <= c < len(styles.columns):
                    styles.iloc[:, c] = "background-color:#dbeafe; color:#1e40af"
        return styles

    st.dataframe(df.style.apply(styler, axis=None), width="stretch", height=min(35 * n + 38, 500))
    if caption:
        st.caption(caption)


HL_STYLE = "background-color:#fef08a; color:#713f12; font-weight:700"


def show_vector(v: np.ndarray, labels=None, caption: str = "", highlight=None) -> None:
    labels = labels if labels else [f"d{i}" for i in range(len(v))]
    df = pd.DataFrame({"Value": [fmt_val(float(x)) for x in v]}, index=labels)

    def styler(_):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        if highlight:
            for h in highlight:
                if 0 <= h < len(styles.index):
                    styles.iloc[h] = "background-color:#fef9c3; color:#854d0e"
        return styles

    st.dataframe(df.style.apply(styler, axis=None), width="stretch", height=min(35 * len(v) + 38, 500))
    if caption:
        st.caption(caption)


def classify_support(fixed_ldofs: List[int]) -> str:
    if len(fixed_ldofs) >= 3:
        return "fixed"
    if 0 in fixed_ldofs and 1 in fixed_ldofs:
        return "pin"
    return "roller"


def draw_frame(
    nodes,
    elements,
    fixed_dofs,
    nodal_loads,
    U=None,
    dof_map=None,
    reactions=None,
    scale=0.3,
    show_dofs=False,
    show_loads=True,
    show_deformed=False,
    show_reactions=False,
    elem_labels=None,
    node_labels=True,
    udl_loads=None,
):
    if udl_loads is None:
        udl_loads = {}
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#f8fafc")
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cbd5e1")

    xs = [n[0] for n in nodes]
    ys = [n[1] for n in nodes]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    arrowsc = span * 0.08
    colors = ["#2563eb", "#d97706", "#7c3aed", "#059669", "#dc2626"]

    for idx, (ni, nj) in enumerate(elements):
        xi, yi = nodes[ni]
        xj, yj = nodes[nj]
        col = colors[idx % len(colors)]
        ax.plot([xi, xj], [yi, yj], color=col, lw=3, solid_capstyle="round", zorder=3)
        if elem_labels and idx < len(elem_labels):
            ax.text(
                (xi + xj) / 2.0,
                (yi + yj) / 2.0,
                elem_labels[idx],
                color=col,
                fontsize=8,
                ha="center",
                va="bottom",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=col, lw=0.8),
            )

    if show_loads:
        for idx, (ni, nj) in enumerate(elements):
            w = udl_loads.get(idx, 0.0)
            if abs(w) < 1e-12:
                continue
            xi, yi = nodes[ni]
            xj, yj = nodes[nj]
            L = math.hypot(xj - xi, yj - yi)
            if L < 1e-9:
                continue
            alpha = math.atan2(yj - yi, xj - xi)
            perp = (-math.sin(alpha), math.cos(alpha))
            arrow_len = span * 0.055 * (1 if w > 0 else -1)
            n_arrows = max(4, int(L / (span * 0.12)))
            t_vals = np.linspace(0.05, 0.95, n_arrows)
            udl_color = "#f59e0b"
            bx0 = xi + 0.05 * (xj - xi) - arrow_len * perp[0]
            by0 = yi + 0.05 * (yj - yi) - arrow_len * perp[1]
            bx1 = xi + 0.95 * (xj - xi) - arrow_len * perp[0]
            by1 = yi + 0.95 * (yj - yi) - arrow_len * perp[1]
            ax.plot([bx0, bx1], [by0, by1], color=udl_color, lw=2.0, zorder=5)
            for t in t_vals:
                px = xi + t * (xj - xi)
                py = yi + t * (yj - yi)
                ax.annotate(
                    "",
                    xy=(px, py),
                    xytext=(px - arrow_len * perp[0], py - arrow_len * perp[1]),
                    arrowprops=dict(arrowstyle="-|>", color=udl_color, lw=1.4),
                    zorder=5,
                )
            ax.text(
                (xi + xj) / 2.0 - arrow_len * perp[0] * 1.6,
                (yi + yj) / 2.0 - arrow_len * perp[1] * 1.6,
                f"w={w:+.1f}\nkN/m",
                color=udl_color,
                fontsize=8,
                ha="center",
                va="center",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.25", fc="#fffbeb", ec=udl_color, lw=0.9),
                zorder=6,
            )

    if show_deformed and U is not None and dof_map is not None:
        disp_max = max(float(np.max(np.abs(U))), 1e-12)
        sc = scale * span / disp_max
        for ni, nj in elements:
            xi, yi = nodes[ni]
            xj, yj = nodes[nj]
            ui = U[dof_map[ni][:2]]
            uj = U[dof_map[nj][:2]]
            ax.plot([xi + sc * ui[0], xj + sc * uj[0]], [yi + sc * ui[1], yj + sc * uj[1]], color="#dc2626", lw=2, ls="--", zorder=4, alpha=0.85)

    for i, (x, y) in enumerate(nodes):
        ax.scatter(x, y, s=60, color="#1e293b", zorder=6)
        if node_labels:
            ax.text(x, y + span * 0.025, f"N{i + 1}", color="#1e293b", fontsize=8, ha="center", va="bottom", fontfamily="monospace", fontweight="bold", zorder=7)

    for nid, ldofs in fixed_dofs.items():
        if nid < 0 or nid >= len(nodes):
            continue
        x, y = nodes[nid]
        stype = classify_support(ldofs)
        if stype == "fixed":
            ax.barh(y, -span * 0.04, height=span * 0.12, left=x, color="#dc2626", alpha=0.35, zorder=2)
            for dy in np.linspace(-span * 0.06, span * 0.06, 5):
                ax.plot([x - span * 0.04, x - span * 0.06], [y + dy, y + dy + span * 0.015], color="#dc2626", lw=1, alpha=0.7)
        elif stype == "pin":
            tri = plt.Polygon([[x, y], [x - span * 0.03, y - span * 0.06], [x + span * 0.03, y - span * 0.06]], closed=True, color="#16a34a", alpha=0.5, zorder=2)
            ax.add_patch(tri)
        else:
            tri = plt.Polygon([[x, y], [x - span * 0.03, y - span * 0.05], [x + span * 0.03, y - span * 0.05]], closed=True, color="#2563eb", alpha=0.35, zorder=2)
            ax.add_patch(tri)
            ax.add_patch(plt.Circle((x, y - span * 0.07), span * 0.022, color="#2563eb", alpha=0.4, zorder=2))

    if show_loads:
        for nid, ldmap in nodal_loads.items():
            if nid < 0 or nid >= len(nodes):
                continue
            x, y = nodes[nid]
            for ld, val in ldmap.items():
                if ld == 0:
                    dx = arrowsc * (1 if val > 0 else -1)
                    ax.annotate("", xy=(x, y), xytext=(x - dx, y), arrowprops=dict(arrowstyle="->", color="#d97706", lw=2))
                    ax.text(x - dx / 2, y + arrowsc * 0.3, f"{val:.0f} kN", color="#d97706", fontsize=8, ha="center", fontfamily="monospace")
                elif ld == 1:
                    dy = arrowsc * (1 if val > 0 else -1)
                    ax.annotate("", xy=(x, y), xytext=(x, y - dy), arrowprops=dict(arrowstyle="->", color="#d97706", lw=2))
                    ax.text(x + arrowsc * 0.4, y - dy / 2, f"{val:.0f} kN", color="#d97706", fontsize=8, ha="left", fontfamily="monospace")

    if show_dofs and dof_map is not None:
        colors_dof = ["#1d4ed8", "#15803d", "#7c3aed"]
        labels_dof = ["u", "v", "θ"]
        for nid, gdofs in dof_map.items():
            x, y = nodes[nid]
            offsets = [(arrowsc * 1.1, 0), (0, arrowsc * 1.1), (arrowsc * 0.6, arrowsc * 0.6)]
            for d, (dx, dy) in enumerate(offsets):
                ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y), arrowprops=dict(arrowstyle="->", color=colors_dof[d], lw=1.2, alpha=0.7))
                ax.text(x + dx * 1.1, y + dy * 1.1, f"d{gdofs[d]}\n({labels_dof[d]})", color=colors_dof[d], fontsize=7, ha="center", fontfamily="monospace")

    if show_reactions and reactions is not None and dof_map is not None:
        for nid, ldofs in fixed_dofs.items():
            x, y = nodes[nid]
            for ld in ldofs:
                gi = dof_map[nid][ld]
                val = reactions.get(gi, 0.0)
                if abs(val) < 1e-4:
                    continue
                if ld == 0:
                    dx = arrowsc * (1 if val > 0 else -1)
                    ax.annotate("", xy=(x + dx, y), xytext=(x, y), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))
                    ax.text(x + dx, y - arrowsc * 0.4, f"Rx={val:.1f}", color="#dc2626", fontsize=7, ha="center", fontfamily="monospace")
                elif ld == 1:
                    dy = arrowsc * (1 if val > 0 else -1)
                    ax.annotate("", xy=(x, y + dy), xytext=(x, y), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))
                    ax.text(x - arrowsc * 0.6, y + dy, f"Ry={val:.1f}", color="#dc2626", fontsize=7, ha="right", fontfamily="monospace")

    ax.set_aspect("equal", adjustable="box")
    ax.relim()
    ax.autoscale_view()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    max_span = max(xmax - xmin, ymax - ymin, 1.0)
    margin = max_span * 0.15
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.grid(True, color="#e2e8f0", lw=0.5, alpha=0.8)
    ax.set_xlabel("X (m)", color="#475569", fontfamily="monospace")
    ax.set_ylabel("Y (m)", color="#475569", fontfamily="monospace")
    plt.tight_layout()
    return fig


def draw_udl_fef_theory(w: float, L: float, E: float = 200e6, A: float = 0.01, I: float = 1e-4):
    """Four-stage picture of how DSM handles a UDL, for a propped cantilever
    (fixed at i, roller at j) of span L under UDL w (local y, negative = down).
    Returns (fig, data) where data holds the vectors shown in the figure."""
    from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon

    res_ex = run_dsm([(0.0, 0.0), (L, 0.0)], [(0, 1)], {0: [0, 1, 2], 1: [0, 1]}, {}, E, A, I, udl_loads={0: w})
    mr = res_ex["member_results"][0]
    fef = mr["fef_local"]                 # equivalent nodal loads (what enters {F})
    f_el = mr["f_local_elastic"]          # k·T·u
    f = mr["f_local"]                     # f_el − FEF
    u = mr["u_local"]

    C_LOAD, C_LOCK, C_EQ, C_EL, C_FIN = "#f59e0b", "#ea580c", "#7c3aed", "#2563eb", "#059669"
    fig = plt.figure(figsize=(15, 8.4), facecolor="#f8fafc")
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 0.12, 1, 0.12, 1], height_ratios=[0.78, 1], hspace=0.38, wspace=0.05, top=0.90, bottom=0.11, left=0.03, right=0.99)
    top = [fig.add_subplot(gs[0, c]) for c in (0, 2, 4)]
    bot = [fig.add_subplot(gs[1, c]) for c in (0, 2, 4)]
    ops = [fig.add_subplot(gs[r, c]) for r in (0, 1) for c in (1, 3)]
    for ax in ops:
        ax.axis("off")
    for ax, s in zip(ops, ["→", "→", "+", "="]):
        ax.text(0.5, 0.5, s, fontsize=30, ha="center", va="center", color="#334155", fontweight="bold", transform=ax.transAxes)

    down = w < 0
    wl2 = abs(w) * L / 2.0
    wl12 = abs(w) * L**2 / 12.0
    h = 0.11 * L  # vertical drawing unit

    def beam(ax, title, sub):
        ax.set_xlim(-0.28 * L, 1.28 * L)
        ax.set_ylim(-3.3 * h, 3.0 * h)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12, color="#1e3a8a", fontweight="bold", loc="left", pad=30)
        ax.text(0.0, 1.02, sub, transform=ax.transAxes, fontsize=9, color="#475569", va="bottom")
        ax.plot([0, L], [0, 0], color="#334155", lw=5, solid_capstyle="butt", zorder=3)
        ax.plot([0, L], [0, 0], "o", color="#0f172a", ms=6, zorder=4)

    def udl(ax, color=C_LOAD):
        y0, y1 = 0.25 * h, 1.15 * h
        ax.add_patch(Rectangle((0, y0), L, y1 - y0, color=color, alpha=0.18, lw=0))
        for x in np.linspace(0.04 * L, 0.96 * L, 9):
            a, b = (y1, y0) if down else (y0, y1)
            ax.annotate("", xy=(x, b), xytext=(x, a), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))
        ax.text(L / 2, y1 + 0.25 * h, f"w = {w:+.1f} kN/m", ha="center", fontsize=9, color=color, fontweight="bold")

    def force(ax, x, up, val, color, label_side):
        a, b = (-1.9 * h, -0.15 * h) if up else (-0.15 * h, -1.9 * h)
        ax.annotate("", xy=(x, b), xytext=(x, a), arrowprops=dict(arrowstyle="-|>", color=color, lw=2.6))
        ax.text(x + label_side * 0.04 * L, -1.15 * h, f"{val:.1f} kN", color=color, fontsize=9,
                ha="left" if label_side > 0 else "right", va="center", fontweight="bold")

    def moment(ax, x, ccw, val, color, label_side):
        r = 0.10 * L
        yA = 0.12 * h
        p_right, p_left = (x + r, yA), (x - r, yA)
        A_, B_ = (p_right, p_left) if ccw else (p_left, p_right)
        rad = 0.95 if ccw else -0.95
        ax.add_patch(FancyArrowPatch(A_, B_, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=16, color=color, lw=2.4, zorder=5))
        ax.text(x + label_side * 0.12 * L, 1.55 * h, f"{val:.1f} kN·m", color=color, fontsize=9,
                ha="left" if label_side > 0 else "right", fontweight="bold")

    def fixed(ax, x, side):
        ax.add_patch(Rectangle((x - (0.05 * L if side < 0 else 0), -1.0 * h), 0.05 * L, 2.0 * h, hatch="////", fill=False, ec="#64748b", lw=1))

    def roller(ax, x):
        ax.add_patch(Polygon([[x, -0.12 * h], [x - 0.05 * L, -0.9 * h], [x + 0.05 * L, -0.9 * h]], closed=True, fc="#e2e8f0", ec="#64748b"))
        ax.plot([x - 0.06 * L, x + 0.06 * L], [-1.05 * h, -1.05 * h], color="#64748b", lw=1.5)

    # ① locked
    ax = top[0]
    beam(ax, "① Lock both ends", "Clamp the member: it carries the UDL alone.")
    fixed(ax, 0, -1); fixed(ax, L, +1); udl(ax)
    force(ax, 0.0, down, wl2, C_LOCK, +1); force(ax, L, down, wl2, C_LOCK, -1)
    moment(ax, 0.0, down, wl12, C_LOCK, -1); moment(ax, L, not down, wl12, C_LOCK, +1)
    ax.text(L / 2, -2.75 * h, "Fixed-end reactions: |w|L/2 and |w|L²/12", ha="center", fontsize=9, color=C_LOCK)

    # ② equivalent nodal loads
    ax = top[1]
    beam(ax, "② Move the load to the nodes", "Reverse the reactions → nodal loads in {F}.")
    force(ax, 0.0, not down, wl2, C_EQ, +1); force(ax, L, not down, wl2, C_EQ, -1)
    moment(ax, 0.0, not down, wl12, C_EQ, -1); moment(ax, L, down, wl12, C_EQ, +1)
    ax.text(L / 2, -2.75 * h, "FEF = [0, wL/2, wL²/12, 0, wL/2, −wL²/12]ᵀ\n{F} += [T]ᵀ·{FEF}", ha="center", fontsize=9, color=C_EQ, family="monospace")

    # ③ solve the real structure (propped cantilever)
    ax = top[2]
    beam(ax, "③ Release and solve", "Real supports, nodal loads only: K·u = F.\nLoads at restrained DOFs go straight into the reactions.")
    fixed(ax, 0, -1); roller(ax, L)
    moment(ax, L, down, wl12, C_EQ, -1)
    xs = np.linspace(0, L, 80); t = xs / L
    N1, N2 = 1 - 3 * t**2 + 2 * t**3, xs * (1 - t) ** 2
    N3, N4 = 3 * t**2 - 2 * t**3, xs * (t**2 - t)
    v = N1 * u[1] + N2 * u[2] + N3 * u[4] + N4 * u[5]
    vs = v / (np.max(np.abs(v)) + 1e-30) * 0.9 * h
    ax.plot(xs, vs, "--", color=C_EL, lw=2)
    ax.text(L / 2, -2.75 * h, f"Free DOF: theta_j = {u[5]:+.3e} rad\nf_elastic = [k]·[T]·{{u}}", ha="center", fontsize=9, color=C_EL, family="monospace")

    # bottom: BMD superposition (beam convention: sagging +, drawn on the tension side)
    xx = np.linspace(0, L, 201); tt = xx / L
    def bmd(Mi, Mj, ww):
        return Mi * (1 - tt) + Mj * tt - ww * xx * (L - xx) / 2.0
    M_lock = bmd(w * L**2 / 12.0, w * L**2 / 12.0, w)
    M_el = bmd(-f_el[2], f_el[5], 0.0)
    M_fin = bmd(-f[2], f[5], w)
    big = max(np.max(np.abs(M_lock)), np.max(np.abs(M_el)), np.max(np.abs(M_fin)))
    sets = [(M_lock, C_LOCK, "Locked BMD (from FEF)", "fixed-fixed beam under UDL"),
            (M_el, C_EL, "Elastic BMD (from k·T·u)", "joint-rotation effect, linear"),
            (M_fin, C_FIN, "Final BMD  f = f_elastic − FEF", "what the member really carries")]
    ends = [(w * L**2 / 12.0, w * L**2 / 12.0, w), (-f_el[2], f_el[5], 0.0), (-f[2], f[5], w)]
    for ax, (M, col, ttl, sub), (Mi_, Mj_, w_) in zip(bot, sets, ends):
        ax.set_facecolor("#ffffff")
        ax.axhline(0, color="#334155", lw=3)
        ax.fill_between(xx, 0, -M, color=col, alpha=0.22)
        ax.plot(xx, -M, color=col, lw=2.4)
        ax.set_ylim(-1.35 * big, 1.35 * big)
        ax.set_xlim(-0.05 * L, 1.05 * L)
        ax.set_title(ttl, fontsize=11.5, color=col, fontweight="bold", loc="left", pad=18)
        ax.text(0.0, 1.01, sub, transform=ax.transAxes, fontsize=9, color="#475569", va="bottom")
        ax.set_yticks([]); ax.set_xlabel("x (m)", fontsize=9)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        for k in (0, -1):
            ax.text(xx[k], -M[k] + np.sign(-M[k] or 1) * 0.1 * big, f"{M[k]:.1f}", color=col, fontsize=9, ha="center",
                    va="bottom" if -M[k] >= 0 else "top", fontweight="bold")
        if abs(w_) > 1e-12:   # parabola: label its stationary point if it lies inside the span
            xs_ = L / 2.0 - (Mj_ - Mi_) / (w_ * L)
            if 0.02 * L < xs_ < 0.98 * L:
                ms_ = Mi_ * (1 - xs_ / L) + Mj_ * xs_ / L - w_ * xs_ * (L - xs_) / 2.0
                ax.plot([xs_], [-ms_], "o", color=col, ms=4)
                ax.text(xs_, -ms_ - np.sign(ms_ or 1) * 0.1 * big, f"{ms_:.1f}", color=col, fontsize=9, ha="center",
                        va="top" if ms_ > 0 else "bottom", fontweight="bold")
    bot[0].text(0.0, -0.20, "BMD drawn on the tension side · M sagging +, hogging −", transform=bot[0].transAxes, fontsize=8.5, color="#64748b")
    fig.suptitle(f"How DSM handles a UDL — propped cantilever, w = {w:+.1f} kN/m, L = {L:.1f} m", fontsize=14, color="#1e40af", fontweight="bold", y=0.985)
    return fig, {"fef": fef, "f_el": f_el, "f": f}



@st.cache_data(show_spinner=False, max_entries=64)
def _udl_fef_theory_png(w: float, L: float):
    """Draw the explainer once per (w, L) and reuse it on later reruns."""
    import io
    fig_t, d = draw_udl_fef_theory(w, L)
    buf = io.BytesIO()
    fig_t.savefig(buf, format="png", dpi=110, facecolor=fig_t.get_facecolor(), bbox_inches="tight")
    plt.close(fig_t)
    return buf.getvalue(), d


def show_udl_fef_explainer(key: str, w_default: float = -10.0, L_default: float = 6.0) -> None:
    """Visual explanation of UDL fixed-end forces, with live numbers."""
    st.markdown(
        "A member with a UDL cannot go straight into **K·u = F**, because {F} only holds loads at the nodes. "
        "DSM solves this in four moves, shown below for a **propped cantilever** "
        "(fixed at i, roller at j). Change w and L to see every number update."
    )
    c1, c2 = st.columns(2)
    with c1:
        w_demo = st.number_input("w (kN/m)  — negative = downward", value=float(w_default), step=1.0, key=f"{key}_w")
    with c2:
        L_demo = st.number_input("L (m)", value=float(L_default), min_value=0.5, step=0.5, key=f"{key}_L")
    if abs(w_demo) < 1e-9:
        st.info("Enter a non-zero w to see the explanation.")
        return
    png, d = _udl_fef_theory_png(float(w_demo), float(L_demo))
    st.image(png, width="stretch")
    st.markdown(
        """
**In plain words**

1. **Lock** — clamp both ends. The member now carries the UDL by itself. The supports push back with
   the *fixed-end reactions*: |w|L/2 at each end and |w|L²/12 as end moments.
2. **Move the load to the nodes** — reverse those reactions. These *equivalent nodal loads* (the FEF vector)
   are added to {F}. Now the structure only has nodal loads, so **K·u = F** can be solved.
3. **Release and solve** — the real supports and joints take over. The solution {u} gives the
   *elastic* end forces **f_elastic = [k]·[T]·{u}**. These describe only the joint movements, not the UDL inside the span.
4. **Add the locked state back** — the true member forces are **f = f_elastic − FEF**.
   In the bottom row: locked BMD **+** elastic BMD **=** final BMD.

*Forgetting step 4 is the most common student error: you would report the blue line instead of the green curve.*
"""
    )
    labels = ["N'i", "V'i", "M'i", "N'j", "V'j", "M'j"]
    st.dataframe(
        pd.DataFrame({"local end force": labels, "f_elastic = k·T·u": np.round(d["f_el"], 3) + 0.0,
                      "− FEF": np.round(-d["fef"], 3) + 0.0, "= f (true)": np.round(d["f"], 3) + 0.0}),
        width="stretch", hide_index=True,
    )
    st.caption("End forces in the local frame (kN, kN·m), counter-clockwise moments positive. "
               "Check with your textbook: for a propped cantilever, M at the fixed end = wL²/8 and the reactions are 5wL/8 and 3wL/8.")


def draw_bmd_sfd(nodes, elements, member_results, elem_labels=None, member_loads=None):
    if member_loads is None:
        member_loads = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#f8fafc")
    titles = ["Bending Moment Diagram (kN·m)", "Shear Force Diagram (kN)"]
    colors = ["#7c3aed", "#0891b2", "#dc2626", "#16a34a", "#d97706"]
    xs = [n[0] for n in nodes]
    ys = [n[1] for n in nodes]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    global_max_bmd = 1e-9
    global_max_sfd = 1e-9
    for idx2, ((ni2, nj2), mr2) in enumerate(zip(elements, member_results)):
        L2 = math.hypot(nodes[nj2][0] - nodes[ni2][0], nodes[nj2][1] - nodes[ni2][1])
        if L2 < 1e-9:
            continue
        w2 = member_loads.get(idx2, 0.0)
        t2 = np.linspace(0, 1, 61)
        x2 = t2 * L2
        m_vals = mr2["M_i"] * (1 - t2) + mr2["M_j"] * t2 - w2 * x2 * (L2 - x2) / 2.0
        v_vals = mr2["V_i"] + w2 * x2 if abs(w2) > 1e-9 else mr2["V_i"] * (1 - t2) + mr2["V_j"] * t2
        global_max_bmd = max(global_max_bmd, float(np.max(np.abs(m_vals))))
        global_max_sfd = max(global_max_sfd, float(np.max(np.abs(v_vals))))

    scale_bmd = span * 0.15 / global_max_bmd
    scale_sfd = span * 0.15 / global_max_sfd

    for ax_idx, ax in enumerate(axes):
        ax.set_facecolor("#ffffff")
        ax.set_title(titles[ax_idx], color="#1e40af", fontfamily="monospace", pad=10)
        ax.grid(True, color="#e2e8f0", lw=0.5)
        for idx, ((ni, nj), mr) in enumerate(zip(elements, member_results)):
            xi, yi = nodes[ni]
            xj, yj = nodes[nj]
            L = math.hypot(xj - xi, yj - yi)
            if L < 1e-9:
                continue
            alpha = math.atan2(yj - yi, xj - xi)
            perp = (-math.sin(alpha), math.cos(alpha))
            t_vals = np.linspace(0, 1, 61)
            x_local = t_vals * L
            w = member_loads.get(idx, 0.0)
            col = colors[idx % len(colors)]

            if ax_idx == 0:
                actual_vals = mr["M_i"] * (1 - t_vals) + mr["M_j"] * t_vals - w * x_local * (L - x_local) / 2.0
                plot_vals = -actual_vals  # tension-side visual convention only
                scale = scale_bmd
            else:
                actual_vals = mr["V_i"] + w * x_local if abs(w) > 1e-9 else mr["V_i"] * (1 - t_vals) + mr["V_j"] * t_vals
                plot_vals = actual_vals
                scale = scale_sfd

            pts_x = xi + t_vals * (xj - xi) + scale * plot_vals * perp[0]
            pts_y = yi + t_vals * (yj - yi) + scale * plot_vals * perp[1]
            bx = xi + t_vals * (xj - xi)
            by = yi + t_vals * (yj - yi)
            ax.plot([xi, xj], [yi, yj], color="#94a3b8", lw=1)
            ax.fill(np.concatenate([bx, pts_x[::-1]]), np.concatenate([by, pts_y[::-1]]), color=col, alpha=0.22)
            ax.plot(pts_x, pts_y, color=col, lw=2)
            ax.text(pts_x[0], pts_y[0], f"{actual_vals[0]:.1f}", color=col, fontsize=7, ha="center", fontfamily="monospace")
            ax.text(pts_x[-1], pts_y[-1], f"{actual_vals[-1]:.1f}", color=col, fontsize=7, ha="center", fontfamily="monospace")
            if ax_idx == 0 and abs(w) > 1e-9:
                # Label the span extreme, where V(x) = Vi + w*x = 0 (if it lies inside the member)
                x_star = -mr["V_i"] / w
                if 1e-6 < x_star < L - 1e-6:
                    t_s = x_star / L
                    m_s = mr["M_i"] * (1 - t_s) + mr["M_j"] * t_s - w * x_star * (L - x_star) / 2.0
                    px = xi + t_s * (xj - xi) - scale * m_s * perp[0]
                    py = yi + t_s * (yj - yi) - scale * m_s * perp[1]
                    ax.plot([px], [py], "o", color=col, ms=3)
                    ax.text(px, py, f"{m_s:.1f}", color=col, fontsize=7, ha="center", va="top", fontfamily="monospace", fontweight="bold")

        ax.set_aspect("equal", adjustable="box")
        ax.relim()
        ax.autoscale_view()
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        max_span = max(xmax - xmin, ymax - ymin, 1.0)
        margin = max_span * 0.15
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Session state and sidebar inputs
# ══════════════════════════════════════════════════════════════════════════════
if "step" not in st.session_state:
    st.session_state.step = 0
if "preset" not in st.session_state:
    st.session_state.preset = list(PRESETS.keys())[0]
if "result" not in st.session_state:
    st.session_state.result = None

with st.sidebar:
    st.markdown("## 🏗️ 2D DSM Frame Analyzer")
    st.caption("Glass-Box Educational Tool for B.Tech / M.Tech")
    st.divider()

    st.markdown("### 📌 Preset Examples")
    chosen = st.selectbox("Load Preset:", list(PRESETS.keys()), index=list(PRESETS.keys()).index(st.session_state.preset))
    if chosen != st.session_state.preset:
        st.session_state.preset = chosen
        st.session_state.result = None
        st.session_state.step = 0
    P = PRESETS[chosen]
    st.info(P["desc"])

    st.divider()
    st.markdown("### ⚙️ Material & Section")
    E_val = st.number_input("E — Elastic Modulus (kN/m²)", value=float(P["E"]), format="%.2e", help="Young's modulus")
    A_val = st.number_input("A — Cross-Section Area (m²)", value=float(P["A"]), format="%.4f")
    I_val = st.number_input("I — Second Moment (m⁴)", value=float(P["I"]), format="%.6f")

    st.divider()
    st.markdown("### 📐 Geometry")
    node_df = pd.DataFrame([{"Node": f"N{i + 1}", "x (m)": x, "y (m)": y} for i, (x, y) in enumerate(P["nodes"])])
    node_df = st.data_editor(node_df, num_rows="dynamic", width="stretch", key=f"nodes_{chosen}")

    elem_df = pd.DataFrame([{"Elem": f"E{i + 1}", "Node i": ni + 1, "Node j": nj + 1, "Label": lbl} for i, ((ni, nj), lbl) in enumerate(zip(P["elements"], P["labels"]))])
    elem_df = st.data_editor(elem_df, num_rows="dynamic", width="stretch", key=f"elems_{chosen}")

    st.divider()
    st.markdown("### 🔩 Boundary Conditions")
    st.caption("Check DOFs to fix (u=horiz, v=vert, θ=rotation)")
    fixed_dofs_ui: Dict[int, List[int]] = {}
    for i in range(len(node_df)):
        def_check = P["fixed_dofs"].get(i, [])
        cols_bc = st.columns(3)
        checks = []
        with cols_bc[0]:
            checks.append(0 if st.checkbox(f"N{i + 1} u", value=0 in def_check, key=f"u_{i}_{chosen}") else None)
        with cols_bc[1]:
            checks.append(1 if st.checkbox(f"N{i + 1} v", value=1 in def_check, key=f"v_{i}_{chosen}") else None)
        with cols_bc[2]:
            checks.append(2 if st.checkbox(f"N{i + 1} θ", value=2 in def_check, key=f"t_{i}_{chosen}") else None)
        fixed = [c for c in checks if c is not None]
        if fixed:
            fixed_dofs_ui[i] = fixed

    st.divider()
    st.markdown("### ➡️ Nodal Point Loads")
    load_rows = []
    for nid, ldmap in P["nodal_loads"].items():
        for ld, val in ldmap.items():
            load_rows.append({"Node": nid + 1, "DOF (0=u,1=v,2=θ)": ld, "Value (kN or kN·m)": val})
    load_df = pd.DataFrame(load_rows if load_rows else [{"Node": 1, "DOF (0=u,1=v,2=θ)": 1, "Value (kN or kN·m)": 0.0}])
    load_df = st.data_editor(load_df, num_rows="dynamic", width="stretch", key=f"loads_{chosen}")

    st.divider()
    st.markdown("### 〰️ Element UDL Loads")
    st.caption(
        "Uniformly Distributed Load (kN/m) in the local transverse y direction. "
        "Negative w = local −y direction (gravity for horizontal beams)."
    )
    udl_rows = [{"Element": f"E{ei + 1}", "w (kN/m)": w} for ei, w in P.get("udl_loads", {}).items()]
    udl_df = pd.DataFrame(udl_rows if udl_rows else [{"Element": "E1", "w (kN/m)": 0.0}])
    udl_df = st.data_editor(udl_df, num_rows="dynamic", width="stretch", key=f"udl_{chosen}")

    with st.container(key="run_bar"):
        run_btn = st.button("🚀 Run DSM Analysis", type="primary", width="stretch")
        run_note = st.empty()


def _is_valid_row(r: pd.Series, keys: List[str]) -> bool:
    for k in keys:
        v = r.get(k)
        if pd.isna(v) or str(v).strip() in ("", "None"):
            return False
    return True


try:
    nodes_parsed = [(float(r["x (m)"]), float(r["y (m)"])) for _, r in node_df.iterrows() if _is_valid_row(r, ["x (m)", "y (m)"])]
    elems_parsed = [(int(r["Node i"]) - 1, int(r["Node j"]) - 1) for _, r in elem_df.iterrows() if _is_valid_row(r, ["Node i", "Node j"])]
    elem_labels_p = [str(r["Label"]) if _is_valid_row(r, ["Label"]) else f"E{i + 1}" for i, (_, r) in enumerate(elem_df.iterrows()) if _is_valid_row(r, ["Node i", "Node j"])]
except Exception as exc:
    st.error(f"Input parsing error: {exc}")
    st.stop()

nodal_loads_parsed: Dict[int, Dict[int, float]] = {}
for _, r in load_df.iterrows():
    if not _is_valid_row(r, ["Node", "DOF (0=u,1=v,2=θ)", "Value (kN or kN·m)"]):
        continue
    try:
        nid = int(r["Node"]) - 1
        ld = int(r["DOF (0=u,1=v,2=θ)"])
        val = float(r["Value (kN or kN·m)"])
        if abs(val) > 1e-10:
            nodal_loads_parsed.setdefault(nid, {})[ld] = val
    except (ValueError, TypeError):
        st.warning("Invalid nodal load row ignored.")

udl_loads_parsed: Dict[int, float] = {}
for _, r in udl_df.iterrows():
    if not _is_valid_row(r, ["Element", "w (kN/m)"]):
        continue
    elem_str = str(r["Element"]).strip().upper()
    if not elem_str.startswith("E"):
        continue
    try:
        ei = int(elem_str[1:]) - 1
        w = float(r["w (kN/m)"])
        if abs(w) > 1e-12:
            udl_loads_parsed[ei] = w
    except (ValueError, TypeError):
        st.warning("Invalid UDL row ignored.")

_input_sig = repr((nodes_parsed, elems_parsed, sorted((k, tuple(v)) for k, v in fixed_dofs_ui.items()),
                   sorted((k, tuple(sorted(v.items()))) for k, v in nodal_loads_parsed.items()),
                   sorted(udl_loads_parsed.items()), E_val, A_val, I_val))

_run_failed = False
if run_btn:
    run_note.info("⏳ Running analysis… assembling matrices and solving.")
    if len(nodes_parsed) < 2:
        st.error("🚨 Need at least 2 nodes.")
        run_note.error("❌ Analysis not run — see the message on the right."); _run_failed = True
    elif len(elems_parsed) < 1:
        st.error("🚨 Need at least 1 element.")
        run_note.error("❌ Analysis not run — see the message on the right."); _run_failed = True
    elif not fixed_dofs_ui:
        st.error("🚨 No boundary conditions defined — structure is a mechanism. Tick at least one support DOF.")
        run_note.error("❌ Analysis not run — see the message on the right."); _run_failed = True
    else:
        max_node = len(nodes_parsed) - 1
        bad = [(i, ni, nj) for i, (ni, nj) in enumerate(elems_parsed) if ni > max_node or nj > max_node or ni < 0 or nj < 0]
        if bad:
            for i, ni, nj in bad:
                st.error(f"🚨 Element E{i + 1} references node {max(ni, nj) + 1} but only {len(nodes_parsed)} nodes exist.")
            run_note.error("❌ Analysis not run — see the message on the right."); _run_failed = True
            st.stop()
        for ei in list(udl_loads_parsed.keys()):
            if ei >= len(elems_parsed) or ei < 0:
                st.warning(f"⚠️ UDL entry E{ei + 1} has no matching element — ignored.")
                del udl_loads_parsed[ei]
        with st.spinner("Assembling stiffness matrices, including UDL Fixed-End Forces…"):
            try:
                st.session_state.result = run_dsm(nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed, E_val, A_val, I_val, udl_loads=udl_loads_parsed)
                st.session_state.result_sig = _input_sig
                _r = st.session_state.result
                _eq_ok = all(abs(v) < 1e-3 for v in _r["eq_check"].values())
                st.session_state.just_ran = (
                    f"Done: {3 * len(nodes_parsed)} DOFs, {len(_r['free_global'])} free. "
                    + ("Equilibrium OK. " if _eq_ok else "Equilibrium FAILED (Step 9). ")
                    + "Click Next ▶"
                )
            except ValueError as err:
                st.error(f"🚨 {err}")
                run_note.error("❌ Analysis not run — see the message on the right."); _run_failed = True
                st.stop()
        st.session_state.step = 0
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Header and initial screen
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<h1 style='text-align:center; font-family:Courier New; color:#1e40af; letter-spacing:2px;'>
  🏗️ 2D FRAME ANALYZER — DIRECT STIFFNESS METHOD
</h1>
<p style='text-align:center; color:#475569; font-family:Courier New; font-size:14px;'>
  Glass-Box Educational Tool &nbsp;|&nbsp; B.Tech / M.Tech Structural Analysis &nbsp;|&nbsp;
  Supports Point Loads &amp; UDL
</p>
""",
    unsafe_allow_html=True,
)
st.divider()

res = st.session_state.result

_done_msg = st.session_state.pop("just_ran", None)
if _done_msg and res is not None:
    st.toast(_done_msg, icon="✅")
if _run_failed:
    pass
elif res is None:
    run_note.caption("👉 Check the inputs above, then click to run the analysis.")
elif st.session_state.get("result_sig") == _input_sig:
    run_note.caption("✅ Results are up to date with these inputs.")

if res is not None and st.session_state.get("result_sig") != _input_sig:
    st.warning("✏️ Inputs have changed since the last analysis. The results below are for the **previous** inputs. "
               "Click **🚀 Run DSM Analysis** to update them.")
    if not _run_failed:
        run_note.caption("⚠️ Inputs changed — click to update the results.")

if res is None:
    fig_welcome = draw_frame(nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed, elem_labels=elem_labels_p, node_labels=True, udl_loads=udl_loads_parsed)
    st.pyplot(fig_welcome, width="stretch")
    plt.close(fig_welcome)
    st.caption("Select a preset in the sidebar and click **🚀 Run DSM Analysis** to begin.")

    with st.expander("📖 What is the Direct Stiffness Method?", expanded=False):
        st.markdown(
            """
The **Direct Stiffness Method (DSM)** is the fundamental matrix algorithm behind commercial
structural analysis software such as STAAD, ETABS, SAP2000, and OpenSees.

This app makes DSM a **glass box**: each intermediate matrix, vector, transformation,
UDL fixed-end-force contribution, and recovery equation is shown step by step.
"""
        )

    with st.expander("📚 UDL Fixed-End Force Theory — click to expand", expanded=False):
        show_udl_fef_explainer("welcome")
        st.markdown("**Formula summary**")
        st.markdown(
            """
<div class='formula-box'>
UDL on a member: w (kN/m) in the local transverse y direction.
Negative w → gravity for a horizontal beam.

FEF_local = [ 0,  wL/2,  wL²/12,  0,  wL/2,  -wL²/12 ]
{F} += [T]ᵀ · {FEF_local}

Member-end force recovery:
{f_local} = [k_local]·[T]·{u_global_elem} − {FEF_local}

Bending moment at distance x from i:
M(x) = Mi·(1 − x/L) + Mj·(x/L) − w·x·(L−x)/2

Shear force at distance x from i:
V(x) = Vi + w·x

Sign convention for internal forces: N tension positive,
M sagging positive, V = dM/dx.
</div>
""",
            unsafe_allow_html=True,
        )
    st.stop()

for msg in res.get("warnings", []):
    st.warning(f"⚠️ {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# Step navigator
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

st.caption("Click any step number to jump straight to it. Results: **7** displacements · **8** member forces and BMD/SFD · **9** reactions.")
tab_cols = st.columns(len(STEP_NAMES))
for i, (col, name) in enumerate(zip(tab_cols, STEP_NAMES)):
    with col:
        if st.button(name.split("·")[0].strip(), key=f"tab_{i}", help=name, width="stretch", type="primary" if i == st.session_state.step else "secondary"):
            st.session_state.step = i
            st.rerun()

st.progress(st.session_state.step / (len(STEP_NAMES) - 1))
nav_c1, nav_mid, nav_c2 = st.columns([1, 6, 1])
with nav_c1:
    if st.button("◀ Prev", width="stretch") and st.session_state.step > 0:
        st.session_state.step -= 1
        st.rerun()
with nav_mid:
    st.markdown(f"<h3 style='text-align:center;margin:0;'>{STEP_NAMES[st.session_state.step]}</h3>", unsafe_allow_html=True)
with nav_c2:
    if st.button("Next ▶", width="stretch") and st.session_state.step < len(STEP_NAMES) - 1:
        st.session_state.step += 1
        st.rerun()

st.divider()
step = st.session_state.step
ed = res["elem_data"]
ne = res["ne"]
nn = res["nn"]
udl_r = res.get("udl_loads", {})


# ══════════════════════════════════════════════════════════════════════════════
# Step content
# ══════════════════════════════════════════════════════════════════════════════
if step == 0:
    col_v, col_t = st.columns([1.4, 1])
    with col_v:
        fig0 = draw_frame(nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed, elem_labels=elem_labels_p, udl_loads=udl_r)
        st.pyplot(fig0, width="stretch")
        plt.close(fig0)
    with col_t:
        st.markdown("<div class='step-card'>", unsafe_allow_html=True)
        st.markdown("<span class='step-title'>Problem Setup</span>", unsafe_allow_html=True)
        st.markdown("<span class='step-subtitle'>Geometry, section, material, loads and UDL</span>", unsafe_allow_html=True)
        st.markdown("**📍 Nodes**")
        st.dataframe(pd.DataFrame([{"ID": f"N{i + 1}", "x (m)": f"{x:.3f}", "y (m)": f"{y:.3f}"} for i, (x, y) in enumerate(nodes_parsed)]), width="stretch", hide_index=True)
        st.markdown("**🔗 Elements**")
        elem_rows = []
        for i, (e, lbl) in enumerate(zip(ed, elem_labels_p)):
            elem_rows.append({"ID": f"E{i + 1}", "Label": lbl, "Ni": f"N{e['ni'] + 1}", "Nj": f"N{e['nj'] + 1}", "L(m)": f"{e['L']:.3f}", "α(°)": f"{e['alpha_deg']:.2f}", "UDL": f"w={udl_r.get(i, 0.0):+.1f} kN/m" if udl_r.get(i, 0.0) != 0.0 else "—"})
        st.dataframe(pd.DataFrame(elem_rows), width="stretch", hide_index=True)
        st.markdown("**⚙️ Material / Section**")
        st.dataframe(pd.DataFrame([{"E (kN/m²)": f"{E_val:.3e}", "A (m²)": f"{A_val:.4f}", "I (m⁴)": f"{I_val:.6f}"}]), width="stretch", hide_index=True)
        st.markdown("**➡️ Nodal Loads**")
        dof_names = {0: "Fx (kN)", 1: "Fy (kN)", 2: "M (kN·m)"}
        lrows = [{"Node": f"N{nid + 1}", "Load": dof_names[ld], "Value": f"{val:.1f}"} for nid, ldmap in nodal_loads_parsed.items() for ld, val in ldmap.items()]
        st.dataframe(pd.DataFrame(lrows if lrows else [{"Note": "No nodal loads"}]), width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif step == 1:
    col_v, col_t = st.columns([1.4, 1])
    with col_v:
        fig1 = draw_frame(nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed, U=res["U"], dof_map=res["dof_map"], show_dofs=True, show_loads=False, udl_loads=udl_r)
        st.pyplot(fig1, width="stretch")
        plt.close(fig1)
    with col_t:
        st.markdown("<div class='step-card'>", unsafe_allow_html=True)
        st.markdown("<span class='step-title'>Step 1 — DOF Numbering</span>", unsafe_allow_html=True)
        st.markdown("<span class='step-subtitle'>Each node gets 3 global DOFs: u, v, θ</span>", unsafe_allow_html=True)
        dof_rows = []
        for i in range(nn):
            dm = res["dof_map"][i]
            tags = ["FREE" if g in res["free_global"] else "FIXED" for g in dm]
            dof_rows.append({"Node": f"N{i + 1}", "u→d": dm[0], "v→d": dm[1], "θ→d": dm[2], "u BC": tags[0], "v BC": tags[1], "θ BC": tags[2]})
        st.dataframe(pd.DataFrame(dof_rows), width="stretch", hide_index=True)
        st.markdown(f"- Total DOFs: **{res['nDOF']}** | Fixed: **{len(res['fixed_global'])}** → {res['fixed_global']} | Free: **{len(res['free_global'])}** → {res['free_global']}")
        st.markdown("</div>", unsafe_allow_html=True)

elif step == 2:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 2 — Local Stiffness Matrix [k] (6×6)</span>", unsafe_allow_html=True)
    st.markdown("<span class='step-subtitle'>Euler–Bernoulli beam-column element in local axes</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    elem_sel = st.selectbox("Select Element:", [f"E{i + 1} — {lbl}" for i, lbl in enumerate(elem_labels_p)])
    ei = int(elem_sel.split("E")[1].split(" ")[0]) - 1
    e = ed[ei]
    dof_lbls = [f"u{e['ni'] + 1}", f"v{e['ni'] + 1}", f"θ{e['ni'] + 1}", f"u{e['nj'] + 1}", f"v{e['nj'] + 1}", f"θ{e['nj'] + 1}"]
    show_matrix(e["k_loc"], row_labels=dof_lbls, col_labels=dof_lbls, caption=f"[k] for E{ei + 1}")

elif step == 3:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 3 — Transformation Matrix [T]</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    elem_sel = st.selectbox("Select Element:", [f"E{i + 1} — {lbl}" for i, lbl in enumerate(elem_labels_p)])
    ei = int(elem_sel.split("E")[1].split(" ")[0]) - 1
    e = ed[ei]
    st.markdown(f"**E{ei + 1}: α = {e['alpha_deg']:.3f}° → c = {e['c']:.4f}, s = {e['s']:.4f}**")
    dof_lbls = [f"u{e['ni'] + 1}", f"v{e['ni'] + 1}", f"θ{e['ni'] + 1}", f"u{e['nj'] + 1}", f"v{e['nj'] + 1}", f"θ{e['nj'] + 1}"]
    show_matrix(e["T6"], row_labels=dof_lbls, col_labels=dof_lbls, caption="[T] — 6×6")

elif step == 4:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 4 — Global Element Stiffness [K]ₑ = [T]ᵀ[k][T]</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    elem_sel = st.selectbox("Select Element:", [f"E{i + 1} — {lbl}" for i, lbl in enumerate(elem_labels_p)])
    ei = int(elem_sel.split("E")[1].split(" ")[0]) - 1
    e = ed[ei]
    dof_lbls = [f"d{g}" for g in e["gdofs"]]
    show_matrix(e["k_glob"], row_labels=dof_lbls, col_labels=dof_lbls, caption=f"[K]ₑ for E{ei + 1} mapped to global DOFs {e['gdofs']}")

elif step == 5:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 5 — Global Assembly [K] and {F}</span>", unsafe_allow_html=True)
    st.markdown("<span class='step-subtitle'>Element matrices plus UDL FEF contributions are scattered into the global system.</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if res.get("fef_summaries"):
        st.markdown("#### 〰️ UDL Fixed-End Force Details")
        fef_rows = []
        for ei, fs in res["fef_summaries"].items():
            fef_rows.append({"Element": f"E{ei + 1}", "w (kN/m)": f"{fs['w']:+.3f}", "L (m)": f"{fs['L']:.3f}", "Vi=wL/2": f"{fs['Vi']:.4f}", "Mi=wL²/12": f"{fs['Mi']:.4f}", "Vj=wL/2": f"{fs['Vj']:.4f}", "Mj=-wL²/12": f"{fs['Mj']:.4f}"})
        st.dataframe(pd.DataFrame(fef_rows), width="stretch", hide_index=True)
    dof_lbls = [f"d{i}" for i in range(res["nDOF"])]
    col_k, col_f = st.columns([3, 1])
    with col_k:
        show_matrix(res["K"], row_labels=dof_lbls, col_labels=dof_lbls, caption=f"Global [K] — {res['nDOF']}×{res['nDOF']}", highlight_rows=res["free_global"], highlight_cols=res["free_global"])
    with col_f:
        show_vector(res["F"], labels=dof_lbls, caption="{F} — point loads + UDL equivalent loads", highlight=[i for i, v in enumerate(res["F"]) if abs(v) > 1e-10])

elif step == 6:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 6 — Partition and Apply Boundary Conditions</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    ff = res["free_global"]
    dof_lbls_f = [f"d{g}" for g in ff]
    col_kff, col_ff = st.columns([3, 1])
    with col_kff:
        show_matrix(res["Kff"], row_labels=dof_lbls_f, col_labels=dof_lbls_f, caption=f"[Kff] — {len(ff)}×{len(ff)}")
    with col_ff:
        show_vector(res["Ff"], labels=dof_lbls_f, caption="{Ff} — reduced loads")
    st.markdown(f"Condition number κ(Kff) = **{res['cond_num']:.3e}**")

elif step == 7:
    col_v, col_t = st.columns([1.4, 1])
    with col_v:
        fig7 = draw_frame(nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed, U=res["U"], dof_map=res["dof_map"], show_deformed=True, show_loads=True, udl_loads=udl_r)
        st.pyplot(fig7, width="stretch")
        plt.close(fig7)
    with col_t:
        st.markdown("<div class='step-card'>", unsafe_allow_html=True)
        st.markdown("<span class='step-title'>Step 7 — Solve {Uf} = [Kff]⁻¹{Ff}</span>", unsafe_allow_html=True)
        rows = []
        for gi in range(res["nDOF"]):
            rows.append({"DOF": f"d{gi}", "Node": f"N{gi // 3 + 1}", "Type": ["u", "v", "θ"][gi % 3], "Value": fmt_val(res["U"][gi], 6), "Status": "FIXED=0" if gi in res["fixed_global"] else "SOLVED"})
        _U = np.asarray(res["U"], dtype=float)
        _tr = [gi for gi in range(res["nDOF"]) if gi % 3 != 2]
        _rt = [gi for gi in range(res["nDOF"]) if gi % 3 == 2]
        _g_tr = max(_tr, key=lambda g: abs(_U[g])) if _tr else None
        _g_rt = max(_rt, key=lambda g: abs(_U[g])) if _rt else None
        _hl7 = {g for g in (_g_tr, _g_rt) if g is not None and abs(_U[g]) > 1e-15}
        st.dataframe(pd.DataFrame(rows).style.apply(lambda r: [HL_STYLE if r.name in _hl7 else "" for _ in r], axis=1),
                     width="stretch", hide_index=True)
        if _g_tr is not None and _g_rt is not None:
            st.markdown(f"**Key results** (highlighted): largest translation **d{_g_tr}** (N{_g_tr // 3 + 1} {['u', 'v'][_g_tr % 3]}) = "
                        f"**{_U[_g_tr]:.4e} m**; largest rotation **d{_g_rt}** (N{_g_rt // 3 + 1} θ) = **{_U[_g_rt]:.4e} rad**.")
        st.markdown("</div>", unsafe_allow_html=True)

elif step == 8:
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.markdown("<span class='step-title'>Step 8 — Member End Forces {f'} = [k][T]{u} − {FEF}</span>", unsafe_allow_html=True)
    st.markdown("<span class='step-subtitle'>FEF is subtracted to obtain true member forces when UDL is present.</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    force_rows = []
    for i, (mr, e) in enumerate(zip(res["member_results"], ed)):
        force_rows.append({"Elem": f"E{i + 1}", "Label": elem_labels_p[i], "UDL": f"w={udl_r.get(i, 0.0):+.2f}" if udl_r.get(i, 0.0) != 0.0 else "—", "N_i (kN)": f"{mr['N_i']:.3f}", "V_i (kN)": f"{mr['V_i']:.3f}", "M_i (kN·m)": f"{mr['M_i']:.3f}", "N_j (kN)": f"{mr['N_j']:.3f}", "V_j (kN)": f"{mr['V_j']:.3f}", "M_j (kN·m)": f"{mr['M_j']:.3f}"})
    _cells = []   # (row, column, value) for end forces
    for i, mr in enumerate(res["member_results"]):
        for col_name, key in (("N_i (kN)", "N_i"), ("N_j (kN)", "N_j"), ("V_i (kN)", "V_i"), ("V_j (kN)", "V_j"), ("M_i (kN·m)", "M_i"), ("M_j (kN·m)", "M_j")):
            _cells.append((i, col_name, key[0], float(mr[key])))
    _hl8 = set()
    for q in ("N", "V", "M"):
        cand = [c for c in _cells if c[2] == q]
        if cand:
            best = max(cand, key=lambda c: abs(c[3]))
            if abs(best[3]) > 1e-9:
                _hl8.add((best[0], best[1]))
    _df8 = pd.DataFrame(force_rows)
    st.dataframe(_df8.style.apply(lambda r: [HL_STYLE if (r.name, c) in _hl8 else "" for c in r.index], axis=1),
                 width="stretch", hide_index=True)
    # moment extremes in each member, including inside UDL spans (where V = 0)
    _ext_rows = []
    _absmax = (None, 0.0)
    for i, (mr, e) in enumerate(zip(res["member_results"], ed)):
        Le = float(e["L"]); we = float(udl_r.get(i, 0.0))
        pts = [(0.0, mr["M_i"]), (Le, mr["M_j"])]
        if abs(we) > 1e-12:
            xs = -mr["V_i"] / we
            if 1e-9 < xs < Le - 1e-9:
                pts.append((xs, mr["M_i"] * (1 - xs / Le) + mr["M_j"] * xs / Le - we * xs * (Le - xs) / 2.0))
        pmax = max(pts, key=lambda q: q[1]); pmin = min(pts, key=lambda q: q[1])
        vmax = max((mr["V_i"], mr["V_j"]), key=abs)
        _ext_rows.append({
            "Elem": f"E{i + 1}", "Label": elem_labels_p[i],
            "Max +M (kN·m)": f"{pmax[1]:.3f}" if pmax[1] > 1e-9 else "—",
            "at x (m)": f"{pmax[0]:.2f}" if pmax[1] > 1e-9 else "—",
            "Max −M (kN·m)": f"{pmin[1]:.3f}" if pmin[1] < -1e-9 else "—",
            "at x  (m)": f"{pmin[0]:.2f}" if pmin[1] < -1e-9 else "—",
            "Max |V| (kN)": f"{abs(vmax):.3f}",
        })
        for x_, m_ in pts:
            if abs(m_) > abs(_absmax[1]):
                _absmax = ((i, x_), m_)
    st.markdown("**Key results — bending moment extremes in each member** (x from end i; + = sagging, − = hogging, about the member's local axis)")
    _df_ext = pd.DataFrame(_ext_rows)
    st.dataframe(_df_ext.style.apply(lambda r: [HL_STYLE if (c in ("Max +M (kN·m)", "Max −M (kN·m)") and r[c] != "—"
                                                and abs(float(r[c]) - _absmax[1]) < 5e-4 and r.name == _absmax[0][0]) else "" for c in r.index], axis=1),
                 width="stretch", hide_index=True)
    if _absmax[0] is not None:
        st.caption(f"Highlighted: the largest bending moment in the structure, {_absmax[1]:.3f} kN·m in E{_absmax[0][0] + 1}. "
                   "Interior values are where V = 0 inside a UDL span.")
    st.divider()

    element_options = [f"E{i + 1} — {lbl}" for i, lbl in enumerate(elem_labels_p)]
    udl_indices = [i for i in range(ne) if abs(udl_r.get(i, 0.0)) > 1e-9]
    default_ei = min(udl_indices[0] if udl_indices else 0, len(element_options) - 1)
    elem_sel = st.selectbox("Detail view:", element_options, index=default_ei, key=f"step8_elem_{chosen}_{tuple(sorted(udl_r.items()))}")
    ei = int(elem_sel.split("E")[1].split(" ")[0]) - 1
    mr = res["member_results"][ei]
    e = ed[ei]
    loc_lbls = ["u'i", "v'i", "θ'i", "u'j", "v'j", "θ'j"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**{u_local} = [T]·{u_global}**")
        show_vector(mr["u_local"], labels=loc_lbls, caption="Local displacements (m/rad)")
    with c2:
        st.markdown("**{f_elastic} = [k]·{u_local}**")
        show_vector(mr["f_local_elastic"], labels=loc_lbls, caption="Before FEF correction")
    with c3:
        w_ei = udl_r.get(ei, 0.0)
        st.markdown(f"**{{FEF_local}}** (w={w_ei:+.2f} kN/m)" if abs(w_ei) > 1e-9 else "**{FEF_local}** = 0 (no UDL)")
        show_vector(mr["fef_local"], labels=loc_lbls, caption="Fixed-end forces (subtracted)")

    st.info(f"✅ **E{ei + 1}** corrected end forces = f_elastic − FEF → M_i = {mr['M_i']:.3f}, V_i = {mr['V_i']:.3f}, M_j = {mr['M_j']:.3f}, V_j = {mr['V_j']:.3f} kN/kN·m")
    st.caption("BMD uses exact formula: M(x) = Mᵢ(1−x/L) + Mⱼ(x/L) − w·x(L−x)/2; SFD uses V(x) = Vᵢ + w·x. Signs: N tension +, M sagging +, V = dM/dx.")
    fig_bmd = draw_bmd_sfd(nodes_parsed, elems_parsed, res["member_results"], elem_labels_p, member_loads=udl_r)
    st.pyplot(fig_bmd, width="stretch")
    plt.close(fig_bmd)
    with st.expander("📚 Why do we subtract the FEF? — visual explanation", expanded=False):
        _w8 = w_ei if abs(w_ei) > 1e-9 else -10.0
        show_udl_fef_explainer(f"step8_{ei}", w_default=_w8, L_default=float(e["L"]))

elif step == 9:
    col_v, col_t = st.columns([1.4, 1])
    with col_v:
        fig9 = draw_frame(nodes_parsed, elems_parsed, fixed_dofs_ui, nodal_loads_parsed, U=res["U"], dof_map=res["dof_map"], show_deformed=True, show_reactions=True, reactions=res["reactions"], udl_loads=udl_r)
        st.pyplot(fig9, width="stretch")
        plt.close(fig9)
    with col_t:
        st.markdown("<div class='step-card'>", unsafe_allow_html=True)
        st.markdown("<span class='step-title'>Step 9 — Support Reactions and Equilibrium</span>", unsafe_allow_html=True)
        rx_rows = []
        dof_names_map = {0: "Rx (kN)", 1: "Ry (kN)", 2: "RM (kN·m)"}
        for gi, val in sorted(res["reactions"].items()):
            rx_rows.append({"Node": f"N{gi // 3 + 1}", "DOF": f"d{gi}", "Type": dof_names_map[gi % 3], "Reaction": f"{val:.4f}"})
        _rx = sorted(res["reactions"].items())
        _hl9 = set()
        for t in (0, 1, 2):
            cand = [(k, gi, v) for k, (gi, v) in enumerate(_rx) if gi % 3 == t]
            if cand:
                kbest = max(cand, key=lambda c: abs(c[2]))
                if abs(kbest[2]) > 1e-9:
                    _hl9.add(kbest[0])
        st.dataframe(pd.DataFrame(rx_rows).style.apply(lambda r: [HL_STYLE if r.name in _hl9 else "" for _ in r], axis=1),
                     width="stretch", hide_index=True)
        st.caption("Highlighted: the largest reaction of each type (Rx, Ry, RM).")
        st.markdown("**⚖️ Global Equilibrium Check**")
        tol = 1e-3
        for label, val in res["eq_check"].items():
            st.markdown(f"{'✅' if abs(val) < tol else '❌'} **{label}** = {val:.6f}")
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
ref_c1, ref_c2, ref_c3 = st.columns(3)
with ref_c1:
    st.markdown(
        """
**📚 References**
- Kassimali A. — *Matrix Analysis of Structures*
- McGuire, Gallagher, Ziemian — *Matrix Structural Analysis*
- Bhavikatti S.S. — *Matrix Methods of Structural Analysis*
"""
    )
with ref_c2:
    st.markdown(
        """
**🧮 DSM + UDL Formula Card**
- [k] = 6×6 Euler–Bernoulli stiffness
- [T] = block-diagonal rotation matrix
- [K]ₑ = [T]ᵀ[k][T]
- FEF_local = [0, wL/2, wL²/12, 0, wL/2, −wL²/12]
- {F} += [T]ᵀ · {FEF_local}
- {f_local} = [k][T]{u} − {FEF_local}
- M(x) = Mi(1−x/L) + Mj(x/L) − w·x(L−x)/2
"""
    )
with ref_c3:
    st.markdown("**🚀 Deploy This App**")
    st.code("pip install streamlit numpy pandas matplotlib\nstreamlit run app2.py", language="bash")
    st.markdown("Or push to GitHub and connect at  \n[share.streamlit.io](https://share.streamlit.io)")
