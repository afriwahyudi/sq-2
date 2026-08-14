"""
Gibbs-Duhem Equation in Vector Space

Demonstrates that Gibbs-Duhem requires v_1 = v_2 = ... = v_N,
but NOT v_j = 0, where v = x^T J, J_ij = d ln(gamma_i)/d x_j.

Run: streamlit run geometric_gd.py
"""

from __future__ import annotations

import hashlib

import numpy as np
import sympy as sp
import streamlit as st
import plotly.graph_objects as go

# =================================================================
st.set_page_config(
    page_title="Gibbs-Duhem Equation in Vector Space",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Gibbs-Duhem Equation in Vector Space")

DEFAULT_GE = {2: "x1*x2", 3: "x1*x2 + x1*x3 + x2*x3"}
DEFAULT_LNG = {
    2: ["x2**2", "x1**2"],
    3: [
        "-x1*x2 - x1*x3 - x2*x3 + x2 + x3",
        "-x1*x2 - x1*x3 + x1 - x2*x3 + x3",
        "-x1*x2 - x1*x3 + x1 - x2*x3 + x2",
    ],
}


def default_gE(N: int) -> str:
    """Symmetric regular-solution gᴱ = Σ_{i<j} xi·xj, generalized to any N."""
    if N in DEFAULT_GE:
        return DEFAULT_GE[N]
    terms = [f"x{i+1}*x{j+1}" for i in range(N) for j in range(i + 1, N)]
    return " + ".join(terms)


def default_lng(N: int) -> list[str]:
    """Ideal (ln γᵢ = 0) default for component counts without a curated example."""
    if N in DEFAULT_LNG:
        return DEFAULT_LNG[N]
    return ["0"] * N

PALETTE = dict(
    simplex="#f7c6a3",
    boundary="#e07a2c",
    composition="#c0392b",
    v_vector="#2e5cb8",
    normal="#2e8b57",
    tangent="#e07a2c",
    tangent2="#8e44ad",
)


# =================================================================
def _symbols_and_locals(N: int):
    xs = sp.symbols(" ".join(f"x{i + 1}" for i in range(N)))
    xs = tuple(xs) if N > 1 else (xs,)
    local_dict = {f"x{i + 1}": xs[i] for i in range(N)}
    local_dict.update({k: getattr(sp, k) for k in ["sin", "cos", "tan", "exp", "log", "sqrt"]})
    local_dict["pi"] = sp.pi
    return xs, local_dict


def _gE_to_lngamma(xs, N, gE):
    """The one true gᴱ -> ln(γᵢ) relation, shared by every code path that needs it."""
    grad_gE = [sp.diff(gE, xs[i]) for i in range(N)]
    correction = sum(xs[i] * grad_gE[i] for i in range(N))
    return [sp.simplify(gE + grad_gE[i] - correction) for i in range(N)]


@st.cache_data(show_spinner=False)
def build_symbolic_system(N: int, mode: str, expr_text: tuple[str, ...]):
    """Parse user expressions and derive ln(gamma), J, and v symbolically.

    Cached on (N, mode, expr_text) so re-running only happens when the
    *model* changes, not when sliders move.
    """
    xs, local_dict = _symbols_and_locals(N)

    if mode == "gᴱ":
        gE = sp.sympify(expr_text[0], locals=local_dict)
        lngamma = _gE_to_lngamma(xs, N, gE)
    else:
        lngamma = [sp.sympify(t, locals=local_dict) for t in expr_text]

    J = sp.Matrix([[sp.diff(lngamma[i], xs[j]) for j in range(N)] for i in range(N)])
    v_symbolic = [sp.simplify((sp.Matrix(1, N, xs) * J)[0, j]) for j in range(N)]

    return xs, lngamma, J, v_symbolic


@st.cache_data(show_spinner=False)
def derive_lngamma_defaults(N: int, gE_str: str) -> list[str]:
    """Derive the ln(γᵢ) default text shown when switching into ln(γᵢ) mode,
    directly from the current gᴱ expression via sympy — so the two modes
    start out numerically identical, and any inconsistency introduced
    afterward comes only from the user editing individual ln(γᵢ) boxes.
    """
    xs, local_dict = _symbols_and_locals(N)
    gE = sp.sympify(gE_str, locals=local_dict)
    lngamma = _gE_to_lngamma(xs, N, gE)
    return [str(e) for e in lngamma]


def evaluate_system(xs, lngamma, J, v_symbolic, x_values, N):
    sub = {xs[i]: float(x_values[i]) for i in range(N)}
    lngamma_values = np.array([float(e.subs(sub)) for e in lngamma])
    J_values = np.array([[float(J[i, j].subs(sub)) for j in range(N)] for i in range(N)])
    v_values = np.array([float(e.subs(sub)) for e in v_symbolic])
    return lngamma_values, J_values, v_values


# =================================================================
with st.sidebar:
    st.header("Model")
    N_choice = st.radio("Number of components", [2, 3, "Custom N"], horizontal=True)
    if N_choice == "Custom N":
        N = st.number_input(
            "N (components)", min_value=2, max_value=30, value=5, step=1,
            help="Any N ≥ 2. The geometric simplex/vector plot is only available "
                 "for N = 2 or 3, but every other computation still works.",
        )
        N = int(N)
    else:
        N = N_choice

    can_visualize = N in (2, 3)
    if not can_visualize:
        st.info("Geometric visualization is unavailable above N = 3; all "
                "other tabs (ln γᵢ, Jacobian, v, tangent test, bar chart) "
                "still work.")

    mode = st.radio("Define ln(γᵢ) using", ["gᴱ", "ln(γᵢ)"])

    st.divider()

    if mode == "gᴱ":
        gE_text = st.text_area(
            "gᴱ =",
            value=st.session_state.get(f"gE_text_{N}", default_gE(N)),
            key=f"gE_text_{N}",
            height=80,
        )
        expr_text = (gE_text,)
    else:

        gE_source = st.session_state.get(f"gE_text_{N}", default_gE(N))
        try:
            derived_defaults = derive_lngamma_defaults(N, gE_source)
            st.caption(f"Defaults derived from gᴱ = {gE_source}")
        except Exception:
            derived_defaults = default_lng(N)
            st.caption("Couldn't derive defaults from gᴱ — showing fallback values.")

        baseline_id = hashlib.md5(gE_source.encode()).hexdigest()[:8]
        expr_text = tuple(
            st.text_input(
                f"ln(γ_{i + 1}) =",
                value=derived_defaults[i],
                key=f"lng_{N}_{baseline_id}_{i}",
            )
            for i in range(N)
        )

    st.divider()
    st.header("Composition")

    if N == 2:
        x1 = st.slider("x₁", 0.001, 0.999, 0.500, 0.001)
        x_values = np.array([x1, 1.0 - x1])
    elif N == 3:
        x1 = st.slider("x₁", 0.001, 0.998, 0.340, 0.001)
        x2 = st.slider("x₂", 0.001, 0.999 - x1, min(0.330, 0.999 - x1), 0.001)
        x_values = np.array([x1, x2, 1.0 - x1 - x2])
    else:
        remaining = 1.0
        x_list = []
        for i in range(N - 1):
            components_left_after_this = N - 1 - i
            max_val = max(round(remaining - 0.001 * components_left_after_this, 3), 0.001)
            default_val = min(max(round(remaining / (N - i), 3), 0.001), max_val)
            xi = st.slider(f"x_{i+1}", 0.001, max_val, default_val, 0.001, key=f"x_slider_{i}")
            x_list.append(xi)
            remaining -= xi
        x_list.append(remaining)
        x_values = np.array(x_list)

    st.caption("Composition")
    st.code(", ".join(f"x{i+1} = {v:.4f}" for i, v in enumerate(x_values)), language=None)
    st.caption(f"Σxᵢ = {np.sum(x_values):.4f}")



# =================================================================
try:
    xs, lngamma, J, v_symbolic = build_symbolic_system(N, mode, expr_text)
except Exception as e:
    st.error(f"Could not parse expression(s): {e}")
    st.stop()

lngamma_values, J_values, v_values = evaluate_system(xs, lngamma, J, v_symbolic, x_values, N)

normal_unit = np.ones(N) / np.sqrt(N)
v_normal = np.dot(v_values, normal_unit) * normal_unit
v_tangent = v_values - v_normal
v_norm = np.linalg.norm(v_values)
max_diff = max(abs(v_values[j] - v_values[-1]) for j in range(N - 1))
gd_satisfied = max_diff < 1e-10

# =================================================================
with st.expander("What does Gibbs-Duhem actually require?", expanded=True):
    st.latex(r"\boxed{\text{Gibbs-Duhem does not necessarily require } v_j = 0}")
    st.markdown("Define the Jacobian and the vector:")
    st.latex(
        r"J_{ij} = \frac{\partial \ln(\gamma_i)}{\partial x_j} \qquad "
        r"v_j = \sum_i x_i \frac{\partial \ln(\gamma_i)}{\partial x_j} "
        r"\qquad \boxed{v = x^T J}"
    )
    st.markdown(
        "For every physical composition displacement, "
        r"$\sum_j dx_j = 0$, so Gibbs-Duhem gives $v\cdot dx = 0$ "
        "for every tangent direction of the simplex."
    )
    st.markdown(
        "The simplex normal is $\\mathbf{1}=(1,\\dots,1)$, so the "
        "actual requirement is:"
    )
    st.latex(r"\boxed{v \parallel \mathbf{1}} \qquad \boxed{v = c\,\mathbf{1}}")
    st.markdown("The scalar $c$ is **not** required to be zero.")

# =================================================================
status_col1, status_col2, status_col3, status_col4 = st.columns(4)
with status_col1:
    st.metric("‖v‖", f"{v_norm:.4f}")
with status_col2:
    st.metric("max |vᵢ − v_N|", f"{max_diff:.2e}")
with status_col3:
    st.metric("Gibbs-Duhem Compliance", "✅ Satisfied" if gd_satisfied else "❌ Violated")
with status_col4:
    st.metric("det(J)", f"{np.linalg.det(J_values):.4g}")

if gd_satisfied:
    st.success(
        f"Gibbs-Duhem is satisfied: all vⱼ are equal (v_j = {v_values[-1]:.4f}). "
        + ("This is also the trivial case v_j = 0." if abs(v_values[-1]) < 1e-10
           else "Note v_j ≠ 0, which is still fully consistent.")
    )
else:
    st.error("Gibbs-Duhem is violated: the vⱼ are not equal.")

# =================================================================
tab_labels = ["Activity coefficient", "Jacobian", "Consistency coefficient", "Tangent test", "Bar chart"]
if can_visualize:
    tab_labels.append("Simplex view")

tabs = st.tabs(tab_labels)
tab_gamma, tab_J, tab_v, tab_test, tab_bar = tabs[:5]
tab_3d = tabs[5] if can_visualize else None

with tab_gamma:
    st.subheader("Activity coefficient expressions")
    for i in range(N):
        st.latex(rf"\ln(\gamma_{{{i+1}}}) = " + sp.latex(lngamma[i]) + f" = {lngamma_values[i]:.4f}")

with tab_J:
    st.subheader("Jacobian matrix")
    st.latex(r"J_{ij} = \frac{\partial \ln(\gamma_i)}{\partial x_j}")
    st.latex(r"J = " + sp.latex(J))
    st.caption("Numerical value at the current composition")

    mat_col, det_col = st.columns([3, 1])
    with mat_col:
        st.dataframe(
            J_values,
            use_container_width=False,
            column_config={i: st.column_config.NumberColumn(f"x{i+1}", format="%.4f") for i in range(N)},
        )
    with det_col:
        det_numeric = float(np.linalg.det(J_values))
        st.metric("det(J)", f"{det_numeric:.6g}")

    show_symbolic_det = st.checkbox(
        "Also compute symbolic det(J)",
        value=(N <= 4),
        help="Symbolic determinants get expensive fast as N grows — "
             "left unchecked by default above N = 4.",
    )
    if show_symbolic_det:
        with st.spinner("Computing symbolic determinant..."):
            try:
                det_symbolic = sp.simplify(J.det())
                st.latex(r"\det(J) = " + sp.latex(det_symbolic))
            except Exception as e:
                st.warning(f"Could not compute symbolic determinant: {e}")

with tab_v:
    st.subheader("Consistency coefficients")
    st.latex(
        r"\boxed{v = x^T J} \qquad "
        r"v_j = \sum_i x_i \frac{\partial \ln(\gamma_i)}{\partial x_j}"
    )
    st.latex(r"v = \left[" + r",\ ".join(sp.latex(e) for e in v_symbolic) + r"\right]")

    vcols = st.columns(N)
    for j in range(N):
        with vcols[j]:
            st.metric(f"v_{j+1}", f"{v_values[j]:.4f}")

    st.caption("Pairwise differences vs. v_N")
    diffcols = st.columns(max(N - 1, 1))
    for j in range(N - 1):
        with diffcols[j]:
            st.metric(f"v_{j+1} − v_{N}", f"{v_values[j] - v_values[-1]:.2e}")

with tab_test:
    st.subheader("Direct Gibbs-Duhem test with tangent directions")
    st.markdown(
        "Random physical displacements are generated with "
        r"$\sum_i dx_i = 0$ (i.e. tangent to the simplex), and $v\cdot dx$ "
        "is evaluated for each — it should vanish to numerical precision "
        "regardless of whether $v$ itself is zero."
    )

    rng = np.random.default_rng(42)
    rows = []
    for _ in range(5):
        raw = rng.normal(size=N)
        dx = raw - np.mean(raw)
        dx /= np.linalg.norm(dx)
        rows.append((np.round(dx, 4).tolist(), np.dot(v_values, dx)))

    for dx, dot in rows:
        st.write(f"dx = `{dx}`  →  v·dx = `{dot:.3e}`")

with tab_bar:
    st.subheader("Components of v")
    fig_bar = go.Figure(
        go.Bar(x=[f"v_{i+1}" for i in range(N)], y=v_values, marker_color=PALETTE["v_vector"])
    )
    fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_bar.update_layout(
        height=380,
        yaxis_title="v_j",
        template="plotly_white",
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

if can_visualize:
    with tab_3d:
        st.subheader("v relative to the composition simplex")

        if N == 2:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[1, 0], mode="lines",
                    line=dict(color=PALETTE["simplex"], width=8), name="Simplex",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[x_values[0]], y=[x_values[1]], mode="markers",
                    marker=dict(size=14, color=PALETTE["composition"]), name="Composition x",
                )
            )

            if v_norm > 1e-14:
                vd = v_values / v_norm * 0.30
                fig.add_trace(
                    go.Scatter(
                        x=[x_values[0], x_values[0] + vd[0]],
                        y=[x_values[1], x_values[1] + vd[1]],
                        mode="lines+markers",
                        line=dict(color=PALETTE["v_vector"], width=6),
                        marker=dict(size=[0, 10]), name="v",
                    )
                )

            t = np.array([1.0, -1.0])
            t = t / np.linalg.norm(t) * 0.25
            fig.add_trace(
                go.Scatter(
                    x=[x_values[0], x_values[0] + t[0]],
                    y=[x_values[1], x_values[1] + t[1]],
                    mode="lines+markers",
                    line=dict(color=PALETTE["tangent"], width=5, dash="dash"),
                    marker=dict(size=[0, 8]), name="Tangent direction",
                )
            )

            fig.update_layout(
                height=650,
                xaxis_title="x₁",
                yaxis_title="x₂",
                template="plotly_white",
                xaxis=dict(range=[-0.35, 1.35], scaleanchor="y"),
                yaxis=dict(range=[-0.35, 1.35]),
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            fig = go.Figure()
            vertices = np.eye(3)
            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                    i=[0], j=[1], k=[2], opacity=0.35, color=PALETTE["simplex"],
                    name="Simplex", showscale=False,
                )
            )

            boundary = np.vstack([vertices, vertices[0]])
            fig.add_trace(
                go.Scatter3d(
                    x=boundary[:, 0], y=boundary[:, 1], z=boundary[:, 2],
                    mode="lines", line=dict(color=PALETTE["boundary"], width=5),
                    name="Simplex boundary",
                )
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[x_values[0]], y=[x_values[1]], z=[x_values[2]],
                    mode="markers", marker=dict(size=9, color=PALETTE["composition"]),
                    name="Composition x",
                )
            )

            if v_norm > 1e-14:
                vd = v_values / v_norm * 0.35
                fig.add_trace(
                    go.Scatter3d(
                        x=[x_values[0], x_values[0] + vd[0]],
                        y=[x_values[1], x_values[1] + vd[1]],
                        z=[x_values[2], x_values[2] + vd[2]],
                        mode="lines+markers",
                        line=dict(color=PALETTE["v_vector"], width=9),
                        marker=dict(size=[0, 9]), name="v",
                    )
                )

            nd = normal_unit * 0.35
            fig.add_trace(
                go.Scatter3d(
                    x=[x_values[0], x_values[0] + nd[0]],
                    y=[x_values[1], x_values[1] + nd[1]],
                    z=[x_values[2], x_values[2] + nd[2]],
                    mode="lines+markers",
                    line=dict(color=PALETTE["normal"], width=6, dash="dash"),
                    marker=dict(size=[0, 8]), name="Simplex normal",
                )
            )

            for t_raw, label, color_key in [
                ([1.0, -1.0, 0.0], "Tangent 1", "tangent"),
                ([1.0, 0.0, -1.0], "Tangent 2", "tangent2"),
            ]:
                t = np.array(t_raw)
                t = t / np.linalg.norm(t) * 0.25
                fig.add_trace(
                    go.Scatter3d(
                        x=[x_values[0], x_values[0] + t[0]],
                        y=[x_values[1], x_values[1] + t[1]],
                        z=[x_values[2], x_values[2] + t[2]],
                        mode="lines", line=dict(color=PALETTE[color_key], width=6), name=label,
                    )
                )

            fig.update_layout(
                height=700,
                template="plotly_white",
                scene=dict(
                    xaxis_title="x₁", yaxis_title="x₂", zaxis_title="x₃",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1]),
                    zaxis=dict(range=[0, 1]),
                    aspectmode="cube",
                ),
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)