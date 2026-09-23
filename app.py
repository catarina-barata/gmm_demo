import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration for Streamlit Cloud
st.set_page_config(
    page_title="GMM: Generative Process vs EM Algorithm",
    page_icon="📊",
    layout="wide"
)

st.title("📊 GMM: Ground Truth Sampling vs. EM Algorithm")
st.markdown("""
Compare the **true generative model** against the **Expectation-Maximization (EM)** inference process, observe how the **incomplete log-likelihood** evolves over iterations, and explore how **covariance constraints** (Spherical, Diagonal, Full) impact cluster fitting.
""")

# --- SIDEBAR: PARAMETERS ---
st.sidebar.header("1. Generative Ground Truth")

N = st.sidebar.slider("Number of Data Points (N)", min_value=100, max_value=1500, value=400, step=50)
K = st.sidebar.slider("Number of Components (K)", min_value=2, max_value=5, value=3, step=1)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.header("2. True Component Parameters")

raw_pis = []
for k in range(K):
    raw_pis.append(st.sidebar.slider(f"Weight (π_{k+1})", 0.1, 1.0, 1.0 / K, 0.05, key=f"pi_gen_{k}"))

pi_true = np.array(raw_pis) / np.sum(raw_pis)

means_true = []
covs_true = []

for k in range(K):
    st.sidebar.markdown(f"**True Component {k+1} Specs**")
    col1, col2 = st.sidebar.columns(2)
    
    default_x = (k - (K - 1) / 2.0) * 4.0
    default_y = (k % 2) * 3.0 - 1.5
    
    mx = col1.number_input(f"μ_x ({k+1})", value=float(default_x), key=f"mx_{k}")
    my = col2.number_input(f"μ_y ({k+1})", value=float(default_y), key=f"my_{k}")
    means_true.append([mx, my])
    
    std_x = col1.slider(f"σ_x ({k+1})", 0.3, 3.0, 1.0, key=f"sx_{k}")
    std_y = col2.slider(f"σ_y ({k+1})", 0.3, 3.0, 1.0, key=f"sy_{k}")
    corr = col2.slider(f"Corr ({k+1})", -0.9, 0.9, 0.0, key=f"corr_{k}")
    
    cov_xy = corr * std_x * std_y
    cov = np.array([[std_x**2, cov_xy], [cov_xy, std_y**2]])
    covs_true.append(cov)

st.sidebar.markdown("---")
st.sidebar.header("3. EM Model Constraints")

# Covariance Type Selection
cov_type = st.sidebar.selectbox(
    "EM Covariance Structure",
    options=["Full", "Diagonal", "Spherical"],
    index=0,
    help="• Full: Arbitrary rotation and variances (3 params/component)\n• Diagonal: Axis-aligned variances (2 params/component)\n• Spherical: Equal variances along both axes (1 param/component)"
)

# --- GENERATION FUNCTION ---
@st.cache_data(show_spinner=False)
def generate_gmm_data(N_val, K_val, pi_vec, means_list, covs_list, random_seed):
    np.random.seed(random_seed)
    z = np.random.choice(K_val, size=N_val, p=pi_vec)
    
    X = np.zeros((N_val, 2))
    for i in range(N_val):
        k_selected = z[i]
        X[i] = np.random.multivariate_normal(means_list[k_selected], covs_list[k_selected])
        
    df_out = pd.DataFrame({
        "x1": X[:, 0],
        "x2": X[:, 1],
        "True Cluster (z_i)": [f"Component {k+1}" for k in z]
    })
    return df_out, X, z

df_gen, X_mat, z_true = generate_gmm_data(N, K, pi_true, means_true, covs_true, seed)

# --- EM ALGORITHM WITH COVARIANCE CONSTRAINTS ---
def gaussian_pdf_2d(x, mean, cov):
    d = 2
    det = max(np.linalg.det(cov), 1e-6)
    inv = np.linalg.inv(cov)
    norm_const = 1.0 / (2.0 * np.pi ** (d / 2.0) * np.sqrt(det))
    diff = x - mean
    exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
    return norm_const * np.exp(exponent)

def run_em_steps(X, K, max_iter=25, constraint="Full"):
    N = X.shape[0]
    np.random.seed(seed + 100) # Distinct seed for EM initialization
    
    pi_hat = np.ones(K) / K
    random_indices = np.random.choice(N, K, replace=False)
    means_hat = X[random_indices].copy()
    covs_hat = [np.eye(2) for _ in range(K)]
    
    history = []
    
    for iteration in range(max_iter):
        # E-step: Responsibilities
        gamma = np.zeros((N, K))
        for k in range(K):
            gamma[:, k] = pi_hat[k] * gaussian_pdf_2d(X, means_hat[k], covs_hat[k])
        
        sum_gamma = np.sum(gamma, axis=1, keepdims=True)
        sum_gamma_safe = np.where(sum_gamma == 0, 1e-10, sum_gamma)
        gamma = gamma / sum_gamma_safe
        
        log_likelihood = np.sum(np.log(sum_gamma_safe))
        
        history.append({
            "iteration": iteration,
            "pi": pi_hat.copy(),
            "means": [m.copy() for m in means_hat],
            "covs": [c.copy() for c in covs_hat],
            "gamma": gamma.copy(),
            "log_likelihood": log_likelihood
        })
        
        # M-step: Parameter updates
        N_k = np.sum(gamma, axis=0)
        for k in range(K):
            pi_hat[k] = N_k[k] / N
            means_hat[k] = np.sum(gamma[:, k, None] * X, axis=0) / max(N_k[k], 1e-6)
            
            diff = X - means_hat[k]
            raw_cov = (diff.T @ (gamma[:, k, None] * diff)) / max(N_k[k], 1e-6)
            
            # Apply Covariance Constraint
            if constraint == "Diagonal":
                # Zero out off-diagonal elements
                covs_hat[k] = np.diag(np.diag(raw_cov))
            elif constraint == "Spherical":
                # Average diagonal elements across dimensions
                avg_var = np.mean(np.diag(raw_cov))
                covs_hat[k] = np.eye(2) * avg_var
            else: # Full
                covs_hat[k] = raw_cov
                
            covs_hat[k] += np.eye(2) * 1e-4 # Regularization for numerical stability
            
    return history

max_em_steps = st.sidebar.slider("Max EM Iterations", min_value=1, max_value=40, value=20)
em_history = run_em_steps(X_mat, K, max_iter=max_em_steps, constraint=cov_type)

# Select Iteration Step
current_step = st.slider("Step Through EM Iterations", min_value=0, max_value=len(em_history)-1, value=len(em_history)-1)
step_data = em_history[current_step]

# --- SECTION 1: LIKELIHOOD EVOLUTION ---
st.subheader(f"1. Incomplete Log-Likelihood Evolution ({cov_type} Covariances)")

col_ll1, col_ll2 = st.columns([2, 1])

with col_ll1:
    ll_series = [h["log_likelihood"] for h in em_history]
    df_ll = pd.DataFrame({"Iteration": list(range(len(ll_series))), "Log-Likelihood": ll_series})
    
    fig_ll = px.line(df_ll, x="Iteration", y="Log-Likelihood", markers=True,
                     title=f"Log-Likelihood Convergence Curve ({cov_type} Structure)")
    
    fig_ll.add_trace(go.Scatter(
        x=[current_step], y=[step_data["log_likelihood"]],
        mode="markers", marker=dict(size=14, color="red", symbol="diamond"),
        name=f"Selected Step ({current_step})"
    ))
    st.plotly_chart(fig_ll, use_container_width=True)

with col_ll2:
    st.metric("Selected Covariance", cov_type)
    st.metric("Current Iteration", f"Step {current_step}")
    st.metric("Current Log-Likelihood", f"{step_data['log_likelihood']:.2f}")
    if current_step > 0:
        gain = step_data['log_likelihood'] - em_history[current_step-1]['log_likelihood']
        st.metric("1-Step Likelihood Improvement", f"+{gain:.4f}")
    else:
        st.metric("1-Step Likelihood Improvement", "Baseline")

st.markdown("---")

# --- SECTION 2: COMPARISON (ORIGINAL SAMPLING VS EM RESULT) ---
st.subheader("2. Visual Comparison: True Generation vs. EM Recovery")

col_left, col_right = st.columns(2)

# LEFT COLUMN: ORIGINAL SAMPLING
with col_left:
    st.markdown("#### A. Original Sampling (Ground Truth)")
    fig_true = px.scatter(
        df_gen, x="x1", y="x2", color="True Cluster (z_i)",
        title="True Latent Cluster Assignments z_i", opacity=0.75
    )
    for k in range(K):
        fig_true.add_trace(go.Scatter(
            x=[means_true[k][0]], y=[means_true[k][1]],
            mode="markers+text",
            marker=dict(size=14, color="black", symbol="x"),
            text=[f"True μ_{k+1}"], textposition="top center",
            name=f"True μ_{k+1}"
        ))
    st.plotly_chart(fig_true, use_container_width=True)

# RIGHT COLUMN: RESULT OF EM
with col_right:
    st.markdown(f"#### B. Result of EM (Step {current_step}, {cov_type} Covariance)")
    
    hard_assignments = np.argmax(step_data["gamma"], axis=1)
    df_em = pd.DataFrame({
        "x1": X_mat[:, 0],
        "x2": X_mat[:, 1],
        "Inferred Cluster": [f"Est. Component {k+1}" for k in hard_assignments]
    })
    
    fig_em = px.scatter(
        df_em, x="x1", y="x2", color="Inferred Cluster",
        title=f"Inferred Clusters (Max Responsibility γ_{{i,k}})", opacity=0.75
    )
    for k in range(K):
        fig_em.add_trace(go.Scatter(
            x=[step_data["means"][k][0]], y=[step_data["means"][k][1]],
            mode="markers+text",
            marker=dict(size=14, color="black", symbol="star"),
            text=[f"Est. μ_{k+1}"], textposition="top center",
            name=f"Est. μ_{k+1}"
        ))
    st.plotly_chart(fig_em, use_container_width=True)

# --- SECTION 3: PARAMETER COMPARISON TABLE ---
st.markdown("---")
st.subheader("3. Parameter Recovery Table (True vs. Estimated)")

param_rows = []
for k in range(K):
    est_cov = step_data['covs'][k]
    param_rows.append({
        "Component": f"K={k+1}",
        "True Weight (π)": f"{pi_true[k]:.3f}",
        "Est Weight (π)": f"{step_data['pi'][k]:.3f}",
        "True Mean (μ)": f"[{means_true[k][0]:.2f}, {means_true[k][1]:.2f}]",
        "Est Mean (μ)": f"[{step_data['means'][k][0]:.2f}, {step_data['means'][k][1]:.2f}]",
        "Est Covariance Σ": f"[{est_cov[0,0]:.2f}, {est_cov[0,1]:.2f}; {est_cov[1,0]:.2f}, {est_cov[1,1]:.2f}]"
    })

st.table(pd.DataFrame(param_rows))
