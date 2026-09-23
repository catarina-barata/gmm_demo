import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration for Cloud
st.set_page_config(
    page_title="GMM Generative & EM Interactive Demo",
    page_icon="🎲",
    layout="wide"
)

st.title("🎲 GMM Generative Process & Expectation-Maximization (EM)")
st.markdown("""
This demo connects **Data Generation** with **Unsupervised Fitting**:
1. **Generative Phase:** Sample data $\mathbf{x}_i \sim \sum_k \pi_k \mathcal{N}(\boldsymbol{\mu}_k, \mathbf{\Sigma}_k)$ with known latent parameters.
2. **Inference Phase (EM):** Hide the true labels and run EM step-by-step to recover parameters $(\hat{\boldsymbol{\pi}}, \hat{\boldsymbol{\mu}}, \hat{\mathbf{\Sigma}})$.
""")

# --- SIDEBAR: PARAMETERS ---
st.sidebar.header("1. Generative Ground Truth")

N = st.sidebar.slider("Number of Data Points (N)", min_value=50, max_value=1000, value=300, step=50)
K = st.sidebar.slider("Number of Components (K)", min_value=2, max_value=4, value=3, step=1)
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

# --- EM ALGORITHM IMPLEMENTATION ---
def gaussian_pdf_2d(x, mean, cov):
    """Calculates 2D multivariate Gaussian density."""
    d = 2
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    norm_const = 1.0 / (2.0 * np.pi ** (d / 2.0) * np.sqrt(max(det, 1e-6)))
    diff = x - mean
    exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
    return norm_const * np.exp(exponent)

def run_em_steps(X, K, max_iter=20):
    """Runs EM and saves history at each step for step-by-step visualization."""
    N = X.shape[0]
    np.random.seed(seed + 1)
    
    # Random Initialization
    pi_hat = np.ones(K) / K
    random_indices = np.random.choice(N, K, replace=False)
    means_hat = X[random_indices].copy()
    covs_hat = [np.eye(2) for _ in range(K)]
    
    history = []
    
    for iteration in range(max_iter):
        # E-STEP: Compute Responsibilities gamma_{i, k} = P(z_i = k | x_i)
        gamma = np.zeros((N, K))
        for k in range(K):
            gamma[:, k] = pi_hat[k] * gaussian_pdf_2d(X, means_hat[k], covs_hat[k])
        
        # Normalize over components (sum over k = 1)
        sum_gamma = np.sum(gamma, axis=1, keepdims=True)
        sum_gamma[sum_gamma == 0] = 1e-10  # Numerical stability
        gamma = gamma / sum_gamma
        
        # Calculate Log-Likelihood
        log_likelihood = np.sum(np.log(sum_gamma))
        
        # Save snapshot before M-Step update
        history.append({
            "iteration": iteration,
            "pi": pi_hat.copy(),
            "means": [m.copy() for m in means_hat],
            "covs": [c.copy() for c in covs_hat],
            "gamma": gamma.copy(),
            "log_likelihood": log_likelihood
        })
        
        # M-STEP: Update Parameters using Responsibilities
        N_k = np.sum(gamma, axis=0)
        
        for k in range(K):
            # Update Pi
            pi_hat[k] = N_k[k] / N
            
            # Update Mean
            means_hat[k] = np.sum(gamma[:, k, None] * X, axis=0) / N_k[k]
            
            # Update Covariance Matrix
            diff = X - means_hat[k]
            covs_hat[k] = (diff.T @ (gamma[:, k, None] * diff)) / N_k[k]
            # Ensure positive definiteness
            covs_hat[k] += np.eye(2) * 1e-4
            
    return history

# --- EM EXECUTION ---
st.markdown("---")
st.header("3. Expectation-Maximization (EM) Execution")

max_em_steps = st.slider("Max EM Iterations", min_value=1, max_value=30, value=15)
em_history = run_em_steps(X_mat, K, max_iter=max_em_steps)

# Select Iteration Step
current_step = st.slider("Step Through EM Iterations", min_value=0, max_value=len(em_history)-1, value=0)
step_data = em_history[current_step]

# --- EM METRICS DISPLAY ---
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Current Iteration", f"Step {step_data['iteration']}")
col_m2.metric("Log-Likelihood", f"{step_data['log_likelihood']:.2f}")

# Measure Convergence
if current_step > 0:
    ll_diff = step_data['log_likelihood'] - em_history[current_step-1]['log_likelihood']
    col_m3.metric("Log-Likelihood Improvement", f"+{ll_diff:.4f}")
else:
    col_m3.metric("Log-Likelihood Improvement", "N/A (Initial Step)")

# --- VISUALIZING EM RESPONSIBILITIES & ESTIMATED MEANS ---
st.subheader("Fitted GMM via Expectation-Maximization")

# Hard assignment based on max responsibility gamma_{i,k}
hard_cluster_assignments = np.argmax(step_data["gamma"], axis=1)

df_em = pd.DataFrame({
    "x1": X_mat[:, 0],
    "x2": X_mat[:, 1],
    "Inferred Cluster": [f"Est. Component {k+1}" for k in hard_cluster_assignments],
    "Max Responsibility": np.max(step_data["gamma"], axis=1)
})

tab1, tab2 = st.tabs(["Inferred Soft Assignments (EM View)", "Log-Likelihood Convergence Curve"])

with tab1:
    fig_em = px.scatter(
        df_em, x="x1", y="x2", 
        color="Inferred Cluster", 
        opacity=0.7,
        title=f"EM Step {current_step}: Soft-Clustering Points to Highest γ_k"
    )
    
    # Overlay Estimated Means
    for k in range(K):
        fig_em.add_trace(go.Scatter(
            x=[step_data["means"][k][0]], 
            y=[step_data["means"][k][1]],
            mode="markers+text",
            marker=dict(size=16, color="black", symbol="star"),
            text=[f"μ_hat_{k+1}"], 
            textposition="top center",
            name=f"Est. Mean μ_{k+1}"
        ))
        
    st.plotly_chart(fig_em, use_container_width=True)

with tab2:
    ll_series = [h["log_likelihood"] for h in em_history]
    df_ll = pd.DataFrame({"Iteration": list(range(len(ll_series))), "Log-Likelihood": ll_series})
    
    fig_ll = px.line(df_ll, x="Iteration", y="Log-Likelihood", markers=True,
                     title="Monotonic Convergence of Log-Likelihood Log P(X | θ)")
    
    # Highlight current step
    fig_ll.add_trace(go.Scatter(
        x=[current_step], y=[step_data["log_likelihood"]],
        mode="markers", marker=dict(size=12, color="red"), name="Current Step"
    ))
    
    st.plotly_chart(fig_ll, use_container_width=True)
