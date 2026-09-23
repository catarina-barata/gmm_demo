import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(page_title="GMM Generative Process Demo", layout="wide")

st.title("🎲 Interactive GMM Generative Process")
st.markdown("""
This demo illustrates how a **Gaussian Mixture Model (GMM)** generates a dataset of $N$ points from $K$ Gaussian components.
""")

# --- SIDEBAR: PARAMETER CONTROLS ---
st.sidebar.header("1. Model Parameters")

# Number of points and components
N = st.sidebar.slider("Number of Data Points (N)", min_value=50, max_value=2000, value=300, step=50)
K = st.sidebar.slider("Number of Components (K)", min_value=2, max_value=5, value=3, step=1)

st.sidebar.markdown("---")
st.sidebar.header("2. Component Parameters")

# Mixing Proportions (Dirichlet/Normalized)
raw_pis = []
for k in range(K):
    raw_pis.append(st.sidebar.slider(f"Weight (π_{k+1})", 0.1, 1.0, 1.0 / K, 0.05))

# Normalize weights so sum(pi) = 1
pi = np.array(raw_pis) / np.sum(raw_pis)

# Interactive Component Means and Covariances
means = []
covs = []

for k in range(K):
    st.sidebar.markdown(f"**Component {k+1} Specs**")
    col1, col2 = st.sidebar.columns(2)
    
    # Custom positions for default spread
    default_x = (k - (K - 1) / 2.0) * 4.0
    default_y = (k % 2) * 3.0 - 1.5
    
    mx = col1.number_input(f"μ_x ({k+1})", value=float(default_x), key=f"mx_{k}")
    my = col2.number_input(f"μ_y ({k+1})", value=float(default_y), key=f"my_{k}")
    means.append([mx, my])
    
    std_x = col1.slider(f"σ_x ({k+1})", 0.3, 3.0, 1.0, key=f"sx_{k}")
    std_y = col2.slider(f"σ_y ({k+1})", 0.3, 3.0, 1.0, key=f"sy_{k}")
    corr = col2.slider(f"Corr ({k+1})", -0.9, 0.9, 0.0, key=f"corr_{k}")
    
    # Construct covariance matrix
    cov_xy = corr * std_x * std_y
    cov = np.array([[std_x**2, cov_xy], [cov_xy, std_y**2]])
    covs.append(cov)

# --- GENERATIVE SAMPLING PROCESS ---
np.random.seed(42)

# Step 1: Sample latent cluster indicators z_i ~ Categorical(pi)
z = np.random.choice(K, size=N, p=pi)

# Step 2: Sample x_i ~ N(mu_{z_i}, Sigma_{z_i})
X = np.zeros((N, 2))
for i in range(N):
    k_selected = z[i]
    X[i] = np.random.multivariate_normal(means[k_selected], covs[k_selected])

df = pd.DataFrame({
    "x1": X[:, 0],
    "x2": X[:, 1],
    "Cluster (z_i)": [f"Component {k+1}" for k in z]
})

# --- MAIN PAGE DISPLAY ---

col_left, col_right = st.columns([1, 1])

# Left Column: Step-by-step math and mixing proportions
with col_left:
    st.subheader("Normalized Mixing Proportions (π)")
    pi_df = pd.DataFrame({"Component": [f"K={k+1}" for k in range(K)], "Probability (π_k)": pi})
    
    fig_pi = px.bar(pi_df, x="Component", y="Probability (π_k)", color="Component", 
                    text_auto=".2f", title="Categorical Prior P(z_i = k)")
    fig_pi.update_layout(showlegend=False, height=250)
    st.plotly_chart(fig_pi, use_container_width=True)

# Right Column: Sample Counts per Component
with col_right:
    st.subheader("Sampled Cluster Counts (z_i)")
    counts = pd.Series(z).value_counts().sort_index()
    count_df = pd.DataFrame({"Component": [f"K={k+1}" for k in range(K)], "Sample Count": [counts.get(k, 0) for k in range(K)]})
    
    fig_counts = px.bar(count_df, x="Component", y="Sample Count", color="Component", 
                        text_auto=True, title=f"Sampled Latent Assignments (N={N})")
    fig_counts.update_layout(showlegend=False, height=250)
    st.plotly_chart(fig_counts, use_container_width=True)

st.markdown("---")

# --- VISUALIZING THE GENERATED DATA ---
st.subheader("Generated Data Visualization")

tab1, tab2 = st.tabs(["Latent View (True Clusters Known)", "Observed View (Unlabeled Data)"])

with tab1:
    st.markdown("**Latent View:** Shows data points color-coded by their generating cluster $z_i$.")
    fig_latent = px.scatter(df, x="x1", y="x2", color="Cluster (z_i)", 
                            title="Sampled Points x_i ~ N(μ_{z_i}, Σ_{z_i})",
                            opacity=0.7, width=800, height=500)
    
    # Overlay Gaussian Center points
    for k in range(K):
        fig_latent.add_trace(go.Scatter(
            x=[means[k][0]], y=[means[k][1]],
            mode="markers+text",
            marker=dict(size=14, color="black", symbol="x"),
            text=[f"μ_{k+1}"], textposition="top center",
            name=f"Mean μ_{k+1}"
        ))
    st.plotly_chart(fig_latent, use_container_width=True)

with tab2:
    st.markdown("**Observed View:** What an unsupervised learning algorithm (like EM) sees before clustering.")
    fig_obs = px.scatter(df, x="x1", y="x2", 
                         title="Observed Data Matrix X (Unlabeled)",
                         opacity=0.6, width=800, height=500)
    fig_obs.update_traces(marker=dict(color="gray"))
    st.plotly_chart(fig_obs, use_container_width=True)
