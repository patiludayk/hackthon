
import streamlit as st
import pandas as pd
import plotly.express as px

# Title
st.title("Advanced FX Forward Behavior Clustering Dashboard")
st.markdown("Explore party-wise trade behavior before and after market scenarios (e.g., tariffs).")

# Load data
df = pd.read_csv("enhanced_party_behavior_features.csv")

# Load clustering output
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

features = [
    'before_trade_count', 'after_trade_count',
    'before_total_notional', 'after_total_notional',
    'before_avg_notional', 'after_avg_notional',
    'before_avg_rate', 'after_avg_rate',
    'notional_change_pct', 'rate_sensitivity', 'trade_count_change'
]

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Sidebar: Cluster filter
cluster_filter = st.sidebar.multiselect("Select Clusters", sorted(df['cluster'].unique()), default=df['cluster'].unique())
filtered_df = df[df['cluster'].isin(cluster_filter)]

# PCA Cluster plot
st.subheader("Client Behavior Clusters (PCA Projection)")
fig = px.scatter(filtered_df, x="pca1", y="pca2", color="cluster", hover_data=["party1", "currency_pair"],
                 title="Behavior Clustering by PCA", size_max=60)
st.plotly_chart(fig)

# Metric summary
st.subheader("Cluster-Level Summary Metrics")
summary = filtered_df.groupby("cluster").agg({
    "before_total_notional": "mean",
    "after_total_notional": "mean",
    "notional_change_pct": "mean",
    "rate_sensitivity": "mean",
    "trade_count_change": "mean"
}).round(2)
st.dataframe(summary)

# Party details
st.subheader("Party-wise Behavior Details")
st.dataframe(filtered_df[[
    "party1", "currency_pair", "before_total_notional", "after_total_notional",
    "notional_change_pct", "rate_sensitivity", "trade_count_change", "cluster"
]])