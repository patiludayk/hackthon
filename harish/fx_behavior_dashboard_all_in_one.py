import os
import openai
from openai import AzureOpenAI
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def openaicall(promptContent):
    os.environ["AZURE_OPENAI_API_KEY"] ='6fe74af0d78e4fa382eceed78433c3a0'
    # gets the API Key from environment variable AZURE_OPENAI_API_KEY
    client = AzureOpenAI(
        api_version="2025-01-01-preview",
        azure_endpoint="https://bh-uk-openai-dataai-lens.openai.azure.com",
    )
     
    completion = client.chat.completions.create(
        model="gpt-4o",  # e.g. gpt-35-instant
        messages=[
            {
                "role": "user",
                "content": f"{promptContent}",
            },
        ],
        max_tokens=150
    )
    print(completion.to_json())
    return completion.to_json();


st.set_page_config(layout="wide")
st.title("📊 FX Forward Trade Behavior Dashboard")
st.markdown("Explore client behavior changes before and after market events like tariffs.")

# Load data
df = pd.read_csv("enhanced_party_behavior_features.csv")

# Recluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = [
    'before_trade_count', 'after_trade_count',
    'before_total_notional', 'after_total_notional',
    'before_avg_notional', 'after_avg_notional',
    'before_avg_rate', 'after_avg_rate',
    'notional_change_pct', 'rate_sensitivity', 'trade_count_change'
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Sidebar filters
selected_clusters = st.sidebar.multiselect("Filter by Cluster", df['cluster'].unique(), default=df['cluster'].unique())
selected_df = df[df['cluster'].isin(selected_clusters)]

# Bar Chart: Notional Comparison
st.subheader("📉 Notional Volume Before vs After Tariff (Grouped by Party)")
bar_fig = px.bar(
    selected_df,
    x='party1',
    y=['before_total_notional', 'after_total_notional'],
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Dark24,
    title="Client Notional Shift"
)
st.plotly_chart(bar_fig, use_container_width=True)

# Prepare prompt
data_str = df.to_markdown(index=False)
prompt = f"""Below is data from a bar chart:

{data_str}

Explain what this bar chart shows. Highlight trends and extremes.
"""
# Ask LLM (e.g., via OpenAI API)
summary = openaicall(prompt)
st.markdown(summary)

# Scatter Chart: Clustered View
st.subheader("🟠 Client Clustering View (2D Projection Not Shown)")
scatter_fig = px.scatter(
    selected_df,
    x='notional_change_pct',
    y='rate_sensitivity',
    color='cluster',
    hover_data=['party1', 'currency_pair'],
    title="Notional Change vs Rate Sensitivity by Cluster",
    size='after_total_notional'
)
st.plotly_chart(scatter_fig, use_container_width=True)

# Sankey Diagram
st.subheader("🔗 Client to Cluster Mapping (Sankey Diagram)")
st.components.v1.html(open("client_cluster_sankey.html", "r").read(), height=500)

# Summary Table
st.subheader("📋 Cluster Behavior Summary")
summary = selected_df.groupby('cluster').agg({
    'before_total_notional': 'mean',
    'after_total_notional': 'mean',
    'notional_change_pct': 'mean',
    'rate_sensitivity': 'mean',
    'trade_count_change': 'mean'
}).round(2).reset_index()
st.dataframe(summary)

# Detailed Table
st.subheader("🔎 Party-Level Behavior Breakdown")
st.dataframe(selected_df.sort_values(by="cluster"))
 