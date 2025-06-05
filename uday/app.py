# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import json
import requests
import graphviz
import os
import openai
from openai import AzureOpenAI
from databricks import sql

# Configure the page
st.set_page_config(
    page_title="FPML Data Analyzer",
    page_icon="📊",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .header {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
    }
    .bot-message {
        background-color: #F5F5F5;
        border-left: 5px solid #424242;
    }
</style>
""", unsafe_allow_html=True)

# Define the fancy separator as a variable
fancy_separator = """
    <style>
        .fancy-separator {
            border: none;
            height: 3px;
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            border-radius: 5px;
            box-shadow: 0 4px 10px rgba(0, 114, 255, 0.3);
            margin: 30px 0;
            position: relative;
            animation: glow 2s infinite ease-in-out;
        }

        .fancy-separator::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: rgba(255, 255, 255, 0.5);
            animation: sparkle 3s infinite;
        }

        @keyframes glow {
            0% { box-shadow: 0 4px 10px rgba(0, 114, 255, 0.3); }
            50% { box-shadow: 0 4px 20px rgba(0, 114, 255, 0.7); }
            100% { box-shadow: 0 4px 10px rgba(0, 114, 255, 0.3); }
        }

        @keyframes sparkle {
            0% { width: 50%; opacity: 0.5; }
            50% { width: 90%; opacity: 1; }
            100% { width: 50%; opacity: 0.5; }
        }
    </style>
    <hr class="fancy-separator">
"""

# API configuration
API_ENDPOINT = "http://localhost:8000/api"  # Change to your actual backend endpoint

#Harish - K-mean
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

# Helper functions
def get_fpml_data(fpml_file=None, xsd_file=None):
    """Get processed FPML data from backend LLM model"""
    try:
        files = {}
        if fpml_file:
            files['fpml'] = fpml_file
        if xsd_file:
            files['xsd'] = xsd_file

        # If no files provided, get sample data
        if not files:
            response = requests.get(f"{API_ENDPOINT}/sample-data")
        else:
            response = requests.post(f"{API_ENDPOINT}/process-fpml", files=files)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching data: {response.status_code}")
            return get_sample_fpml_data()  # Fallback to sample data
    except Exception as e:
        st.error(f"Error communicating with backend: {str(e)}")
        return get_sample_fpml_data()  # Fallback to sample data

def read_uploaded_file(file):
    """Reads the uploaded file and returns its content as a string."""
    try:
        # Read the file content as a string
        file_content = file.getvalue().decode("utf-8")  # assuming it's a text-based file (XML/FPML)
        # print(f"uploaded FPML content: {file_content}")
        return file_content
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def create_3d_scatter(data):
    if "trades" not in data or not data["trades"]:
        return px.scatter_3d(title="No data available")

    df = pd.DataFrame(data["trades"])
    if len(df) < 3:
        return px.scatter_3d(title="Insufficient data for 3D visualization")

    # Create 3D scatter plot
    fig = px.scatter_3d(
        df,
        x="amount1",
        y="rate" if "rate" in df.columns else "strikeRate",
        z="tradeDate" if "tradeDate" in df.columns else "expiryDate",
        color="type",
        size="amount1",
        hover_name="tradeId",
        opacity=0.7,
        title="3D Trade Visualization"
    )

    fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)))
    return fig

def create_sunburst(data):
    if "trades" not in data or not data["trades"]:
        return px.sunburst(title="No data available")

    df = pd.DataFrame(data["trades"])

    # Create sunburst chart
    fig = px.sunburst(
        df,
        path=["type", "pair", "currency1"],
        values="amount1",
        color="type",
        title="Hierarchical View of Trades"
    )

    return fig

def create_animated_bubble(data):
    if "trades" not in data or not data["trades"]:
        return px.scatter(title="No data available")

    df = pd.DataFrame(data["trades"])
    if "tradeDate" not in df.columns:
        return px.scatter(title="Trade date information missing")

    df["tradeDate"] = pd.to_datetime(df["tradeDate"])

    # Create animated bubble chart
    fig = px.scatter(
        df,
        x="amount1",
        y="rate" if "rate" in df.columns else "strikeRate",
        size="amount1",
        color="type",
        hover_name="tradeId",
        animation_frame=df["tradeDate"].dt.strftime("%Y-%m"),
        animation_group="tradeId",
        size_max=60,
        range_x=[df["amount1"].min()*0.9, df["amount1"].max()*1.1],
        range_y=[df["rate"].min()*0.9 if "rate" in df.columns else df["strikeRate"].min()*0.9,
                 df["rate"].max()*1.1 if "rate" in df.columns else df["strikeRate"].max()*1.1],
        title="Trade Evolution Over Time"
    )

    return fig

def create_radar_chart(data):
    if "trades" not in data or not data["trades"]:
        return go.Figure(title="No data available")

    df = pd.DataFrame(data["trades"])

    # Group by trade type and calculate metrics
    grouped = df.groupby("type").agg({
        "amount1": "mean",
        "rate" if "rate" in df.columns else "strikeRate": "mean",
        "tradeId": "count"
    }).reset_index()

    # Normalize the values for radar chart
    for col in grouped.columns[1:]:
        grouped[col] = grouped[col] / grouped[col].max()

    # Create radar chart
    fig = go.Figure()

    for i, trade_type in enumerate(grouped["type"]):
        fig.add_trace(go.Scatterpolar(
            r=grouped.iloc[i, 1:].values.tolist() + [grouped.iloc[i, 1]],  # Close the loop
            theta=grouped.columns[1:].tolist() + [grouped.columns[1]],  # Close the loop
            fill='toself',
            name=trade_type
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Trade Type Comparison"
    )

    return fig

def create_network_graph(data):
    if "trades" not in data or not data["trades"]:
        return go.Figure(title="No data available")

    df = pd.DataFrame(data["trades"])

    # Create edges between currencies
    edges = []
    for _, row in df.iterrows():
        if "currency1" in row and "currency2" in row:
            edges.append((row["currency1"], row["currency2"], row["amount1"]))

    # Create nodes and links for network graph
    nodes = list(set([e[0] for e in edges] + [e[1] for e in edges]))
    links = [{"source": e[0], "target": e[1], "value": e[2]} for e in edges]

    # Create network graph
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes
        ),
        link=dict(
            source=[nodes.index(e[0]) for e in edges],
            target=[nodes.index(e[1]) for e in edges],
            value=[e[2] for e in edges]
        )
    )])

    fig.update_layout(title_text="Currency Flow Network")
    return fig

def get_sample_fpml_data():
    """Generate sample FPML data for demonstration"""
    # Sample FX trades data
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="W")

    # Create sample trades
    trades = []
    currency_pairs = ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", "AUD/USD"]
    trade_types = ["Spot", "Forward", "Option", "Swap"]

    for i, date in enumerate(dates):
        trade_type = trade_types[i % len(trade_types)]
        pair = currency_pairs[i % len(currency_pairs)]
        currencies = pair.split('/')

        # Base trade data
        trade = {
            "tradeId": f"T{i+1000}",
            "client": f"Client_{i % 5 + 1}",  # Simulate 5 clients
            "tradeDate": date.strftime("%Y-%m-%d"),
            "type": trade_type,
            "pair": pair,
            "currency1": currencies[0],
            "currency2": currencies[1],
            "amount1": round(np.random.uniform(1000000, 10000000), 2),
        }

        # Add trade-specific fields
        if trade_type == "Spot" or trade_type == "Forward":
            trade["valueDate"] = (date + pd.Timedelta(days=2 if trade_type == "Spot" else 30)).strftime("%Y-%m-%d")
            trade["rate"] = round(np.random.uniform(0.8, 1.5), 4)
            trade["amount2"] = round(trade["amount1"] * trade["rate"], 2)

        elif trade_type == "Option":
            trade["expiryDate"] = (date + pd.Timedelta(days=90)).strftime("%Y-%m-%d")
            trade["strikeRate"] = round(np.random.uniform(0.8, 1.5), 4)
            trade["premium"] = round(trade["amount1"] * 0.02, 2)
            trade["optionType"] = "Call" if i % 2 == 0 else "Put"

        elif trade_type == "Swap":
            trade["nearLegValueDate"] = (date + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            trade["farLegValueDate"] = (date + pd.Timedelta(days=90)).strftime("%Y-%m-%d")
            trade["nearLegRate"] = round(np.random.uniform(0.8, 1.5), 4)
            trade["farLegRate"] = round(trade["nearLegRate"] + np.random.uniform(-0.05, 0.05), 4)

        trades.append(trade)

    return {
        "trades": trades,
        "summary": {
            "totalTrades": len(trades),
            "tradesByType": {t: len([tr for tr in trades if tr["type"] == t]) for t in trade_types},
            "tradesByPair": {p: len([tr for tr in trades if tr["pair"] == p]) for p in currency_pairs}
        }
    }

def create_client_trade_graph(data):
    """Generate a Graphviz diagram showing clients and their trade types"""
    if "trades" not in data or not data["trades"]:
        st.warning("No trade data available for Graphviz.")
        return

    df = pd.DataFrame(data["trades"])

    if "client" not in df.columns or "type" not in df.columns:
        st.warning("Missing 'client' or 'type' data.")
        return

    dot = graphviz.Digraph()
    dot.attr(rankdir="LR", size="8")

    # Add unique clients and trade types
    clients = df["client"].unique()
    trade_types = df["type"].unique()

    for client in clients:
        dot.node(client, shape="box", style="filled", color="lightblue")

    for ttype in trade_types:
        dot.node(ttype, shape="ellipse", style="filled", color="lightgray")

    # Add edges from clients to trade types
    edge_data = df.groupby(["client", "type"]).size().reset_index(name="count")

    for _, row in edge_data.iterrows():
        dot.edge(row["client"], row["type"], label=str(row["count"]))

    # Render
    st.graphviz_chart(dot)

def create_fpml_chart(data, chart_type):
    """Create a chart based on the selected type and FPML data"""
    if not data or "trades" not in data or not data["trades"]:
        return px.bar(title="No data available")

    trades = data["trades"]
    df = pd.DataFrame(trades)

    if chart_type == "Trade Types":
        # Count by trade type
        type_counts = df["type"].value_counts().reset_index()
        type_counts.columns = ["Trade Type", "Count"]
        fig = px.pie(
            type_counts,
            values="Count",
            names="Trade Type",
            title="Distribution of FX Trade Types",
            color_discrete_sequence=px.colors.qualitative.Bold
        )

    elif chart_type == "Currency Pairs":
        # Count by currency pair
        pair_counts = df["pair"].value_counts().reset_index()
        pair_counts.columns = ["Currency Pair", "Count"]
        fig = px.bar(
            pair_counts,
            x="Currency Pair",
            y="Count",
            title="Distribution of Currency Pairs",
            color="Currency Pair",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

    elif chart_type == "Trade Timeline":
        # Timeline of trades
        df["tradeDate"] = pd.to_datetime(df["tradeDate"])
        timeline_data = df.groupby(["tradeDate", "type"]).size().reset_index(name="Count")
        fig = px.line(
            timeline_data,
            x="tradeDate",
            y="Count",
            color="type",
            title="FX Trades Over Time",
            markers=True
        )

    elif chart_type == "Exchange Rates":
        # Exchange rates for spots and forwards
        rate_data = df[df["type"].isin(["Spot", "Forward"])].copy()
        if not rate_data.empty and "rate" in rate_data.columns:
            rate_data["tradeDate"] = pd.to_datetime(rate_data["tradeDate"])
            fig = px.scatter(
                rate_data,
                x="tradeDate",
                y="rate",
                color="pair",
                size="amount1",
                hover_data=["type", "tradeId"],
                title="Exchange Rates by Trade Date",
                labels={"rate": "Exchange Rate", "tradeDate": "Trade Date"}
            )
        else:
            fig = px.scatter(title="No exchange rate data available")

    elif chart_type == "Option Strikes":
        # Option strike rates
        option_data = df[df["type"] == "Option"].copy()
        if not option_data.empty and "strikeRate" in option_data.columns:
            option_data["expiryDate"] = pd.to_datetime(option_data["expiryDate"])
            fig = px.scatter(
                option_data,
                x="expiryDate",
                y="strikeRate",
                color="pair",
                size="amount1",
                hover_data=["optionType", "tradeId"],
                title="Option Strike Rates by Expiry Date",
                labels={"strikeRate": "Strike Rate", "expiryDate": "Expiry Date"}
            )
        else:
            fig = px.scatter(title="No option data available")

    elif chart_type == "3D Scatter":
        return create_3d_scatter(data)
    elif chart_type == "Sunburst":
        fig = create_sunburst(data)
    elif chart_type == "Animated Bubble":
        fig = create_animated_bubble(data)
    elif chart_type == "Radar Chart":
        fig = create_radar_chart(data)
    elif chart_type == "Network Graph":
        fig = create_network_graph(data)

    else:  # Default chart
        # Trade amounts by currency
        amount_data = df.groupby(["currency1", "type"])["amount1"].sum().reset_index()
        fig = px.bar(
            amount_data,
            x="currency1",
            y="amount1",
            color="type",
            title="Trade Amounts by Currency",
            labels={"currency1": "Currency", "amount1": "Amount"}
        )

    return fig

def get_chat_response(question, fpml_data=None):
    """Get response from backend LLM about FPML data"""
    try:
        payload = {
            "question": question,
            "context": json.dumps(fpml_data) if fpml_data else None
        }

        response = requests.post(f"{API_ENDPOINT}/chat", json=payload)

        if response.status_code == 200:
            return response.json().get("answer", "I couldn't process your question.")
        else:
            return f"Error getting response: {response.status_code}"
    except Exception as e:
        return f"Error communicating with backend: {str(e)}"

def visualize_trade_types(data):
    """Create visualization based on trade types"""
    if not data or "trades" not in data or not data["trades"]:
        st.warning("No trade data available")
        return

    # Extract trade types from the data
    trades = data["trades"]
    trade_types = {}

    for trade in trades:
        if "type" in trade:
            trade_type = trade["type"]
            trade_types[trade_type] = trade_types.get(trade_type, 0) + 1

    # Create DataFrame for visualization
    df = pd.DataFrame({
        "Trade Type": list(trade_types.keys()),
        "Count": list(trade_types.values())
    })

    # Create visualization
    fig = px.pie(
        df,
        values="Count",
        names="Trade Type",
        title="Distribution of FX Trade Types",
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Show the data table
    with st.expander("View Trade Type Data"):
        st.dataframe(df)

def visualize_client_behavior(data):
    """Visualize client-to-trade relationships using Graphviz"""
    if not data or "trades" not in data or not data["trades"]:
        st.warning("No client data available.")
        return

    df = pd.DataFrame(data["trades"])

    if "client" not in df.columns:
        st.warning("Client data not available in trade records.")
        return

    st.subheader("Client-to-Trade Graph (Graphviz)")

    # Call the function to generate the client-to-trade graph (assuming this is implemented elsewhere)
    create_client_trade_graph(data)

def main():
    st.markdown('<div class="header">FPML Data Analyzer</div>', unsafe_allow_html=True)

    # Create tabs with only Client Behavior tab
    tabs = st.tabs(["📈 Client Behavior"])

    # Tab 3: Client Behavior
    with tabs[0]:
        st.header("Client Behavior Analytics")

        # Use filtered data if available
        if 'fpml_data' in st.session_state:
            trades_df = pd.DataFrame(st.session_state.fpml_data['trades'])
            if 'tradeDate' in trades_df.columns:
                trades_df['tradeDate'] = pd.to_datetime(trades_df['tradeDate'])
                filtered_trades = trades_df[(trades_df['tradeDate'].dt.date >= start_date) &
                                            (trades_df['tradeDate'].dt.date <= end_date)]
                filtered_data = {
                    'trades': filtered_trades.to_dict('records'),
                    'summary': st.session_state.fpml_data.get('summary', {})
                }
            else:
                filtered_data = st.session_state.fpml_data
        else:
            filtered_data = get_sample_fpml_data()

        # Only visualize the client-to-trade graph, remove other charts
        visualize_client_behavior(filtered_data)

def explain_trade(trade_id):
    """
    Simulated LLM explanation of a trade using hardcoded logic.
    """
    if True:
        print(f"faking response for graphviz")
        # Read JSON data from file
        try:
            with open('graphviz_res1.json', 'r') as file:
                return json.load(file)  # Parse JSON data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
        except FileNotFoundError:
            print(f"File not found: graphviz_res1.json")
            return None

    response = get_fpml_data(None)
    trades = response.get('trades', [])
    # print(f"explain trade trades: {trades}")
    for trade in trades:
        if str(trade.get('tradeId')) == trade_id:
            trade_type = trade.get('tradeType', 'unknown')
            notional = trade.get('notional', 'N/A')
            currency = trade.get('currency', 'N/A')
            party = trade.get('party', 'N/A')
            direction = trade.get('buySell', 'N/A')
            date = trade.get('tradeDate', 'N/A')
            return (
                f"**Trade ID:** {trade_id}\n\n"
                f"This is a **{direction}** trade of type **{trade_type}** executed on **{date}**. "
                f"The notional amount is **{notional} {currency}**. "
                f"The counterparty involved is **{party}**.\n\n"
                f"This trade may be used for risk management, hedging, or speculative purposes depending on your portfolio."
            )
    return f"🚫 No trade found with ID `{trade_id}` in the current FPML data."


def explain_my_trade(result):
    """
    Simulated LLM explanation of a trade using hardcoded logic.
    """
    print(f"--------->{result}")
    return result
    # response = get_fpml_data(None)
    # trades = response.get('trades', [])
    # # print(f"explain trade trades: {trades}")
    # for trade in trades:
    #     if str(trade.get('tradeId')) == trade_id:
    #         trade_type = trade.get('tradeType', 'unknown')
    #         notional = trade.get('notional', 'N/A')
    #         currency = trade.get('currency', 'N/A')
    #         party = trade.get('party', 'N/A')
    #         direction = trade.get('buySell', 'N/A')
    #         date = trade.get('tradeDate', 'N/A')
    #         return (
    #             f"**Trade ID:** {trade_id}\n\n"
    #             f"This is a **{direction}** trade of type **{trade_type}** executed on **{date}**. "
    #             f"The notional amount is **{notional} {currency}**. "
    #             f"The counterparty involved is **{party}**.\n\n"
    #             f"This trade may be used for risk management, hedging, or speculative purposes depending on your portfolio."
    #         )
    # return f"🚫 No trade found with ID `{trade_id}` in the current FPML data."
    return f"🚫 No trade found with ID `{trade_id}` in the current FPML data."


def create_trade_graph(trade_id, fpml_data):
    # Extracting the trade data
    trade_date = fpml_data.get("trade_date", "Unknown")
    value_date = fpml_data.get("value_date", "Unknown")
    exchange_rate = fpml_data.get("exchange_rate", {})
    exchange_currency = exchange_rate.get("currency_pair", "Unknown")
    rate = exchange_rate.get("rate", "Unknown")
    parties = fpml_data.get("parties", [])

    # Check if the required data is available
    if not parties or not exchange_rate:
        print("Error: Missing required trade data.")
        return None

    # Create the Graphviz graph
    graph = graphviz.Digraph(format='png', engine='dot')
    # graph = graphviz.Digraph()
    graph.attr(rankdir="LR", size="8")

    # Add trade and value dates as metadata
    graph.attr(label=f"Trade ID: {trade_id}\nTrade Date: {trade_date}\nValue Date: {value_date}", labelloc="t", fontsize="20")

    # Add parties (Payers and Receivers) as nodes with unique identifiers (name + role)
    for party in parties:
        name = party.get("name", "Unknown")
        role = party.get("role", "Unknown")
        currency = party.get("currency", "Unknown")
        amount = party.get("amount", 0)

        # Create a unique node ID by appending the role to the name
        node_id = f"{name} ({role})"
        graph.node(node_id, f"{name}\nRole: {role}\nPays/Receives: {amount} {currency}")

    # Add edges between payers and receivers
    payer_party = [p for p in parties if p.get("role") == "Payer"]
    receiver_party = [p for p in parties if p.get("role") == "Receiver"]

    for payer, receiver in zip(payer_party, receiver_party):
        payer_name = payer.get("name", "Unknown")
        receiver_name = receiver.get("name", "Unknown")

        # Use the unique node identifiers (name + role)
        payer_node = f"{payer_name} (Payer)"
        receiver_node = f"{receiver_name} (Receiver)"

        graph.edge(payer_node, receiver_node, label=f"Rate: {rate}\nCurrency Pair: {exchange_currency}")

    # Additional formatting
    graph.attr(dpi='70')

    # Return the graph source for rendering
    return graph.source

def create_trade_graph1(trade_id, summary, trade_data):
    # Extracting the trade data
    trade_data = json.loads(trade_data)
    print(f"*****************{trade_data}")
    trade_date = trade_data.get("trade_date", "Unknown")
    value_date = trade_data.get("value_date", "Unknown")
    exchange_rate = trade_data.get("exchange_rate", {})
    exchange_currency = exchange_rate.get("currency_pair", "Unknown")
    rate = exchange_rate.get("rate", "Unknown")
    parties = trade_data.get("parties", [])

    # Check if the required data is available
    if not parties or not exchange_rate:
        print("Error: Missing required trade data.")
        return None

    # Create the Graphviz graph
    graph = graphviz.Digraph(format='png', engine='dot')
    # graph = graphviz.Digraph()
    graph.attr(rankdir="LR", size="8")

    # Add trade and value dates as metadata
    graph.attr(label=f"Trade ID: {trade_id}\nTrade Date: {trade_date}\nValue Date: {value_date}", labelloc="t", fontsize="20")

    # Add parties (Payers and Receivers) as nodes with unique identifiers (name + role)
    for party in parties:
        name = party.get("name", "Unknown")
        role = party.get("role", "Unknown")
        currency = party.get("currency", "Unknown")
        amount = party.get("amount", 0)

        # Create a unique node ID by appending the role to the name
        node_id = f"{name} ({role})"
        graph.node(node_id, f"{name}\nRole: {role}\nPays/Receives: {amount} {currency}")

    # Add edges between payers and receivers
    payer_party = [p for p in parties if p.get("role") == "Payer"]
    receiver_party = [p for p in parties if p.get("role") == "Receiver"]

    for payer, receiver in zip(payer_party, receiver_party):
        payer_name = payer.get("name", "Unknown")
        receiver_name = receiver.get("name", "Unknown")

        # Use the unique node identifiers (name + role)
        payer_node = f"{payer_name} (Payer)"
        receiver_node = f"{receiver_name} (Receiver)"

        graph.edge(payer_node, receiver_node, label=f"Rate: {rate}\nCurrency Pair: {exchange_currency}")

    # Additional formatting
    graph.attr(dpi='70')

    # Return the graph source for rendering
    return graph.source


def format_response(res):
    # print(f"{res}")
    print(f"------------1-------------")
    # print(f"{res}")
    # res = (res.replace("\\n", "").replace("\\","").replace("```","").replace("json","").replace("\"{","{").replace("}\"","}"))
    # res = (res.replace("\\n", "").replace("\\","").replace("```","").replace("json","").replace("\"{","{").replace("}\"","}").replace("\"dotdigraph G", "").replace("\"plaintextdigraph G", ""))
    res = (res.replace("\\n", "").replace("\\","").replace("```","").replace("json","").replace("\"{","{").replace("}\"","}").replace(";},", ";}\","))
    # res = (res.replace("\\n", ""))
    # print(f"----------2---------------")
    print(res)
    print(f"----------2.0.1---------------")
    # Step 1: Extract message content from the response
    # message_content = json.loads(res)['choices'][0]['message']['content']
    message_content = res['choices'][0]['message']['content']
    print(f"message_content: {message_content}")
    print(f"----------2.1---------------")

    return message_content


def prepare_for_fpml_upload_or_explain_trade(trade_id=None, fpml_file=None):
    # print(f"prepare_for_fpml_upload_or_explain_trade: trade_id: {trade_id}, fpml_file: {fpml_file}")
    input_to_llm = trade_id

    GRAPH_PROMPT = """You are an AI assistant specialized in analyzing market post trade data in FpML message.
        Your task is to extract relevant information from a given FpML document.
        Your output must be a structured JSON object.
         
        Instructions:
        1. Carefully read the entire FpML trade document provided at the end of this prompt.
        2. Extract the relevant information.
        3. Present your findings in JSON format as specified below.
         
        Important Notes:
        - Extract only relevant information.
        - Consider the context of the entire FpML message when determining the json.
        - Do not be verbose, only respond with the correct format and information.
        - Some questions may have no relevant excerpts. Just return "N/A" or ["N/A"] depending on the expected type in this case.
        - Do not include additional JSON keys beyond the ones listed here.
        - Do not include the same key multiple times in the JSON.
         
        Expected sample JSON is as below :
         
        {
          "trade_type": "fxForward",
          "trade_date": "2025-04-27",
          "value_date": "2025-05-04",
          "party1": {
            "name": "Barclays Capital",
            "pays_currency": "CNH",
            "pays_amount": 58267695.46,
            "receives_currency": "USD",
            "receives_amount": 8216902
          },
          "party2": {
            "name": "JPMorgan Chase",
            "pays_currency": "USD",
            "pays_amount": 8216902,
            "receives_currency": "CNH",
            "receives_amount": 58267695.46
          },
          "rate": 7.0912
        }
         
        FpML document to analyze: 
        """ + input_to_llm

    GRAPH_PROMPT_1 = """You are an AI assistant specialized in analyzing market post trade data in FpML message.
        Your task is to extract relevant information from a given FpML document.
        Your output must be a structured JSON object.
         
        Instructions:
        1. Carefully read the entire FpML trade document provided at the end of this prompt.
        2. Extract the relevant information.
        3. Return base64-encoded graphviz(DOT) source code. Ensure the final JSON is valid and do not add additional double quotes.
         
        Important Notes:
        - Extract only relevant information.
        - Consider the context of the entire FpML message when determining the json.
        - Do not be verbose, only respond with the correct format and information.
        - Do not include additional JSON keys beyond the ones listed here.
        - Do not include the same key multiple times in the JSON.
         
        Expected output is graphviz(DOT) source code sequence diagram to render graph.
        Sample code as below:
        digraph G {

          subgraph cluster_0 {
            style=filled;
            color=lightgrey;
            node [style=filled,color=white];
            a0 -> a1 -> a2 -> a3;
            label = "process #1";
          }
        
          subgraph cluster_1 {
            node [style=filled];
            b0 -> b1 -> b2 -> b3;
            label = "process #2";
            color=blue
          }
          start -> a0;
          start -> b0;
          a1 -> b3;
          b2 -> a3;
          a3 -> a0;
          a3 -> end;
          b3 -> end;
        
          start [shape=Mdiamond];
          end [shape=Msquare];
        }
        
        FpML document to analyze: 
        """ + input_to_llm

    # print(f"GRAPH_PROMPT: {GRAPH_PROMPT}")

    # Set your OpenAI API key and Azure endpoint URL as environment variables
    # Example: os.environ["OPENAI_API_KEY"] = "your-api-key"
    # Example: os.environ["OPENAI_API_BASE"] = "https://your-azure-endpoint.openai.azure.com/"

    results = []
    if fpml_file is not None:
        input_to_llm = fpml_file
        os.environ["AZURE_OPENAI_API_KEY"] ='6fe74af0d78e4fa382eceed78433c3a0'
        # gets the API Key from environment variable AZURE_OPENAI_API_KEY
        client = AzureOpenAI(
            # https://learn.microsoft.com/azure/ai-services/openai/reference#rest-api-versioning
            api_version="2025-01-01-preview",
            # https://learn.microsoft.com/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
            azure_endpoint="https://bh-uk-openai-dataai-lens.openai.azure.com",
        )

        response = client.chat.completions.create(
            model="gpt-4o",  # e.g. gpt-35-instant
            messages=[
                {
                    "role": "user",
                    "content": f"{GRAPH_PROMPT_1}",
                },
            ],
        )
        print(f"*******************LLM response**********************")
        print(response)
        print(f"---------------*--------")
        # Print the JSON structure
        print(response.to_json())
        print(f"-----------------------")
        # formatted_json = response.to_json()
        print(f"*******************LLM response**********************")
        # Extract the content from the message field
        # message_content = response['choices'][0]['message']['content']
        formatted_json = format_response(response.to_json())
        # formatted_json = format_response(response)
        print(formatted_json)
        print(f"*******************LLM response**********************")
    else:
        # calling Databricks for explain my trade
        print(f"Build Databrick client")
        connection = sql.connect(
            server_hostname = "adb-7336075840475686.6.azuredatabricks.net",
            http_path = "/sql/1.0/warehouses/b65f083626cc1906",
            access_token = "dapi76c6cd629f927bd814e8fdb6d41fd91f")
        cursor = connection.cursor()
        output = cursor.execute("SELECT summary, response_json from hackathon.dataai_lens.silver_layer where tradeid='FXTRADE-ATLAS-FX6_trade_100'")
        # Fetch all results
        # print(f"fetchall: {cursor.fetchall()}")
        results = cursor.fetchall()

        # Printing the fetched results
        print(f"results: {results}")

        response_json= []
        summary= []
        for row in results:
            summary = row[0].replace("```json", "").replace("```", "")  # Assuming 'summary' is the first column
            response_json = row[1].replace("```json", "").replace("```", "")  # Assuming 'response_json' is the second column
            print(f"Summary: {summary}")
            print(f"Response JSON: {response_json}")

        cursor.close()
        connection.close()
        return summary, response_json


    if fpml_file is None:
        # explain my trade
        return explain_my_trade(results)
    else:
        # fpml uplaod
        # return get_fpml_data(fpml_file)
        return explain_trade(trade_id)


def main():
    st.markdown('<div class="header">Market Trade FPML Data Analyzer</div>', unsafe_allow_html=True)

    # Create tabs
    # tabs = st.tabs(["📊 FPML Visualization", "💬 FPML Q&A", "📈 Client Behavior"])
    tabs = st.tabs(["📊 FPML Visualization", "📈 Client Behavior"])

    # Tab 1: FPML Visualization
    with tabs[0]:
        st.header("FPML Data Visualization")

        # Sidebar controls
        with st.sidebar:
            st.header("FPML Controls")

            # 📦 Explain My Trade Section
            with st.container():
                st.markdown("<h4 style='margin-top: 0;'>🔍 Explain My Trade</h4>", unsafe_allow_html=True)

                # Streamlit components inside the visual box
                trade_id_input = st.text_input("Enter Trade ID", key="trade_id_box")

                if st.button("Explain My Trade"):
                    if trade_id_input:
                        # explanation = explain_trade(trade_id_input)
                        summary, json = prepare_for_fpml_upload_or_explain_trade(trade_id_input)
                        st.session_state.trade_explanation = json
                        # print(f"explanation received: {st.session_state.trade_explanation }")
                        st.session_state.graphviz_data = create_trade_graph1(trade_id_input, summary, json)
                        st.success("Trade explanation and Graphviz generated!")
                    else:
                        st.warning("Please enter a valid Trade ID and upload FPML data first.")

            # Adding a thick line separator
            st.markdown(fancy_separator, unsafe_allow_html=True)

            # 📦 Upload FPML File Section
            st.subheader("Upload FPML File")
            fpml_file = st.file_uploader("Upload FPML File", type=["xml", "fpml"])

            # Process FPML file automatically when uploaded
            if fpml_file:
                # Read and display the content of the uploaded file as a string
                file_content = read_uploaded_file(fpml_file)
                st.session_state.process_fpml = True
                # st.session_state.fpml_data = get_fpml_data(fpml_file)  # Process and extract data immediately
                explanation = prepare_for_fpml_upload_or_explain_trade(None, file_content)  # Process and extract data immediately
                # explanation = prepare_for_fpml_upload_or_explain_trade(None, fpml_file)  # Process and extract data immediately
                st.session_state.fpml_data = create_trade_graph(12345, explanation)
                # print(f"st.session_state.fpml_data: {st.session_state.fpml_data}")
                st.success("FPML file successfully uploaded! Now select the chart type.")

            # Adding a thick line separator
            st.markdown(fancy_separator, unsafe_allow_html=True)

            # 📦 Filter Date Range Section
            st.subheader("Filter by Date Range")
            # Extract date range from FPML data if available
            if 'fpml_data' in st.session_state and 'trades' in st.session_state.fpml_data:
                trades_df = pd.DataFrame(st.session_state.fpml_data['trades'])
                if 'tradeDate' in trades_df.columns:
                    trades_df['tradeDate'] = pd.to_datetime(trades_df['tradeDate'])
                    min_date = trades_df['tradeDate'].min().date()
                    max_date = trades_df['tradeDate'].max().date()
                else:
                    min_date = datetime(2023, 1, 1).date()
                    max_date = datetime(2023, 12, 31).date()
            else:
                min_date = datetime(2023, 1, 1).date()
                max_date = datetime(2023, 12, 31).date()

            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            start_date, end_date = date_range if len(date_range) == 2 else (date_range[0], date_range[0])

            # 📦 Chart Selection
            chart_type = st.selectbox(
                "Select Chart Type",
                options=["Trade Types", "Currency Pairs", "Trade Timeline", "Exchange Rates",
                         "Option Strikes", "Trade Amounts", "3D Scatter", "Sunburst",
                         "Animated Bubble", "Radar Chart", "Network Graph"]
            )

        # Main Visualization Display (Based on File Upload or Trade ID Input)
        try:
            filtered_data = {}  # Default empty dictionary, to avoid UnboundLocalError

            # Check if FPML data is available and process it
            if 'fpml_data' in st.session_state and st.session_state.fpml_data:
                filtered_data = st.session_state.fpml_data
                # if 'process_fpml' in st.session_state and st.session_state.process_fpml:
                #     # If FPML data is uploaded and processed, show charts
                #     if chart_type in ["3D Scatter", "Sunburst", "Animated Bubble", "Radar Chart", "Network Graph"]:
                #         # Handle jazzy charts
                #         st.subheader(f"{chart_type} Visualization")
                #         fig = create_fpml_chart(filtered_data, chart_type)
                #         st.plotly_chart(fig, use_container_width=True)
                #     elif chart_type == "Trade Types":
                #         visualize_trade_types(filtered_data)
                #     else:
                #         fig = create_fpml_chart(filtered_data, chart_type)
                #         st.plotly_chart(fig, use_container_width=True)
                # print(f"Graphviz Visualization for Trade: {st.session_state.fpml_data}")
                st.subheader("Graphviz Visualization for fpml upload")
                st.graphviz_chart(st.session_state.fpml_data)

            # If a trade explanation was generated, show the Graphviz diagram
            if 'graphviz_data' in st.session_state:
                print(f"Graphviz Visualization for explain my trade: {st.session_state.graphviz_data}")
                st.subheader("Graphviz Visualization for Explain my Trade")
                st.graphviz_chart(st.session_state.graphviz_data)

        except Exception as e:
            st.error(f"Error processing FPML data: {str(e)}")
            st.write("Data structure:", filtered_data)
            st.error(f"graphviz error: {st.session_state.graphviz_data}")

        # Show data summary
        if "summary" in filtered_data:
            with st.expander("FPML Data Summary"):
                st.json(filtered_data["summary"])

        # Show raw data
        with st.expander("View Raw Trade Data"):
            if 'graphviz_data' in st.session_state:
                st.text(st.session_state.graphviz_data)
            else:
                st.write("No trade data available")

        # Add this to debug your sample data
        with st.expander("Debug Sample Data Structure"):
            if 'graphviz_data' in st.session_state:
                st.text(st.session_state.graphviz_data)
            else:
                st.write("No trade data available")

    # Tab 2: FPML Q&A
    # with tabs[1]:
    #     st.header("Chat with FPML Assistant")
    #
    #     # Initialize chat history
    #     if 'chat_history' not in st.session_state:
    #         st.session_state.chat_history = []
    #
    #     # Display chat history
    #     for message in st.session_state.chat_history:
    #         if message['role'] == 'user':
    #             st.markdown(f'<div class="chat-message user-message"><b>You:</b> {message["content"]}</div>', unsafe_allow_html=True)
    #         else:
    #             st.markdown(f'<div class="chat-message bot-message"><b>Assistant:</b> {message["content"]}</div>', unsafe_allow_html=True)
    #
    #     # User input
    #     user_input = st.text_input("Ask a question about the FPML data:", key="user_question")
    #
    #     # Process user input
    #     if user_input:
    #         # Add user message to chat history
    #         st.session_state.chat_history.append({"role": "user", "content": user_input})
    #
    #         # Get response from LLM
    #         response = get_chat_response(user_input, st.session_state.fpml_data)
    #
    #         # Add assistant response to chat history
    #         st.session_state.chat_history.append({"role": "assistant", "content": response})
    #
    #         # Rerun to update the UI
    #         st.experimental_rerun()
    #
    #     # Sample questions
    #     st.sidebar.header("Sample FPML Questions")
    #     sample_questions = [
    #         "What are the most common currency pairs in the data?",
    #         "What's the average exchange rate for EUR/USD trades?",
    #         "How many option trades are there and what are their characteristics?",
    #         "What's the total notional value of all trades?",
    #         "Explain the distribution of trade types in the data"
    #     ]
    #
    #     for question in sample_questions:
    #         if st.sidebar.button(question, key=f"q_{question[:20]}"):
    #             st.session_state.user_question = question
    #             st.experimental_rerun()

    # Tab 3: Client Behavior
    with tabs[1]:
        st.header("📊 FX Forward Trade Behavior Dashboard")
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
        # Parse the string into a dictionary
        import json
        summary_str = json.loads(summary)
        # Access the content and print
        content = summary_str['choices'][0]['message']['content']
        print(content)
        st.markdown(content)

        # Use filtered data if available
        # if 'fpml_data' in st.session_state:
        #     trades_df = pd.DataFrame(st.session_state.fpml_data['trades'])
        #     if 'tradeDate' in trades_df.columns:
        #         trades_df['tradeDate'] = pd.to_datetime(trades_df['tradeDate'])
        #         filtered_trades = trades_df[(trades_df['tradeDate'].dt.date >= start_date) &
        #                                     (trades_df['tradeDate'].dt.date <= end_date)]
        #         filtered_data = {
        #             'trades': filtered_trades.to_dict('records'),
        #             'summary': st.session_state.fpml_data.get('summary', {})
        #         }
        #     else:
        #         filtered_data = st.session_state.fpml_data
        # else:
        #     filtered_data = get_sample_fpml_data()

        # visualize_client_behavior(filtered_data)
        # st.subheader("Client-to-Trade Graph (Graphviz)")
        # create_client_trade_graph(filtered_data)

if __name__ == "__main__":
    main()