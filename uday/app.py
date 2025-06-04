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

# Configure the page
st.set_page_config(
    page_title="FPML Data Analyzer & Chat",
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

# API configuration
API_ENDPOINT = "http://localhost:8000/api"  # Change to your actual backend endpoint

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
    st.markdown('<div class="header">FPML Data Analyzer & Chat</div>', unsafe_allow_html=True)

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

def explain_trade(trade_id, fpml_data):
    """
    Simulated LLM explanation of a trade using hardcoded logic.
    """
    trades = fpml_data.get('trades', [])
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


def main():
    st.markdown('<div class="header">FPML Data Analyzer & Chat</div>', unsafe_allow_html=True)

    # Create tabs
    tabs = st.tabs(["📊 FPML Visualization", "💬 FPML Q&A", "📈 Client Behavior"])

    # Tab 1: FPML Visualization
    with tabs[0]:
        st.header("FPML Data Visualization")

        # Sidebar controls
        with st.sidebar:
            st.header("FPML Controls")

            # 📦 Upload FPML File Section
            st.subheader("Upload FPML File")
            fpml_file = st.file_uploader("Upload FPML File", type=["xml", "fpml"])

            # Process FPML file automatically when uploaded
            if fpml_file:
                st.session_state.process_fpml = True
                st.session_state.fpml_data = get_fpml_data(fpml_file)  # Process and extract data immediately
                st.success("FPML file successfully uploaded! Now select the chart type.")

            # Adding a thick line separator
            st.markdown("<hr style='border: 2px solid #333; margin: 20px 0;'>", unsafe_allow_html=True)

            # 📦 Explain My Trade Section
            with st.container():
                st.markdown("<h4 style='margin-top: 0;'>🔍 Explain My Trade</h4>", unsafe_allow_html=True)

                # Streamlit components inside the visual box
                trade_id_input = st.text_input("Enter Trade ID", key="trade_id_box")

                if st.button("Explain My Trade"):
                    if trade_id_input and 'fpml_data' in st.session_state:
                        explanation = explain_trade(trade_id_input, st.session_state.fpml_data)
                        st.session_state.trade_explanation = explanation
                        st.session_state.graphviz_data = create_trade_graph(trade_id_input, st.session_state.fpml_data)
                        st.success("Trade explanation and Graphviz generated!")

                    else:
                        st.warning("Please enter a valid Trade ID and upload FPML data first.")

            # Adding a thick line separator
            st.markdown("<hr style='border: 2px solid #333; margin: 20px 0;'>", unsafe_allow_html=True)

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
                if 'process_fpml' in st.session_state and st.session_state.process_fpml:
                    # If FPML data is uploaded and processed, show charts
                    if chart_type in ["3D Scatter", "Sunburst", "Animated Bubble", "Radar Chart", "Network Graph"]:
                        # Handle jazzy charts
                        st.subheader(f"{chart_type} Visualization")
                        fig = create_fpml_chart(filtered_data, chart_type)
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "Trade Types":
                        visualize_trade_types(filtered_data)
                    else:
                        fig = create_fpml_chart(filtered_data, chart_type)
                        st.plotly_chart(fig, use_container_width=True)

            # If a trade explanation was generated, show the Graphviz diagram
            if 'graphviz_data' in st.session_state:
                st.subheader("Graphviz Visualization for Trade")
                st.graphviz_chart(st.session_state.graphviz_data)

        except Exception as e:
            st.error(f"Error processing FPML data: {str(e)}")
            st.write("Data structure:", filtered_data)

        # Show data summary
        if "summary" in filtered_data:
            with st.expander("FPML Data Summary"):
                st.json(filtered_data["summary"])

        # Show raw data
        with st.expander("View Raw Trade Data"):
            if "trades" in filtered_data:
                st.dataframe(pd.DataFrame(filtered_data["trades"]))
            else:
                st.write("No trade data available")

        # Add this to debug your sample data
        with st.expander("Debug Sample Data Structure"):
            st.json(filtered_data)

    # Tab 2: FPML Q&A
    with tabs[1]:
        st.header("Chat with FPML Assistant")

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="chat-message user-message"><b>You:</b> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message"><b>Assistant:</b> {message["content"]}</div>', unsafe_allow_html=True)

        # User input
        user_input = st.text_input("Ask a question about the FPML data:", key="user_question")

        # Process user input
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Get response from LLM
            response = get_chat_response(user_input, st.session_state.fpml_data)

            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Rerun to update the UI
            st.experimental_rerun()

        # Sample questions
        st.sidebar.header("Sample FPML Questions")
        sample_questions = [
            "What are the most common currency pairs in the data?",
            "What's the average exchange rate for EUR/USD trades?",
            "How many option trades are there and what are their characteristics?",
            "What's the total notional value of all trades?",
            "Explain the distribution of trade types in the data"
        ]

        for question in sample_questions:
            if st.sidebar.button(question, key=f"q_{question[:20]}"):
                st.session_state.user_question = question
                st.experimental_rerun()

    # Tab 3: Client Behavior
    with tabs[2]:
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

        visualize_client_behavior(filtered_data)
        st.subheader("Client-to-Trade Graph (Graphviz)")
        create_client_trade_graph(filtered_data)

if __name__ == "__main__":
    main()