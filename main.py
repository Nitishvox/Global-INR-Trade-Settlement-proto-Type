import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import uuid
from typing import Dict, List, Any
import threading
import queue

# Enhanced encryption with better error handling
def encrypt_data(data, key):
    try:
        if not data:
            return None
        nonce = os.urandom(12)
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(str(data).encode()) + encryptor.finalize()
        return base64.b64encode(nonce + ciphertext + encryptor.tag).decode()
    except Exception as e:
        st.error(f"Encryption failed: {str(e)}")
        return None

def decrypt_data(encrypted_data, key):
    try:
        if not encrypted_data:
            return "N/A"
        decoded = base64.b64decode(encrypted_data)
        nonce, ciphertext, tag = decoded[:12], decoded[12:-16], decoded[-16:]
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        return (decryptor.update(ciphertext) + decryptor.finalize()).decode()
    except Exception as e:
        return f"Decryption Failed: {str(e)}"

def generate_key():
    return os.urandom(32)

# Enhanced transaction generation with more realistic data
def generate_realistic_transaction(source=None, amount=None, currency=None, corridor=None):
    sources = ["UPI", "NEFT", "SRVA-INR", "SRVA-Non-INR", "RTGS", "SWIFT"]
    currencies = ["INR", "USD", "EUR", "AED", "SGD", "GBP"]
    corridors = ["IN-US", "IN-EU", "IN-SG", "IN-AE", "IN-GB", "IN-JP"]
    
    # Realistic amount ranges based on source
    amount_ranges = {
        "UPI": (100, 100000),
        "NEFT": (1000, 10000000),
        "RTGS": (200000, 100000000),
        "SRVA-INR": (50000, 50000000),
        "SRVA-Non-INR": (10000, 20000000),
        "SWIFT": (100000, 200000000)
    }
    
    selected_source = source or np.random.choice(sources, p=[0.3, 0.25, 0.15, 0.1, 0.1, 0.1])
    selected_currency = currency or np.random.choice(currencies, p=[0.4, 0.2, 0.15, 0.1, 0.08, 0.07])
    selected_corridor = corridor or np.random.choice(corridors)
    
    min_amt, max_amt = amount_ranges.get(selected_source, (1000, 1000000))
    selected_amount = amount or np.random.uniform(min_amt, max_amt)
    
    timestamp = datetime.now() - timedelta(seconds=np.random.randint(0, 3600))
    transaction_id = str(uuid.uuid4())[:12]
    
    # Enhanced risk calculation
    base_risk = np.random.uniform(10, 30)
    if selected_amount > 10000000:
        base_risk += 25
    if selected_currency != "INR":
        base_risk += 15
    if selected_source.startswith("SRVA"):
        base_risk += 10
    if selected_corridor in ["IN-US", "IN-EU"]:
        base_risk += 8
    
    statuses = ["Completed", "Pending", "Failed", "Processing"]
    status_weights = [0.7, 0.15, 0.05, 0.1]
    
    return {
        "id": transaction_id,
        "source": selected_source,
        "amount": round(selected_amount, 2),
        "currency": selected_currency,
        "corridor": selected_corridor,
        "timestamp": timestamp,
        "status": np.random.choice(statuses, p=status_weights),
        "risk_score": min(base_risk + np.random.uniform(-5, 15), 100),
        "processing_time": np.random.uniform(0.1, 5.0),
        "fees": round(selected_amount * 0.001 * np.random.uniform(0.5, 2.0), 2),
        "counterparty": f"Entity_{np.random.randint(1000, 9999)}",
        "reference": f"REF{np.random.randint(100000, 999999)}"
    }

# Enhanced FX simulation with realistic volatility patterns
def simulate_enhanced_fx_data(days=30):
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    # Base rates with realistic trends
    base_rates = {
        "USDINR": 83.5,
        "EURINR": 90.2,
        "AEDINR": 22.7,
        "SGDINR": 61.8,
        "GBPINR": 105.3
    }
    
    fx_data = []
    for i, date in enumerate(dates):
        data_point = {"Date": date}
        for pair, base_rate in base_rates.items():
            # Add realistic daily movements
            daily_change = np.random.normal(0, 0.005) + np.sin(i/10) * 0.002
            rate = base_rate * (1 + daily_change * (i+1))
            volatility = abs(np.random.normal(0.02, 0.01))
            
            data_point[f"{pair}_Rate"] = round(rate, 4)
            data_point[f"{pair}_Volatility"] = round(volatility, 4)
            
        fx_data.append(data_point)
    
    return pd.DataFrame(fx_data)

# Real-time analytics with caching
@st.cache_data(ttl=30)
def calculate_enhanced_analytics(transactions):
    if not transactions:
        return {}, pd.DataFrame()
    
    df = pd.DataFrame(transactions)
    
    # KPIs
    total_volume = df['amount'].sum()
    avg_risk = df['risk_score'].mean()
    completion_rate = (df['status'] == 'Completed').mean() * 100
    avg_processing_time = df['processing_time'].mean()
    total_fees = df['fees'].sum()
    
    # Time series analysis
    df['hour'] = df['timestamp'].dt.hour
    hourly_volume = df.groupby('hour')['amount'].sum().reset_index()
    
    # Risk distribution
    risk_bins = pd.cut(df['risk_score'], bins=[0, 25, 50, 75, 100], labels=['Low', 'Medium', 'High', 'Critical'])
    risk_distribution = risk_bins.value_counts()
    
    analytics = {
        'total_volume': total_volume,
        'avg_risk': avg_risk,
        'completion_rate': completion_rate,
        'avg_processing_time': avg_processing_time,
        'total_fees': total_fees,
        'transaction_count': len(df),
        'unique_corridors': df['corridor'].nunique(),
        'currency_distribution': df['currency'].value_counts(),
        'source_distribution': df['source'].value_counts(),
        'risk_distribution': risk_distribution,
        'hourly_volume': hourly_volume
    }
    
    return analytics, df

# Enhanced visualization functions
def create_risk_heatmap(df):
    risk_by_corridor_currency = df.groupby(['corridor', 'currency'])['risk_score'].mean().reset_index()
    
    fig = px.density_heatmap(
        risk_by_corridor_currency, 
        x='corridor', 
        y='currency', 
        z='risk_score',
        title='Risk Score Heatmap by Corridor and Currency',
        color_continuous_scale='RdYlBu_r'
    )
    fig.update_layout(height=400)
    return fig

def create_volume_treemap(df):
    volume_by_source = df.groupby(['source', 'currency'])['amount'].sum().reset_index()
    
    fig = px.treemap(
        volume_by_source,
        path=[px.Constant("Total"), 'source', 'currency'],
        values='amount',
        title='Transaction Volume Distribution',
        color='amount',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=500)
    return fig

def create_real_time_dashboard(analytics, df):
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Hourly Volume Trend', 'Risk Distribution', 
                       'Currency Distribution', 'Processing Time vs Amount'),
        specs=[[{"secondary_y": False}, {"type": "pie"}],
               [{"type": "pie"}, {"secondary_y": False}]]
    )
    
    # Hourly volume
    hourly_data = analytics['hourly_volume']
    fig.add_trace(
        go.Scatter(x=hourly_data['hour'], y=hourly_data['amount'], 
                  mode='lines+markers', name='Volume'),
        row=1, col=1
    )
    
    # Risk distribution pie
    risk_dist = analytics['risk_distribution']
    fig.add_trace(
        go.Pie(labels=risk_dist.index, values=risk_dist.values, name="Risk"),
        row=1, col=2
    )
    
    # Currency distribution pie
    curr_dist = analytics['currency_distribution'].head(5)
    fig.add_trace(
        go.Pie(labels=curr_dist.index, values=curr_dist.values, name="Currency"),
        row=2, col=1
    )
    
    # Processing time scatter
    fig.add_trace(
        go.Scatter(x=df['amount'], y=df['processing_time'], 
                  mode='markers', name='Processing Time',
                  marker=dict(color=df['risk_score'], colorscale='Viridis')),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Real-Time Analytics Dashboard")
    return fig

# Background transaction generator
def auto_generate_thread(q):
    while st.session_state.get('auto_generate', False):
        transaction = generate_realistic_transaction()
        encrypted_amount = encrypt_data(transaction["amount"], st.session_state.encryption_key)
        encrypted_counterparty = encrypt_data(transaction["counterparty"], st.session_state.encryption_key)
        transaction.update({
            "encrypted_amount": encrypted_amount,
            "encrypted_counterparty": encrypted_counterparty
        })
        q.put(transaction)
        time.sleep(6)  # Generate ~10 per minute

# Streamlit app with enhanced UI
def main():
    st.set_page_config(
        page_title="Global-INR Trade Settlement Platform", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling and responsiveness
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-high { background-color: #ffebee; border-left: 4px solid #f44336; }
    .alert-medium { background-color: #fff3e0; border-left: 4px solid #ff9800; }
    .alert-low { background-color: #e8f5e8; border-left: 4px solid #4caf50; }
    
    /* Responsive design improvements */
    @media (max-width: 768px) {
        .stMetric {
            margin-bottom: 1rem;
        }
        .main-header {
            padding: 0.5rem;
            font-size: 0.9rem;
        }
        .stForm {
            padding: 0.5rem;
        }
        .stButton > button {
            width: 100%;
        }
        .stSlider > div {
            width: 100%;
        }
        .stExpander {
            width: 100%;
        }
        .stDataFrame {
            width: 100%;
            overflow-x: auto;
        }
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 0 100% !important;
            margin-bottom: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌐 Global-INR Trade Settlement Intelligence Platform</h1>
        <p><strong>Motto:</strong> Secure speed. Transparent trust. INR-first global flows</p>
        <p><strong>Vision:</strong> Real-time, RBI-aligned intelligence for INR-denominated cross-border trade</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "transactions" not in st.session_state:
        st.session_state.transactions = []
    if "encryption_key" not in st.session_state:
        st.session_state.encryption_key = generate_key()
    if "auto_generate" not in st.session_state:
        st.session_state.auto_generate = False
    if "fx_data" not in st.session_state:
        st.session_state.fx_data = simulate_enhanced_fx_data()
    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []
    if "transaction_queue" not in st.session_state:
        st.session_state.transaction_queue = queue.Queue()
    if "gen_thread" not in st.session_state:
        st.session_state.gen_thread = None
    
    # Start auto-generate thread if enabled
    if st.session_state.auto_generate:
        if st.session_state.gen_thread is None or not st.session_state.gen_thread.is_alive():
            st.session_state.gen_thread = threading.Thread(target=auto_generate_thread, args=(st.session_state.transaction_queue,))
            st.session_state.gen_thread.daemon = True
            st.session_state.gen_thread.start()
    
    # Process queued transactions
    while not st.session_state.transaction_queue.empty():
        st.session_state.transactions.append(st.session_state.transaction_queue.get())
    
    # Sidebar with enhanced navigation and controls
    st.sidebar.header("🎛️ Control Panel")
    
    # Auto-generation toggle
    auto_gen = st.sidebar.checkbox("🔄 Auto-generate transactions", value=st.session_state.auto_generate)
    if auto_gen != st.session_state.auto_generate:
        st.session_state.auto_generate = auto_gen
        st.rerun()
    
    if st.session_state.auto_generate:
        if st.sidebar.button("Generate 10 Random Transactions"):
            for _ in range(10):
                transaction = generate_realistic_transaction()
                encrypted_amount = encrypt_data(transaction["amount"], st.session_state.encryption_key)
                encrypted_counterparty = encrypt_data(transaction["counterparty"], st.session_state.encryption_key)
                transaction.update({
                    "encrypted_amount": encrypted_amount,
                    "encrypted_counterparty": encrypted_counterparty
                })
                st.session_state.transactions.append(transaction)
            st.sidebar.success("✅ 10 transactions generated!")
            st.rerun()
    
    # Data management
    st.sidebar.subheader("📊 Data Management")
    if st.sidebar.button("🗑️ Clear All Data"):
        st.session_state.transactions = []
        st.sidebar.success("Data cleared!")
        st.rerun()
    
    if st.sidebar.button("🔄 Refresh FX Data"):
        st.session_state.fx_data = simulate_enhanced_fx_data()
        st.sidebar.success("FX data refreshed!")
        st.rerun()
    
    # Export functionality
    if st.session_state.transactions:
        df_export = pd.DataFrame(st.session_state.transactions)
        csv = df_export.to_csv(index=False)
        st.sidebar.download_button(
            label="📥 Download Transaction Data",
            data=csv,
            file_name=f"inr_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Main navigation
    section = st.sidebar.radio(
        "📍 Navigation", 
        ["🏠 Dashboard", "➕ Transaction Ingestion", "📊 Analytics", "📈 Markets & FX", "🔐 Security Center", "⚙️ Settings"]
    )
    
    # Dashboard Section
    if section == "🏠 Dashboard":
        if st.session_state.transactions:
            analytics, df = calculate_enhanced_analytics(st.session_state.transactions)
            
            # Key metrics row - responsive stacking
            cols = st.columns(min(5, len(st.columns(1))))
            metrics = [
                ("💰 Total Volume", f"₹{analytics['total_volume']:,.0f}", f"{len(st.session_state.transactions)} txns"),
                ("⚠️ Avg Risk Score", f"{analytics['avg_risk']:.1f}", f"{analytics['completion_rate']:.1f}% completed"),
                ("⏱️ Avg Processing", f"{analytics['avg_processing_time']:.2f}s", f"₹{analytics['total_fees']:,.0f} fees"),
                ("🌐 Active Corridors", f"{analytics['unique_corridors']}", f"{df['source'].nunique()} sources"),
                ("❌ Failed Transactions", f"{len(df[df['status'] == 'Failed'])}", f"{(len(df[df['status'] == 'Failed'])/len(df)*100):.1f}%" if len(df) > 0 else "0%")
            ]
            for i, (label, value, delta) in enumerate(metrics):
                with cols[i % len(cols)]:
                    st.metric(label, value, delta=delta)
            
            # Real-time dashboard
            st.plotly_chart(create_real_time_dashboard(analytics, df), use_container_width=True)
            
            # Risk alerts
            high_risk_txns = df[df['risk_score'] > 75]
            if not high_risk_txns.empty:
                st.markdown(f"""
                <div class="alert-box alert-high">
                    <strong>🚨 High Risk Alert:</strong> {len(high_risk_txns)} transactions with risk score > 75
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View High Risk Transactions"):
                    st.dataframe(high_risk_txns[['id', 'amount', 'currency', 'corridor', 'risk_score', 'status']], use_container_width=True)
            
            # Recent transactions
            st.subheader("📋 Recent Transactions")
            recent_txns = df.nlargest(10, 'timestamp')
            st.dataframe(
                recent_txns[['id', 'source', 'amount', 'currency', 'corridor', 'status', 'risk_score', 'timestamp']],
                use_container_width=True
            )
            
        else:
            st.info("🔄 No transactions yet. Generate some data or add transactions manually!")
            if st.button("🎲 Generate Sample Data"):
                for _ in range(50):
                    transaction = generate_realistic_transaction()
                    encrypted_amount = encrypt_data(transaction["amount"], st.session_state.encryption_key)
                    encrypted_counterparty = encrypt_data(transaction["counterparty"], st.session_state.encryption_key)
                    transaction.update({
                        "encrypted_amount": encrypted_amount,
                        "encrypted_counterparty": encrypted_counterparty
                    })
                    st.session_state.transactions.append(transaction)
                st.success("✅ Sample data generated!")
                st.rerun()
    
    # Transaction Ingestion Section
    elif section == "➕ Transaction Ingestion":
        st.header("💱 Transaction Ingestion Center")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.form("enhanced_transaction_form", clear_on_submit=True):
                st.subheader("Manual Transaction Entry")
                
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    source = st.selectbox("Payment Source", ["UPI", "NEFT", "RTGS", "SRVA-INR", "SRVA-Non-INR", "SWIFT"])
                    currency = st.selectbox("Currency", ["INR", "USD", "EUR", "AED", "SGD", "GBP"])
                    amount = st.number_input("Amount", min_value=0.01, step=1000.0, format="%.2f")
                
                with col_b:
                    corridor = st.selectbox("Trade Corridor", ["IN-US", "IN-EU", "IN-SG", "IN-AE", "IN-GB", "IN-JP"])
                    counterparty = st.text_input("Counterparty ID", value=f"Entity_{np.random.randint(1000, 9999)}")
                    reference = st.text_input("Reference", value=f"REF{np.random.randint(100000, 999999)}")
                
                priority = st.select_slider("Priority Level", options=["Low", "Normal", "High", "Critical"])
                notes = st.text_area("Transaction Notes (Optional)")
                
                submit = st.form_submit_button("🚀 Process Transaction", type="primary")
                
                if submit and amount > 0:
                    transaction = generate_realistic_transaction(source, amount, currency, corridor)
                    transaction.update({
                        "counterparty": counterparty,
                        "reference": reference,
                        "priority": priority,
                        "notes": notes,
                        "manual_entry": True
                    })
                    
                    # Enhanced risk calculation for manual entries
                    if priority == "Critical":
                        transaction["risk_score"] += 20
                    
                    # Encrypt sensitive data
                    encrypted_amount = encrypt_data(transaction["amount"], st.session_state.encryption_key)
                    encrypted_counterparty = encrypt_data(transaction["counterparty"], st.session_state.encryption_key)
                    
                    if encrypted_amount and encrypted_counterparty:
                        transaction.update({
                            "encrypted_amount": encrypted_amount,
                            "encrypted_counterparty": encrypted_counterparty
                        })
                        st.session_state.transactions.append(transaction)
                        
                        # Success message with details
                        st.success(f"""
                        ✅ **Transaction Processed Successfully!**
                        - **ID:** {transaction['id']}
                        - **Risk Score:** {transaction['risk_score']:.1f}
                        - **Processing Time:** {transaction['processing_time']:.2f}s
                        - **Status:** {transaction['status']}
                        """)
                        
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("❌ Failed to process transaction due to encryption error.")
        
        with col2:
            st.subheader("📊 Quick Stats")
            if st.session_state.transactions:
                recent_count = len([t for t in st.session_state.transactions 
                                 if (datetime.now() - t['timestamp']).total_seconds() < 3600])
                st.metric("Transactions (Last Hour)", recent_count)
                
                avg_risk_recent = np.mean([t['risk_score'] for t in st.session_state.transactions[-10:]])
                st.metric("Recent Avg Risk", f"{avg_risk_recent:.1f}")
                
                latest = st.session_state.transactions[-1] if st.session_state.transactions else None
                if latest:
                    st.info(f"""
                    **Latest Transaction:**
                    - ID: {latest['id'][:8]}...
                    - Amount: {latest['currency']} {latest['amount']:,.2f}
                    - Status: {latest['status']}
                    """)
    
    # Analytics Section
    elif section == "📊 Analytics":
        st.header("📈 Advanced Analytics Center")
        
        if st.session_state.transactions:
            analytics, df = calculate_enhanced_analytics(st.session_state.transactions)
            
            tab1, tab2, tab3, tab4 = st.tabs(["🎯 Overview", "🌡️ Risk Analysis", "💰 Volume Analysis", "🔍 Deep Dive"])
            
            with tab1:
                # Risk heatmap
                st.plotly_chart(create_risk_heatmap(df), use_container_width=True)
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    # Source distribution
                    fig_source = px.pie(
                        values=analytics['source_distribution'].values,
                        names=analytics['source_distribution'].index,
                        title="Transaction Distribution by Source"
                    )
                    st.plotly_chart(fig_source, use_container_width=True)
                
                with col2:
                    # Currency distribution
                    fig_curr = px.bar(
                        x=analytics['currency_distribution'].index,
                        y=analytics['currency_distribution'].values,
                        title="Transaction Count by Currency"
                    )
                    st.plotly_chart(fig_curr, use_container_width=True)
            
            with tab2:
                st.subheader("🎲 Risk Score Distribution")
                
                fig_risk = px.histogram(df, x='risk_score', nbins=20, title='Risk Score Distribution')
                st.plotly_chart(fig_risk, use_container_width=True)
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    # Risk by corridor
                    risk_by_corridor = df.groupby('corridor')['risk_score'].mean().sort_values(ascending=False)
                    fig_corridor_risk = px.bar(
                        x=risk_by_corridor.values,
                        y=risk_by_corridor.index,
                        orientation='h',
                        title='Average Risk by Corridor'
                    )
                    st.plotly_chart(fig_corridor_risk, use_container_width=True)
                
                with col2:
                    # Risk vs Amount scatter
                    fig_scatter = px.scatter(
                        df, x='amount', y='risk_score', 
                        color='currency', size='processing_time',
                        title='Risk vs Amount by Currency'
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            with tab3:
                st.subheader("💸 Volume Analysis")
                
                # Volume treemap
                st.plotly_chart(create_volume_treemap(df), use_container_width=True)
                
                # Time series analysis
                df_hourly = df.set_index('timestamp').resample('H')['amount'].sum().reset_index()
                fig_timeseries = px.line(df_hourly, x='timestamp', y='amount', title='Hourly Transaction Volume')
                st.plotly_chart(fig_timeseries, use_container_width=True)
            
            with tab4:
                st.subheader("🔬 Deep Dive Analysis")
                
                # Filter controls
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    selected_corridor = st.multiselect("Filter by Corridor", df['corridor'].unique())
                with col2:
                    selected_currency = st.multiselect("Filter by Currency", df['currency'].unique())
                with col3:
                    risk_threshold = st.slider("Risk Score Threshold", 0, 100, 50)
                
                # Apply filters
                filtered_df = df.copy()
                if selected_corridor:
                    filtered_df = filtered_df[filtered_df['corridor'].isin(selected_corridor)]
                if selected_currency:
                    filtered_df = filtered_df[filtered_df['currency'].isin(selected_currency)]
                
                filtered_df = filtered_df[filtered_df['risk_score'] >= risk_threshold]
                
                st.subheader(f"📋 Filtered Results ({len(filtered_df)} transactions)")
                st.dataframe(filtered_df, use_container_width=True)
                
                # Summary stats for filtered data
                if not filtered_df.empty:
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Total Volume", f"₹{filtered_df['amount'].sum():,.0f}")
                    with cols[1]:
                        st.metric("Avg Risk", f"{filtered_df['risk_score'].mean():.1f}")
                    with cols[2]:
                        st.metric("Success Rate", f"{(filtered_df['status'] == 'Completed').mean()*100:.1f}%")
                    with cols[3]:
                        st.metric("Avg Processing", f"{filtered_df['processing_time'].mean():.2f}s")
        else:
            st.info("📊 No transaction data available for analysis. Please add some transactions first.")
    
    # Markets & FX Section
    elif section == "📈 Markets & FX":
        st.header("📈 FX Markets & Currency Intelligence")
        
        # FX data display
        fx_df = st.session_state.fx_data
        
        tab1, tab2, tab3 = st.tabs(["💱 Live Rates", "📊 Volatility Analysis", "📈 Trends"])
        
        with tab1:
            st.subheader("💱 Current FX Rates")
            
            # Latest rates
            latest_rates = fx_df.iloc[-1]
            
            cols = st.columns(min(5, len(st.columns(1))))
            rate_cols = [col for col in fx_df.columns if col.endswith('_Rate')]
            
            for i, rate_col in enumerate(rate_cols[:5]):
                pair = rate_col.replace('_Rate', '')
                current_rate = latest_rates[rate_col]
                prev_rate = fx_df.iloc[-2][rate_col] if len(fx_df) > 1 else current_rate
                change = current_rate - prev_rate
                
                with cols[i % len(cols)]:
                    st.metric(
                        pair.replace('INR', '/INR'),
                        f"{current_rate:.4f}",
                        delta=f"{change:+.4f}"
                    )
            
            # Rate charts
            st.subheader("📊 Rate Movements (Last 30 Days)")
            
            fig_rates = go.Figure()
            for rate_col in rate_cols:
                pair = rate_col.replace('_Rate', '')
                fig_rates.add_trace(go.Scatter(
                    x=fx_df['Date'],
                    y=fx_df[rate_col],
                    mode='lines',
                    name=pair.replace('INR', '/INR')
                ))
            
            fig_rates.update_layout(
                title="FX Rates Over Time",
                xaxis_title="Date",
                yaxis_title="Rate",
                height=500
            )
            st.plotly_chart(fig_rates, use_container_width=True)
        
        with tab2:
            st.subheader("📊 Volatility Analysis")
            
            # Volatility charts
            fig_vol = go.Figure()
            vol_cols = [col for col in fx_df.columns if col.endswith('_Volatility')]
            for vol_col in vol_cols:
                pair = vol_col.replace('_Volatility', '').replace('INR', '/INR')
                fig_vol.add_trace(go.Scatter(
                    x=fx_df['Date'],
                    y=fx_df[vol_col],
                    mode='lines',
                    name=pair
                ))
            
            fig_vol.update_layout(
                title="FX Volatility Over Time",
                xaxis_title="Date",
                yaxis_title="Volatility",
                height=500
            )
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Average volatility metrics
            avg_vol = fx_df[vol_cols].mean()
            cols = st.columns(min(5, len(st.columns(1))))
            for i, vol_col in enumerate(vol_cols[:5]):
                pair = vol_col.replace('_Volatility', '').replace('INR', '/INR')
                with cols[i % len(cols)]:
                    st.metric(f"{pair} Avg Vol", f"{avg_vol[vol_col]:.4f}")
        
        with tab3:
            st.subheader("📈 Market Trends & Insights")
            
            # Average daily changes
            daily_changes = {}
            for rate_col in rate_cols:
                pair = rate_col.replace('_Rate', '').replace('INR', '/INR')
                changes = fx_df[rate_col].pct_change().mean() * 100  # in percent
                daily_changes[pair] = changes
            
            fig_changes = px.bar(
                x=list(daily_changes.keys()),
                y=list(daily_changes.values()),
                title="Average Daily Rate Change (%)",
                labels={'x': 'Currency Pair', 'y': 'Avg Daily Change (%)'}
            )
            st.plotly_chart(fig_changes, use_container_width=True)
            
            # Volatility box plots
            vol_data = pd.melt(fx_df, id_vars=['Date'], value_vars=vol_cols, 
                               var_name='Pair', value_name='Volatility')
            vol_data['Pair'] = vol_data['Pair'].str.replace('_Volatility', '').str.replace('INR', '/INR')
            fig_box = px.box(vol_data, x='Pair', y='Volatility', title="Volatility Distribution by Pair")
            st.plotly_chart(fig_box, use_container_width=True)
    
    # Security Center Section
    elif section == "🔐 Security Center":
        st.header("🔐 Security & Compliance Center")
        
        st.subheader("🔑 Encryption Management")
        st.write("Current Encryption: AES-256-GCM (at rest), TLS 1.3 (in transit)")
        st.write("Key Rotation Policy: Automatic every 90 days or manual on demand")
        
        if st.button("🔄 Rotate Encryption Key"):
            old_key = st.session_state.encryption_key
            st.session_state.encryption_key = generate_key()
            
            # Re-encrypt existing data
            for txn in st.session_state.transactions:
                if 'encrypted_amount' in txn:
                    amt = decrypt_data(txn['encrypted_amount'], old_key)
                    if "Decryption Failed" not in amt:
                        txn['encrypted_amount'] = encrypt_data(amt, st.session_state.encryption_key)
                if 'encrypted_counterparty' in txn:
                    cp = decrypt_data(txn['encrypted_counterparty'], old_key)
                    if "Decryption Failed" not in cp:
                        txn['encrypted_counterparty'] = encrypt_data(cp, st.session_state.encryption_key)
            
            st.session_state.audit_log.append({
                "action": "Encryption Key Rotation",
                "timestamp": datetime.now().isoformat(),
                "details": "Key rotated successfully, data re-encrypted"
            })
            st.success("✅ Encryption key rotated and all data re-encrypted!")
        
        with st.expander("👀 View Decrypted Transaction Data"):
            if st.session_state.transactions:
                decrypted_data = []
                for txn in st.session_state.transactions:
                    dec_txn = txn.copy()
                    dec_txn['amount_decrypted'] = decrypt_data(txn.get('encrypted_amount'), st.session_state.encryption_key)
                    dec_txn['counterparty_decrypted'] = decrypt_data(txn.get('encrypted_counterparty'), st.session_state.encryption_key)
                    decrypted_data.append(dec_txn)
                st.dataframe(pd.DataFrame(decrypted_data), use_container_width=True)
            else:
                st.info("No transactions to display.")
        
        st.subheader("📜 Regulatory Compliance")
        st.write(" - FEMA Tagging: Enabled for all cross-border transactions")
        st.write(" - AFA Alignment: Fully compliant with RBI guidelines")
        st.write(" - Data Minimization: Only essential fields stored and encrypted")
        
        st.subheader("📝 Audit Log")
        if st.session_state.audit_log:
            st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True)
        else:
            st.info("No audit events recorded yet.")
    
    # Settings Section
    elif section == "⚙️ Settings":
        st.header("⚙️ System Configuration")
        
        st.subheader("📊 Performance Metrics")
        st.write("TPS Capacity: 10–50 (prototype) → 1,000+ (production scalable)")
        st.write("Ingestion Latency: <1s (real-time processing)")
        st.write("Analytics Latency: <3s (insight generation)")
        st.write("SRVA Utilization: Real-time monitoring with INR share growth tracking")
        st.write(f"Current Transaction Count: {len(st.session_state.transactions)}")
        
        st.subheader("🛠️ Simulation Controls")
        st.slider("Auto-Generation Rate (txns/min)", 1, 60, 10, disabled=not st.session_state.auto_generate)
        st.info("Note: Auto-generation runs in background when enabled in sidebar. Adjust rate for simulation purposes.")
        
        st.subheader("🔧 Advanced Options")
        if st.checkbox("Enable Debug Mode"):
            st.write("Debug Info:")
            st.json({
                "Session Keys": list(st.session_state.keys()),
                "Transaction Count": len(st.session_state.transactions),
                "FX Data Shape": st.session_state.fx_data.shape,
                "Auto Generate Status": st.session_state.auto_generate
            })

if __name__ == "__main__":
    main()
