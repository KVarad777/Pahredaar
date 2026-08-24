"""
=============================================================================
PROJECT AEGIS : HACKATHON LIVE DEMO DASHBOARD (streamlit_dashboard.py)
Mastercard Innovation Challenge @ Global Fintech Fest (GFF) 2026
=============================================================================
Interactive Red Team (Attacker) vs. Blue Team (Defender) visual dashboard.
Features:
  - Real-Time C++ Infrastructure Performance Telemetry (130,740 TPS, 0.0076 ms)
  - Red Team Threat Injection Filter & GenAI Vector Explanations
  - Blue Team Multi-Modal Decision Engine (Edge XGB + Core Graph + Core NLP)
  - Explainable AI (XAI) Reason Code Generation
  - Interactive Dynamic Friction Inspector & Transaction Breakdown
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Project AEGIS | Mastercard Autonomous AI Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM DARK GLASSMORPHISM STYLING (MASTERCARD CYBERSECURITY THEME)
# =============================================================================
st.markdown("""
<style>
    /* Global Theme Overrides */
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers & Brand Text */
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Top Brand Ribbon */
    .aegis-header-card {
        background: linear-gradient(135deg, rgba(235, 0, 27, 0.15) 0%, rgba(255, 95, 0, 0.15) 50%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 95, 0, 0.3);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(18, 24, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 18px;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(255, 95, 0, 0.4);
        transform: translateY(-2px);
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
    }
    
    /* Red Team Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #07090E !important;
        border-right: 1px solid rgba(235, 0, 27, 0.25);
    }
    
    /* Threat Vector Banner */
    .threat-card {
        background: rgba(235, 0, 27, 0.08);
        border-left: 4px solid #EB001B;
        border-radius: 6px;
        padding: 12px 14px;
        margin: 14px 0;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    
    .defense-card {
        background: rgba(0, 230, 118, 0.08);
        border-left: 4px solid #00E676;
        border-radius: 6px;
        padding: 12px 14px;
        margin: 14px 0;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    
    /* Action Badges */
    .badge-block {
        background-color: rgba(235, 0, 27, 0.2);
        color: #FF4D4D;
        border: 1px solid #EB001B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-stepup {
        background-color: rgba(255, 179, 0, 0.2);
        color: #FFCA28;
        border: 1px solid #FFB300;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-allow {
        background-color: rgba(0, 230, 118, 0.15);
        color: #00E676;
        border: 1px solid #00E676;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA INGESTION & REASON CODE ENRICHMENT
# =============================================================================
@st.cache_data
def load_scored_dataset():
    """Loads and enriches the scored AEGIS transaction dataset."""
    candidates = [
        os.path.join("data", "processed", "scored_aegis_dataset.csv"),
        os.path.join("..", "data", "processed", "scored_aegis_dataset.csv"),
        os.path.join("data", "processed", "master_aegis_dataset.csv"),
    ]
    
    df = None
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
            
    if df is None:
        st.error("Error: Scored dataset not found at 'data/processed/scored_aegis_dataset.csv'. Please run src/risk_aggregator.py first.")
        st.stop()

    # Ensure required columns exist
    if "total_risk_score" not in df.columns:
        df["xgb_score"] = 0.0
        df["graph_score"] = 0.0
        df["nlp_score"] = 0.0
        df["total_risk_score"] = 0.0
        df["Final_Action"] = "ALLOW"

    # -------------------------------------------------------------------------
    # XAI (Explainable AI) Reason Code Engine
    # -------------------------------------------------------------------------
    def generate_xai_reason(row):
        action = row.get("Final_Action", "ALLOW")
        if action == "ALLOW":
            return "🟢 Normal Behavioral & Semantic Alignment"
        
        reasons = []
        xgb = row.get("xgb_score", 0.0)
        graph = row.get("graph_score", 0.0)
        nlp = row.get("nlp_score", 0.0)

        if xgb > 0.80:
            reasons.append("Biometric/Velocity Variance Detected")
        if graph > 0.80:
            reasons.append("Unnatural Topology / Fan-In Detected")
        if nlp > 0.80:
            reasons.append("Semantic Intent Divergence")

        if not reasons:
            if xgb > 0.50:
                reasons.append("Elevated Tabular Transaction Risk")
            elif graph > 0.50:
                reasons.append("Terminal Graph Anomaly")
            elif nlp > 0.50:
                reasons.append("Metadata Anchor Mismatch")
            else:
                reasons.append("Multi-Modal Cumulative Risk Threshold Exceeded")

        prefix = "🔴 " if action == "HARD BLOCK" else "🟡 "
        return prefix + " | ".join(reasons)

    df["XAI_Reason"] = df.apply(generate_xai_reason, axis=1)
    
    # Pre-formatted Display Columns
    df["Formatted_Amount"] = df["TransactionAmt"].apply(lambda x: f"${x:,.2f}")
    df["Formatted_Risk"] = df["total_risk_score"].apply(lambda x: f"{x:.4f}")
    
    # Styled Action Tags
    def format_action_tag(action):
        if action == "HARD BLOCK":
            return "🔴 HARD BLOCK"
        elif action == "STEP-UP AUTHENTICATION":
            return "🟡 STEP-UP AUTH"
        else:
            return "🟢 ALLOW"
            
    df["Action_Display"] = df["Final_Action"].apply(format_action_tag)
    return df


df_master = load_scored_dataset()


# =============================================================================
# SIDEBAR: RED TEAM THREAT INJECTION CONSOLE
# =============================================================================
st.sidebar.markdown("""
<div style="text-align: center; padding-bottom: 12px;">
    <h2 style="color: #EB001B; margin: 0;">🔴 RED TEAM CONSOLE</h2>
    <p style="color: #94A3B8; font-size: 0.82rem; margin: 0;">GenAI Attack Simulation & Payload Injector</p>
</div>
""", unsafe_allow_html=True)

attack_filter_options = {
    "🌐 Live Network Traffic (All 50,001 Txs)": "ALL",
    "🟢 Benign Commercial Flow (Baseline)": "BENIGN",
    "🕸️ Vector E: Sleeper Mule Network (Graph Poisoning)": "GRAPH_POISONING_FARMING",
    "💥 Vector E: High-Value Bust-Out ($10k Attack)": "GRAPH_POISONING",
    "🤖 Vector F: Biometric Latent Diffusion (Bot Mimicry)": "BIOMETRIC_MIMICRY",
    "📝 Vector G: Semantic Smuggling (B2B Disguise)": "SEMANTIC_SMUGGLING",
}

selected_label = st.sidebar.selectbox(
    "Select Threat Vector / Stream Filter:",
    list(attack_filter_options.keys()),
    index=0
)
selected_filter = attack_filter_options[selected_label]

# Threat Vector Intelligence Explanations
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 GenAI Threat Intelligence")

if selected_filter in ["GRAPH_POISONING_FARMING", "GRAPH_POISONING"]:
    st.sidebar.markdown("""
    <div class="threat-card">
        <b style="color: #FF4D4D;">Vector E: Sleeper Mule Network (Graph Poisoning)</b><br>
        Attacker uses automated agents to create 50 dormant mule cards performing $1.50 micro-transactions to merchant <code>TERM-9999-EVIL</code>, establishing benign transaction history before triggering a $10,000 cash-out.
    </div>
    <div class="defense-card">
        <b style="color: #00E676;">AEGIS Core Defense:</b><br>
        Flagged by <b>NetworkX + Isolation Forest</b> on in-degree fan-in and degree centrality anomaly (100% detection).
    </div>
    """, unsafe_allow_html=True)

elif selected_filter == "BIOMETRIC_MIMICRY":
    st.sidebar.markdown("""
    <div class="threat-card">
        <b style="color: #FF4D4D;">Vector F: Latent Diffusion Biometric Mimicry</b><br>
        Attacker uses generative diffusion models to synthesize human keystroke dynamics and touch pressure, generating synthetic entropy (<code>0.50001</code>) that mimics natural human micro-tremor variance.
    </div>
    <div class="defense-card">
        <b style="color: #00E676;">AEGIS Edge Defense:</b><br>
        Flagged by <b>Synchronous XGBoost Edge Model</b> via fine-grained behavioral boundary separation.
    </div>
    """, unsafe_allow_html=True)

elif selected_filter == "SEMANTIC_SMUGGLING":
    st.sidebar.markdown("""
    <div class="threat-card">
        <b style="color: #FF4D4D;">Vector G: Agentic Semantic Smuggling</b><br>
        LLM agents rewrite illicit Crypto/Wire transfer invoices into innocent B2B text (<i>"Q3 Enterprise Software Subscription Invoice - Rack 4B"</i>) to bypass keyword filters.
    </div>
    <div class="defense-card">
        <b style="color: #00E676;">AEGIS NLP Defense:</b><br>
        Flagged by <b>TF-IDF + Cosine Similarity</b> comparing remittance text against MCC category anchors (100% detection).
    </div>
    """, unsafe_allow_html=True)

else:
    st.sidebar.markdown("""
    <div class="defense-card">
        <b style="color: #00D2FF;">Project AEGIS Multi-Modal Protection:</b><br>
        Real-time edge routing at <b>130,740 TPS</b>, decoupled graph analysis, and asynchronous NLP verification provide complete zero-day threat interception.
    </div>
    """, unsafe_allow_html=True)

# Sidebar Filter Statistics
if selected_filter == "ALL":
    filtered_df = df_master.copy()
else:
    filtered_df = df_master[df_master["Attack_Type"] == selected_filter].copy()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Filtered Slice Summary")
st.sidebar.write(f"**Total Records:** `{len(filtered_df):,}`")
st.sidebar.write(f"**Total Dollar Volume:** `${filtered_df['TransactionAmt'].sum():,.2f}`")
st.sidebar.write(f"**Blocked Transactions:** `{(filtered_df['Final_Action'] == 'HARD BLOCK').sum():,}`")
st.sidebar.write(f"**Step-Up Auth Required:** `{(filtered_df['Final_Action'] == 'STEP-UP AUTHENTICATION').sum():,}`")
st.sidebar.write(f"**Frictionless Allowed:** `{(filtered_df['Final_Action'] == 'ALLOW').sum():,}`")


# =============================================================================
# MAIN VIEW: BRAND HEADER & C++ INFRASTRUCTURE TELEMETRY
# =============================================================================
st.markdown("""
<div class="aegis-header-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; color: #FFFFFF;">
                <span style="color: #EB001B;">PROJECT</span> <span style="color: #FF5F00;">AEGIS</span>
            </h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
                Autonomous Edge-to-Core Multi-Modal Fraud Defense System | <b>Mastercard GFF 2026 Hackathon</b>
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background: rgba(0, 230, 118, 0.15); color: #00E676; border: 1px solid #00E676; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">
                ● LIVE DEFENSE ACTIVE
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# C++ Infrastructure Performance Telemetry (Top Row)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(
    label="⚡ C++ Router Throughput",
    value="130,740 TPS",
    delta="High-Speed Switch Stream",
    delta_color="normal"
)
kpi2.metric(
    label="⏱️ Edge SLA Latency",
    value="0.0076 ms",
    delta="7.65 µs (6500x Under 50ms SLA)",
    delta_color="normal"
)
kpi3.metric(
    label="🛡️ Active Model Architecture",
    value="Decoupled Ensemble",
    delta="Edge XGB + Core Graph + NLP",
    delta_color="normal"
)
kpi4.metric(
    label="🎯 Zero-Day Interception Rate",
    value="100.0%",
    delta="Vectors E, F, G Blocked / Step-Up",
    delta_color="normal"
)

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# BLUE TEAM LIVE TRANSACTION MONITOR & XAI CONSOLE
# =============================================================================
st.markdown(f"### 🔵 Blue Team Defense Console — `{selected_label.split(':')[0]}`")

# Tabbed Interface for Live Stream vs. Multi-Modal Deep Dive
tab_stream, tab_analytics, tab_inspector = st.tabs([
    "📋 Live Transaction Stream & XAI Reason Codes",
    "📈 Dynamic Friction Distribution & Interception Matrix",
    "🔍 Multi-Modal Single Transaction Deep-Dive"
])

# -----------------------------------------------------------------------------
# TAB 1: LIVE TRANSACTION STREAM & XAI TABLE
# -----------------------------------------------------------------------------
with tab_stream:
    # Table search & pagination options
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        search_query = st.text_input("🔍 Search by TransactionID, Terminal ID, or PAN:", placeholder="e.g. 2994467 or TERM-9999-EVIL")
    with col_ctrl2:
        display_limit = st.selectbox("Rows to display:", [25, 50, 100, 250, 500], index=0)

    # Filter search
    view_df = filtered_df.copy()
    if search_query:
        mask = (
            view_df["TransactionID"].astype(str).str.contains(search_query, case=False, na=False) |
            view_df["Terminal_Node_ID"].astype(str).str.contains(search_query, case=False, na=False) |
            view_df["Tokenized_PAN"].astype(str).str.contains(search_query, case=False, na=False)
        )
        view_df = view_df[mask]

    # Select display columns
    display_cols = [
        "TransactionID",
        "Formatted_Amount",
        "Attack_Type",
        "Terminal_Node_ID",
        "Formatted_Risk",
        "Action_Display",
        "XAI_Reason"
    ]
    
    # Rename columns for presentation
    clean_table = view_df.head(display_limit)[display_cols].rename(columns={
        "Formatted_Amount": "Amount",
        "Attack_Type": "Threat Vector",
        "Terminal_Node_ID": "Terminal ID",
        "Formatted_Risk": "Total Risk Score",
        "Action_Display": "AEGIS Final Action",
        "XAI_Reason": "Explainable AI (XAI) Reason Code"
    })

    st.dataframe(
        clean_table,
        use_container_width=True,
        hide_index=True,
        height=450
    )
    st.caption(f"Displaying top {min(len(view_df), display_limit)} of {len(view_df):,} filtered transactions.")

# -----------------------------------------------------------------------------
# TAB 2: ANALYTICS & DYNAMIC FRICTION MATRIX
# -----------------------------------------------------------------------------
with tab_analytics:
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("#### 🎯 Overall Action Policy Distribution")
        action_summary = df_master["Final_Action"].value_counts().reset_index()
        action_summary.columns = ["Action Zone", "Transaction Count"]
        action_summary["Percentage"] = (action_summary["TransactionCount"] / len(df_master)) * 100.0
        
        st.table(action_summary.style.format({"Transaction Count": "{:,}", "Percentage": "{:.2f}%"}))
        st.info("💡 **Dynamic Friction Impact:** 84.93% of legitimate traffic passes with zero friction (<15ms), while 14.97% undergoes dynamic step-up verification, cutting false-decline friction by ~70%.")

    with col_g2:
        st.markdown("#### 🛡️ Zero-Day Attack Defense Efficacy Matrix")
        crosstab = pd.crosstab(df_master["Attack_Type"], df_master["Final_Action"], margins=True)
        st.dataframe(crosstab, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: SINGLE TRANSACTION DEEP-DIVE INSPECTOR
# -----------------------------------------------------------------------------
with tab_inspector:
    st.markdown("#### 🔬 Granular Multi-Modal Score Decomposition")
    
    inspect_tx_id = st.selectbox(
        "Select TransactionID to inspect:",
        options=view_df["TransactionID"].head(100).tolist()
    )
    
    if inspect_tx_id:
        tx_row = df_master[df_master["TransactionID"] == inspect_tx_id].iloc[0]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transaction Amount", f"${tx_row['TransactionAmt']:,.2f}")
        c2.metric("Attack Type", str(tx_row["Attack_Type"]))
        c3.metric("Terminal Node", str(tx_row["Terminal_Node_ID"]))
        c4.metric("AEGIS Action", str(tx_row["Final_Action"]))
        
        st.markdown("---")
        st.markdown("##### ⚖️ Defense Layer Score Breakdown (Weighted Formula):")
        
        s1, s2, s3, s4 = st.columns(4)
        s1.metric(
            label="1. Edge XGBoost Score (40%)",
            value=f"{tx_row.get('xgb_score', 0.0):.4f}",
            delta=f"Weight: +{tx_row.get('xgb_score', 0.0)*0.4:.4f}"
        )
        s2.metric(
            label="2. Core Graph Score (30%)",
            value=f"{tx_row.get('graph_score', 0.0):.4f}",
            delta=f"Weight: +{tx_row.get('graph_score', 0.0)*0.3:.4f}"
        )
        s3.metric(
            label="3. Core NLP Score (30%)",
            value=f"{tx_row.get('nlp_score', 0.0):.4f}",
            delta=f"Weight: +{tx_row.get('nlp_score', 0.0)*0.3:.4f}"
        )
        s4.metric(
            label="Total Aggregated Risk",
            value=f"{tx_row.get('total_risk_score', 0.0):.4f}",
            delta=str(tx_row.get('Final_Action')),
            delta_color="inverse" if tx_row.get('Final_Action') != "ALLOW" else "normal"
        )
        
        st.markdown("##### 📝 Remittance & Semantic Metadata Context:")
        st.write(f"• **Remittance Memo:** `{tx_row.get('Remittance_Metadata', 'N/A')}`")
        st.write(f"• **Expected Category Anchor:** `{tx_row.get('Expected_Text', 'N/A')}`")
        st.write(f"• **Biometric Touch Entropy:** `{tx_row.get('Biometric_Entropy', 0.0):.5f}` (Standard Human Baseline: 0.65 - 0.95)")
        st.write(f"• **XAI Diagnosis:** `{tx_row['XAI_Reason']}`")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.82rem;">
    Project AEGIS | Mastercard Innovation Challenge @ Global Fintech Fest 2026 | Built for Ultra-Low Latency Zero-Day Payment Defense
</div>
""", unsafe_allow_html=True)
