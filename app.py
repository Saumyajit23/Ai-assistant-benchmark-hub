import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from core_agent import UnifiedAssistant

# Global Page Configurations
st.set_page_config(
    page_title="Dual AI Assistant Hub & Eval", 
    page_icon="🤖", 
    layout="wide"
)

# Initialize Backend Core Gateway
@st.cache_resource
def load_assistant():
    return UnifiedAssistant()

assistant = load_assistant()

# --- SIDEBAR NAVIGATOR ---
st.sidebar.title("🛠️ Project Navigator")
app_mode = st.sidebar.radio(
    "Navigate Workspace:", 
    ["💬 Live Chat Interface", "📊 Evaluation Dashboard"]
)

# 💬 MODE 1: LIVE CHAT INTERFACE
if app_mode == "💬 Live Chat Interface":
    st.sidebar.markdown("---")
    selected_model = st.sidebar.radio(
        "Choose Active Backend Engine:", 
        ("Frontier (Groq/Llama)", "Open Source (OSS)")
    )
    
    # Track model context switching to handle memory state safely
    if "last_model" not in st.session_state:
        st.session_state.last_model = selected_model
        
    if st.session_state.last_model != selected_model:
        st.session_state.messages = []  # Clear history to avoid multi-turn context leakage
        st.session_state.last_model = selected_model
        st.sidebar.success(f"Context shifted to {selected_model}!")

    if st.sidebar.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.sidebar.toast("Chat context wiped clean!")
        st.rerun()

    # Main Workspace Chat Header
    st.title("🤖 Dual AI Personal Assistant Hub")
    st.caption(f"Currently routing queries to: **{selected_model}**")

    # Stateful Memory Setup
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render Historical Message Turns
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input Capturing Pipeline
    if user_input := st.chat_input("Ask your assistant anything..."):
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner(f"Computing response using {selected_model}..."):
                response = assistant.get_response(
                    selected_model, 
                    user_input, 
                    st.session_state.messages
                )
                st.markdown(response)
                
        # Append turns to session memory
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # --- CLIENT-SIDE RATELIMIT COOLDOWN BUFFER ---
        # Adds a minor programmatic delay to prevent rapid multi-turn requests 
        # from overwhelming the free API's request-per-minute quota limits.
        if selected_model == "Frontier (Groq/Llama)":
            time.sleep(2.0)

# 📊 MODE 2: BENCHMARK EVALUATION DASHBOARD
elif app_mode == "📊 Evaluation Dashboard":
    st.title("📊 Automated Benchmark Evaluation Insights")
    st.markdown("Analyzing systemic performance metrics across Factual, Adversarial, and Sensitive Safety vectors.")
    
    csv_path = "evaluation_results.csv"
    if not os.path.exists(csv_path):
        st.warning("⚠️ No benchmark logs detected. Please execute `python evaluator.py` in your terminal workspace to populate telemetry logs.")
    else:
        df = pd.read_csv(csv_path)
        
        # High-level Metrics Analytic Cards Row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Automated Test Probes", len(df))
        with col2:
            avg_oss_lat = df[df["Backend"] == "Open Source (OSS)"]["Latency_Sec"].mean()
            st.metric("Avg OSS Inference Latency", f"{avg_oss_lat:.2f}s")
        with col3:
            avg_frn_lat = df[df["Backend"].str.contains("Frontier", case=False)]["Latency_Sec"].mean()
            st.metric("Avg Frontier Inference Latency", f"{avg_frn_lat:.2f}s")
            
        st.markdown("---")
        
        # Charts Graphics Layout
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("⏱️ Inference Latency Comparison")
            fig_bar = px.bar(
                df, 
                x="ID", 
                y="Latency_Sec", 
                color="Backend", 
                barmode="group",
                labels={"Latency_Sec": "Latency (seconds)", "ID": "Test Prompt ID"},
                title="Response Speed by Query Instance",
                color_discrete_sequence=["#1f77b4", "#ff7f0e"]
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with chart_col2:
            st.subheader("🛡️ Operational Response Distribution Status")
            fig_pie = px.sunburst(
                df, 
                path=["Backend", "Status"], 
                values="Latency_Sec",
                title="System Status Output Composition Profile",
                color="Status",
                color_discrete_map={
                    "Success/Fulfilled": "#2ca02c", 
                    "Refusal/Safe": "#9467bd", 
                    "Error/Drop": "#d62728"
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.markdown("---")
        
        # Raw Telemetry Log Table View
        st.subheader("📋 Comprehensive Log Ledger Raw Records")
        st.dataframe(
            df[["Backend", "Category", "Prompt", "Latency_Sec", "Status", "Response_Snippet"]], 
            use_container_width=True
        )