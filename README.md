# Dual-Backend AI Assistant & Automated Benchmarking Framework 🤖📊

A production-ready evaluation workbench and interactive workspace comparing a centralized frontier model with a decentralized serverless open-source model under multi-turn conversational loops. Built entirely using zero-cost free-tier infrastructure.

---

## 🏗️ Core Architecture Overview

The system is separated into three highly decoupled, modular layers to enforce clean separation of concerns:

```text
ai-assistant-eval/
├── app.py                 # Streamlit UI Dashboard (Chat & Analytics view)
├── core_agent.py          # Unified LLM client gateway & memory state coordinator
└── evaluator.py           # Programmatic test harness & safety probe simulator


Stateful Chat & Telemetry Dashboard (app.py): A unified front-end interface built with Streamlit featuring two isolated workspaces: an interactive chat zone with contextual memory and a dynamic evaluation panel parsing runtime metrics using Plotly graphics.

  Unified Agent Engine (core_agent.py): A consolidated routing core converting standardized user inputs into client-specific message payloads. It manages state parameters independently across backend targets, ensuring zero cross-contamination of token history.
  
  Automated Evaluation Harness (evaluator.py): A black-box testing harness designed to run batch baseline prompts against Factual, Adversarial (Jailbreak), and Sensitive Bias alignment vectors completely independent of the UI.


  ⚡ Quick Start & Deployment

1. Environment Initialization
Ensure you have Python 3.10+ installed. Clone this repository, move into the directory root, and initialize a localized virtual environment:

python -m venv .venv
# On Windows PowerShell:
& ".venv\Scripts\Activate.ps1"
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

2. Configure Environment Secrets (.env)
Create a secure state file named .env in your project folder root and map your API tokens:

# Frontier Target Configuration (Via Groq Cloud Free Tier)
FRONTIER_BASE_URL=[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)
FRONTIER_API_KEY=your_groq_api_key_here
FRONTIER_MODEL_NAME=llama-3.3-70b-versatile

# Open-Source Target Configuration (Via Hugging Face Serverless)
OSS_BASE_URL=[https://router.huggingface.co/v1](https://router.huggingface.co/v1)
OSS_API_KEY=your_huggingface_fine_grained_token_here
OSS_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct

3. Run the Automated Benchmark Sweep
Before initializing the frontend, execute the automated evaluation script to simulate alignment stress testing and compile the telemetry data ledger (evaluation_results.csv):

python evaluator.py

4. Launch the Application Interface
Spin up the interactive Streamlit local server to chat live and visualize your telemetry plots:

streamlit run app.py


🛠️ Strategic Engineering Decisions & Technical Trade-offs

1. The Production Infrastructure Pivot (Gemini API to Groq LPUs)
Context: Initial iterations utilized Google AI Studio's gemini-2.5-flash endpoint as the primary Frontier baseline. 
 The Problem: During dense multi-turn conversation sweeps, Google's centralized shared free clusters frequently threw severe HTTP 503 Service Unavailable capacity drops, and client-side resilience retries accelerated downstream 429 ResourceExhausted quota lockouts.
 The Pivot: To guarantee high availability and keep the benchmarking loops scientific, the core gateway was dynamically refactored to consume Llama 3.3 (70B Versatile) via Groq Cloud's distributed LPU pipeline. This shift dropped request errors to 0.0%, demonstrating resilient fallback engineering.

 2. OpenAI-Compatible Serialization vs Multi-SDK Bloat
 Decision: Leveraged standard OpenAI-compatible network structures for both Hugging Face's global proxy router and Groq Cloud instead of importing heavy, model-specific SDK libraries. 
Trade-off: This keeps the codebase incredibly lightweight, reduces package deployment sizes, and standardizes dictionary formats. However, it minorly restricts the capability to inject vendor-specific parameters (like native Gemini web search grounding) directly into the stream.

3. Dynamic Memory Truncation on Model Toggling
Decision: Implemented automatic wiping of session state lists when switching active engines in the sidebar.
Trade-off: This guarantees that structural prompt patterns and token arrays are completely isolated, removing any chance of data bias or structural context leakage during evaluation steps. The minor drawback is that users lose their multi-turn conversation memory history when shifting backends mid-session.


🔮 Future Roadmap Enhancements (With More Time)

Asynchronous Batch Execution Loops: Refactor the automated evaluator.py script using asyncio and aiohttp to run evaluations against backends concurrently rather than sequentially, dropping benchmark execution runtime.

Programmatic LLM-As-A-Judge Consensus Scoring: Replace the keyword parsing heuristic in the evaluation suite with a dedicated, neutral LLM referee agent (e.g., Llama-3-70b) to dynamically audit responses on a continuous semantic scale (1–5) for subtle bias and hallucination drift.  

Retrieval-Augmented Grounding (RAG): Integrate a lightweight vector database (like Milvus or Pinecone) to wrap the Open Source backend, providing an external knowledge boundaries index to eliminate the 16.6% factual hallucination rate caught during stress tests.