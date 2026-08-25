# 🎸 Guitar Shopping Multi-Agent AI System

A multi-agentic AI system using **LangGraph** where specialized agents collaborate to provide a guitar shopping assistant experience — recommendations, information retrieval, and price negotiation all working together.

---

## ✨ Overview

- **3 Specialized AI Agents** orchestrated via LangGraph
- **RAG (Retrieval-Augmented Generation)** powered by a local embedding model + FAISS vector store
- **Hybrid Search**: Vector embeddings with keyword fallback for reliability
- **Conversation Memory**: Maintains context across multiple turns
- **Streamlit UI**: Chat interface with multi-conversation support

---

## 🏗️ Architecture

### Agents

| Agent | Role |
|-------|------|
| 📚 **Information Agent** | Retrieves specs, brand info, and features from the catalog using RAG |
| ✨ **Recommendation Agent** | Matches user preferences (skill, budget, genre) to guitars |
| 💰 **Negotiator Agent** | Handles pricing, discounts, and bundle suggestions |

### Orchestration (LangGraph)

- Parses user intent and activates only relevant agents
- Injects conversation history and preferences into each agent
- Synthesizes multi-agent outputs into a single cohesive response

---

## 📁 Project Structure

```
guitar-agents-lab/
├── agents/
│   ├── orchestrator.py           # LangGraph workflow orchestration
│   ├── information_agent.py      # Knowledge-based agent (RAG)
│   ├── recommendation_agent.py   # Preference matching agent
│   └── negotiator_agent.py       # Pricing & negotiation agent
├── evals/
│   ├── answer_evaluator.py       # Response quality evaluation (LLM-as-judge)
│   ├── prompt_evaluator.py       # Agent prompt effectiveness evaluation
│   └── run_evals.py              # CLI runner for evaluations
├── utils/
│   ├── rag.py                    # RAG system (local embeddings + FAISS)
│   ├── llm_setup.py              # Azure OpenAI initialization
│   └── data_utils.py             # Utility functions
├── ui/
│   └── streamlit_app.py          # Streamlit chat interface
├── data/
│   └── guitar_catalog.xlsx       # Structured guitar catalog (180 entries)
├── models/
│   └── all-MiniLM-L6-v2/        # Local embedding model (download separately)
├── config.py                     # Central configuration
├── requirements.txt              # Python dependencies
├── download_model.py             # Helper script to download embedding model
├── .env.example                  # Environment variable template
└── main.py                       # CLI entry point
```

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.10+
- Azure OpenAI resource with a deployed model (e.g., `gpt-4.1`)

### Step 1: Clone and create virtual environment

```bash
git clone https://github.com/your-username/guitar-agents-lab.git
cd guitar-agents-lab

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### Step 2: Configure environment variables

```bash
copy .env.example .env
```

Edit `.env` and fill in your Azure OpenAI credentials:

```env
AZURE_OPENAI_API_KEY=your_actual_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
API_VERSION=2025-01-01-preview
TEMPERATURE=0.5
```

#### Azure OpenAI Setup

1. Go to [Azure Portal](https://portal.azure.com) → create an **Azure OpenAI** resource
2. In Azure OpenAI Studio, deploy a model (e.g., `gpt-4.1` or `gpt-4o`)
3. Copy the **API Key** and **Endpoint** from the resource's "Keys and Endpoint" page
4. The **Deployment Name** is what you named your deployment in step 2

### Step 3: Download the embedding model

The RAG system uses `all-MiniLM-L6-v2` (~80 MB) for generating vector embeddings locally. This model is **not included in the repo** due to GitHub's file size limits.

**Option A — Run the helper script (requires internet access):**

```bash
python download_model.py
```

This downloads the model and saves it to `models/all-MiniLM-L6-v2/`.

**Option B — Manual download (if behind a corporate proxy):**

On any machine with unrestricted internet access, run:

```bash
pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); model.save('./all-MiniLM-L6-v2'); print('Done!')"
```

Then copy the generated `all-MiniLM-L6-v2/` folder into `models/` in your project:

```
guitar-agents-lab/
└── models/
    └── all-MiniLM-L6-v2/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── ... (other files)
```

Transfer via USB drive, OneDrive, email (zip it first — compresses to ~60 MB), or any file sharing method.

---

## 🚀 Running the Application

### Streamlit UI (Recommended)

```bash
streamlit run ui/streamlit_app.py
```

Opens a browser at `http://localhost:8501` with the full chat interface.

### Interactive CLI

```bash
python main.py --mode interactive
```

### Single Query

```bash
python main.py --mode cli --query "recommend a beginner guitar for blues under $500"
```

---

## 📊 Evaluation Framework

The project includes a separate evaluation system (`evals/`) to measure quality of both the system outputs and the agent prompts.

### Answer Evaluation

Evaluates live system responses against user queries using LLM-as-a-judge. Scores across 5 dimensions:

| Dimension | What it measures |
|-----------|-----------------|
| Relevance | Does the answer address what was actually asked? |
| Completeness | Are all aspects of the question covered? |
| Groundedness | Is the response based on catalog data, not hallucinated? |
| Helpfulness | Would this help a customer make a decision? |
| Clarity | Is it well-structured and easy to understand? |

### Prompt Evaluation

Evaluates agent system prompts (personas) to assess whether they effectively guide the LLM. Scores across 6 dimensions:

| Dimension | What it measures |
|-----------|-----------------|
| Role Clarity | Is the agent's identity clearly defined? |
| Constraint Effectiveness | Are guardrails against hallucination well-defined? |
| Task Alignment | Does the prompt match the agent's responsibility? |
| Tone Guidance | Does it establish the right conversational style? |
| Completeness | Does it cover edge cases and fallback behavior? |
| Conciseness | Is it focused without unnecessary verbosity? |

### Running Evaluations

```bash
# Evaluate agent prompts only
python -m evals.run_evals --type prompt

# Evaluate system answers (generates live responses + evaluates)
python -m evals.run_evals --type answer

# Run both
python -m evals.run_evals --type all

# Save results to JSON
python -m evals.run_evals --type all --output eval_results.json
```

---

## 💡 Usage Tips

- Ask natural questions like *"I want a guitar for fingerpicking, budget around $800"*
- All three agents collaborate automatically based on your query
- The FAISS index is cached after first run — subsequent starts are near-instant
- If the embedding model is not available, the system falls back to keyword search

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph |
| LLM | Azure OpenAI (GPT-4) |
| Embeddings | all-MiniLM-L6-v2 (local, offline) |
| Vector Store | FAISS |
| RAG Framework | LangChain |
| Data | Structured Excel catalog |
| Frontend | Streamlit |
| Language | Python 3.10+ |
