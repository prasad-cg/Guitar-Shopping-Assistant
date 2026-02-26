# 🎸 Guitar Shopping Multi-Agent AI System

A comprehensive demonstration of **multi-agentic AI orchestration** using LangGraph, showcasing how specialized agents collaborate to provide an exceptional customer experience in an e-commerce setting.

---

## ✨ Overview

This project implements a sophisticated multi-agent system where:

- **3 Specialized AI Agents** work collaboratively through LangGraph orchestration.
- **RAG (Retrieval-Augmented Generation)** provides knowledge-grounded responses from a structured Excel catalog.
- **Hybrid Search**: Uses vector embeddings (FAISS) with a keyword fallback to ensure reliability in any network environment.
- **State Persistence**: Maintains conversation history and user preferences across multiple turns.
- **Modern UI**: A premium Streamlit interface with a responsive chat experience.

---

## 🏗️ System Architecture

### Multi-Agent Components

1. 📚 **Information Agent**: Retrieves technical specs and brand info from the catalog using RAG.
2. ✨ **Recommendation Agent**: Analyzes user needs (skill level, budget, genre) to find the perfect match.
3. 💰 **Negotiator Agent**: Handles pricing inquiries, discounts, and value-added bundles.

### Orchestration Layer

Managed by **LangGraph**, the workflow dynamically routes queries:

- **Intent Parsing**: Analyzes the user's message to activate only the relevant agents.
- **Contextual Memory**: History and sidebar preferences are injected into each agent's reasoning.
- **Synthesis**: Combines specialized agent outputs into a single, cohesive response.

---

## 📁 Project Structure

```
guitar-agents-lab/
├── agents/
│   ├── orchestrator.py           # LangGraph workflow orchestration
│   ├── information_agent.py      # Knowledge-based agent (RAG)
│   ├── recommendation_agent.py   # Preference matching agent
│   └── negotiator_agent.py       # Pricing & negotiation agent
├── utils/
│   ├── rag.py                    # RAG system (FAISS + Keyword Fallback)
│   ├── llm_setup.py              # Azure OpenAI initialization
│   └── data_utils.py             # Utility functions
├── ui/
│   └── streamlit_app.py          # Interactive Streamlit interface
├── data/
│   └── guitar_catalog.xlsx       # The "Brain" - Structured guitar data
├── config.py                      # Central configuration
├── requirements.txt               # Project dependencies
└── main.py                        # Alternative CLI entry point
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites

- Python 3.10+
- Azure OpenAI Access (or modify `llm_setup.py` for standard OpenAI)

### 2. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the root directory:

```env
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
API_VERSION=2025-01-01-preview
TEMPERATURE=0.5
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
```

---

## 🚀 Running the Application

### Streamlit UI (Recommended)

```bash
streamlit run ui/streamlit_app.py
```

This launches a browser window where you can interact with the agents, set preferences in the sidebar, and see the multi-agent collaboration in real-time.

### Interactive CLI

```bash
python main.py --mode interactive
```

---

## 🔑 Key Features & Fixes

- **Robust RAG**: Custom logic handles SSL/Proxy issues and provides a keyword-based fallback if vector search is unavailable.
- **Intelligent Recommendations**: Specifically tuned to identify "Beginner" vs "Professional" models from the catalog.
- **Agent Grounding**: Strict system prompts prevent hallucinations; agents only discuss guitars found in the attached Excel catalog.
- **Clean Codebase**: Minimal dependencies and modular architecture ready for production-like deployments.

---

## 📝 Usage Tips

- **Sidebar Preferences**: Set your budget and skill level in the sidebar; the Recommendation Agent will automatically factor these into its suggestions.
- **Natural Conversation**: You can ask complex questions like _"I want to buy a beginner guitar for rock music, what's a good deal?"_ and all three agents will collaborate to answer.
