import sys
import os
import traceback
import logging
import certifi

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add parent directory to path so we can import agents and utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure httpx/openssl can verify certificates on Windows by pointing to certifi bundle
if not os.environ.get("SSL_CERT_FILE"):
    os.environ["SSL_CERT_FILE"] = certifi.where()

import streamlit as st

# ---------------------------------------------------------------------------
# Page config – MUST be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Guitar Shopping Assistant",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded",
)

"""
Streamlit UI for Guitar Shopping Multi-Agent System
Main demo interface showcasing the multi-agentic AI system
"""

from typing import Dict, Any
from agents.orchestrator import GuitarShoppingOrchestrator
from config import APP_THEME, AGENTS
from utils import (
    get_guitar_categories,
    get_price_ranges,
    get_playing_styles,
    format_conversation_history,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CUSTOM CSS                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def inject_custom_css():
    st.markdown("""
    <style>
    /* ---------- Import Google Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ---------- Global ---------- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Hero Banner ---------- */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 24px;
        border: 1px solid rgba(255,107,53,0.25);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50%; right: -30%;
        width: 60%; height: 200%;
        background: radial-gradient(circle, rgba(255,107,53,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.6em;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B35, #FFC947);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 6px 0;
    }
    .hero-subtitle {
        color: #b0b8c8;
        font-size: 1.05em;
        font-weight: 400;
        margin-top: 4px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,107,53,0.15);
        color: #FF6B35;
        font-weight: 600;
        font-size: 0.82em;
        padding: 5px 14px;
        border-radius: 20px;
        margin-top: 12px;
        border: 1px solid rgba(255,107,53,0.3);
    }

    /* ---------- Agent Cards ---------- */
    .agent-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 22px 20px;
        margin: 8px 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .agent-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border-color: rgba(255,107,53,0.35);
    }
    .agent-card-icon {
        font-size: 1.7em;
        margin-bottom: 8px;
    }
    .agent-card-title {
        font-size: 1.05em;
        font-weight: 700;
        color: #FF6B35;
        margin-bottom: 6px;
    }
    .agent-card-desc {
        font-size: 0.88em;
        color: #9ba4b5;
        line-height: 1.5;
    }

    /* ---------- Chat area ---------- */
    .chat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .chat-title {
        font-size: 1.3em;
        font-weight: 700;
        color: #e0e0e0;
    }

    /* ---------- New-chat floating button ---------- */
    .new-chat-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #FF6B35, #e05520);
        color: #fff !important;
        font-weight: 600;
        font-size: 0.88em;
        padding: 8px 18px;
        border-radius: 24px;
        border: none;
        cursor: pointer;
        text-decoration: none;
        transition: box-shadow 0.2s ease;
    }
    .new-chat-btn:hover {
        box-shadow: 0 4px 18px rgba(255,107,53,0.45);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #141428 100%);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* ---------- Footer ---------- */
    .footer-text {
        text-align: center;
        color: #555;
        font-size: 0.82em;
        padding: 20px 10px 8px 10px;
    }

    /* ---------- Misc ---------- */
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  COMPONENTS                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def render_hero():
    """Render the hero banner at the very top."""
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">{APP_THEME['title']}</div>
        <div class="hero-subtitle">{APP_THEME['subtitle']}</div>
    </div>
    """, unsafe_allow_html=True)


def render_agent_overview():
    """Render the multi-agent system overview cards."""
    st.markdown("#### 🤖 Multi-Agent System Overview")

    cols = st.columns(3)
    icons = ["📚", "✨", "💰"]

    for idx, (agent_key, agent_info) in enumerate(AGENTS.items()):
        with cols[idx]:
            st.markdown(f"""
            <div class="agent-card">
                <div class="agent-card-icon">{icons[idx]}</div>
                <div class="agent-card-title">{agent_info['name']}</div>
                <div class="agent-card-desc">{agent_info['description']}</div>
            </div>
            """, unsafe_allow_html=True)


def render_sidebar_preferences():
    """Render the preferences sidebar (display only — NOT used by agents)."""
    st.sidebar.markdown("### ⚙️ Preferences")
    st.sidebar.caption("ℹ️ These preferences are for your reference only. "
                       "The AI assistant will ask you directly during the conversation.")
    st.sidebar.markdown("---")

    budget_range = st.sidebar.select_slider(
        "💰 Budget Range",
        options=["Budget ($0-500)", "Mid-Range ($500-1500)",
                 "Premium ($1500-5000)", "Ultra-Premium ($5000+)"],
        value="Budget ($0-500)",
    )

    skill_level = st.sidebar.selectbox(
        "🎓 Your Skill Level",
        ["Beginner", "Intermediate", "Advanced", "Professional"],
    )

    music_style = st.sidebar.multiselect(
        "🎵 Music Style(s) You Play",
        get_playing_styles(),
        default=["Rock"],
    )

    guitar_type = st.sidebar.selectbox(
        "🎸 Guitar Type",
        get_guitar_categories(),
    )

    other_considerations = st.sidebar.text_input(
        "📝 Any other considerations?",
        placeholder="e.g., small hands, prefer vintage style...",
    )

    return {
        "budget": budget_range,
        "skill_level": skill_level,
        "music_style": music_style,
        "guitar_type": guitar_type,
        "other_considerations": other_considerations,
    }


def _init_chat_state():
    """Initialise session-state keys used for multi-chat support."""
    if "chats" not in st.session_state:
        # chats is a dict of {chat_id: [messages]}
        st.session_state.chats = {"Chat 1": []}
    if "active_chat" not in st.session_state:
        st.session_state.active_chat = "Chat 1"


def render_chat_interface(orchestrator: GuitarShoppingOrchestrator):
    """Render the chat interface with a + (new chat) button."""
    _init_chat_state()

    # ---- Chat header with new-chat button ----
    hcol1, hcol2 = st.columns([8, 2])
    with hcol1:
        st.markdown(f'<div class="chat-title">💬 {st.session_state.active_chat}</div>',
                    unsafe_allow_html=True)
    with hcol2:
        if st.button("➕ New Chat", key="new_chat_btn", type="primary"):
            chat_num = len(st.session_state.chats) + 1
            new_name = f"Chat {chat_num}"
            st.session_state.chats[new_name] = []
            st.session_state.active_chat = new_name
            st.rerun()

    # ---- Sidebar chat list ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💬 Conversations")
    for chat_name in st.session_state.chats:
        is_active = chat_name == st.session_state.active_chat
        label = f"▶ {chat_name}" if is_active else chat_name
        if st.sidebar.button(label, key=f"switch_{chat_name}", use_container_width=True):
            st.session_state.active_chat = chat_name
            st.rerun()

    # ---- Active messages ----
    messages = st.session_state.chats[st.session_state.active_chat]

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---- User input ----
    user_query = st.chat_input("Ask me anything about guitars...")

    if user_query:
        messages.append({"role": "user", "content": user_query})

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.status("🤖 Processing your question...", expanded=True) as status:
            # Get existing history (excluding the query we just added)
            history_to_pass = messages[:-1]
            
            # Get preferences from current session (actually we need to capture them from the sidebar)
            # Since render_sidebar_preferences is called in main, we might need to store them in session_state
            preferences = st.session_state.get("sidebar_prefs", {})
            
            result = orchestrator.process_customer_query(
                user_query, 
                history=history_to_pass,
                preferences=preferences
            )

            # Show which agents were activated
            active = result.get("metadata", {}).get("active_agents", [])
            if "information" in active:
                st.write("📚 Information Agent – Retrieved guitar knowledge")
            if "recommendation" in active:
                st.write("✨ Recommendation Agent – Analysed your needs")
            if "negotiation" in active:
                st.write("💰 Negotiator Agent – Checked deals & pricing")
            status.update(label="✅ Complete", state="complete")

        # Display response
        response_text = result["final_response"]
        with st.chat_message("assistant"):
            st.markdown(response_text)
        messages.append({"role": "assistant", "content": response_text})

        # Individual agent responses (only agents that actually ran)
        agents_data = result["agents_involved"]
        if agents_data:
            with st.expander("� View Individual Agent Responses"):
                for agent_resp in agents_data:
                    agent_name = agent_resp.get("agent", "Agent")
                    st.markdown(f"**🤖 {agent_name}**")
                    st.markdown(agent_resp.get("content", "No response"))
                    st.markdown("---")


def render_about_tab():
    """Render the About tab with full agent information."""
    st.markdown("### ℹ️ About This System")

    st.markdown("""
    **Multi-Agentic AI for Guitar Shopping**

    This system demonstrates how multiple specialised AI agents collaborate
    to provide an exceptional customer experience — just like walking into a
    real guitar store and chatting with knowledgeable staff.
    """)

    st.markdown("---")
    st.markdown("#### 🤖 Agent Details")

    # Detailed breakdown of each agent
    agent_details = {
        "📚 Information Agent": {
            "role": "Guitar Knowledge Expert",
            "description": (
                "Retrieves and presents comprehensive information about guitars from our catalog. "
                "Uses RAG (Retrieval-Augmented Generation) to search through the guitar knowledge base "
                "and provide accurate, context-aware answers about types, features, specifications, "
                "brands, and models."
            ),
            "capabilities": [
                "Catalog search & retrieval",
                "Feature & specification explanations",
                "Category overviews",
                "Brand & model comparisons",
            ],
        },
        "✨ Recommendation Agent": {
            "role": "Personalised Guitar Matchmaker",
            "description": (
                "Analyses what the customer describes in conversation — their playing style, "
                "experience level, genre preferences, and budget — then recommends the best-matching "
                "guitars from the catalog."
            ),
            "capabilities": [
                "Conversational preference detection",
                "Use-case analysis",
                "Personalised top-pick recommendations",
                "Side-by-side comparisons",
            ],
        },
        "💰 Price Negotiator Agent": {
            "role": "Deal & Value Specialist",
            "description": (
                "Discusses pricing, bundles, discounts, and trade-in opportunities. "
                "Helps customers understand the value they're getting and finds the "
                "best deal structures."
            ),
            "capabilities": [
                "Price range breakdown",
                "Bundle & accessory suggestions",
                "Discount strategies",
                "Value justification",
            ],
        },
    }

    for title, details in agent_details.items():
        caps_list = "".join(f"<li>{cap}</li>" for cap in details["capabilities"])
        st.markdown(f"""
        <div class="agent-card" style="margin-bottom:12px;">
            <div class="agent-card-title">{title} — {details['role']}</div>
            <div class="agent-card-desc">{details['description']}</div>
            <div style="margin-top:8px; color:#b0b8c8; font-size:0.88em;">
                <b>Key Capabilities:</b>
                <ul style="margin-top:4px;">{caps_list}</ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🛠️ Tech Stack")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown("""
        - **Orchestration:** LangGraph
        - **LLM:** Azure OpenAI (GPT-4)
        - **Retrieval:** FAISS + LangChain RAG
        """)
    with tcol2:
        st.markdown("""
        - **Data:** Structured Excel guitar catalog
        - **Frontend:** Streamlit
        - **Language:** Python 3.10+
        """)


def render_footer():
    st.markdown('<div class="footer-text"><b>🚀 AI Futures Lab Demonstration</b></div>',
                unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def main():
    inject_custom_css()
    render_hero()

    try:
        orchestrator = GuitarShoppingOrchestrator()

        # Sidebar: preferences + conversation list
        st.session_state.sidebar_prefs = render_sidebar_preferences()

        # Agent overview cards
        render_agent_overview()
        st.markdown("---")

        # Chat interface (must be at top level – cannot live inside tabs)
        render_chat_interface(orchestrator)

        # About section below chat
        st.markdown("---")
        with st.expander("ℹ️ About This System", expanded=False):
            render_about_tab()

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error initializing system: {error_traceback}")
        print(error_traceback)
        st.error(f"❌ Error initializing system: {str(e)}")
        st.error("Full Error Details:")
        st.code(error_traceback, language="python")
        st.info("💡 Tip: Make sure you have set your Azure OpenAI credentials in the .env file")

    render_footer()


if __name__ == "__main__":
    main()
