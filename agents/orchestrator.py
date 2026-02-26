"""
Multi-Agent Orchestrator using LangGraph
Orchestrates the workflow between Information, Recommendation, and Negotiator agents.

KEY DESIGN:
  * Intent-based routing — only the agents relevant to the customer's message
    are activated.  If the customer is just asking about brands / info, the
    negotiator stays silent.  If they are asking about prices, the info +
    negotiator respond but the recommendation agent stays quiet.
  * Agents behave like friendly guitar-shop staff and converse naturally.
  * The sidebar preferences are NEVER forwarded to agents.
"""
from typing import Dict, List, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from agents.information_agent import InformationAgent
from agents.recommendation_agent import RecommendationAgent
from agents.negotiator_agent import NegotiatorAgent
from utils import format_agent_response


# ── Intent constants ──────────────────────────────────────────────────────
INTENT_INFO        = "information_request"
INTENT_RECOMMEND   = "recommendation_request"
INTENT_PRICE       = "price_inquiry"
INTENT_COMPARE     = "comparison_request"
INTENT_GENERAL     = "general_inquiry"

# Which agents are needed per intent
INTENT_AGENT_MAP: Dict[str, List[str]] = {
    INTENT_INFO:      ["information"],
    INTENT_RECOMMEND: ["information", "recommendation"],
    INTENT_PRICE:     ["information", "negotiation"],
    INTENT_COMPARE:   ["information", "recommendation"],
    INTENT_GENERAL:   ["information"],
}


class AgentState(TypedDict):
    """State object for the agent workflow"""
    customer_query: str
    conversation_history: List[Dict[str, str]]
    intent: str
    active_agents: List[str]            # which agents should run
    information_response: Dict[str, Any]
    recommendation_response: Dict[str, Any]
    negotiation_response: Dict[str, Any]
    final_response: str
    current_stage: str
    workflow_complete: bool
    preferences: Dict[str, Any]


class GuitarShoppingOrchestrator:
    """Multi-agent orchestrator for guitar shopping assistance"""

    def __init__(self):
        self.information_agent = InformationAgent()
        self.recommendation_agent = RecommendationAgent()
        self.negotiator_agent = NegotiatorAgent()
        self.graph = self._build_workflow_graph()

    # ──────────────────────────────────────────────────────────────────────
    # Graph construction
    # ──────────────────────────────────────────────────────────────────────
    def _build_workflow_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("parse_intent", self._parse_customer_intent)
        workflow.add_node("information_node", self._run_information_agent)
        workflow.add_node("recommendation_node", self._run_recommendation_agent)
        workflow.add_node("negotiation_node", self._run_negotiation_agent)
        workflow.add_node("synthesize_response", self._synthesize_response)

        # Entry
        workflow.set_entry_point("parse_intent")

        # After intent parsing → always go to information first
        workflow.add_edge("parse_intent", "information_node")

        # After information → conditional: recommendation or negotiation or synthesize
        workflow.add_conditional_edges(
            "information_node",
            self._route_after_information,
            {
                "recommendation_node": "recommendation_node",
                "negotiation_node": "negotiation_node",
                "synthesize_response": "synthesize_response",
            },
        )

        # After recommendation → conditional: negotiation or synthesize
        workflow.add_conditional_edges(
            "recommendation_node",
            self._route_after_recommendation,
            {
                "negotiation_node": "negotiation_node",
                "synthesize_response": "synthesize_response",
            },
        )

        # After negotiation → always synthesize
        workflow.add_edge("negotiation_node", "synthesize_response")
        workflow.add_edge("synthesize_response", END)

        return workflow.compile()

    # ──────────────────────────────────────────────────────────────────────
    # Routing helpers
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _route_after_information(state: AgentState) -> str:
        active = state.get("active_agents", [])
        if "recommendation" in active:
            return "recommendation_node"
        if "negotiation" in active:
            return "negotiation_node"
        return "synthesize_response"

    @staticmethod
    def _route_after_recommendation(state: AgentState) -> str:
        active = state.get("active_agents", [])
        if "negotiation" in active:
            return "negotiation_node"
        return "synthesize_response"

    # ──────────────────────────────────────────────────────────────────────
    # Nodes
    # ──────────────────────────────────────────────────────────────────────
    def _parse_customer_intent(self, state: AgentState) -> AgentState:
        query = state["customer_query"]
        intent = self._classify_intent(query)
        active_agents = INTENT_AGENT_MAP.get(intent, ["information"])

        state["intent"] = intent
        state["active_agents"] = active_agents
        state["current_stage"] = "intent_parsing"
        state["conversation_history"].append({
            "role": "system",
            "content": f"Detected intent: {intent} → activating: {', '.join(active_agents)}"
        })
        return state

    def _run_information_agent(self, state: AgentState) -> AgentState:
        response = self.information_agent.process_information_request(
            state["customer_query"],
            history=state["conversation_history"][:-1] # pass history excluding current query
        )
        state["information_response"] = response
        state["current_stage"] = "information_gathering"
        state["conversation_history"].append({
            "role": "agent",
            "agent": "Information Agent",
            "content": response["content"],
        })
        return state

    def _run_recommendation_agent(self, state: AgentState) -> AgentState:
        # Use both the query, preferences and history
        response = self.recommendation_agent.recommend_guitars(
            preferences=state.get("preferences", {}),
            history=state["conversation_history"][:-1]
        )
        state["recommendation_response"] = response
        state["current_stage"] = "recommendation"
        state["conversation_history"].append({
            "role": "agent",
            "agent": "Recommendation Agent",
            "content": response["content"],
        })
        return state

    def _run_negotiation_agent(self, state: AgentState) -> AgentState:
        response = self.negotiator_agent.handle_price_inquiry(
            state["customer_query"],
            quantity=1,
        )
        state["negotiation_response"] = response
        state["current_stage"] = "negotiation"
        state["conversation_history"].append({
            "role": "agent",
            "agent": "Price Negotiator Agent",
            "content": response["content"],
        })
        return state

    def _synthesize_response(self, state: AgentState) -> AgentState:
        """Build a single, clean response using ONLY the agents that actually ran."""
        active = state.get("active_agents", [])
        sections: List[str] = []

        if "information" in active and state.get("information_response"):
            content = state["information_response"].get("content", "")
            if content:
                sections.append(f"### 📚 Guitar Info\n{content}")

        if "recommendation" in active and state.get("recommendation_response"):
            content = state["recommendation_response"].get("content", "")
            if content:
                sections.append(f"### ✨ Recommendations\n{content}")

        if "negotiation" in active and state.get("negotiation_response"):
            content = state["negotiation_response"].get("content", "")
            if content:
                sections.append(f"### 💰 Pricing & Deals\n{content}")

        body = "\n\n".join(sections) if sections else "I'm not sure how to help with that — could you rephrase?"

        final_response = f"""{body}

---
*Feel free to ask me more — I'm here to help you find your perfect guitar!* 🎸
"""
        state["final_response"] = final_response
        state["current_stage"] = "complete"
        state["workflow_complete"] = True
        return state

    # ──────────────────────────────────────────────────────────────────────
    # Intent classifier
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _classify_intent(query: str) -> str:
        q = query.lower()

        price_kw = ["price", "cost", "expensive", "cheap", "discount", "deal",
                     "afford", "budget", "how much", "negotiate", "offer"]
        rec_kw   = ["recommend", "suggest", "best", "suitable", "good for",
                     "which guitar", "what should", "help me choose", "pick",
                     "looking for", "want to buy", "buying", "need a guitar",
                     "purchase", "buy", "shop"]
        info_kw  = ["tell me", "explain", "what is", "how does", "difference",
                     "type", "brand", "brands", "models", "features",
                     "specification", "specs", "about", "info", "information",
                     "browse", "show", "list", "available"]
        comp_kw  = ["compare", "vs", "versus", "between", "or"]

        # Score each intent
        scores = {
            INTENT_PRICE:     sum(1 for kw in price_kw if kw in q),
            INTENT_RECOMMEND: sum(1 for kw in rec_kw   if kw in q),
            INTENT_INFO:      sum(1 for kw in info_kw  if kw in q),
            INTENT_COMPARE:   sum(1 for kw in comp_kw  if kw in q),
        }

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best] > 0:
            return best
        return INTENT_GENERAL  # fallback → info agent only

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    def process_customer_query(
        self, 
        query: str, 
        history: List[Dict[str, str]] = None,
        preferences: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process customer query using conversation history and preferences."""
        
        # Initialize history if none provided
        if history is None:
            history = []
            
        # Add current query to history
        current_history = history + [{"role": "user", "content": query}]

        initial_state: AgentState = {
            "customer_query": query,
            "conversation_history": current_history,
            "intent": "",
            "active_agents": [],
            "information_response": {},
            "recommendation_response": {},
            "negotiation_response": {},
            "final_response": "",
            "current_stage": "initialized",
            "workflow_complete": False,
            "preferences": preferences or {},
        }

        result = self.graph.invoke(initial_state)

        # Only return agent responses that actually ran
        agents_involved = []
        for key in ("information_response", "recommendation_response", "negotiation_response"):
            resp = result.get(key, {})
            if resp and resp.get("content"):
                agents_involved.append(resp)

        return {
            "status": "success",
            "final_response": result["final_response"],
            "conversation_history": result["conversation_history"],
            "agents_involved": agents_involved,
            "metadata": {
                "intent": result.get("intent", ""),
                "active_agents": result.get("active_agents", []),
                "workflow_complete": result["workflow_complete"],
                "final_stage": result["current_stage"],
            },
        }

    def interactive_mode(self):
        """Run orchestrator in interactive mode"""
        print("🎸 Welcome to the AI Guitar Shopping Assistant!")
        print("=" * 60)
        while True:
            print("\n📝 Enter your query (or 'quit'):")
            q = input("> ").strip()
            if q.lower() == "quit":
                print("Thanks for visiting! 🎸")
                break
            if not q:
                continue
            print("\n🤖 Processing...\n")
            result = self.process_customer_query(q)
            print(result["final_response"])
            print("\n" + "=" * 60)
