"""
Recommendation Agent - Recommends guitars based on customer preferences
"""
from typing import Dict, Any, List
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from utils import get_llm, get_rag_system, format_agent_response


class RecommendationAgent:
    """Agent responsible for recommending guitars based on preferences"""
    
    def __init__(self):
        self.llm = get_llm()
        self.rag = get_rag_system()
        self.name = "Recommendation Agent"
        self.system_prompt = """You are a friendly guitar matchmaker at a guitar shop.
Talk to the customer naturally, like a real person would. Understand what they need
from their words – their playing style, experience, what kind of music they enjoy,
their budget, and any other preferences they mention in conversation.
Make personalised recommendations that are practical, well-justified, and feel human.
CRITICAL: You MUST recommend ONLY guitar brands and models found in the provided Knowledge Base Context.
You MAY use your general expertise to determine which models best fit the user's needs.
Do NOT invent recommendations or use external knowledge to recommend brands not in the context.
Always respond in complete sentences and paragraphs – never cut off mid-sentence."""
    
    def recommend_guitars(self, 
                          preferences: Dict[str, Any],
                          history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate personalized guitar recommendations with conversation context"""
        
        # Build preference summary
        preference_summary = self._build_preference_summary(preferences)
        
        # Retrieve relevant information
        context = self.rag.retrieve_with_context(preference_summary, k=5)
        
        # Format history for the prompt
        formatted_history = ""
        if history:
            formatted_history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt + "\n\nCRITICAL: If the retrieved catalog data below is empty or doesn't contain relevant guitars, politely inform the customer that we don't have those specific models in our current catalog instead of making them up."),
            ("user", f"""Previous Conversation:
{{history}}

Please recommend the best guitars for this customer based on their preferences:

CUSTOMER PREFERENCES from Sidebar:
{{preference_summary}}

KNOWLEDGE BASE CONTEXT from Catalog:
{{context}}

Provide 3-5 specific guitar recommendations (ONLY from the catalog context above) with:
1. Guitar model/type name
2. Why it matches their preferences
3. Key features that align with their needs
4. Approximate price range
5. Best use cases for this guitar""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "preference_summary": preference_summary,
            "context": context,
            "history": formatted_history
        })
        
        return format_agent_response(
            self.name,
            response.content,
            {
                "preferences": preferences,
                "recommendation_type": "personalized",
                "recommendation_count": "3-5",
                "knowledge_base_used": True
            }
        )
    
    def compare_guitars(self, guitar_list: List[str]) -> Dict[str, Any]:
        """Compare multiple guitars based on customer request"""
        
        guitars_text = ", ".join(guitar_list)
        query = f"Compare these guitars: {guitars_text}"
        
        context = self.rag.retrieve_with_context(query, k=6)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", """Please provide a detailed comparison of these guitars:
{guitars_text}

KNOWLEDGE BASE CONTEXT:
{context}

Create a structured comparison covering:
1. Sound characteristics
2. Build quality
3. Price point
4. Best suited for
5. Pros and cons of each
6. Overall recommendation""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "guitars_text": guitars_text,
            "context": context
        })
        
        return format_agent_response(
            self.name,
            response.content,
            {
                "guitars_compared": guitar_list,
                "comparison_count": len(guitar_list),
                "recommendation_type": "comparative"
            }
        )
    
    def analyze_use_case(self, use_case: str, budget: str = None, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Analyze use case and recommend suitable guitars with context"""
        
        query = f"Guitars suitable for {use_case}" + (f" within {budget} budget" if budget else "")
        context = self.rag.retrieve_with_context(query, k=5)
        
        budget_text = f'Budget: {budget}' if budget else 'Budget: Not specified'
        
        # Format history for the prompt
        formatted_history = ""
        if history:
            formatted_history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt + "\n\nCRITICAL: Carefully scan the 'KNOWLEDGE BASE CONTEXT' below for the exact phrase 'Skill Level: Beginner' or 'Beginner-friendly response'. If you see these, emphasize them as the best choices for beginner users. Mention ONLY specific guitar names and brands found in the provided catalog data."),
            ("user", """Previous Conversation:
{history}

A customer is looking for a guitar for this specific use case:
Use Case: {use_case}
{budget_text}

KNOWLEDGE BASE CONTEXT:
{context}

Based on this use case, recommend the most suitable guitars with:
1. Specific model recommendations (ONLY from the context above)
2. Why they're perfect for this use case
3. Key features to look for
4. Alternative options if available
5. What to avoid

If no suitable guitars are found in the catalog context, admit that we don't have exact matches.""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "use_case": use_case,
            "budget_text": budget_text,
            "context": context,
            "history": formatted_history
        })
        
        return format_agent_response(
            self.name,
            response.content,
            {
                "use_case": use_case,
                "budget": budget,
                "recommendation_type": "use_case_based"
            }
        )
    
    @staticmethod
    def _build_preference_summary(preferences: Dict[str, Any]) -> str:
        """Build a summary of customer preferences"""
        summary = []
        
        if "budget" in preferences:
            summary.append(f"- Budget: {preferences['budget']}")
        if "skill_level" in preferences:
            summary.append(f"- Skill Level: {preferences['skill_level']}")
        if "music_style" in preferences:
            summary.append(f"- Music Style: {preferences['music_style']}")
        if "guitar_type" in preferences:
            summary.append(f"- Preferred Type: {preferences['guitar_type']}")
        if "features" in preferences:
            summary.append(f"- Desired Features: {', '.join(preferences['features'])}")
        if "use_case" in preferences:
            summary.append(f"- Use Case: {preferences['use_case']}")
        if "other_considerations" in preferences:
            summary.append(f"- Special Considerations: {preferences['other_considerations']}")
        
        return "\n".join(summary) if summary else "No specific preferences provided"
