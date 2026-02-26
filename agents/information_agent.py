"""
Information Agent - Provides guitar information using RAG
"""
from typing import Dict, Any, List
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import HumanMessage, AIMessage
from utils import get_llm, get_rag_system, format_agent_response


class InformationAgent:
    """Agent responsible for providing guitar information"""
    
    def __init__(self):
        self.llm = get_llm()
        self.rag = get_rag_system()
        self.name = "Information Agent"
        self.system_prompt = """You are a friendly, knowledgeable guitar shop assistant.
Talk to the customer naturally, like a real person working at a guitar store.
When a customer asks about guitars, ALWAYS mention specific guitar names, brands, and models
from the knowledge base — don't give vague generic answers.
If the customer hasn't shared enough details yet (like their budget, skill level, or preferred genre),
ask friendly follow-up questions to understand them better — just like a real shop assistant would.
Keep your tone warm, conversational, and concise.
CRITICAL: You MUST only discuss guitars present in the provided knowledge base context.
Never invent brands or models not in the context.
Always respond in complete sentences – never cut off mid-sentence."""
    
    def process_information_request(self, customer_query: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process information requests from customers with conversation context"""
        # Retrieve relevant information from knowledge base
        context = self.rag.retrieve_with_context(customer_query, k=5)
        
        # Format history for the prompt
        formatted_history = ""
        if history:
            formatted_history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])

        # Create the message for the LLM with proper variable placeholders
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt + "\n\nCRITICAL: Carefully scan the 'Guitar Catalog Excerpts' below for the exact phrase 'Skill Level: Beginner' or 'Beginner-friendly response'. If you see these, emphasize them as the best choices. Mention ONLY specific guitar names and brands found in the provided catalog data. If NO entries mention beginners, only then state we don't have them."),
            ("user", """Previous Conversation:
{history}

Customer says: {customer_query}
            
Our Guitar Catalog Data:
{context}

Respond naturally as a shop assistant. Mention ONLY specific guitar names and brands from the catalog data provided above. 
If the information is not in the catalog, say you don't have that information.""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "customer_query": customer_query,
            "context": context,
            "history": formatted_history
        })
        
        return format_agent_response(
            self.name,
            response.content,
            {
                "query": customer_query,
                "knowledge_base_used": True,
                "retrieved_chunks": 5
            }
        )
    
    def get_guitar_recommendations(self, category: str) -> Dict[str, Any]:
        """Get guitar recommendations for a specific category"""
        query = f"What guitars are available in the {category} category? Provide detailed information."
        
        context = self.rag.retrieve_with_context(query, k=5)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", """Please provide comprehensive information about guitars in the '{category}' category.

Knowledge Base Context:
{context}

Include details about different options, their characteristics, and suitable use cases.""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "category": category,
            "context": context,
        })
        
        return format_agent_response(
            self.name,
            response.content,
            {
                "category": category,
                "request_type": "category_overview"
            }
        )
    
    def answer_specification_question(self, question: str) -> Dict[str, Any]:
        """Answer detailed specification questions"""
        context = self.rag.retrieve_with_context(question, k=4)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", """A customer has a specific question about guitar specifications:
{question}

Knowledge Base Context:
{context}

Provide a detailed, technical but understandable answer.""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "question": question,
            "context": context,
        })
        
        return format_agent_response(
            self.name,
            response.content,
            {
                "question": question,
                "request_type": "specification_inquiry"
            }
        )
