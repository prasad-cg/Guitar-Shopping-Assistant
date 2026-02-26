"""
Negotiator Agent - Handles pricing and deals
"""
from typing import Dict, Any
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from utils import get_llm_with_custom_temp, get_rag_system, format_agent_response


class NegotiatorAgent:
    """Agent responsible for price negotiation and dealing"""
    
    def __init__(self):
        # Use higher temperature for negotiation creativity
        self.llm = get_llm_with_custom_temp(temperature=0.8)
        self.rag = get_rag_system()
        self.name = "Price Negotiator Agent"
        self.system_prompt = """You are a friendly, savvy salesperson at a guitar shop.
You naturally talk about pricing, deals, and value with customers —
just like a real person behind the counter would.
You understand pricing dynamics, can suggest bundles, mention current deals,
and find creative ways to help the customer feel great about their purchase.
Be warm, relatable, and genuinely helpful.
CRITICAL: You MUST base all models strictly on the provided KNOWLEDGE BASE CONTEXT.
Do not invent outside brands or models not present in the context.
Always respond in complete sentences and paragraphs – never cut off mid-sentence."""
    
    def handle_price_inquiry(self, 
                             guitar_model: str, 
                             quantity: int = 1) -> Dict[str, Any]:
        """Handle customer price inquiries"""
        
        query = f"Price range for {guitar_model}"
        context = self.rag.retrieve_with_context(query, k=3)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", """A customer is inquiring about the price for:
Guitar Model: {guitar_model}
Quantity: {quantity}

KNOWLEDGE BASE CONTEXT:
{context}

Please provide:
1. The typical price range for this guitar
2. Factors that affect pricing
3. Any current promotions or deals available
4. Volume discounts if applicable
5. Bundle options that could improve value""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "guitar_model": guitar_model,
            "quantity": str(quantity),
            "context": context,
        })
        
        return format_agent_response(
            self.name,
            response.content,
            {
                "guitar_model": guitar_model,
                "quantity": quantity,
                "inquiry_type": "price_inquiry"
            }
        )
    
    def negotiate_discount(self, 
                          guitars: list, 
                          customer_budget: str,
                          reason: str = None) -> Dict[str, Any]:
        """Negotiate discounts based on customer context"""
        
        guitars_text = ", ".join(guitars)
        context = self.rag.retrieve_with_context(f"Pricing and discounts for {guitars_text}", k=4)
        
        reason_text = f"\nCustomer's reason: {reason}" if reason else ""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", """Please help find the best deal for this customer:

Guitars of Interest: {guitars_text}
Customer Budget: {customer_budget}
{reason_text}

KNOWLEDGE BASE CONTEXT:
{context}

Please suggest:
1. Best possible pricing for these guitars
2. Available discount strategies
3. Bundle combinations that fit the budget
4. Trade-in opportunities
5. Extended warranty/service packages to add value
6. Timeline suggestions for best deals""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "guitars_text": guitars_text,
            "customer_budget": customer_budget,
            "reason_text": reason_text,
            "context": context
        })
        
        return format_agent_response(
            self.name,
            response.content,
            {
                "guitars": guitars,
                "customer_budget": customer_budget,
                "reason": reason,
                "negotiation_type": "discount_negotiation"
            }
        )
    
    def create_custom_deal(self, 
                          selections: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom deal package for customer"""
        
        guitars = selections.get("guitars", [])
        accessories = selections.get("accessories", [])
        services = selections.get("services", [])
        budget = selections.get("budget", "Not specified")
        
        guitars_text = ', '.join(guitars) if guitars else 'None selected'
        accessories_text = ', '.join(accessories) if accessories else 'None'
        services_text = ', '.join(services) if services else 'None'
        
        context = self.rag.retrieve_with_context(
            f"Guitar deals and bundles for {', '.join(guitars) if guitars else 'guitars'}", 
            k=5
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", """Please create a custom deal package for this customer:

CUSTOMER SELECTIONS:
Guitars: {guitars_text}
Accessories: {accessories_text}
Services: {services_text}
Total Budget: {budget}

KNOWLEDGE BASE CONTEXT:
{context}

Please create:
1. A complete package proposal with itemized pricing
2. Total package price and savings vs. individual purchases
3. Payment plan options if needed
4. Warranty and service inclusions
5. Any additional value-adds to sweeten the deal
6. Clear next steps for the customer""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "guitars_text": guitars_text,
            "accessories_text": accessories_text,
            "services_text": services_text,
            "budget": budget,
            "context": context
        })
        
        return format_agent_response(
            self.name,
            response.content,
            {
                "selections": selections,
                "negotiation_type": "custom_deal"
            }
        )
    
    def handle_customer_concern(self, concern: str, 
                               related_guitar: str = None) -> Dict[str, Any]:
        """Handle customer concerns about pricing or value"""
        
        search_query = f"Pricing concerns and solutions for {related_guitar}" if related_guitar else concern
        context = self.rag.retrieve_with_context(search_query, k=4)
        
        related_guitar_text = f'\nRelated to: {related_guitar}' if related_guitar else ''
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", """A customer has expressed this concern about pricing/value:
{concern}
{related_guitar_text}

KNOWLEDGE BASE CONTEXT:
{context}

Please:
1. Acknowledge and validate their concern
2. Provide relevant pricing/value information
3. Suggest alternatives or solutions
4. Explain what they get for the price
5. Offer next steps or compromise options""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "concern": concern,
            "related_guitar_text": related_guitar_text,
            "context": context
        })
        
        return format_agent_response(
            self.name,
            response.content,
            {
                "concern": concern,
                "related_guitar": related_guitar,
                "negotiation_type": "concern_handling"
            }
        )
