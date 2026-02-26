"""
Quick start guide and testing utilities for the multi-agent system
"""
from agents.orchestrator import GuitarShoppingOrchestrator
from utils import get_guitar_categories, get_playing_styles, get_price_ranges


def run_demo_queries():
    """Run a series of demo queries to showcase the system"""
    
    orchestrator = GuitarShoppingOrchestrator()
    
    demo_queries = [
        {
            "query": "I'm a beginner wanting to learn acoustic guitar. What's a good starting option?",
            "preferences": {
                "skill_level": "Beginner",
                "guitar_type": "Acoustic Guitars",
                "music_style": ["Folk", "Strumming"],
                "budget": "Budget ($0-500)"
            }
        },
        {
            "query": "What's the difference between a Stratocaster and a Telecaster?",
            "preferences": {
                "skill_level": "Intermediate",
                "guitar_type": "Electric Guitars"
            }
        },
        {
            "query": "I have $2000 to spend on a professional quality electric guitar for rock music. What do you recommend?",
            "preferences": {
                "skill_level": "Professional",
                "music_style": ["Rock"],
                "guitar_type": "Electric Guitars",
                "budget": "Premium ($1500-5000)"
            }
        },
        {
            "query": "Can you negotiate a better price on a Les Paul-style guitar?",
            "preferences": {
                "budget": "Mid-Range ($500-1500)"
            }
        }
    ]
    
    print("🎸 Guitar Shopping Multi-Agent System - Demo Queries")
    print("=" * 70)
    
    for idx, demo in enumerate(demo_queries, 1):
        print(f"\n{'='*70}")
        print(f"DEMO {idx}: {demo['query'][:50]}...")
        print(f"{'='*70}")
        
        result = orchestrator.process_customer_query(
            demo["query"],
            demo.get("preferences", {})
        )
        
        print(f"\n✅ Response received:")
        print(result["final_response"][:1000])
        print("\n[Response truncated for demo purposes]")
        
        input("\nPress Enter to continue to next demo...")


def display_system_info():
    """Display information about the system"""
    
    print("\n🎸 Guitar Shopping Multi-Agent System - System Info")
    print("=" * 70)
    
    print("\n📚 Available Guitar Categories:")
    for cat in get_guitar_categories():
        print(f"  • {cat}")
    
    print("\n🎵 Available Music Styles:")
    styles = get_playing_styles()
    for i, style in enumerate(styles, 1):
        print(f"  {i}. {style}", end="  ")
        if i % 3 == 0:
            print()
    print()
    
    print("\n💰 Price Ranges:")
    for range_name, (min_price, max_price) in get_price_ranges().items():
        if max_price == float('inf'):
            print(f"  • {range_name}: ${min_price}+")
        else:
            print(f"  • {range_name}: ${min_price} - ${max_price}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys
    
    print("\n🎸 Quick Start and Testing")
    print("=" * 70)
    print("Options:")
    print("1. Run demo queries (requires OpenAI API)")
    print("2. Display system information")
    print("=" * 70)
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == "1":
        try:
            run_demo_queries()
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Make sure your OPENAI_API_KEY is set in the .env file")
    elif choice == "2":
        display_system_info()
    else:
        print("Invalid option")
