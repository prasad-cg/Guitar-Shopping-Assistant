"""
Main entry point for Guitar Shopping Multi-Agent System
Run this file to start the Streamlit interface or use CLI mode
"""
import sys
import argparse
from agents.orchestrator import GuitarShoppingOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Guitar Shopping Multi-Agent AI System")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="streamlit",
        choices=["streamlit", "interactive", "cli"],
        help="Run mode: streamlit (UI), interactive (CLI), or cli (single query)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process (for CLI mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "streamlit":
        print("🎸 Starting Streamlit UI...")
        print("To start the app, run: streamlit run ui/streamlit_app.py")
        import subprocess
        import os
        os.chdir(os.path.dirname(__file__))
        subprocess.run(["streamlit", "run", "ui/streamlit_app.py"])
    
    elif args.mode == "interactive":
        orchestrator = GuitarShoppingOrchestrator()
        orchestrator.interactive_mode()
    
    elif args.mode == "cli":
        if not args.query:
            print("❌ Error: --query is required for CLI mode")
            sys.exit(1)
        
        orchestrator = GuitarShoppingOrchestrator()
        result = orchestrator.process_customer_query(args.query)
        print("\n" + "=" * 60)
        print(result["final_response"])
        print("=" * 60)


if __name__ == "__main__":
    main()
