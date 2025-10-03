import json
import sys
import os

# This line adds the project root to Python's path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.agents.search_and_filter_agent import saf_agent

def debug_search_agent():
    """
    A simplified test to debug the search agent with a basic query.
    """
    print("\n--- Debugging the Search & Filter Sub-Agent ---")
    
    # 1. Define a very simple, single research question.
    # We are bypassing the multi-step LLM query expansion for this test.
    input_data = {
        "research_questions": [
            "Artificial Intelligence" 
        ]
    }
    
    print(f"Invoking agent with simple query: {input_data['research_questions']}")
    
    # 2. Invoke the agent directly
    # The agent will still run its *internal* query expansion on this simple phrase.
    result = saf_agent.invoke(input_data)
    
    # 3. Check the final result
    if result and result.get("filtered_papers"):
        papers = result["filtered_papers"]
        print(f"\n✅ SUCCESS: Agent found {len(papers)} papers.")
        if papers:
            print("--- Example Paper ---")
            print(json.dumps(papers[0], indent=2))
    else:
        print("\n❌ FAILURE: Agent still found 0 papers, even with a simple query.")

if __name__ == "__main__":
    debug_search_agent()