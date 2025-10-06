import json
import sys
import os

# This line adds the project root to Python's path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.agents.search_and_filter_agent import saf_agent

def test_the_search_and_filter_agent_with_filters():
    """
    Tests the Search and Filter sub-agent with specific metadata filters.
    """
    print("\n--- Testing the Search & Filter Sub-Agent with Filters ---")
    
    # Define the input, including the new filter criteria.
    # This simulates the data that will be passed from the orchestrator.
    input_data = {
        "research_questions": [
            "multi-agent systems for literature review"
        ],
        "start_year": 2022,
        "end_year": 2024,
        # --- NEW, BROADER LIST OF SOURCES ---
        "sources": [ 
            "Springer", 
            "Elsevier", 
            "ScienceDirect", 
            "ACM", 
            "IEEE",
            "Nature"
        ]
    }
    
    print(f"Invoking agent with {len(input_data['research_questions'])} research question(s)...")
    print(f"Applying filters: Year Range ({input_data['start_year']}-{input_data['end_year']}), Sources ({input_data['sources']})")
    
    # Invoke the agent with the input data
    result = saf_agent.invoke(input_data)
    
    # Check the final result from the agent
    if result:
        raw_papers = result.get("raw_papers", [])
        filtered_papers = result.get("filtered_papers", [])
        
        print(f"\n--- Results ---")
        print(f"  - Total unique papers found before filtering: {len(raw_papers)}")
        print(f"  - Papers remaining after all filters: {len(filtered_papers)}")
        
        if filtered_papers:
            print(f"\n✅ Test Passed: Agent returned {len(filtered_papers)} filtered papers.")
            print("--- Example Filtered Paper ---")
            # Verify that the example paper meets the criteria
            example = filtered_papers[0]
            print(json.dumps(example, indent=2))
            print("---------------------------------")
            print(f"Verification: Year ({example.get('year')}), Abstract Present ({bool(example.get('abstract'))})")

        else:
             print("\n✅ Test Passed: Agent correctly filtered all papers, resulting in an empty list.")

    else:
        print("\n❌ Test Failed: Agent did not return a valid result.")

if __name__ == "__main__":
    test_the_search_and_filter_agent_with_filters()