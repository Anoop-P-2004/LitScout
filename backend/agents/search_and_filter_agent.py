import os
import time
import requests
from dotenv import load_dotenv
from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. CONFIGURATION ---
load_dotenv()
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not SEMANTIC_SCHOLAR_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("One or more required API keys are not set in the .env file.")

llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest", google_api_key=GOOGLE_API_KEY)


# --- 2. SIMPLIFIED QUERY GENERATION LOGIC ---
def generate_simple_query(research_question: str) -> str:
    """
    Uses an LLM to distill a research question into a simple keyword query.
    """
    prompt = (
        "You are an expert academic researcher. Rephrase the following research question "
        "into a simple, effective keyword query for a search engine like Google Scholar. "
        "Use only the 3-5 most important keywords or quoted phrases. "
        "Do not use Boolean operators like AND/OR. "
        f"Research Question: '{research_question}'"
    )
    try:
        response = llm.invoke(prompt)
        # Clean up the response, removing quotes if the LLM adds them
        return response.content.strip().replace('"', '')
    except Exception as e:
        print(f"Error during query generation: {e}")
        # Fallback to just using the original question
        return research_question


# --- 3. SEARCH LOGIC ---
def paper_search(state: dict) -> dict:
    """The main search node for the sub-graph."""
    print("---SUB-AGENT: Starting paper search---")
    
    research_questions = state.get("research_questions", [])
    all_unique_papers = {}

    for question in research_questions:
        print(f"\nProcessing question: {question}")
        simple_query = generate_simple_query(question)
        print(f"Simplified query: {simple_query}")
        
        current_offset = 0
        limit_per_request = 100
        max_papers_per_question = 200 # Limit to 200 papers per question to be efficient

        while len(all_unique_papers) < (len(all_unique_papers) + max_papers_per_question):
            params = {
                'query': simple_query,
                'fields': 'title,abstract,authors.name,year,paperId',
                'limit': limit_per_request,
                'offset': current_offset
            }
            try:
                response = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", params=params, headers={'x-api-key': SEMANTIC_SCHOLAR_API_KEY})
                response.raise_for_status()
                results = response.json()
                batch_papers = results.get('data', [])
                
                if not batch_papers: break
                
                for paper in batch_papers:
                    all_unique_papers[paper['paperId']] = paper
                
                print(f"Fetched {len(batch_papers)} papers. Total unique papers now: {len(all_unique_papers)}")
                
                current_offset += limit_per_request
                if current_offset >= results.get('total', 0) or current_offset >= 1000: # Respect API's 1000 result limit
                    break
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                print(f"Error during paper search: {e}")
                break
        time.sleep(2) # Polite delay between different question searches

    final_paper_list = list(all_unique_papers.values())
    print(f"---SUB-AGENT: Found a total of {len(final_paper_list)} unique papers---")
    return {"raw_papers": final_paper_list, "filtered_papers": final_paper_list}


# --- 4. SUB-GRAPH DEFINITION ---
class SearchFilterState(TypedDict):
    research_questions: List[str]
    raw_papers: List[Dict]
    filtered_papers: List[Dict]

workflow = StateGraph(SearchFilterState)
workflow.add_node("paper_search", paper_search)
workflow.set_entry_point("paper_search")
workflow.add_edge("paper_search", END)
saf_agent = workflow.compile()