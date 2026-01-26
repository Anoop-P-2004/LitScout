import sys
import os
import json

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.agents.search_and_filter_agent import saf_agent
from backend.agents.screening_agent import screening_agent
from backend.utils.fetch_papers import fetch_papers


def test_pipeline_until_paper_fetching():
    """
    End-to-end test:
    Search & Filter -> Screening -> Paper Fetching
    """

    print("\n=== END-TO-END PIPELINE TEST (UNTIL PAPER FETCHING) ===\n")

    # -----------------------------
    # 1. SEARCH & FILTER
    # -----------------------------
    search_input = {
        "research_questions": [
            "multi agent system for automated literature review"
        ],
        "start_year": 2021,
        "end_year": 2024,
        "sources": [
            "IEEE",
            "ACM",
            "Springer",
            "Elsevier",
            "Wiley",
            "Taylor & Francis",
            "Oxford Academic",
            "Cambridge University Press",
            "Nature",
            "Science",
            "PNAS",
            "IET",
            "ASME",
            "ASCE",
            "AIAA"
        ],
        "raw_papers": [],
        "filtered_papers": []
    }

    print("🔍 Running Search & Filter Agent...")
    saf_result = saf_agent.invoke(search_input)

    filtered_papers = saf_result.get("filtered_papers", [])
    raw_papers = saf_result.get("raw_papers", [])

    print(f"   Raw papers found: {len(raw_papers)}")
    print(f"   Papers after filtering: {len(filtered_papers)}")

    if not filtered_papers:
        print("\n⚠️ No papers found after filtering. Test ends early.")
        return

    # -----------------------------
    # 2. SCREENING
    # -----------------------------
    screening_input = {
        "filtered_papers": filtered_papers,
        "research_questions": search_input["research_questions"],
        "inclusion_criteria": "multi-agent systems, automated literature review, AI-based screening",
        "exclusion_criteria": "non-academic articles, opinion pieces",

        # Adaptive thresholds (screening agent will set these)
        "keyword_high_threshold": 0.0,
        "keyword_medium_threshold": 0.0,
        "tfidf_threshold": 0.0,
        "use_llm_screening": False,

        # Internal placeholders
        "metadata_filtered_papers": [],
        "high_relevance_papers": [],
        "medium_relevance_papers": [],
        "borderline_papers": [],
        "screened_papers": [],
        "screening_results": {}
    }

    print("\n🧠 Running Screening Agent...")
    screening_result = screening_agent.invoke(screening_input)

    screened_papers = screening_result.get("screened_papers", [])

    print(f"   Papers after screening: {len(screened_papers)}")

    if not screened_papers:
        print("\n⚠️ No papers passed screening. Test ends early.")
        return

    # Show one example paper
    print("\n--- Example Screened Paper ---")
    print(json.dumps(screened_papers[0], indent=2)[:1500])
    print("--------------------------------")

    # -----------------------------
    # 3. PAPER FETCHING
    # -----------------------------
    print("\n📥 Fetching papers (PDFs / texts)...")
    fetch_result = fetch_papers(screened_papers)

    print("\n=== PAPER FETCHING RESULTS ===")
    print(json.dumps(fetch_result, indent=2))

    extracted_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "backend",
        "utils",
        "Extracted Papers"
    )

    if os.path.exists(extracted_dir):
        print(f"\n📁 Extracted Papers directory exists:")
        print(os.path.abspath(extracted_dir))
    else:
        print("\n❌ Extracted Papers directory not found!")

    print("\n✅ END-TO-END TEST COMPLETED\n")


if __name__ == "__main__":
    test_pipeline_until_paper_fetching()
