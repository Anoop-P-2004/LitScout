import json
import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool, render_text_description
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# --- AGENT IMPORT AND MOCKING ---
class MockAgent:
    def invoke(self, input_data):
        return {"status": "mock_response", "data": input_data}

saf = MockAgent()
screening_agent = MockAgent()
extraction_agent = MockAgent()
thematic_agent = MockAgent()
synthesis_agent = MockAgent()
qa_agent = MockAgent()
print("🤖 Mock agents created for development.")

try:
    from agents.search_and_filter_agent import saf_agent as saf
    print("✅ Real Search & Filter agent imported successfully.")
except ImportError as e:
    print(f"⚠️ Warning: Could not import a real agent, will use mock instead: {e}")


# --- Environment Setup ---
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please create a .env file.")

# ==============================================================================
# 1. STATE DEFINITION (UPDATED)
# ==============================================================================

class LitScoutState(BaseModel):
    """Complete state management for LitScout"""
    
    # Input
    user_prompt: str = ""
    
    # Planning
    plan: List[str] = Field(default_factory=list)
    current_step: int = 0
    next_agent: str = ""
    observations: List[str] = Field(default_factory=list)
    
    # Research Planning Phase (with new filters)
    research_questions: List[str] = Field(default_factory=list)
    inclusion_criteria: str = ""
    exclusion_criteria: str = ""
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    sources: Optional[List[str]] = Field(default_factory=list)
    
    # Search & Filter Phase
    raw_papers: List[Dict] = Field(default_factory=list)
    filtered_papers: List[Dict] = Field(default_factory=list)
    
    # Other phases remain the same
    screened_papers: List[Dict] = Field(default_factory=list)
    screening_results: Dict = Field(default_factory=dict)
    extracted_data: Dict = Field(default_factory=dict)
    extraction_results: Dict = Field(default_factory=dict)
    themes: List[Dict] = Field(default_factory=list)
    thematic_results: Dict = Field(default_factory=dict)
    synthesis_draft: str = ""
    synthesis_results: Dict = Field(default_factory=dict)
    quality_report: Dict = Field(default_factory=dict)
    quality_passed: bool = False
    final_report: str = ""
    workflow_complete: bool = False
    error_messages: List[str] = Field(default_factory=list)

# ==============================================================================
# 2. LLM and PARSER SETUP
# ==============================================================================

llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest")

class PlannerResponse(BaseModel):
    plan: List[str] = Field(description="The updated step-by-step plan.")
    next_agent: str = Field(description="The name of the tool to call next, or 'END'.")

parser = PydanticOutputParser(pydantic_object=PlannerResponse)

# ==============================================================================
# 3. AGENT TOOL DEFINITIONS (UPDATED)
# ==============================================================================

# --- NEW: Tool for parsing the initial user request ---
class RequestParameters(BaseModel):
    start_year: Optional[int] = Field(description="The starting year for the search, if specified.")
    end_year: Optional[int] = Field(description="The ending year for the search, if specified.")
    sources: Optional[List[str]] = Field(description="A list of specific publication sources, if mentioned.")

@tool
def parse_initial_request(user_prompt: str) -> Dict[str, Any]:
    """
    Parses the user's initial prompt to extract structured filter criteria. This is the very first step.
    """
    print("---TOOL: Parsing Initial User Request---")
    try:
        param_parser = PydanticOutputParser(pydantic_object=RequestParameters)
        prompt_template = """
        Analyze the user's request and extract the following filter criteria if they exist.
        - start_year: The beginning of a date range (e.g., "from 2020 onwards" means start_year is 2020).
        - end_year: The end of a date range. If only a start year is given, default this to the current year.
        - sources: A list of academic sources (e.g., "arXiv", "IEEE", "ScienceDirect").
        
        User Prompt: {user_prompt}
        Current Year: {current_year}

        {format_instructions}
        """
        prompt = ChatPromptTemplate.from_template(
            prompt_template,
            partial_variables={"format_instructions": param_parser.get_format_instructions()},
        )
        chain = prompt | llm | param_parser
        response = chain.invoke({
            "user_prompt": user_prompt,
            "current_year": datetime.now().year
        })
        result = response.model_dump()
        print(f"✅ Extracted Filters: {result}")
        return result
    except Exception as e:
        print(f"❌ Error parsing initial request: {e}. Using default filters.")
        return {}

class ResearchPlan(BaseModel):
    research_questions: List[str] = Field(description="List of 3-5 specific research questions.")
    inclusion_criteria: str = Field(description="Concise inclusion criteria for the literature review.")
    exclusion_criteria: str = Field(description="Concise exclusion criteria for the literature review.")

@tool
def formulate_research_plan(user_prompt: str) -> Dict[str, Any]:
    """
    Formulates a research plan based on the user_prompt. This is the first step.
    It generates research questions and defines inclusion/exclusion criteria using an LLM.
    """
    print("---TOOL: Formulating Research Plan---")
    # This tool's internal logic is unchanged
    try:
        plan_parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        prompt_template_string = "Based on the user's research prompt... {format_instructions}"
        prompt = ChatPromptTemplate.from_template(prompt_template_string, partial_variables={"format_instructions": plan_parser.get_format_instructions()})
        chain = prompt | llm | plan_parser
        response_obj = chain.invoke({"user_prompt": user_prompt})
        result = response_obj.model_dump()
        print(f"✅ Generated {len(result['research_questions'])} research questions")
        return result
    except Exception as e:
        print(f"❌ Error in formulate_research_plan: {e}")
        return {"research_questions": [f"Research question for: {user_prompt}"], "inclusion_criteria": "", "exclusion_criteria": "", "error": str(e)}

# --- MODIFIED: search_and_filter_tool now takes more arguments ---
@tool
def search_and_filter_tool(research_questions: List[str], start_year: Optional[int], end_year: Optional[int], sources: Optional[List[str]]) -> Dict[str, Any]:
    """
    Search and Filter Agent Tool - Delegates to the specialized SAF agent.
    The SAF agent has its own graph structure for complex search and filtering operations.
    """
    print("---TOOL: Search and Filter Agent---")
    try:
        saf_input = {
            "research_questions": research_questions,
            "start_year": start_year,
            "end_year": end_year,
            "sources": sources
        }
        print(f"🔍 Invoking SAF agent with {len(research_questions)} research questions and filters.")
        saf_result = saf.invoke(saf_input)
        print(f"✅ SAF agent completed successfully")
        return saf_result
    except Exception as e:
        print(f"❌ Error in search_and_filter_tool: {e}")
        return {"raw_papers": [], "filtered_papers": [], "error": str(e)}

# --- The rest of the tools (screening, etc.) remain as they were ---
@tool
def screening_tool(filtered_papers: List[Dict], research_questions: List[str], inclusion_criteria: str) -> Dict[str, Any]:
    """
    Screening Agent Tool - Delegates to the specialized screening agent.
    Performs detailed relevance screening of filtered papers based on their abstracts.
    """
    print("---TOOL: Screening Agent---")
    return screening_agent.invoke({"filtered_papers": filtered_papers})

# (... other mock tools ...)

tools = [
    parse_initial_request, # NEW
    formulate_research_plan, 
    search_and_filter_tool, 
    screening_tool,
    # ... extraction_tool, thematic_analysis_tool, etc.
]
tool_map = {t.name: t for t in tools}

# ==============================================================================
# 4. PLANNER & EXECUTOR NODES (UPDATED)
# ==============================================================================

PLANNER_PROMPT = """You are LitScout Orchestrator... Review your progress and decide the next agent.

**Current State & Progress**
User Prompt: {user_prompt}
Start Year: {start_year}
End Year: {end_year}
Sources: {sources}
Research Questions: {research_questions_count}
Filtered Papers: {filtered_papers_count}
Observations: {observations}

**Available Agents (Tools)**
{tool_descriptions}

**Instructions**
- The first step is ALWAYS 'parse_initial_request'.
- Then, follow the sequence: formulate_research_plan -> search_and_filter_tool -> screening_tool...
- If screening produces too few papers, go back to search_and_filter_tool.

{format_instructions}
"""
prompt_template = ChatPromptTemplate.from_template(PLANNER_PROMPT)
planner_chain = prompt_template | llm | parser

def planner_node(state: LitScoutState) -> Dict[str, Any]:
    """The central planner node that coordinates all specialized agents."""
    print("\n---ORCHESTRATOR PLANNER: Reviewing state and selecting next agent---")
    try:
        response = planner_chain.invoke({
            "user_prompt": state.user_prompt,
            "start_year": state.start_year,
            "end_year": state.end_year,
            "sources": state.sources,
            "research_questions_count": len(state.research_questions),
            "filtered_papers_count": len(state.filtered_papers),
            "observations": "\n".join(state.observations[-3:]) if state.observations else "No observations yet.",
            "tool_descriptions": render_text_description(tools),
            "format_instructions": parser.get_format_instructions()
        })
        return {"plan": response.plan, "next_agent": response.next_agent}
    except Exception as e:
        print(f"❌ Error in planner_node: {e}")
        return {"next_agent": "END", "error_messages": state.error_messages + [f"Planner error: {e}"]}

def tool_executor_node(state: LitScoutState) -> Dict[str, Any]:
    """Executes the specialized agent selected by the planner."""
    tool_name = state.next_agent
    if tool_name not in tool_map:
        return {"error_messages": state.error_messages + [f"Unknown tool: {tool_name}"], "next_agent": "END"}
    
    selected_tool = tool_map[tool_name]
    print(f"---ORCHESTRATOR EXECUTOR: Invoking agent '{tool_name}'---")
    
    try:
        if tool_name == "parse_initial_request":
            result = selected_tool.invoke({"user_prompt": state.user_prompt})
        elif tool_name == "formulate_research_plan":
            result = selected_tool.invoke({"user_prompt": state.user_prompt})
        elif tool_name == "search_and_filter_tool":
            result = selected_tool.invoke({
                "research_questions": state.research_questions,
                "start_year": state.start_year,
                "end_year": state.end_year,
                "sources": state.sources
            })
        elif tool_name == "screening_tool":
             result = selected_tool.invoke({
                "filtered_papers": state.filtered_papers,
                "research_questions": state.research_questions,
                "inclusion_criteria": state.inclusion_criteria
            })
        else: # For other mock tools
            result = selected_tool.invoke({})
        
        observation = f"Successfully invoked agent '{tool_name}'"
        if "error" in result:
            observation += f" with errors: {result['error']}"
        
        state_update = result.copy()
        state_update["observations"] = state.observations + [observation]
        state_update["current_step"] = state.current_step + 1
        
        print(f"✅ {observation}")
        return state_update
    except Exception as e:
        error_msg = f"Error invoking agent '{tool_name}': {e}"
        return {"observations": state.observations + [error_msg], "error_messages": state.error_messages + [error_msg]}

# ==============================================================================
# 5. GRAPH ASSEMBLY & MAIN EXECUTION
# ==============================================================================
def router(state: LitScoutState) -> str:
    if state.next_agent == "END": return "END"
    return "execute_tool"

workflow = StateGraph(LitScoutState)
workflow.add_node("planner", planner_node)
workflow.add_node("execute_tool", tool_executor_node)
workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", router, {"execute_tool": "execute_tool", "END": END})
workflow.add_edge("execute_tool", "planner")
app = workflow.compile()

if __name__ == "__main__":
    # --- The sample query is given here ---
    initial_state = LitScoutState(
        user_prompt="Perform a literature survey on 'multi-agent systems for literature review' from 2022 onwards, focusing on papers from Springer, Elsevier, ScienceDirect, ACM, IEEE and Nature."
    )
    
    print(f"🚀 Starting LitScout Orchestrator...\nQuery: {initial_state.user_prompt}\n")
    
    try:
        for step in app.stream(initial_state, {"recursion_limit": 20}):
            node, output = next(iter(step.items()))
            print(f"\n---[STEP {output.get('current_step', 0)}: {node}]---")
            # This is just for printing the current state, not part of the logic
            if 'filtered_papers' in output: print(f"  📄 Filtered Papers: {len(output['filtered_papers'])}")
            if 'screened_papers' in output: print(f"  🔍 Screened Papers: {len(output['screened_papers'])}")
    except Exception as e:
        print(f"\n❌ Fatal orchestration error: {e}")