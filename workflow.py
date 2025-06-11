from typing import Dict, TypedDict, Annotated, Sequence, Optional
from langgraph.graph import Graph, StateGraph
from agents import MarketAnalysisAgent, RiskScoringAgent, ProjectStatusAgent, ReportingAgent
import logging
from datetime import datetime
import json
from functools import lru_cache
import os

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    market_analysis: Dict
    risk_scoring: Dict
    project_status: Dict
    final_report: Dict
    messages: Sequence[str]
    error: Optional[str]
    timestamp: str
    metrics: Dict

def create_workflow() -> Graph:
    # Initialize agents with caching
    market_agent = MarketAnalysisAgent()
    risk_agent = RiskScoringAgent()
    project_agent = ProjectStatusAgent()
    reporting_agent = ReportingAgent()

    # Define the workflow
    workflow = StateGraph(AgentState)

    @lru_cache(maxsize=100)
    def get_cached_state(state_key: str) -> Optional[AgentState]:
        try:
            with open(f"./cache/{state_key}.json", "r") as f:
                return json.load(f)
        except:
            return None

    def save_state_to_cache(state_key: str, state: AgentState):
        try:
            os.makedirs("./cache", exist_ok=True)
            with open(f"./cache/{state_key}.json", "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Error saving state to cache: {str(e)}")

    # Market Analysis Node
    def analyze_market(state: AgentState) -> AgentState:
        try:
            state_key = f"market_{hash(json.dumps(state, sort_keys=True))}"
            cached_state = get_cached_state(state_key)
            if cached_state:
                return cached_state

            market_data = {
                "market_conditions": "current_market_data",
                "trends": "market_trends",
                "indicators": "economic_indicators",
                "it_industry": {
                    "technology_trends": [],
                    "competitor_analysis": [],
                    "market_demand": []
                }
            }
            analysis_result = market_agent.analyze(market_data)
            
            new_state = {
                **state,
                "market_analysis": analysis_result,
                "timestamp": datetime.now().isoformat()
            }
            
            save_state_to_cache(state_key, new_state)
            return new_state
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return {**state, "error": str(e)}

    # Risk Scoring Node
    def score_risk(state: AgentState) -> AgentState:
        try:
            if "error" in state:
                return state
            
            state_key = f"risk_{hash(json.dumps(state, sort_keys=True))}"
            cached_state = get_cached_state(state_key)
            if cached_state:
                return cached_state
            
            risk_data = {
                "market_analysis": state["market_analysis"],
                "project_data": {
                    "technical_risks": [],
                    "project_risks": [],
                    "business_risks": [],
                    "operational_risks": []
                }
            }
            risk_result = risk_agent.score(risk_data)
            
            new_state = {
                **state,
                "risk_scoring": risk_result,
                "timestamp": datetime.now().isoformat()
            }
            
            save_state_to_cache(state_key, new_state)
            return new_state
        except Exception as e:
            logger.error(f"Error in risk scoring: {str(e)}")
            return {**state, "error": str(e)}

    # Project Status Node
    def track_project(state: AgentState) -> AgentState:
        try:
            if "error" in state:
                return state
            
            state_key = f"project_{hash(json.dumps(state, sort_keys=True))}"
            cached_state = get_cached_state(state_key)
            if cached_state:
                return cached_state
            
            project_data = {
                "milestones": "current_milestones",
                "resources": {
                    "team_composition": {},
                    "skill_gaps": [],
                    "knowledge_transfer": []
                },
                "technical": {
                    "architecture_status": [],
                    "code_quality": [],
                    "security_status": []
                },
                "schedule": "schedule_data"
            }
            status_result = project_agent.track(project_data)
            
            new_state = {
                **state,
                "project_status": status_result,
                "timestamp": datetime.now().isoformat()
            }
            
            save_state_to_cache(state_key, new_state)
            return new_state
        except Exception as e:
            logger.error(f"Error in project tracking: {str(e)}")
            return {**state, "error": str(e)}

    # Reporting Node
    def generate_report(state: AgentState) -> AgentState:
        try:
            if "error" in state:
                return state
            
            state_key = f"report_{hash(json.dumps(state, sort_keys=True))}"
            cached_state = get_cached_state(state_key)
            if cached_state:
                return cached_state
            
            report_data = {
                "market_analysis": state["market_analysis"],
                "risk_scoring": state["risk_scoring"],
                "project_status": state["project_status"]
            }
            report_result = reporting_agent.generate_report(report_data)
            
            # Calculate metrics
            metrics = {
                "risk_score": report_result.get("risk_assessment", {}).get("overall_risk_score", 0),
                "project_health": report_result.get("project_status", {}).get("overall_status", "UNKNOWN"),
                "market_sentiment": report_result.get("market_analysis", {}).get("market_sentiment", "NEUTRAL")
            }
            
            new_state = {
                **state,
                "final_report": report_result,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            save_state_to_cache(state_key, new_state)
            return new_state
        except Exception as e:
            logger.error(f"Error in report generation: {str(e)}")
            return {**state, "error": str(e)}

    # Error Handling Node
    def handle_error(state: AgentState) -> AgentState:
        if "error" in state:
            logger.error(f"Workflow error: {state['error']}")
            return {
                **state,
                "final_report": {
                    "error": state["error"],
                    "status": "ERROR",
                    "message": "An error occurred during processing",
                    "timestamp": datetime.now().isoformat()
                }
            }
        return state

    # Add nodes to the workflow
    workflow.add_node("analyze_market", analyze_market)
    workflow.add_node("score_risk", score_risk)
    workflow.add_node("track_project", track_project)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("handle_error", handle_error)

    # Define edges
    workflow.add_edge("analyze_market", "score_risk")
    workflow.add_edge("score_risk", "track_project")
    workflow.add_edge("track_project", "generate_report")
    workflow.add_edge("generate_report", "handle_error")

    # Add conditional edges for error handling
    workflow.add_conditional_edges(
        "analyze_market",
        lambda x: "error" in x,
        {
            True: "handle_error",
            False: "score_risk"
        }
    )
    workflow.add_conditional_edges(
        "score_risk",
        lambda x: "error" in x,
        {
            True: "handle_error",
            False: "track_project"
        }
    )
    workflow.add_conditional_edges(
        "track_project",
        lambda x: "error" in x,
        {
            True: "handle_error",
            False: "generate_report"
        }
    )
    workflow.add_conditional_edges(
        "generate_report",
        lambda x: "error" in x,
        {
            True: "handle_error",
            False: "handle_error"
        }
    )

    # Set entry point
    workflow.set_entry_point("analyze_market")

    # Compile the workflow
    return workflow.compile()

# Create the workflow instance
workflow = create_workflow() 