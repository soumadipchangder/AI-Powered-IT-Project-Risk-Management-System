import gradio as gr
import json
from workflow import workflow
from agents import MarketAnalysisAgent, RiskScoringAgent, ProjectStatusAgent, ReportingAgent
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

def format_json_output(data):
    if isinstance(data, dict):
        return json.dumps(data, indent=2)
    return str(data)

def generate_market_analysis():
    try:
        market_agent = MarketAnalysisAgent()
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
        result = market_agent.analyze(market_data)
        return format_json_output(result)
    except Exception as e:
        logger.error(f"Error in market analysis: {str(e)}")
        return f"Error: {str(e)}"

def generate_risk_scoring():
    try:
        risk_agent = RiskScoringAgent()
        risk_data = {
            "market_analysis": {},
            "project_data": {
                "technical_risks": [],
                "project_risks": [],
                "business_risks": [],
                "operational_risks": []
            }
        }
        result = risk_agent.score(risk_data)
        return format_json_output(result)
    except Exception as e:
        logger.error(f"Error in risk scoring: {str(e)}")
        return f"Error: {str(e)}"

def generate_project_status():
    try:
        project_agent = ProjectStatusAgent()
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
        result = project_agent.track(project_data)
        return format_json_output(result)
    except Exception as e:
        logger.error(f"Error in project status: {str(e)}")
        return f"Error: {str(e)}"

def generate_risk_analysis():
    try:
        initial_state = {
            "market_analysis": {},
            "risk_scoring": {},
            "project_status": {},
            "final_report": {},
            "messages": [],
            "metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        result = workflow.invoke(initial_state)
        
        # Extract metrics and report
        metrics = result.get("metrics", {})
        report = result.get("final_report", {})
        
        return format_json_output(report), format_json_output(metrics)
    except Exception as e:
        logger.error(f"Error in risk analysis: {str(e)}")
        return f"Error: {str(e)}", "{}"

def chat_with_system(message, history):
    try:
        initial_state = {
            "market_analysis": {},
            "risk_scoring": {},
            "project_status": {},
            "final_report": {},
            "messages": [message],
            "metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        result = workflow.invoke(initial_state)
        response = result.get("final_report", {}).get("executive_summary", {}).get("risk_overview", "No response available")
        return response
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return f"Error: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="AI Project Risk Management", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # AI-Powered IT Project Risk Management System
    
    This system helps identify, assess, and mitigate risks in IT projects by analyzing:
    - Market trends and technology adoption
    - Technical and project risks
    - Resource availability and team dynamics
    - Business impact and ROI metrics
    """)
    
    with gr.Tabs():
        with gr.TabItem("Dashboard"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Market Analysis")
                    market_analysis_btn = gr.Button("Generate Market Analysis", variant="primary")
                    market_analysis_output = gr.JSON(label="Market Analysis Results")
                
                with gr.Column():
                    gr.Markdown("### Risk Scoring")
                    risk_scoring_btn = gr.Button("Generate Risk Scoring", variant="primary")
                    risk_scoring_output = gr.JSON(label="Risk Scoring Results")
                
                with gr.Column():
                    gr.Markdown("### Project Status")
                    project_status_btn = gr.Button("Generate Project Status", variant="primary")
                    project_status_output = gr.JSON(label="Project Status Results")
            
            market_analysis_btn.click(
                fn=generate_market_analysis,
                outputs=market_analysis_output
            )
            risk_scoring_btn.click(
                fn=generate_risk_scoring,
                outputs=risk_scoring_output
            )
            project_status_btn.click(
                fn=generate_project_status,
                outputs=project_status_output
            )
        
        with gr.TabItem("Risk Analysis"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Generate Comprehensive Risk Analysis")
                    risk_analysis_btn = gr.Button("Generate Analysis", variant="primary")
                    risk_analysis_output = gr.JSON(label="Risk Analysis Report")
                with gr.Column():
                    gr.Markdown("### Key Metrics")
                    metrics_output = gr.JSON(label="Project Metrics")
            
            risk_analysis_btn.click(
                fn=generate_risk_analysis,
                outputs=[risk_analysis_output, metrics_output]
            )
        
        with gr.TabItem("Chat Interface"):
            chatbot = gr.ChatInterface(
                fn=chat_with_system,
                title="Chat with Risk Management System",
                description="Ask questions about project risks and get AI-powered responses.",
                examples=[
                    "What are the current technical risks?",
                    "How is the project progressing?",
                    "What are the main risk factors?",
                    "What mitigation strategies do you recommend?",
                    "What is the current risk score?",
                    "How is the team performing?",
                    "What are the security risks?",
                    "What is the project ROI?"
                ],
                theme="soft"
            )

if __name__ == "__main__":
    # Create cache directory if it doesn't exist
    os.makedirs("./cache", exist_ok=True)
    demo.launch(share=True) 