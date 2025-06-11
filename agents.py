from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import Tool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import json
import logging
from datetime import datetime
import time
from functools import lru_cache
import backoff

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize Groq LLM with retry mechanism
@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def initialize_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="gemma2-9b-it",
        temperature=0.7,
        max_tokens=1000
    )

llm = initialize_llm()

# Initialize tools with error handling
try:
    yfinance_tool = YahooFinanceNewsTool()
    search_tool = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
except Exception as e:
    logger.error(f"Error initializing tools: {str(e)}")
    yfinance_tool = None
    search_tool = None

# Initialize ChromaDB with persistence
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
except Exception as e:
    logger.error(f"Error initializing vector store: {str(e)}")
    vector_store = None

# Market Analysis Agent
MARKET_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an advanced Market Analysis Agent specialized in analyzing IT industry trends and market conditions.
    Your task is to:
    1. Analyze current IT market trends and technology adoption rates
    2. Identify potential market risks and opportunities for IT projects
    3. Evaluate economic indicators affecting IT investments
    4. Monitor competitor activities and market positioning
    5. Track technology stack evolution and industry standards
    
    Available Tools:
    - Yahoo Finance: For IT company performance and market data
    - Web Search: For latest IT industry news and trends
    
    Format your response as a JSON with the following structure:
    {{
        "market_trends": {{
            "technology_trends": [],
            "market_demand": [],
            "competitor_analysis": [],
            "market_sentiment": "POSITIVE/NEGATIVE/NEUTRAL"
        }},
        "risk_analysis": {{
            "market_risks": [],
            "technology_risks": [],
            "competition_risks": [],
            "risk_severity": "LOW/MEDIUM/HIGH",
            "risk_impact": "SHORT_TERM/LONG_TERM"
        }},
        "economic_indicators": {{
            "it_spending_trends": [],
            "investment_climate": [],
            "market_volatility": float,
            "growth_projections": []
        }},
        "recommendations": {{
            "technology_adoption": [],
            "market_positioning": [],
            "risk_mitigation": [],
            "opportunity_capture": []
        }},
        "data_sources": [],
        "timestamp": str
    }}
    """),
    ("human", "{input}")
])

# Risk Scoring Agent
RISK_SCORING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a sophisticated Risk Scoring Agent specialized in assessing IT project risks.
    Your task is to:
    1. Evaluate technical risks (architecture, scalability, security)
    2. Assess project management risks (timeline, budget, resources)
    3. Analyze business risks (ROI, market fit, stakeholder alignment)
    4. Identify operational risks (maintenance, support, compliance)
    5. Calculate comprehensive risk scores with confidence levels
    
    Format your response as a JSON with the following structure:
    {{
        "risk_assessment": {{
            "technical_risk_score": float,
            "project_risk_score": float,
            "business_risk_score": float,
            "operational_risk_score": float,
            "overall_risk_score": float,
            "risk_confidence": float
        }},
        "risk_factors": {{
            "technical_factors": [],
            "project_factors": [],
            "business_factors": [],
            "operational_factors": [],
            "external_factors": []
        }},
        "risk_patterns": {{
            "identified_patterns": [],
            "pattern_confidence": float,
            "historical_correlation": float
        }},
        "mitigation_strategies": {{
            "technical_mitigation": [],
            "project_mitigation": [],
            "business_mitigation": [],
            "operational_mitigation": [],
            "contingency_plans": []
        }},
        "data_sources": [],
        "timestamp": str
    }}
    """),
    ("human", "{input}")
])

# Project Status Tracking Agent
PROJECT_STATUS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Project Status Tracking Agent specialized in monitoring IT project progress and risks.
    Your task is to:
    1. Track project milestones and technical deliverables
    2. Monitor resource availability and skill requirements
    3. Identify technical debt and quality issues
    4. Assess team dynamics and knowledge transfer
    5. Track budget utilization and ROI metrics
    
    Format your response as a JSON with the following structure:
    {{
        "project_status": {{
            "overall_status": "ON_TRACK/AT_RISK/DELAYED",
            "milestone_completion": float,
            "technical_debt": float,
            "quality_metrics": float,
            "budget_utilization": float,
            "roi_metrics": float
        }},
        "resource_analysis": {{
            "team_composition": {{}},
            "skill_gaps": [],
            "knowledge_transfer": [],
            "team_capacity": float,
            "resource_utilization": float
        }},
        "technical_analysis": {{
            "architecture_status": [],
            "code_quality": [],
            "security_status": [],
            "performance_metrics": []
        }},
        "risk_assessment": {{
            "technical_risks": [],
            "resource_risks": [],
            "schedule_risks": [],
            "budget_risks": [],
            "risk_impact": "LOW/MEDIUM/HIGH",
            "mitigation_status": []
        }},
        "recommendations": {{
            "technical_actions": [],
            "resource_actions": [],
            "process_improvements": [],
            "risk_mitigation": []
        }},
        "timestamp": str
    }}
    """),
    ("human", "{input}")
])

# Reporting Agent
REPORTING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Reporting Agent specialized in generating comprehensive IT project risk reports.
    Your task is to:
    1. Compile and analyze risk data from multiple sources
    2. Generate technical and business insights
    3. Create stakeholder-specific reports
    4. Provide actionable recommendations
    5. Track risk trends and patterns
    
    Format your response as a JSON with the following structure:
    {{
        "executive_summary": {{
            "key_findings": [],
            "risk_overview": str,
            "technical_status": str,
            "business_impact": str,
            "recommended_actions": []
        }},
        "detailed_analysis": {{
            "technical_analysis": {{}},
            "project_analysis": {{}},
            "business_analysis": {{}},
            "risk_analysis": {{}},
            "trend_analysis": []
        }},
        "recommendations": {{
            "technical_recommendations": [],
            "project_recommendations": [],
            "business_recommendations": [],
            "risk_mitigation": []
        }},
        "action_items": {{
            "critical_actions": [],
            "technical_actions": [],
            "project_actions": [],
            "business_actions": [],
            "risk_mitigation_actions": []
        }},
        "data_sources": [],
        "timestamp": str
    }}
    """),
    ("human", "{input}")
])

class BaseAgent:
    def __init__(self):
        self.vector_store = vector_store
        self.cache = {}
        self.last_request_time = {}
        self.rate_limit = 60  # requests per minute

    def _check_rate_limit(self, key: str) -> bool:
        current_time = time.time()
        if key not in self.last_request_time:
            self.last_request_time[key] = current_time
            return True
        
        time_diff = current_time - self.last_request_time[key]
        if time_diff < 60 / self.rate_limit:
            time.sleep(60 / self.rate_limit - time_diff)
        
        self.last_request_time[key] = time.time()
        return True

    @lru_cache(maxsize=100)
    def _get_cached_result(self, key: str) -> Optional[Dict]:
        return self.cache.get(key)

    def _cache_result(self, key: str, result: Dict, ttl: int = 3600):
        self.cache[key] = {
            'result': result,
            'expiry': time.time() + ttl
        }

    def _get_cached_or_fresh(self, key: str, func: callable, *args, **kwargs) -> Dict:
        cached = self._get_cached_result(key)
        if cached and cached['expiry'] > time.time():
            return cached['result']
        
        result = func(*args, **kwargs)
        self._cache_result(key, result)
        return result

class MarketAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.chain = MARKET_ANALYSIS_PROMPT | llm | JsonOutputParser()

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def analyze(self, market_data: Dict) -> Dict:
        try:
            self._check_rate_limit("market_analysis")
            
            # Generate cache key
            cache_key = f"market_analysis_{hash(json.dumps(market_data, sort_keys=True))}"
            
            return self._get_cached_or_fresh(
                cache_key,
                self._analyze_impl,
                market_data
            )
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _analyze_impl(self, market_data: Dict) -> Dict:
        # Store market data in vector store for future reference
        if self.vector_store:
            market_text = json.dumps(market_data)
            texts = text_splitter.split_text(market_text)
            self.vector_store.add_texts(texts)
            
            # Get similar historical data
            similar_docs = self.vector_store.similarity_search(market_text, k=3)
            context = "\n".join([doc.page_content for doc in similar_docs])
        else:
            context = ""

        # Analyze with context
        result = self.chain.invoke({
            "input": f"Market Data: {market_data}\nHistorical Context: {context}"
        })
        result["timestamp"] = datetime.now().isoformat()
        return result

class RiskScoringAgent:
    def __init__(self):
        self.chain = RISK_SCORING_PROMPT | llm | JsonOutputParser()
        self.vector_store = Chroma(embedding_function=embeddings)
    
    def score(self, risk_data: Dict) -> Dict:
        try:
            # Store risk data in vector store
            risk_text = json.dumps(risk_data)
            texts = text_splitter.split_text(risk_text)
            self.vector_store.add_texts(texts)
            
            # Get similar historical risk assessments
            similar_docs = self.vector_store.similarity_search(risk_text, k=3)
            context = "\n".join([doc.page_content for doc in similar_docs])
            
            # Score with context
            result = self.chain.invoke({
                "input": f"Risk Data: {risk_data}\nHistorical Context: {context}"
            })
            result["timestamp"] = datetime.now().isoformat()
            return result
        except Exception as e:
            logger.error(f"Error in risk scoring: {str(e)}")
            return {"error": str(e)}

class ProjectStatusAgent:
    def __init__(self):
        self.chain = PROJECT_STATUS_PROMPT | llm | JsonOutputParser()
        self.vector_store = Chroma(embedding_function=embeddings)
    
    def track(self, project_data: Dict) -> Dict:
        try:
            # Store project data in vector store
            project_text = json.dumps(project_data)
            texts = text_splitter.split_text(project_text)
            self.vector_store.add_texts(texts)
            
            # Get similar historical project data
            similar_docs = self.vector_store.similarity_search(project_text, k=3)
            context = "\n".join([doc.page_content for doc in similar_docs])
            
            # Track with context
            result = self.chain.invoke({
                "input": f"Project Data: {project_data}\nHistorical Context: {context}"
            })
            result["timestamp"] = datetime.now().isoformat()
            return result
        except Exception as e:
            logger.error(f"Error in project tracking: {str(e)}")
            return {"error": str(e)}

class ReportingAgent:
    def __init__(self):
        self.chain = REPORTING_PROMPT | llm | JsonOutputParser()
        self.vector_store = Chroma(embedding_function=embeddings)
    
    def generate_report(self, analysis_data: Dict) -> Dict:
        try:
            # Store analysis data in vector store
            analysis_text = json.dumps(analysis_data)
            texts = text_splitter.split_text(analysis_text)
            self.vector_store.add_texts(texts)
            
            # Get similar historical reports
            similar_docs = self.vector_store.similarity_search(analysis_text, k=3)
            context = "\n".join([doc.page_content for doc in similar_docs])
            
            # Generate report with context
            result = self.chain.invoke({
                "input": f"Analysis Data: {analysis_data}\nHistorical Context: {context}"
            })
            result["timestamp"] = datetime.now().isoformat()
            return result
        except Exception as e:
            logger.error(f"Error in report generation: {str(e)}")
            return {"error": str(e)} 