# 🚀 AI-Powered IT Project Risk Management System

An intelligent multi-agent system that monitors **market conditions and internal project signals** to identify, assess, and report **IT project risks in real time** using LLMs, RAG, and agentic workflows.

🔗 **Live Demo (Hugging Face Space):**  
https://huggingface.co/spaces/Soumya79/AI_PROJECT_RISK_MANAGEMENT_System

🔗 **GitHub Repository:**  
https://github.com/soumadipchangder/AI-Powered-IT-Project-Risk-Management-System

---

## 📌 Problem Statement

IT projects often fail due to delayed risk identification, lack of real-time monitoring, and fragmented decision-making.  
Traditional tools are reactive and heavily manual.

This system provides an **AI-driven, proactive risk monitoring framework** that continuously evaluates both **external market risks** and **internal operational risks**, helping managers take preventive action early.

---

## ✨ Key Features

- 🤖 Multi-Agent Architecture with specialized agents
- 📚 RAG Pipeline using ChromaDB for contextual reasoning
- 📊 Risk Scoring & Classification
- 🧠 LLM-based Market & Financial Analysis
- 🧾 Automated PDF Risk Reports
- 🖥️ Interactive Streamlit Dashboard
- ⚡ Parallel agent execution using LangGraph workflows

---

## 🧠 System Architecture

### 🔹 Agents

1. **Market Analysis Agent**  
   - Analyzes market trends, company news, and financial indicators  
   - Uses LLM + web data for macro risk detection

2. **Risk Scoring Agent**  
   - Evaluates probability and impact of risks  
   - Assigns severity levels

3. **Project Status Tracking Agent**  
   - Tracks internal risks such as schedule delays and resource constraints  
   - Simulates Jira/API-based project signals

4. **Reporting Agent**  
   - Consolidates agent outputs  
   - Generates summaries and PDF reports

---

## 🛠 Tech Stack

| Category | Tools |
|--------|--------|
| Language | Python |
| UI | Streamlit |
| LLM APIs | Groq API, Mistral |
| Embeddings | LLaMA Embeddings |
| Vector DB | ChromaDB |
| Agent Framework | LangGraph, CrewAI |
| Retrieval | RAG (Retrieval-Augmented Generation) |
| Reporting | PDF generation |

---

## ⚙️ How It Works

1. User enters company/project details via UI  
2. Query is enriched using RAG from vector database  
3. Task is routed to relevant agents using LangGraph  
4. Agents run in parallel and analyze different risk dimensions  
5. Risk scores and mitigation suggestions are generated  
6. Final report is shown in UI and downloadable as PDF

---

## ▶️ Run Locally

### ✅ 1. Clone Repository

git clone https://github.com/soumadipchangder/AI-Powered-IT-Project-Risk-Management-System.git
cd AI-Powered-IT-Project-Risk-Management-System 

### ✅ 2. Install Dependencies

pip install -r requirements.txt

###✅ 3. Set Environment Variables

Create a .env file:

GROQ_API_KEY=your_api_key_here
MISTRAL_API_KEY=your_api_key_here

###✅ 4. Run App

streamlit run app.py

📁 Project Structure
├── app.py          # Streamlit UI
├── agents.py       # Agent definitions
├── workflow.py     # LangGraph workflow orchestration
├── requirements.txt
└── README.md

📈 Future Enhancements

🔗 Jira & Trello API integration

📡 Real-time financial feeds

📊 Advanced risk visualization dashboards

📱 Mobile-friendly UI

🧠 Reinforcement learning for adaptive risk strategies

👨‍💻 Author

Soumyadip Changder
Final Year B.Tech CSE Student
AI/ML | Generative AI | Multi-Agent Systems | RAG

🔗 GitHub: https://github.com/soumadipchangder

🔗 Hugging Face: https://huggingface.co/Soumya79
