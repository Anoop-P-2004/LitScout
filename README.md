# LitScout: Multi-Agent System for Automated Literature Review

LitScout is a full-stack, AI-powered application designed to automate the complex and labor-intensive process of conducting a Systematic Literature Review (SLR). By leveraging a multi-agent system, the project aims to save researchers significant time and effort, reduce human error, and increase the transparency of AI-assisted research.
This README provides an overview of the project's architecture, tech stack, and setup instructions.

# File Structure with Purpose
```
LitScout/
├── package.json                # Node.js (frontend) project metadata and dependencies
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies for backend
├── backend/                    # Backend code and data
│   ├── orch.py                     # Orchestrater agent to manage workflow
│   ├── test.py                     # search and filter test scripts
│   ├── agents/                     # Agents for literature review
│   │   ├── __init__.py                 # Marks directory as Python package
│   │   ├── screening_agent.py           # Agent for screening literature
│   │   └── search_and_filter_agent.py   # Agent for searching and filtering papers
│   ├── app/                        # Backend application logic
│   │   ├── __init__.py                 # Marks directory as Python package
│   │   ├── auth.py                     # Handles authentication logic
│   │   ├── database.py                 # Database connection and operations
│   │   ├── main.py                     # Entry point for backend API (FastAPI)
│   │   ├── models.py                   # Database models and ORM classes
│   │   └── schemas.py                  # Pydantic schemas for data validation
│   └── utils/                      # Utility functions
│       └── __init__.py                 # Utility helpers
├── frontend/                    # Frontend code
│   ├── index.html                   # Main HTML file for frontend app
│   ├── package.json                 # Frontend project metadata and dependencies
│   ├── postcss.config.js            # PostCSS configuration
│   ├── tailwind.config.js           # Tailwind CSS configuration
│   └── src/                         # Frontend source code
│       ├── App.jsx                      # Main React component
│       ├── index.css                    # Global CSS styles
│       ├── main.jsx                     # Entry point for React app
│       └── components/                  # React UI components
│           ├── About.jsx                    # About page component
│           ├── Dashboard.jsx                # Dashboard UI component
│           ├── Layout.jsx                   # Layout wrapper for pages
│           ├── Login.jsx                    # Login form component
│           └── SignUp.jsx                   # Signup form component
```
## Tech Stack

- **Backend**: FastAPI (Python), SQLite, Uvicorn, LangGraph
- **Frontend**: React, Tailwind CSS