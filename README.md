# BUYorBYE 🛍️

> AI-powered shopping assistant that finds, compares, and ranks real products and deals from the web.

BUYorBYE turns natural-language shopping requests into intelligent product recommendations.

Instead of simply searching for products, BUYorBYE understands what the user wants, searches real shopping sources, compares prices, checks ratings and availability, and ranks the best available options.

## ✨ Features

- 🤖 Natural-language shopping queries
- 🔎 Real-time product discovery
- 💰 India-focused INR pricing
- 📊 Price comparison
- 🏷️ Deal detection
- ⭐ Rating and availability comparison
- 🧠 AI-powered query understanding with Groq
- 🎯 Requirement-aware product ranking
- 🖼️ Real product images
- 🔗 Direct product links when available
- ❤️ Wishlist
- 📂 Product categories
- 🔥 Deals section
- 🔄 Relevance, price and rating sorting
- 📱 Responsive premium UI
- ⚡ FastAPI backend

## 🧠 How It Works

User Request
↓
Groq AI
↓
Understand Requirements
↓
Google Shopping / Serper
↓
Product Extraction
↓
Normalize & Filter
↓
Compare Price, Rating & Availability
↓
Deal Detection & Ranking
↓
BUYorBYE Recommendation

## 🛒 Example

User:

"Find me a wireless mouse under ₹1500."

BUYorBYE searches real shopping results and ranks suitable products using:

- Requirement match
- Budget compliance
- Current price
- Rating
- Availability
- Deal quality

The interface then displays the best available products with their price, image, details and purchase link when available.

## 🏗️ Tech Stack

### Frontend

- HTML
- CSS
- JavaScript
- Responsive UI
- Glassmorphism
- CSS animations

### Backend

- Python
- FastAPI
- LangChain
- LangGraph
- SQLAlchemy
- SQLite

### AI

- Groq
- openai/gpt-oss-20b
- LangChain tool calling

### Search & Product Discovery

- Serper API
- Google Shopping
- Real-time shopping results

## 📁 Project Structure

BUYorBYE-v2/
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── README.md
├── src/
│   ├── agent/
│   ├── api/
│   ├── database/
│   ├── mcp/
│   ├── services/
│   └── utils/
├── alembic/
├── requirements.txt
├── start_server.py
├── alembic.ini
├── .env.example
└── README.md

## ⚙️ Local Setup

### 1. Clone

git clone https://github.com/anudeep-max/BUYorBYE-v2.git
cd BUYorBYE-v2

### 2. Create virtual environment

python3 -m venv venv
source venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Configure environment variables

Create a .env file containing:

GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
LLM_TEMPERATURE=0.3
DATABASE_URL=sqlite:///./shopping_assistant.db
CACHE_ENABLED=false
LANGFUSE_ENABLED=false
DEEPEVAL_ENABLED=false
XRAY_ENABLED=false
CLOUDWATCH_ENABLED=false
BEDROCK_ENABLED=false

Never commit API keys or .env files to GitHub.

### 5. Initialize the database

alembic upgrade head

### 6. Start the backend

python start_server.py

Backend:
http://localhost:3565

### 7. Start the frontend

Open another terminal:

cd frontend
python3 -m http.server 8080

Frontend:
http://localhost:8080

## 🔐 Security

API keys are stored using environment variables and are never intended to be committed to the repository.

BUYorBYE does not intentionally fabricate product prices, ratings, sellers or purchase links. Product information is based on available shopping/search evidence.

## 🚀 Deployment

BUYorBYE is designed to use a separate frontend and backend deployment.

Frontend:
GitHub Pages

Backend:
Render or another cloud platform supporting FastAPI

The frontend communicates with the deployed backend through the API.

## 🎯 Hackathon

BUYorBYE was developed as an AI-powered shopping and deal-finding agent for the WCC Launchpad 30 hackathon.

### The idea

Don't just find a product. Find the right deal.

## 📄 License

BUYorBYE is based on an MIT-licensed open-source foundation and has been substantially modified and extended for this project.

The applicable license and attribution information are retained in the repository.
