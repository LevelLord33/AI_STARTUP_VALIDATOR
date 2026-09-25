# StartupSense AI – NLP-Based Conversational Business Advisor

**Academic AI & Natural Language Processing (NLP) Project**  
**Repository:** `LevelLord33/AI_STARTUP_VALIDATOR`  

---

## 📌 Project Overview

**StartupSense AI** is a conversational AI and Natural Language Processing system designed to provide practical, step-by-step business advisory to entrepreneurs. Instead of requiring users to fill tedious 15–20 field forms and questionnaires with complex business terminology, StartupSense AI adheres to a core philosophy:

> **"ENTER YOUR BUSINESS IDEA ONCE → CHAT WITH THE AI → GET STEP-BY-STEP GUIDANCE"**

Business advisory, financial modeling, competitor discovery, and risk analysis are structured application modules built directly on top of an academic NLP pipeline.

---

## 🧠 Core Academic NLP Pipeline

```
USER NATURAL LANGUAGE (Unstructured Description)
        ↓
1. TEXT PREPROCESSING (Tokenization, Lowercasing, Stopword Removal, N-grams)
        ↓
2. INFORMATION EXTRACTION (Regex currency ₹/Rs/Lakh, capacity entities, location parsing)
        ↓
3. BUSINESS CONCEPT & ENTITY RECOGNITION (Domain ontology matching: Dairy, Agri, Food, Apparel, Tech, etc.)
        ↓
4. INTENT CLASSIFICATION (Financial, Funding, Competition, Risk, Customer Acquisition, Action, Pricing)
        ↓
5. DIALOGUE STATE TRACKING & DATA PROVENANCE (USER PROVIDED, NLP INFERRED, AI ESTIMATED, PYTHON CALCULATED)
        ↓
6. CONVERSATIONAL AI & SINGLE TARGETED QUESTIONING (Hugging Face Qwen / Fallback Heuristic Dialogue Manager)
        ↓
PERSONALIZED STEP-BY-STEP ADVISORY
```

### Data Provenance Tracking
Every parameter in the internal business profile is explicitly tagged with its origin to ensure academic integrity:
- `USER PROVIDED`: Directly stated by the user (e.g., initial investment, location).
- `NLP INFERRED`: Derived from natural language text using entity extraction rules.
- `AI ESTIMATED`: Heuristically estimated industry baseline when information is missing.
- `PYTHON CALCULATED`: Computed deterministically using transparent mathematical formulas (e.g., surplus, runway, loan EMI).

---

## ⚙️ Key Technical Features

### 1. Adaptive Dialogue Management (One Question at a Time)
- The system checks what is already known and identifies missing context.
- Formulates **ONE primary, practical question per turn** (e.g. animal capacity for dairy, equipment setup for tailoring).
- Never repeats previously answered questions.

### 2. Deterministic Python Financial Engine
- **Monthly Operating Surplus:** `Monthly Revenue - Monthly Expenses`
- **Runway:** Calculates runway in months based on available funds and monthly burn rate.
- **Estimated Break-Even:** Calculates months to recover initial investment.
- **Capital Allocation Breakdown:** Practical guidance on founder savings, government schemes (PMEGP subsidy, Mudra loans), and bank debt.

### 3. Supporting Historical ML Signal (Random Forest)
- Features a pre-trained **Random Forest Classifier** (`startup_model.pkl`, `startup_features.pkl`) trained on historical venture outcomes (Crunchbase dataset).
- Evaluates operational ventures on 31 quantitative signals as a supporting analytical benchmark.
- Gracefully handles missing dependencies without interrupting conversational advisory.

### 4. Transparent 5-Dimension Validation Assessment (/100)
- **Problem Strength (20%)**
- **Market Potential (20%)**
- **Financial Feasibility (25%)**
- **Competitive Position (15%)**
- **Execution Readiness (20%)**

### 5. Academic "How NLP is Used" Inspector
- An interactive inspector inside the application UI that reveals real-time preprocessed tokens, extracted entities, detected intent, and confidence metrics.

---

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.9+
- Virtual environment (`.venv`)

### 1. Clone the Repository
```bash
git clone https://github.com/LevelLord33/AI_STARTUP_VALIDATOR.git
cd AI_STARTUP_VALIDATOR
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Secrets
Create a `.streamlit/secrets.toml` file with your Hugging Face API key:
```toml
HF_TOKEN = "your_huggingface_api_token_here"
```
*(Note: If `HF_TOKEN` is not provided, the application automatically engages its deterministic Heuristic Intelligence Engine so all features continue to work seamlessly!)*

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
├── app.py                  # Main Streamlit application with NLP pipeline, Chatbot UI, and Financial Engine
├── startup_model.pkl       # Trained Scikit-Learn Random Forest Classifier
├── startup_features.pkl    # 31-feature list for historical benchmark
├── train-model.py          # Model training pipeline
├── train-data.csv          # Training dataset for historical venture benchmark
├── test-data.csv           # Evaluation dataset
├── requirements.txt        # Python package dependencies
├── .streamlit/
│   └── secrets.toml        # Hugging Face token configuration (excluded from git)
└── README.md               # Project documentation
```

---

## 🎓 Academic Integrity & Disclaimer
StartupSense AI is designed as an educational decision-support tool demonstrating Natural Language Processing, Information Extraction, and Conversational AI. It is not an automated predictor of guaranteed business success and does not replace certified professional legal or financial advice.
