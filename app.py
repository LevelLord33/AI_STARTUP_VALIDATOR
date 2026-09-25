"""
StartupSense AI – NLP-Based Conversational Business Advisor
Academic AI & NLP Project | LevelLord33/AI_STARTUP_VALIDATOR

Core Architecture:
1. Natural Language Onboarding (Single Minimal Input: Idea, Location, Stage, Optional Finances)
2. Comprehensive NLP Pipeline:
   - Text Preprocessing & Cleaning (Tokenization, Stopword Filtering, N-grams)
   - Information Extraction (Currency ₹/Rs/Lakh, Quantities, Pricing, Locations)
   - Business Entity & Concept Recognition (Ontology classification, Customer, Problem, Product)
   - Intent Classification (Financial, Funding, Competition, Risk, Customer Acquisition, Action, Pricing, Validation)
   - Context Memory & Provenance Tracking (USER PROVIDED, NLP INFERRED, AI ESTIMATED, PYTHON CALCULATED)
3. Dynamic Conversational AI (Hugging Face Qwen / Fallback Heuristic Intelligence)
4. Adaptive Single-Question Dialogue Flow (Step-by-step human advisor experience)
5. Deterministic Python Financial Engine (Unit economics, Surplus, Runway, Break-even, Capital structure)
6. Supporting Historical Random Forest Signal (Crunchbase Benchmark for Operating Ventures)
7. Transparent 5-Dimension Validation Assessment (/100)
8. Academic "How NLP is Used" Inspector & System Engines Configuration
"""

import streamlit as st
from huggingface_hub import InferenceClient
import pandas as pd
import numpy as np
import joblib
import json
import re
import os
from datetime import datetime

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="StartupSense AI – NLP Conversational Business Advisor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# MODERN SAAS / DARK-COMPATIBLE CSS
# ==========================================================

def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .block-container {
        max-width: 1240px;
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
    }
    
    /* Brand Header Styles */
    .brand-hero {
        text-align: center;
        padding: 1.6rem 1.2rem 2rem 1.2rem;
        background: radial-gradient(circle at top, rgba(99, 102, 241, 0.18) 0%, rgba(15, 23, 42, 0.6) 70%);
        border: 1px solid #334155;
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    
    .academic-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.45);
        font-size: 11.5px;
        font-weight: 700;
        padding: 4px 14px;
        border-radius: 9999px;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        text-transform: uppercase;
    }
    
    .brand-title {
        font-size: 40px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }
    
    .brand-gradient {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .brand-subtitle {
        font-size: 18px;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 6px;
    }
    
    .brand-desc {
        font-size: 14.5px;
        color: #94a3b8;
        max-width: 740px;
        margin: 0 auto;
        line-height: 1.55;
    }
    
    /* Academic Pipeline Flow Bar */
    .flow-badge-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: center;
        gap: 8px;
        margin-top: 16px;
    }
    
    .flow-step-pill {
        background: #1e293b;
        color: #cbd5e1;
        border: 1px solid #334155;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .flow-step-arrow {
        color: #64748b;
        font-size: 11px;
        font-weight: 800;
    }
    
    /* Status Header in Chat */
    .chat-status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #1e293b;
        border: 1px solid #334155;
        padding: 12px 18px;
        border-radius: 12px;
        margin-bottom: 18px;
    }
    
    /* Snapshot & Metric Cards in Sidebar */
    .snapshot-box {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25) !important;
        color: #f8fafc !important;
    }
    
    .snapshot-title {
        font-size: 11.5px !important;
        font-weight: 800 !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
        margin-bottom: 10px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
    }
    
    .snapshot-row {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        padding: 7px 0 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07) !important;
        font-size: 13px !important;
    }
    
    .snapshot-row:last-child {
        border-bottom: none !important;
    }
    
    .snapshot-label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }
    
    .snapshot-value {
        color: #f8fafc !important;
        font-weight: 700 !important;
        text-align: right !important;
    }
    
    .dimension-row {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        color: #cbd5e1 !important;
        font-size: 12px !important;
        padding: 3px 0 !important;
    }
    .dimension-row b {
        color: #818cf8 !important;
    }
    
    /* Tag Provenance Pill */
    .tag-prov {
        display: inline-block;
        font-size: 9.5px;
        font-weight: 800;
        padding: 2px 6px;
        border-radius: 4px;
        letter-spacing: 0.3px;
        margin-left: 5px;
    }
    .prov-user { background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.45); }
    .prov-calc { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.45); }
    .prov-inferred { background: rgba(168, 85, 247, 0.2); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.45); }
    .prov-est { background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.45); }
    
    /* High-Contrast Conversational Advisor Cards */
    .advisor-question-box {
        background: rgba(99, 102, 241, 0.14) !important;
        border: 1.5px solid #6366f1 !important;
        border-left: 5px solid #818cf8 !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        margin-top: 14px !important;
        color: #ffffff !important;
    }
    .advisor-q-badge {
        color: #a5b4fc !important;
        font-size: 11px !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 6px !important;
    }
    .advisor-q-text {
        color: #ffffff !important;
        font-size: 15.5px !important;
        font-weight: 700 !important;
        line-height: 1.5 !important;
    }
    
    .advisor-step-box {
        background: rgba(16, 185, 129, 0.12) !important;
        border: 1.5px solid #10b981 !important;
        border-left: 5px solid #34d399 !important;
        border-radius: 10px !important;
        padding: 11px 16px !important;
        margin-top: 10px !important;
        color: #ffffff !important;
        font-size: 13.5px !important;
    }
    .advisor-step-badge {
        color: #34d399 !important;
        font-weight: 800 !important;
        margin-right: 6px !important;
    }
    .advisor-step-text {
        color: #ecfdf5 !important;
        font-weight: 600 !important;
    }
    
    /* Financial Quick Pills */
    .fin-pill-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
        margin: 14px 0 10px 0;
    }
    
    .fin-pill {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 9px 12px !important;
        color: #f8fafc !important;
    }
    .fin-pill-lbl {
        font-size: 11px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    .fin-pill-val {
        font-size: 16px !important;
        color: #38bdf8 !important;
        font-weight: 800 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* NLP Token Tag */
    .nlp-token {
        display: inline-block;
        background: rgba(139, 92, 246, 0.2);
        color: #c4b5fd;
        border: 1px solid rgba(139, 92, 246, 0.4);
        border-radius: 4px;
        padding: 2px 7px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        margin: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# MACHINE LEARNING ENGINE (RANDOM FOREST BENCHMARK)
# ==========================================================

@st.cache_resource
def load_ml_assets():
    """Loads supporting Random Forest benchmark trained on historical venture outcomes."""
    model_path = "startup_model.pkl"
    features_path = "startup_features.pkl"
    if os.path.exists(model_path) and os.path.exists(features_path):
        try:
            model = joblib.load(model_path)
            features = joblib.load(features_path)
            return model, features, True
        except Exception:
            return None, None, False
    return None, None, False

ml_model, ml_features, ml_available = load_ml_assets()

def evaluate_random_forest_signal(prof, fin):
    """
    Computes historical pattern match signal using Crunchbase random forest benchmark.
    Gracefully handles missing inputs and returns supporting probability.
    """
    if not ml_available or ml_model is None or ml_features is None:
        return None
    try:
        cur_yr = datetime.now().year
        founded = cur_yr - 1 if prof.get("business_stage") == "Already Operating" else cur_yr
        age = max(0, cur_yr - founded)
        rounds = 1 if prof.get("business_stage") == "Already Operating" else 0
        tot_fund_inr = float(fin.get("initial_investment") or 200000.0)
        # Convert INR to USD equivalent (~83 INR/USD)
        funding_usd = tot_fund_inr / 83.0
        
        ind = prof.get("industry", "Technology").lower()
        is_software = 1 if any(k in ind for k in ["saas", "software", "tech", "digital"]) else 0
        is_ecommerce = 1 if any(k in ind for k in ["food", "retail", "fmcg", "dairy", "apparel"]) else 0
        is_biotech = 1 if any(k in ind for k in ["health", "bio", "med"]) else 0
        is_othercategory = 1 if not any([is_software, is_ecommerce, is_biotech]) else 0
        
        feat_dict = {
            "age_first_funding_year": 1.0,
            "age_last_funding_year": float(age),
            "age_first_milestone_year": 1.0,
            "age_last_milestone_year": float(age),
            "relationships": 5,
            "funding_rounds": rounds,
            "funding_total_usd": funding_usd,
            "milestones": max(1, min(5, int(age) + 1)),
            "is_CA": 0, "is_NY": 0, "is_MA": 0, "is_TX": 0, "is_otherstate": 1,
            "is_software": is_software, "is_web": 0, "is_mobile": 0, "is_enterprise": 0,
            "is_advertising": 0, "is_gamesvideo": 0, "is_ecommerce": is_ecommerce,
            "is_biotech": is_biotech, "is_consulting": 0, "is_othercategory": is_othercategory,
            "has_VC": 0, "has_angel": 1 if rounds > 0 else 0, "has_roundA": 0,
            "has_roundB": 0, "has_roundC": 0, "has_roundD": 0,
            "avg_participants": 2.0 if rounds > 0 else 0.0, "is_top500": 0
        }
        input_row = [feat_dict.get(f, 0) for f in ml_features]
        df_input = pd.DataFrame([input_row], columns=ml_features)
        prob = ml_model.predict_proba(df_input)[0]
        return {
            "success_probability": round(float(prob[1] * 100.0), 1),
            "status": "Active Crunchbase Model",
            "is_available": True
        }
    except Exception:
        return None

# ==========================================================
# NLP PIPELINE ENGINE: PREPROCESSING, EXTRACTION & INTENTS
# ==========================================================

STOPWORDS = {
    'this', 'that', 'with', 'from', 'have', 'were', 'which', 'their', 'there',
    'about', 'would', 'could', 'these', 'other', 'into', 'first', 'after',
    'while', 'where', 'startup', 'business', 'product', 'service', 'start',
    'want', 'will', 'sell', 'people', 'currently', 'each', 'make', 'also', 'some',
    'i', 'we', 'my', 'our', 'to', 'and', 'a', 'in', 'is', 'for', 'of', 'on', 'at'
}

def clean_and_tokenize(text):
    """Tokenization and stopword removal NLP stage."""
    raw_tokens = re.findall(r'\b[a-zA-Z]{3,}\b', (text or "").lower())
    clean_tokens = [t for t in raw_tokens if t not in STOPWORDS]
    freq = {}
    for t in clean_tokens:
        freq[t] = freq.get(t, 0) + 1
    top_keywords = [k.capitalize() for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8]]
    return clean_tokens, top_keywords

def extract_nlp_profile(raw_idea, location_input, stage_input, investment_input=None, revenue_input=None):
    """
    Academic NLP Information Extraction Engine.
    Maps unstructured natural language to structured business entities,
    customer profiles, problems, products, and provenance tags.
    """
    text = (raw_idea or "").strip()
    text_lower = text.lower()
    loc = (location_input or "").strip()
    
    clean_tokens, top_keywords = clean_and_tokenize(text)
    
    # 1. Business Category & Industry Inference
    inferred_industry = "General Enterprise & Retail"
    inferred_type = "Local Enterprise"
    
    if any(k in text_lower for k in ['milk', 'dairy', 'curd', 'ghee', 'cow', 'buffalo', 'cattle']):
        inferred_industry = "Dairy & Agriculture"
        inferred_type = "Dairy & Milk Distribution"
    elif any(k in text_lower for k in ['tailor', 'stitch', 'boutique', 'cloth', 'garment', 'apparel', 'fashion', 'sewing']):
        inferred_industry = "Apparel & Tailoring Services"
        inferred_type = "Custom Tailoring & Apparel Hub"
    elif any(k in text_lower for k in ['farm', 'crop', 'vegetable', 'fruit', 'farmer', 'harvest', 'organic', 'fpo', 'mandi']):
        inferred_industry = "AgriTech & Fresh Food Supply"
        inferred_type = "Direct Farm-to-Consumer Hub"
    elif any(k in text_lower for k in ['snack', 'food', 'millet', 'sweet', 'bakery', 'tea', 'cafe', 'cloud kitchen', 'cook']):
        inferred_industry = "Food & Packaged Goods (FMCG)"
        inferred_type = "Packaged Food & Local Delicacies"
    elif any(k in text_lower for k in ['software', 'app', 'platform', 'ai', 'algorithm', 'saas', 'cloud', 'portal', 'api']):
        inferred_industry = "Technology & Software (SaaS)"
        inferred_type = "Digital Software Platform"
    elif any(k in text_lower for k in ['doctor', 'clinic', 'health', 'medicine', 'hospital', 'patient', 'therapy', 'care']):
        inferred_industry = "HealthTech & Wellness"
        inferred_type = "Healthcare & Clinic Services"
    elif any(k in text_lower for k in ['school', 'student', 'teach', 'learn', 'tutor', 'course', 'exam', 'coaching']):
        inferred_industry = "EdTech & Education"
        inferred_type = "Education & Skill Center"
    elif any(k in text_lower for k in ['delivery', 'logistics', 'transport', 'courier', 'van', 'fleet']):
        inferred_industry = "Logistics & Supply Chain"
        inferred_type = "Hyperlocal Courier & Delivery"
        
    # 2. Target Customer Segment Extraction
    customers_found = []
    if any(k in text_lower for k in ['household', 'family', 'families', 'home', 'resident', 'neighborhood', 'village people']):
        customers_found.append("Local households and families")
    if any(k in text_lower for k in ['tea shop', 'tea stall', 'restaurant', 'cafe', 'hotel', 'shopkeeper', 'retailer', 'store']):
        customers_found.append("Local tea stalls, cafes & shops")
    if any(k in text_lower for k in ['student', 'college', 'school', 'children', 'youth']):
        customers_found.append("Students and school learners")
    if any(k in text_lower for k in ['farmer', 'grower', 'producer']):
        customers_found.append("Smallholder farmers & producers")
    if any(k in text_lower for k in ['working', 'professional', 'office', 'corporate', 'urban']):
        customers_found.append("Working professionals & office employees")
    if any(k in text_lower for k in ['clothing shop', 'boutique', 'b2b']):
        customers_found.append("Local clothing retailers & boutiques")
        
    inferred_customer = ", ".join(customers_found) if customers_found else "Local community consumers and end-users"
    
    # 3. Product / Service Extraction
    products_found = []
    if 'milk' in text_lower: products_found.append("Fresh Milk")
    if 'curd' in text_lower: products_found.append("Fresh Curd")
    if any(k in text_lower for k in ['tailor', 'stitch', 'blouse', 'shirt', 'dress']): products_found.append("Custom Tailoring & Stitching")
    if any(k in text_lower for k in ['vegetable', 'fruit', 'produce']): products_found.append("Farm-Fresh Produce")
    if any(k in text_lower for k in ['snack', 'millet', 'sweet']): products_found.append("Healthy Millet Snacks & Sweets")
    if any(k in text_lower for k in ['app', 'software', 'platform']): products_found.append("Digital Software / App")
    inferred_product = ", ".join(products_found) if products_found else "Core product/service line"
    
    # 4. Problem Statement Extraction
    pain_points = []
    if any(k in text_lower for k in ['outside', 'far away', 'distance', 'travel']):
        pain_points.append("Dependence on distant outside suppliers causing delays and stockouts")
    if any(k in text_lower for k in ['price change', 'fluctuate', 'expensive', 'costly', 'high price']):
        pain_points.append("Unpredictable pricing and frequent price hikes")
    if any(k in text_lower for k in ['middlemen', 'intermediar', 'commission', 'broker', 'mandi']):
        pain_points.append("Multiple layers of intermediaries taking away producer margins")
    if any(k in text_lower for k in ['spoil', 'stale', 'rotten', 'perish', 'fresh']):
        pain_points.append("Perishability and lack of guaranteed daily freshness")
    if any(k in text_lower for k in ['fit', 'quality', 'delay', 'alteration']):
        pain_points.append("Poor fitting, long turnaround times, or unreliable local alternatives")
        
    if not pain_points:
        pain_points.append("Lack of reliable, high-quality, and fairly-priced local direct supply")
    inferred_problem = " | ".join(pain_points)
    
    # 5. Numerical Extraction: Currency & Budget
    currency_matches = re.findall(r'(?:₹|rs\.?|inr)\s?(\d+(?:,\d+)*(?:\.\d+)?)\s*(lakh|k|cr)?', text_lower)
    extracted_inv_from_text = None
    if currency_matches:
        try:
            val_str, unit = currency_matches[0]
            val = float(val_str.replace(',', ''))
            if unit == 'lakh': val *= 100000.0
            elif unit == 'k': val *= 1000.0
            elif unit == 'cr': val *= 10000000.0
            extracted_inv_from_text = val
        except Exception:
            pass
            
    final_inv = investment_input if (investment_input and investment_input > 0) else extracted_inv_from_text
    
    # 6. Market Scope
    loc_lower = loc.lower()
    if any(k in loc_lower for k in ['village', 'town', 'taluka', 'mohalla', 'bazaar', 'colony']):
        inferred_market = "Local / Hyperlocal Community"
    elif any(k in loc_lower for k in ['district', 'pune', 'mumbai', 'bengaluru', 'state', 'delhi', 'indore']):
        inferred_market = "Regional District / Urban Cluster"
    else:
        inferred_market = "Local & Surrounding Hubs"
        
    # Provenance Mapping
    provenance = {
        "business_type": "NLP INFERRED",
        "industry": "NLP INFERRED",
        "location": "USER PROVIDED" if loc else "NLP INFERRED",
        "business_stage": "USER PROVIDED" if stage_input else "NLP INFERRED",
        "target_customer": "NLP INFERRED" if customers_found else "AI ESTIMATED",
        "problem": "NLP INFERRED" if pain_points else "AI ESTIMATED",
        "product_service": "NLP INFERRED" if products_found else "AI ESTIMATED",
        "market": "NLP INFERRED",
        "revenue_model": "NLP INFERRED",
        "pricing": "USER PROVIDED" if currency_matches else "AI ESTIMATED",
        "competition": "AI ESTIMATED",
        "initial_investment": "USER PROVIDED" if final_inv else "AI ESTIMATED",
        "monthly_revenue": "USER PROVIDED" if (revenue_input and revenue_input > 0) else "AI ESTIMATED"
    }
    
    return {
        "business_type": inferred_type,
        "industry": inferred_industry,
        "location": loc if loc else "Local Regional Cluster",
        "business_stage": stage_input if stage_input else "Planning",
        "product_service": inferred_product,
        "target_customer": inferred_customer,
        "problem": inferred_problem,
        "market": inferred_market,
        "revenue_model": "Direct Unit Sales + Local Scheduled Supply",
        "pricing": f"₹{final_inv:,.0f} budget context" if final_inv else "Fair market pricing (To be validated)",
        "competition": "Traditional informal sellers & distant commercial brands",
        "competitive_advantage": "Direct daily local supply, verified freshness, and no intermediary markups.",
        "initial_investment": final_inv,
        "monthly_revenue": revenue_input if (revenue_input and revenue_input > 0) else None,
        "monthly_expenses": None,
        "funding_requirement": None,
        "provenance": provenance,
        "clean_tokens": clean_tokens,
        "top_keywords": top_keywords,
        "confidence": "High" if len(clean_tokens) >= 12 else "Medium"
    }

def detect_user_intent(message):
    """
    Academic Intent Detection Classifier.
    Maps free-form user message to defined business intent classes.
    """
    msg = (message or "").lower().strip()
    
    # Financial Intent
    if any(k in msg for k in ['how much money', 'initial investment', 'cost', 'expense', 'budget', 'capital', 'unit economics', 'financials']):
        return "FINANCIAL_INTENT"
    # Funding Intent
    if any(k in msg for k in ['loan', 'fund', 'subsidy', 'pmegp', 'mudra', 'bank loan', 'borrow', 'investor', 'equity', 'grant', 'finance it']):
        return "FUNDING_INTENT"
    # Competition Intent
    if any(k in msg for k in ['competitor', 'competition', 'who else', 'rival', 'alternative', 'substitute', 'other shop', 'amul']):
        return "COMPETITION_INTENT"
    # Risk Intent
    if any(k in msg for k in ['risk', 'what could go wrong', 'fail', 'danger', 'threat', 'problem', 'pitfall', 'challenge']):
        return "RISK_INTENT"
    # Customer Acquisition
    if any(k in msg for k in ['get customer', 'find customer', 'marketing', 'sell to', 'acquire', 'sales', 'promote', 'outreach']):
        return "CUSTOMER_ACQUISITION_INTENT"
    # Pricing Intent
    if any(k in msg for k in ['charge', 'price', 'pricing', 'rate', 'cost per liter', 'how much to charge', 'margin']):
        return "PRICING_INTENT"
    # Action Intent
    if any(k in msg for k in ['what should i do next', 'next step', 'action', 'roadmap', 'where to start', 'first step', 'begin', 'this week']):
        return "ACTION_INTENT"
    # Validation Intent
    if any(k in msg for k in ['is my idea good', 'is this good', 'validate', 'feasibility', 'will it work', 'feedback', 'rate my idea']):
        return "VALIDATION_INTENT"
    # Operational Clarification Answer (User replying to question)
    if any(k in msg for k in ['home', 'kitchen', 'shop', 'cows', 'buffalo', 'shed', 'machine', 'mandi', 'farmer', 'yes', 'no', 'rent', 'lease']):
        return "CLARIFICATION_ANSWER"
        
    return "GENERAL_QA"

# ==========================================================
# DETERMINISTIC PYTHON FINANCIAL ENGINE
# ==========================================================

def compute_financial_engine(prof, user_inv=None, user_rev=None):
    """
    Deterministic Unit Economics & Scenario Modeling in Python.
    Strictly labels each value: USER PROVIDED, PYTHON CALCULATED, or AI ESTIMATE.
    """
    ind = prof.get("industry", "").lower()
    
    # Baseline CapEx heuristic by sector
    if "dairy" in ind or "milk" in ind:
        def_inv = 200000.0
        def_rev = 75000.0
        def_exp = 48000.0
    elif "tailor" in ind or "apparel" in ind:
        def_inv = 150000.0
        def_rev = 55000.0
        def_exp = 30000.0
    elif "agri" in ind or "food" in ind:
        def_inv = 250000.0
        def_rev = 95000.0
        def_exp = 62000.0
    elif "saas" in ind or "software" in ind or "tech" in ind:
        def_inv = 150000.0
        def_rev = 80000.0
        def_exp = 45000.0
    else:
        def_inv = 180000.0
        def_rev = 65000.0
        def_exp = 42000.0
        
    # Initial Investment
    if user_inv and user_inv > 0:
        inv = float(user_inv)
        inv_tag = "USER PROVIDED"
    else:
        inv = def_inv
        inv_tag = "AI ESTIMATE"
        
    # Monthly Revenue
    if user_rev and user_rev > 0:
        rev = float(user_rev)
        rev_tag = "USER PROVIDED"
        exp = rev * 0.65
        exp_tag = "PYTHON CALCULATED (65% OPEX)"
    else:
        rev = def_rev
        rev_tag = "AI ESTIMATE"
        exp = def_exp
        exp_tag = "AI ESTIMATE"
        
    surplus = rev - exp
    working_cap = exp * 2.5
    emergency_buf = inv * 0.15
    
    # Runway in months (Python Calculated)
    runway_months = round(inv / exp, 1) if exp > 0 else 6.0
    
    # Break-even in months (Python Calculated)
    if surplus > 0:
        be_months = round(inv / surplus, 1)
        be_text = f"~{be_months} months based on projected operating surplus"
        status = "Projected Operating Surplus"
    else:
        be_text = "Burn / Reinvestment Phase"
        status = "Pre-profit / Establishing Traction"
        
    funding_req = max(0.0, (inv + working_cap) - (inv if inv_tag == "USER PROVIDED" else 0.0))
    
    return {
        "initial_investment": inv,
        "initial_investment_tag": inv_tag,
        "monthly_revenue": rev,
        "monthly_revenue_tag": rev_tag,
        "monthly_expenses": exp,
        "monthly_expenses_tag": exp_tag,
        "monthly_surplus": surplus,
        "monthly_surplus_tag": "PYTHON CALCULATED",
        "working_capital": working_cap,
        "working_capital_tag": "PYTHON CALCULATED",
        "emergency_reserve": emergency_buf,
        "emergency_reserve_tag": "PYTHON CALCULATED",
        "runway_months": runway_months,
        "runway_text": f"~{runway_months} months before considering new revenue",
        "runway_tag": "PYTHON CALCULATED",
        "break_even_text": be_text,
        "break_even_tag": "PYTHON CALCULATED",
        "financial_status": status,
        "funding_required": funding_req
    }

def calculate_loan_emi(principal, annual_interest_rate=9.5, tenure_years=5):
    """Calculates standard monthly loan EMI in Python."""
    p = float(principal)
    if p <= 0: return 0.0
    r = (annual_interest_rate / 100.0) / 12.0
    n = tenure_years * 12
    emi = (p * r * ((1 + r) ** n)) / (((1 + r) ** n) - 1)
    return round(emi, 0)

# ==========================================================
# TRANSPARENT ASSESSMENT SCORING ENGINE (/100)
# ==========================================================

def calculate_assessment_score(prof, fin, ml_result=None):
    """
    Transparent Python assessment score across 5 core dimensions:
    1. Problem Strength (20%)
    2. Market Potential (20%)
    3. Financial Feasibility (25%)
    4. Competitive Position (15%)
    5. Execution Readiness (20%)
    """
    prob_score = 78.0 if len(prof.get("problem", "")) > 15 else 65.0
    mkt_score = 80.0 if "Local" in prof.get("market", "") else 72.0
    fin_score = 82.0 if fin.get("monthly_surplus", 0) > 0 else 60.0
    comp_score = 75.0
    exec_score = 85.0 if prof.get("business_stage") == "Already Operating" else (74.0 if prof.get("business_stage") == "Starting" else 68.0)
    
    weights = [0.20, 0.20, 0.25, 0.15, 0.20]
    sub_scores = [prob_score, mkt_score, fin_score, comp_score, exec_score]
    base_score = sum(w * s for w, s in zip(weights, sub_scores))
    
    if prof.get("business_stage") == "Already Operating" and ml_result and ml_result.get("is_available"):
        ml_prob = ml_result["success_probability"]
        final = round(base_score * 0.75 + ml_prob * 0.25, 1)
        note = f"Blended: 75% Qualitative Model ({base_score:.1f}) + 25% Historical ML Signal ({ml_prob}%)."
    else:
        final = round(base_score, 1)
        note = "100% Deterministic Multi-Dimensional Evaluation."
        
    return {
        "final_score": final,
        "problem_strength": prob_score,
        "market_potential": mkt_score,
        "financial_feasibility": fin_score,
        "competitive_position": comp_score,
        "execution_readiness": exec_score,
        "note": note
    }

# ==========================================================
# TOP 3-5 RISKS, COMPETITION, CAPITAL REASONING
# ==========================================================

def get_top_3_5_risks(prof):
    """Identifies top 3-5 curated, highly relevant business risks with mitigations."""
    ind = prof.get("industry", "").lower()
    if "dairy" in ind or "milk" in ind:
        return [
            {
                "risk": "Cattle Disease or Sudden Milk Yield Drop",
                "category": "Operational",
                "likelihood": "Medium",
                "impact": "High",
                "why": "Single infected animal immediately halts revenue and causes high medical cost.",
                "mitigation": "Partner with a local veterinary doctor for routine vaccination and secure livestock insurance."
            },
            {
                "risk": "Feed & Fodder Cost Inflation",
                "category": "Financial",
                "likelihood": "High",
                "impact": "Medium",
                "why": "Green and dry fodder prices fluctuate seasonally, eating up unit profit margins.",
                "mitigation": "Buy seasonal fodder in bulk with advance contracts or grow green fodder locally."
            },
            {
                "risk": "Milk Spoilage Due to Cold Chain Failure",
                "category": "Quality & Supply Chain",
                "likelihood": "Medium",
                "impact": "High",
                "why": "Unchilled milk sours in 4-5 hours during summer, leading to customer churn.",
                "mitigation": "Schedule direct morning/evening doorstep deliveries within 90 minutes of milking."
            }
        ]
    elif "tailor" in ind or "apparel" in ind:
        return [
            {
                "risk": "Seasonal Demand Volatility",
                "category": "Market",
                "likelihood": "High",
                "impact": "Medium",
                "why": "Orders peak during wedding/festival seasons and drop sharply during off-seasons.",
                "mitigation": "Introduce school uniform contracts or corporate workwear to guarantee year-round volume."
            },
            {
                "risk": "Skilled Stitcher Absenteeism",
                "category": "Operational",
                "likelihood": "Medium",
                "impact": "High",
                "why": "Reliance on individual stitchers leads to delayed customer orders and reputation damage.",
                "mitigation": "Standardize cutting templates and cross-train 2 part-time assistant operators."
            },
            {
                "risk": "Fabric Alteration Disputes",
                "category": "Customer Satisfaction",
                "likelihood": "Medium",
                "impact": "Medium",
                "why": "Customer fabric disputes can lead to refund demands and negative local word-of-mouth.",
                "mitigation": "Use written measurement cards signed by customer before cutting expensive fabrics."
            }
        ]
    elif "food" in ind or "snack" in ind or "millet" in ind:
        return [
            {
                "risk": "Shelf-Life Degradation & Moisture Spoilage",
                "category": "Quality Control",
                "likelihood": "High",
                "impact": "High",
                "why": "Roasted snacks lose crispness quickly if packaging foil seals are imperfect.",
                "mitigation": "Use 3-ply nitrogen-flushed or aluminum-lined foil pouches with heat-seal verification."
            },
            {
                "risk": "Unsold Inventory Returns from Grocery Stores",
                "category": "Cash Flow",
                "likelihood": "Medium",
                "impact": "High",
                "why": "Retail shopkeepers often insist on returning unsold packets after 30 days.",
                "mitigation": "Deliver in small batches of 15 packets per store to ensure fast sell-through before restocking."
            },
            {
                "risk": "Delayed Payments on Credit Sales",
                "category": "Financial",
                "likelihood": "High",
                "impact": "Medium",
                "why": "Retail stores often delay 30-day payment settlements, freezing your working capital.",
                "mitigation": "Incentivize cash-on-delivery with an extra 2% trade discount for immediate settlement."
            }
        ]
    else:
        return [
            {
                "risk": "Weak Initial Customer Adoption",
                "category": "Market",
                "likelihood": "Medium",
                "impact": "High",
                "why": "Customers continue using existing habits or traditional suppliers out of inertia.",
                "mitigation": "Offer a 7-day trial or small free sample to the first 25 target neighborhood customers."
            },
            {
                "risk": "Working Capital Exhaustion Before Cash Breakeven",
                "category": "Financial",
                "likelihood": "Medium",
                "impact": "High",
                "why": "Cash inflows lag behind upfront raw material purchase and operating expenses.",
                "mitigation": "Maintain a strict 60-day operating expense buffer in founder savings."
            },
            {
                "risk": "Price Competition from Established Incumbents",
                "category": "Competitive",
                "likelihood": "Medium",
                "impact": "Medium",
                "why": "Incumbent suppliers may drop prices temporarily to push new local entrants out.",
                "mitigation": "Compete on freshness, personalized service, and reliability rather than discount pricing."
            }
        ]

def get_competitor_analysis(prof):
    """Provides practical competitor categories labeled as AI-Suggested Alternatives."""
    ind = prof.get("industry", "").lower()
    if "dairy" in ind or "milk" in ind:
        return [
            {
                "category": "Direct Competitor Category",
                "entity": "Packaged Commercial Milk Brands (Amul, Nandini, Mother Dairy)",
                "who": "Urban & suburban households buying pouch milk",
                "alternative": "Standardized pasteurized pouch milk available at local booths",
                "strength": "Massive brand trust, long shelf life, and deep cold chain distribution",
                "differentiation": "Offer 100% farm-fresh, unadulterated raw milk delivered within 2 hours of morning milking."
            },
            {
                "category": "Traditional Alternative",
                "entity": "Informal Local Dairy Vendors / Cattle Herders",
                "who": "Local villagers and nearby tea stalls",
                "alternative": "Loose milk delivery in cans directly from small farms",
                "strength": "Existing personal relationships and flexible credit terms",
                "differentiation": "Guarantee transparent hygiene, strict lactometer/fat testing, and dependable daily delivery time."
            }
        ]
    elif "food" in ind or "snack" in ind or "millet" in ind:
        return [
            {
                "category": "Direct Competitor Category",
                "entity": "Mass-Market Commercial Snack Brands (Haldiram's, Balaji, Bikaji)",
                "who": "General grocery shoppers looking for savory snacks",
                "alternative": "Commercial fried namkeen and extruded snacks",
                "strength": "High brand visibility, long shelf life, and aggressive distributor retail margins",
                "differentiation": "Promote zero palm oil, 100% roasted millet nutrition, and fresh small-batch authentic taste."
            },
            {
                "category": "Traditional Alternative",
                "entity": "Local Neighborhood Bakeries & Sweet Marts",
                "who": "Families and tea drinkers looking for fresh evening snacks",
                "alternative": "Fresh fried samosas, farsan, and loose bakery items",
                "strength": "Immediate hot serving and local walk-in footfall",
                "differentiation": "Airtight hygienic packaging tailored for office snacking and guilt-free healthy daily consumption."
            }
        ]
    elif "tailor" in ind or "apparel" in ind:
        return [
            {
                "category": "Direct Competitor Category",
                "entity": "Local Independent Tailors & Street Boutiques",
                "who": "Nearby residents needing custom blouse, kurti, and suit stitching",
                "alternative": "Conventional neighborhood tailoring shops",
                "strength": "Familiarity and established local footfall",
                "differentiation": "Guaranteed on-time 48-hour turnaround and digital measurement logs for repeat orders."
            },
            {
                "category": "Substitute Solution",
                "entity": "Ready-Made Fast Fashion & E-Commerce",
                "who": "Younger generation and office workers buying off-the-rack",
                "alternative": "Online ready-to-wear clothing (Meesho, Myntra, local retail shops)",
                "strength": "Immediate delivery and lower perceived initial price point",
                "differentiation": "Perfect custom fit, fabric personalization, and local alteration service."
            }
        ]
    else:
        return [
            {
                "category": "Direct Competitor Category",
                "entity": "Incumbent Local Suppliers & Retail Outlets",
                "who": "Existing community buyers and walk-in customers",
                "alternative": "Traditional physical stores or existing vendors",
                "strength": "Longstanding local presence and immediate proximity",
                "differentiation": "Superior customer service, doorstep convenience, and transparent pricing."
            }
        ]

def get_capital_structure_breakdown(fin):
    """Explains practical funding options in simple language."""
    inv = fin["initial_investment"]
    return [
        {
            "source": "Founder Savings (Equity)",
            "share": "40% - 50%",
            "amount": f"₹{inv * 0.45:,.0f}",
            "purpose": "Essential equipment, upfront setup, and early deposits.",
            "advantage": "Zero debt burden; you retain 100% control and flexibility.",
            "tradeoff": "Personal financial risk if the initial tests fail."
        },
        {
            "source": "Government Subsidy / Scheme (PMEGP / Mudra)",
            "share": "30% - 35%",
            "amount": f"₹{inv * 0.35:,.0f}",
            "purpose": "Capital expenditure for machinery and productive assets.",
            "advantage": "15% to 35% capital subsidy under PMEGP; no collateral required under Mudra Shishu/Kishore.",
            "tradeoff": "Requires formal documentation (Aadhaar, Udyam registration, Project Report) and 4-8 weeks approval."
        },
        {
            "source": "Bank Working Capital Loan / Micro-Credit",
            "share": "15% - 20%",
            "amount": f"₹{inv * 0.20:,.0f}",
            "purpose": "Buffer for feed/raw materials, packaging, and operating cashflow.",
            "advantage": "Ensures the business does not run out of cash before customer revenue stabilizes.",
            "tradeoff": f"Requires monthly EMI repayment (approx. ₹{calculate_loan_emi(inv * 0.20, 9.5, 3):,.0f}/month at 9.5% for 3 years)."
        }
    ]

# ==========================================================
# CONVERSATIONAL AI: ADAPTIVE DIALOGUE & REASONING
# ==========================================================

def generate_initial_turn_response(prof, fin):
    """
    Academic Dialogue Manager - Turn 1 (Orientation & First Targeted Question).
    Accurately acknowledges founder idea, location, and budget.
    Does NOT quote arbitrary premature profits before knowing operational scale.
    Asks the single most critical operational setup question.
    """
    b_type = prof.get("business_type", "venture")
    cust = prof.get("target_customer", "local customers")
    loc = prof.get("location", "your area")
    inv = fin.get("initial_investment")
    budget_phrase = f" with an initial budget of ₹{inv:,.0f}" if (inv and inv > 0) else ""
    ind = prof.get("industry", "").lower()
    
    # Sector-specific critical first question
    if "dairy" in ind or "milk" in ind:
        first_q = "How many cows or buffaloes are you planning to start with?"
        why_ask = "Before we calculate your daily feed costs, milk yield, and operating surplus, I need one important operational detail:"
        next_step = "Estimate your available space and daily water supply capacity for animal housing."
    elif "tailor" in ind or "apparel" in ind:
        first_q = "Do you already own stitching machines and a dedicated workspace, or will they need to be acquired?"
        why_ask = "I see two critical areas to check first: equipment setup and monthly operating costs. To structure your startup budget accurately:"
        next_step = "List down the specific stitching/finishing tools you already have versus what you need to purchase."
    elif "snack" in ind or "food" in ind or "millet" in ind or "sweet" in ind:
        first_q = "Will you prepare these snacks at home initially using existing kitchen equipment, or do you plan to rent a dedicated commercial production unit?"
        why_ask = "Before we calculate your recipe batch costs and profitability, we must verify your facility overheads:"
        next_step = "Test small 500g sample batches to calculate the exact raw ingredient cost per packet."
    elif "farm" in ind or "agri" in ind or "vegetable" in ind:
        first_q = "Do you already have direct supply tie-ups with local farmers, or will you source produce from the wholesale mandi?"
        why_ask = "Before estimating margins and logistical transit costs, we need to know your sourcing model:"
        next_step = "Visit 5 local farmers or cooperative societies to confirm seasonal harvest availability."
    elif "saas" in ind or "software" in ind or "tech" in ind or "app" in ind:
        first_q = "Do you already have an early prototype/MVP developed, or are you currently designing the concept?"
        why_ask = "Before estimating development runway and cloud hosting expenses, we need to verify technical readiness:"
        next_step = "Create a simple clickable wireframe or Figma mockup of the core 2 features."
    else:
        first_q = "Will you operate from a physical roadside commercial shop with monthly rent, or distribute directly to customers?"
        why_ask = "Before we project your monthly breakeven and cash burn, I need to know your operational channel:"
        next_step = "Survey 10 target customers in your area to confirm current pricing they pay for alternatives."

    guidance = (
        f"I understand that you are planning a **{b_type}** in **{loc}** serving **{cust}**{budget_phrase}.\n\n"
        f"Our NLP engine has extracted your business entities, target segment, and value proposition. We will systematically explore your unit economics, customer acquisition, competitors, and key business risks step by step.\n\n"
        f"{why_ask}"
    )
    
    return {
        "guidance": guidance,
        "next_question": first_q,
        "next_step": next_step
    }

def call_conversational_llm(history, prof, fin, intent, user_msg, selected_model="Qwen/Qwen2.5-Coder-32B-Instruct", temperature=0.25):
    """
    Hugging Face Inference with conversational context history.
    Falls back gracefully to deterministic heuristic response if API is unavailable.
    """
    token = st.secrets.get("HF_TOKEN")
    if not token:
        return generate_heuristic_response(history, prof, fin, intent, user_msg)
        
    client = InferenceClient(api_key=token, timeout=30.0)
    defaults = [
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ]
    models = [selected_model] + [m for m in defaults if m != selected_model]
    
    # Format dialogue history context
    history_str = ""
    for m in history[-6:]:
        role = "Founder" if m.get("role") == "user" else "Advisor"
        txt = re.sub(r'<[^>]+>', '', m.get("content", "")).replace("\n", " ")
        history_str += f"{role}: {txt[:160]}\n"
        
    prompt = f"""
You are StartupSense AI, an experienced academic business advisor conversing directly with an entrepreneur.

BUSINESS PROFILE:
- Business: {prof.get('business_type')} in {prof.get('location')}
- Target Customer: {prof.get('target_customer')}
- Budget: ₹{fin.get('initial_investment', 0):,.0f} ({fin.get('initial_investment_tag')})
- Stage: {prof.get('business_stage')}

CONVERSATION HISTORY:
{history_str}

LATEST FOUNDER MESSAGE:
"{user_msg}"

DETECTED INTENT: {intent}

INSTRUCTIONS:
1. Act like a thoughtful, practical human mentor having an ongoing conversation.
2. Directly address the founder's message in 2 to 3 concise, actionable paragraphs.
3. If they answered your previous question about setup/scale, acknowledge the exact choice and explain its business/cost implication.
4. Distinguish verified facts from AI estimates.
5. Provide ONE specific immediate next step they can do this week.
6. Formulate ONE clear, practical next question to advance their planning.
7. Return strictly valid JSON:
{{
  "business_guidance": "Concise advisory text addressing the message...",
  "immediate_next_step": "Actionable single step for this week...",
  "next_question": "Single targeted question to ask next..."
}}
"""
    for m in models:
        try:
            res = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": "You are a professional business advisor. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=900,
                temperature=temperature
            )
            raw = res.choices[0].message.content
            if raw:
                clean = re.sub(r"^```[a-zA-Z]*\n?", "", raw.strip())
                clean = re.sub(r"\n?```$", "", clean).strip()
                parsed = json.loads(clean)
                return parsed
        except Exception:
            continue
            
    return generate_heuristic_response(history, prof, fin, intent, user_msg)

def generate_heuristic_response(history, prof, fin, intent, user_msg):
    """
    Deterministic Heuristic Conversational Reasoning Engine.
    Ensures 100% reliable, grounded responses across all intents and turns.
    """
    b_type = prof.get("business_type", "venture")
    cust = prof.get("target_customer", "local consumers")
    loc = prof.get("location", "local market")
    inv = fin.get("initial_investment", 200000.0)
    surplus = fin.get("monthly_surplus", 25000.0)
    runway = fin.get("runway_months", 4.2)
    msg_lower = (user_msg or "").lower()
    
    # Clarification Answers: User answering previous operational question
    if intent == "CLARIFICATION_ANSWER" or any(k in msg_lower for k in ["home", "kitchen", "cows", "buffalo", "shed", "machine", "mandi"]):
        if any(k in msg_lower for k in ["home", "kitchen"]):
            guidance = (
                f"Preparing your products from **home initially** is a smart strategic decision for your **₹{inv:,.0f} budget**.\n\n"
                f"- **Cost Savings:** It immediately eliminates **₹15,000–₹20,000/month** in commercial rent, high electricity security deposits, and advance lease commitments.\n"
                f"- **Capital Focus:** You can channel your capital towards bulk procurement of high-grade raw ingredients, moisture-proof aluminum-foil packaging pouches, and a basic local **FSSAI registration** (~₹100–₹2,000).\n\n"
                f"Now that fixed overheads are minimized, let's examine your sales distribution channels."
            )
            next_q = "Will you sell directly to households/consumers at doorstep, or supply wholesale to neighborhood grocery stores on margin?"
            step = "Draft a cost sheet calculating the raw material cost per 100g packet from home."
        elif any(k in msg_lower for k in ["cows", "buffalo", "cattle"]):
            guidance = (
                f"Starting with this livestock count gives your {b_type} a clear daily output baseline:\n\n"
                f"- **Production Expectation:** Expect approximately 45–55 liters of fresh milk daily.\n"
                f"- **Projected Revenue:** At local rates of ₹55–₹65/liter, gross monthly revenue is around **₹75,000–₹95,000**.\n"
                f"- **Main Operating Costs:** Fodder and cattle feed concentrates (~₹35,000–₹42,000/month), veterinary care, and utility costs."
            )
            next_q = "Do you already have a cattle shed and dedicated water connection, or will the shed need to be constructed?"
            step = "Contact your local veterinary dispensary to schedule livestock insurance and vaccination."
        elif any(k in msg_lower for k in ["shed", "space"]):
            guidance = (
                f"Having your shed/facility already available eliminates **₹60,000–₹80,000** in upfront construction CapEx.\n\n"
                f"This preserves your ₹{inv:,.0f} capital for purchasing productive assets and leaves a healthy **2–3 month operating reserve**."
            )
            next_q = "Will you deliver milk in cans directly to households every morning, or supply in bulk to nearby tea stalls and shops?"
            step = "Inspect your cattle housing for proper drainage, shade, and fresh water storage."
        elif any(k in msg_lower for k in ["machine", "sewing"]):
            guidance = (
                f"Commercial equipment setup is a one-time capital investment.\n\n"
                f"Budgeting approx. **₹45,000–₹60,000** for 2 commercial stitching machines and an overlock machine leaves sufficient capital for fabric rolls, accessories, and promotional signage."
            )
            next_q = "Will you operate from home initially to minimize overheads, or lease a roadside commercial shop?"
            step = "Get written quotation from 2 local sewing machine dealers for commercial motor models."
        else:
            guidance = (
                f"Thank you for confirming that setup context for your **{b_type}** in **{loc}**.\n\n"
                f"With these operational parameters defined, your working capital requirement is significantly derisked. We can now focus on customer acquisition channels and unit pricing."
            )
            next_q = "Who will be your primary paying customer: individual neighborhood households or commercial bulk buyers?"
            step = "List your top 3 operational priorities for the upcoming week."
            
    elif intent == "FINANCIAL_INTENT":
        guidance = (
            f"Here is the deterministic financial breakdown for your **{b_type}** in **{loc}**:\n\n"
            f"- **Initial Setup Budget:** ₹{inv:,.0f} ({fin.get('initial_investment_tag')})\n"
            f"- **Estimated Monthly Revenue:** ₹{fin.get('monthly_revenue', 0):,.0f} ({fin.get('monthly_revenue_tag')})\n"
            f"- **Estimated Monthly Operating Costs:** ₹{fin.get('monthly_expenses', 0):,.0f} ({fin.get('monthly_expenses_tag')})\n"
            f"- **Estimated Monthly Surplus:** ₹{surplus:,.0f} [PYTHON CALCULATED]\n\n"
            f"**Operating Runway:** At your projected operating cost of ₹{fin.get('monthly_expenses', 0):,.0f}/month, your initial capital provides approximately **{runway} months of runway**, before counting new incoming sales."
        )
        next_q = "Would you like to examine how to arrange funding (loans & subsidies), or look into your unit pricing margins?"
        step = "Draft a 1-page budget splitting your capital into Setup CapEx (60%) and Working Capital Reserve (40%)."
        
    elif intent == "FUNDING_INTENT":
        guidance = (
            f"To arrange your **₹{inv:,.0f}** capital, avoid taking high-interest informal loans. Use this recommended capital structure:\n\n"
            f"1. **Founder Savings (40–50% ~ ₹{inv*0.45:,.0f}):** Fund initial equipment and security deposits with your own money to avoid debt pressure.\n"
            f"2. **PMEGP Scheme (Prime Minister’s Employment Generation Programme):** Provides **15% to 35% capital subsidy** for manufacturing and rural enterprise projects.\n"
            f"3. **Mudra Shishu / Kishore Loan:** Provides bank micro-credit up to ₹50,000 (Shishu) or up to ₹5 lakh (Kishore) without collateral at low interest rates."
        )
        next_q = "Do you have your Aadhaar, bank passbook, and basic project description ready to apply on the Udyam / PMEGP portal?"
        step = "Register your enterprise for free on the official MSME Udyam registration portal (udyamregistration.gov.in)."
        
    elif intent == "COMPETITION_INTENT":
        guidance = (
            f"In **{loc}**, your main competitive landscape consists of two key alternatives:\n\n"
            f"1. **Commercial Packaged Brands:** Big brands with strong distribution and shelf life, but perceived as mass-produced, chemically preserved, or lacking fresh authentic taste.\n"
            f"2. **Informal Local Vendors:** Established local sellers who have existing personal relationships with customers.\n\n"
            f"**How to Win:** Do not compete on price discounts. Win on **guaranteed morning delivery, verified freshness, and clean, transparent packaging**."
        )
        next_q = "What specific quality or delivery advantage can you offer that local buyers cannot get from existing shops?"
        step = "Visit 5 neighborhood grocery stores and note the prices and packet sizes of the top-selling snacks."
        
    elif intent == "RISK_INTENT":
        top_r = get_top_3_5_risks(prof)[0]
        guidance = (
            f"The single most critical risk for your {b_type} right now is **{top_r['risk']}** ({top_r['category']} Risk):\n\n"
            f"- **Likelihood:** {top_r['likelihood']} | **Impact:** {top_r['impact']}\n"
            f"- **Why it matters:** {top_r['why']}\n"
            f"- **Practical Mitigation:** {top_r['mitigation']}\n\n"
            f"Additionally, avoid giving long credit periods (more than 7 days) to local retail stores, as unpaid dues can freeze your working capital."
        )
        next_q = "How do you plan to handle payment settlements: upfront cash-on-delivery or weekly billing?"
        step = "Establish a strict cash-on-delivery or maximum 7-day payment policy for your first 15 retail clients."
        
    elif intent == "CUSTOMER_ACQUISITION_INTENT":
        guidance = (
            f"To acquire your first 25 customers in **{loc}**, do NOT spend money on online ads. Focus on hyper-local community outreach:\n\n"
            f"1. **Free Sample Tasting:** Prepare 30 small sample pouches (50g) and distribute them to target households and office workers with a small feedback card.\n"
            f"2. **Consignment Retail Tie-Up:** Place a display jar with 15 packets in 3 popular neighborhood tea stalls and grocery shops on a 20% commission basis.\n"
            f"3. **Prepaid Weekly Subscriptions:** Offer a 5% extra discount for families who subscribe for weekly regular doorstep delivery."
        )
        next_q = "Can you identify 3 high-footfall local shops or tea stalls in your area where you can place sample jars this week?"
        step = "Pack 20 free tasting samples and give them to prospective buyers to collect immediate taste feedback."
        
    elif intent == "ACTION_INTENT":
        guidance = (
            f"Here is your clear, actionable roadmap for the next 14 days:\n\n"
            f"- **Days 1–3:** Finalize your equipment quotes and confirm your production space setup.\n"
            f"- **Days 4–7:** Produce a small test batch to calculate the exact unit cost of raw ingredients and packaging.\n"
            f"- **Days 8–10:** Secure commitments from 10 households or 2 local retail shops for your first batch.\n"
            f"- **Days 11–14:** Register on the MSME Udyam portal and apply for a basic FSSAI registration."
        )
        next_q = "Which of these 4 steps would you like help planning out first?"
        step = "Calculate the total cost of raw materials required to produce your very first 100 packets."
        
    else:
        guidance = (
            f"I understand your focus regarding your **{b_type}** in **{loc}** serving **{cust}**.\n\n"
            f"With your projected operating surplus of **₹{surplus:,.0f}/month**, the business has solid fundamental potential provided you keep upfront fixed costs low and secure repeat local buyers."
        )
        next_q = "Would you like to review how to arrange funding, or examine your competitor alternatives?"
        step = "Confirm your primary supplier or equipment distributor to get exact wholesale pricing."

    return {
        "business_guidance": guidance,
        "next_question": next_q,
        "immediate_next_step": step
    }

# ==========================================================
# STREAMLIT UI SETUP: SIDEBAR (SYSTEM ENGINES & LLM CONFIG)
# ==========================================================

apply_custom_css()

# Session State Initialization
if "session_active" not in st.session_state:
    st.session_state["session_active"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "business_profile" not in st.session_state:
    st.session_state["business_profile"] = {}
if "financials" not in st.session_state:
    st.session_state["financials"] = {}
if "nlp_data" not in st.session_state:
    st.session_state["nlp_data"] = {}
if "assessment" not in st.session_state:
    st.session_state["assessment"] = {}
if "ml_result" not in st.session_state:
    st.session_state["ml_result"] = None
if "input_idea_text" not in st.session_state:
    st.session_state["input_idea_text"] = ""
if "input_loc_text" not in st.session_state:
    st.session_state["input_loc_text"] = ""
if "input_stage_text" not in st.session_state:
    st.session_state["input_stage_text"] = "Planning"
if "input_inv_val" not in st.session_state:
    st.session_state["input_inv_val"] = 0.0
if "input_rev_val" not in st.session_state:
    st.session_state["input_rev_val"] = 0.0

# SIDEBAR: PERSISTENT ENGINES, CONFIGURATION & SNAPSHOT
with st.sidebar:
    st.markdown("### 🧠 **StartupSense AI**")
    st.caption("NLP-Based Conversational Business Advisor")
    
    # If active chat session, show consultation controls & snapshot
    if st.session_state["session_active"]:
        if st.button("🔄 Start New Business Consultation", use_container_width=True):
            st.session_state["session_active"] = False
            st.session_state["messages"] = []
            st.session_state["business_profile"] = {}
            st.session_state["financials"] = {}
            st.rerun()
            
        st.divider()
        
        prof = st.session_state["business_profile"]
        fin = st.session_state["financials"]
        score = st.session_state["assessment"]
        ml_res = st.session_state["ml_result"]
        
        # 1. Validation Assessment Score Box
        st.markdown(f"""
        <div class="snapshot-box" style="border-top: 3px solid #6366f1;">
            <div class="snapshot-title">
                <span>Business Assessment</span>
                <span style="font-size:17px; color:#818cf8; font-weight:800;">{score.get('final_score', 75)}/100</span>
            </div>
            <div style="font-size:11px; color:#94a3b8; margin-bottom:8px;">
                Transparent Python calculation across 5 key dimensions:
            </div>
            <div style="display:flex; flex-direction:column; gap:4px; font-size:12px;">
                <div class="dimension-row"><span>Problem Strength</span><b>{score.get('problem_strength', 75):.0f}%</b></div>
                <div class="dimension-row"><span>Market Potential</span><b>{score.get('market_potential', 75):.0f}%</b></div>
                <div class="dimension-row"><span>Financial Feasibility</span><b>{score.get('financial_feasibility', 75):.0f}%</b></div>
                <div class="dimension-row"><span>Competitive Position</span><b>{score.get('competitive_position', 75):.0f}%</b></div>
                <div class="dimension-row"><span>Execution Readiness</span><b>{score.get('execution_readiness', 75):.0f}%</b></div>
            </div>
            <div style="font-size:10px; color:#64748b; margin-top:8px; line-height:1.3;">
                *{score.get('note', '')}*
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Compact Live Business Snapshot
        st.markdown(f"""
        <div class="snapshot-box">
            <div class="snapshot-title">Live Business Snapshot</div>
            <div class="snapshot-row">
                <span class="snapshot-label">Business</span>
                <span class="snapshot-value">{prof.get('business_type')} <span class="tag-prov prov-inferred">INFERRED</span></span>
            </div>
            <div class="snapshot-row">
                <span class="snapshot-label">Location</span>
                <span class="snapshot-value">{prof.get('location')}</span>
            </div>
            <div class="snapshot-row">
                <span class="snapshot-label">Stage</span>
                <span class="snapshot-value">{prof.get('business_stage')}</span>
            </div>
            <div class="snapshot-row">
                <span class="snapshot-label">Initial Budget</span>
                <span class="snapshot-value">₹{fin.get('initial_investment', 0):,.0f} <span class="tag-prov prov-user">{fin.get('initial_investment_tag')}</span></span>
            </div>
            <div class="snapshot-row">
                <span class="snapshot-label">Est. Revenue</span>
                <span class="snapshot-value">₹{fin.get('monthly_revenue', 0):,.0f}/mo <span class="tag-prov prov-est">{fin.get('monthly_revenue_tag')}</span></span>
            </div>
            <div class="snapshot-row">
                <span class="snapshot-label">Est. Surplus</span>
                <span class="snapshot-value">₹{fin.get('monthly_surplus', 0):,.0f}/mo <span class="tag-prov prov-calc">CALCULATED</span></span>
            </div>
            <div class="snapshot-row">
                <span class="snapshot-label">Operating Runway</span>
                <span class="snapshot-value">~{fin.get('runway_months', 0)} mo <span class="tag-prov prov-calc">CALCULATED</span></span>
            </div>
            <div class="snapshot-row">
                <span class="snapshot-label">Financial Status</span>
                <span class="snapshot-value" style="color:#34d399;">{fin.get('financial_status')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Academic NLP Inspector Section
        with st.expander("🔬 How NLP is Used (Academic Inspector)", expanded=False):
            st.markdown("""
            **NLP Pipeline Flow:**
            1. `Raw Natural Language`
            2. `Tokenization & Stopword Filtering`
            3. `Entity & Domain Concept Extraction`
            4. `Intent Classification`
            5. `Context & Dialogue State Tracking`
            """)
            st.markdown("---")
            st.markdown("**Detected NLP Concepts:**")
            st.markdown(f"- **Business Type:** `{prof.get('business_type')}` `[INFERRED]`")
            st.markdown(f"- **Target Customer:** `{prof.get('target_customer')}` `[INFERRED]`")
            st.markdown(f"- **Core Problem:** `{prof.get('problem')}` `[INFERRED]`")
            st.markdown(f"- **Product / Line:** `{prof.get('product_service')}` `[INFERRED]`")
            st.markdown(f"- **Confidence:** `{prof.get('confidence', 'High')}`")
            st.markdown("---")
            st.markdown("**Key Semantic Tokens Extracted:**")
            tokens_html = " ".join([f"<span class='nlp-token'>{t}</span>" for t in prof.get("top_keywords", [])])
            st.markdown(f"<div>{tokens_html}</div>", unsafe_allow_html=True)
            
        st.divider()

    # 3. Persistent System Engines Section
    st.markdown("##### ⚙️ **System Engines**")
    hf_token_present = bool(st.secrets.get("HF_TOKEN"))
    if hf_token_present:
        st.markdown(
            """<div style="padding: 8px 12px; background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.35); border-radius: 8px; margin-bottom: 7px;">
            <b style="color:#34d399; font-size:12.5px;">● Hugging Face LLM Active</b><br>
            <span style="font-size:11px; color:#94a3b8;">Qwen 2.5 / DeepSeek / Llama 3.3 Connected</span>
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div style="padding: 8px 12px; background: rgba(244,63,94,0.12); border: 1px solid rgba(244,63,94,0.35); border-radius: 8px; margin-bottom: 7px;">
            <b style="color:#fb7185; font-size:12.5px;">○ HF_TOKEN Missing</b><br>
            <span style="font-size:11px; color:#94a3b8;">Add HF_TOKEN in secrets.toml (Heuristic Engine Active)</span>
            </div>""",
            unsafe_allow_html=True
        )
        
    if ml_available:
        st.markdown(
            """<div style="padding: 8px 12px; background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.35); border-radius: 8px; margin-bottom: 7px;">
            <b style="color:#60a5fa; font-size:12.5px;">● Random Forest ML Loaded</b><br>
            <span style="font-size:11px; color:#94a3b8;">31-Feature Crunchbase Model Benchmark</span>
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div style="padding: 8px 12px; background: #1e293b; border: 1px solid #334155; border-radius: 8px; margin-bottom: 7px;">
            <b style="color:#94a3b8; font-size:12.5px;">○ ML Model Benchmark</b><br>
            <span style="font-size:11px; color:#64748b;">Crunchbase historical model not found</span>
            </div>""",
            unsafe_allow_html=True
        )
        
    st.markdown(
        """<div style="padding: 8px 12px; background: rgba(139,92,246,0.12); border: 1px solid rgba(139,92,246,0.35); border-radius: 8px; margin-bottom: 7px;">
        <b style="color:#a78bfa; font-size:12.5px;">● NLP Concept Parser</b><br>
        <span style="font-size:11px; color:#94a3b8;">Entity, Intent & Provenance Engine Active</span>
        </div>""",
        unsafe_allow_html=True
    )
    
    st.divider()

    # 4. LLM Engine Configuration
    st.markdown("##### 🤖 **LLM Engine Configuration**")
    ai_backend_model = st.selectbox(
        "Active AI Model:",
        [
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        ],
        index=0
    )
    llm_temperature = st.slider("Model Temperature", min_value=0.0, max_value=0.8, value=0.25, step=0.05)

    st.divider()
    
    # 5. Academic Principles
    st.markdown("##### 📌 **Academic Principles**")
    st.info(
        "**Primary UX Principle:**\n\n"
        "*'Ask the user less. Analyze more.'*\n\n"
        "The user is not required to understand business terminology. The NLP pipeline extracts structured parameters automatically from free-form natural language."
    )

# Predefined Demo Ideas for Instant 1-Click Testing
DEMOS = {
    "Village Dairy": {
        "idea": "I want to start a small dairy business in my village. I have about ₹2 lakh and want to sell fresh milk and curd to nearby households and tea shops. People currently depend on outside suppliers and prices change frequently.",
        "location": "Narayangaon, Pune, Maharashtra",
        "stage": "Planning",
        "inv": 200000.0,
        "rev": 0.0
    },
    "Local Tailoring Hub": {
        "idea": "I want to open a tailoring and custom stitching shop in my village with ₹1.5 lakh budget. I want to stitch clothes for local households and school students because people currently travel 12 km to town for quality fitting.",
        "location": "Shirwal, Satara, Maharashtra",
        "stage": "Planning",
        "inv": 150000.0,
        "rev": 0.0
    },
    "Healthy Millet Snacks": {
        "idea": "I want to manufacture and sell roasted healthy millet snacks and traditional sweets to neighborhood grocery stores and working professionals. I have ₹3 lakh initial savings.",
        "location": "Indore, Madhya Pradesh",
        "stage": "Starting",
        "inv": 300000.0,
        "rev": 0.0
    },
    "Farm-to-Consumer Hub": {
        "idea": "I want to connect 20 local organic vegetable farmers directly to residential apartment societies in the city. Eliminating middlemen will give better rates to farmers and fresh produce to families.",
        "location": "Bengaluru, Karnataka",
        "stage": "Already Operating",
        "inv": 500000.0,
        "rev": 120000.0
    }
}

# ----------------------------------------------------------
# ONBOARDING VIEW (MINIMAL NATURAL LANGUAGE INPUT)
# ----------------------------------------------------------

if not st.session_state["session_active"]:
    st.markdown("""
    <div class="brand-hero">
        <div class="academic-badge">🎓 Academic AI & NLP Project | Natural Language Understanding Engine</div>
        <div class="brand-title"><span class="brand-gradient">StartupSense AI</span></div>
        <div class="brand-subtitle">NLP-Based Conversational Business Advisor</div>
        <div class="brand-desc">
            Describe your business idea in your own everyday words. Our NLP system extracts your business model, customer segments, unit economics, and guides you step by step through interactive conversation.
        </div>
        <div class="flow-badge-container">
            <span class="flow-step-pill">1. User Natural Language</span>
            <span class="flow-step-arrow">→</span>
            <span class="flow-step-pill">2. Text Preprocessing</span>
            <span class="flow-step-arrow">→</span>
            <span class="flow-step-pill">3. Entity & Concept Extraction</span>
            <span class="flow-step-arrow">→</span>
            <span class="flow-step-pill">4. Intent Detection</span>
            <span class="flow-step-arrow">→</span>
            <span class="flow-step-pill">5. Step-by-Step Advisor</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 1-Click Demo Buttons Row
    st.markdown("##### 💡 **Instant Academic Demonstrations (1-Click Test):**")
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        if st.button("🥛 Village Dairy (₹2L)", use_container_width=True):
            d = DEMOS["Village Dairy"]
            st.session_state["input_idea_text"] = d["idea"]
            st.session_state["input_loc_text"] = d["location"]
            st.session_state["input_stage_text"] = d["stage"]
            st.session_state["input_inv_val"] = d["inv"]
            st.session_state["input_rev_val"] = d["rev"]
            st.rerun()
    with col_d2:
        if st.button("🧵 Local Tailoring (₹1.5L)", use_container_width=True):
            d = DEMOS["Local Tailoring Hub"]
            st.session_state["input_idea_text"] = d["idea"]
            st.session_state["input_loc_text"] = d["location"]
            st.session_state["input_stage_text"] = d["stage"]
            st.session_state["input_inv_val"] = d["inv"]
            st.session_state["input_rev_val"] = d["rev"]
            st.rerun()
    with col_d3:
        if st.button("🍪 Healthy Snacks (₹3L)", use_container_width=True):
            d = DEMOS["Healthy Millet Snacks"]
            st.session_state["input_idea_text"] = d["idea"]
            st.session_state["input_loc_text"] = d["location"]
            st.session_state["input_stage_text"] = d["stage"]
            st.session_state["input_inv_val"] = d["inv"]
            st.session_state["input_rev_val"] = d["rev"]
            st.rerun()
    with col_d4:
        if st.button("🥦 Farm-to-Consumer", use_container_width=True):
            d = DEMOS["Farm-to-Consumer Hub"]
            st.session_state["input_idea_text"] = d["idea"]
            st.session_state["input_loc_text"] = d["location"]
            st.session_state["input_stage_text"] = d["stage"]
            st.session_state["input_inv_val"] = d["inv"]
            st.session_state["input_rev_val"] = d["rev"]
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Minimal Onboarding Container
    with st.container(border=True):
        st.markdown("### 📝 Enter Your Business Idea Once")
        st.caption("No business terminology or long questionnaires required. Express your thoughts naturally.")
        
        # A. Large Natural Language Text Area
        idea_val = st.text_area(
            "What is your business idea? *",
            value=st.session_state["input_idea_text"],
            placeholder="Describe what you want to sell or provide, who you want to serve, where you want to operate, and the problem you want to solve.",
            height=140
        )
        
        col_b1, col_b2 = st.columns([1.5, 1])
        with col_b1:
            # B. Location
            loc_val = st.text_input(
                "Where will you operate? *",
                value=st.session_state["input_loc_text"],
                placeholder="Village / Town / District / State (e.g., Narayangaon, Pune, Maharashtra)"
            )
        with col_b2:
            # C. Stage
            stages = ["Planning", "Starting", "Already Operating"]
            idx = stages.index(st.session_state["input_stage_text"]) if st.session_state["input_stage_text"] in stages else 0
            stage_val = st.selectbox("Current Stage *", stages, index=idx)
            
        st.markdown("##### 🪙 Optional Financial Context *(Leave empty if unsure — AI will estimate realistic unit economics)*")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            # D. Initial Investment
            inv_val = st.number_input(
                "Approximate Initial Budget (₹ INR) [Optional]",
                min_value=0.0,
                value=float(st.session_state["input_inv_val"]),
                step=25000.0,
                help="Total money you plan to invest or have invested."
            )
        with col_f2:
            # E. Current Monthly Revenue
            if stage_val == "Already Operating":
                rev_val = st.number_input(
                    "Current Monthly Revenue (₹ INR) [Optional for Operating]",
                    min_value=0.0,
                    value=float(st.session_state["input_rev_val"]),
                    step=25000.0
                )
            else:
                rev_val = 0.0
                st.caption("ℹ️ *Monthly revenue is only applicable for businesses that are Already Operating.*")
                
        st.divider()
        col_btn, col_note = st.columns([1.2, 2.5])
        with col_btn:
            start_btn = st.button("🚀 Start AI Advisory Conversation", type="primary", use_container_width=True)
        with col_note:
            st.markdown("""
            <div style="font-size: 13px; color: #94a3b8; padding-top: 6px;">
            ⚡ <b>Core Philosophy:</b> Enter your idea once → Chat with the AI advisor → Get step-by-step guidance.
            </div>
            """, unsafe_allow_html=True)
            
    if start_btn:
        if not idea_val.strip():
            st.error("⚠️ Please describe your business idea before starting.")
            st.stop()
            
        # Step 1: Run Academic NLP Pipeline
        with st.spinner("🧠 Running NLP entity extraction, intent detection & concept mapping..."):
            nlp_res = extract_nlp_profile(idea_val, loc_val, stage_val, inv_val, rev_val)
            nlp_res["business_idea"] = idea_val
            
            # Step 2: Compute Deterministic Python Financials
            fin_res = compute_financial_engine(nlp_res, user_inv=inv_val, user_rev=rev_val)
            
            # Step 3: Run Optional Scikit-Learn Historical Benchmark (for Existing Business)
            ml_res = None
            if stage_val == "Already Operating" and ml_available:
                ml_res = evaluate_random_forest_signal(nlp_res, fin_res)
                
            # Step 4: Transparent Multi-Dimensional Assessment Score
            score_res = calculate_assessment_score(nlp_res, fin_res, ml_res)
            
            # Step 5: Initial Conversational Advice (Turn 1)
            init_turn = generate_initial_turn_response(nlp_res, fin_res)
            
            # Save into Session State
            st.session_state["business_profile"] = nlp_res
            st.session_state["financials"] = fin_res
            st.session_state["nlp_data"] = nlp_res
            st.session_state["assessment"] = score_res
            st.session_state["ml_result"] = ml_res
            st.session_state["session_active"] = True
            
            # Add to Chat Messages
            st.session_state["messages"] = [
                {
                    "role": "user",
                    "content": f"**Business Idea:** {idea_val}\n\n**Location:** {loc_val or 'Local'} | **Stage:** {stage_val}" + (f" | **Budget:** ₹{inv_val:,.0f}" if inv_val > 0 else "")
                },
                {
                    "role": "assistant",
                    "content": init_turn["guidance"],
                    "next_step": init_turn["next_step"],
                    "next_question": init_turn["next_question"],
                    "intent": "INITIAL_UNDERSTANDING"
                }
            ]
            st.rerun()

# ----------------------------------------------------------
# ACTIVE CONVERSATIONAL WORKSPACE (CHATBOT UI)
# ----------------------------------------------------------

else:
    prof = st.session_state["business_profile"]
    fin = st.session_state["financials"]
    score = st.session_state["assessment"]
    ml_res = st.session_state["ml_result"]

    # MAIN AREA: CHAT CONVERSATION
    st.markdown(f"""
    <div class="chat-status-bar">
        <div>
            <span style="font-weight:700; color:#f8fafc; font-size:16px;">{prof.get('business_type')}</span>
            <span style="color:#94a3b8; font-size:13px; margin-left:10px;">📍 {prof.get('location')} &nbsp;|&nbsp; 🏷️ {prof.get('business_stage')}</span>
        </div>
        <div style="font-size:12px; font-weight:700; color:#818cf8; background:rgba(99, 102, 241, 0.15); border:1px solid rgba(99, 102, 241, 0.35); padding:4px 12px; border-radius:20px;">
            ● AI Conversational Advisor Active
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display Chat History
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(msg["content"])
                
                # Show financial quick-pill only if financial topic was discussed
                if msg.get("intent") in ["FINANCIAL_INTENT", "FUNDING_INTENT"]:
                    st.markdown(f"""
                    <div class="fin-pill-grid">
                        <div class="fin-pill">
                            <div class="fin-pill-lbl">Initial Budget</div>
                            <div class="fin-pill-val">₹{fin.get('initial_investment', 0):,.0f}</div>
                        </div>
                        <div class="fin-pill">
                            <div class="fin-pill-lbl">Est. Monthly Surplus</div>
                            <div class="fin-pill-val" style="color:#34d399;">+₹{fin.get('monthly_surplus', 0):,.0f}</div>
                        </div>
                        <div class="fin-pill">
                            <div class="fin-pill-lbl">Calculated Runway</div>
                            <div class="fin-pill-val">{fin.get('runway_months', 0)} Months</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # High-contrast Advisor Question Box
                if msg.get("next_question"):
                    st.markdown(f"""
                    <div class="advisor-question-box">
                        <div class="advisor-q-badge">💬 Advisor Question (Next Step in Planning)</div>
                        <div class="advisor-q-text">{msg['next_question']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # High-contrast Immediate Next Step Box
                if msg.get("next_step"):
                    st.markdown(f"""
                    <div class="advisor-step-box">
                        <span class="advisor-step-badge">🎯 Immediate Action:</span>
                        <span class="advisor-step-text">{msg['next_step']}</span>
                    </div>
                    """, unsafe_allow_html=True)

    # QUICK ACTION BUTTONS (Clean 3x2 Grid)
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    st.markdown("##### ⚡ **Quick Strategic Questions:**")
    qc1, qc2, qc3 = st.columns(3)
    quick_query = None
    
    with qc1:
        if st.button("💰 How much money do I need?", use_container_width=True):
            quick_query = "How much money do I need to start and what are the operating expenses?"
        if st.button("⚠️ What are my top risks?", use_container_width=True):
            quick_query = "What are the biggest risks that could make this business fail?"
    with qc2:
        if st.button("🏦 How should I arrange funds?", use_container_width=True):
            quick_query = "How should I arrange the money? Should I take a bank loan or government subsidy like PMEGP/Mudra?"
        if st.button("🎯 How do I get first customers?", use_container_width=True):
            quick_query = "How should I acquire my first 20 paying customers?"
    with qc3:
        if st.button("⚔️ Who are my competitors?", use_container_width=True):
            quick_query = "Who are my main competitors and alternatives in this area?"
        if st.button("🚀 What should I do this week?", use_container_width=True):
            quick_query = "What is the exact next step I should execute this week?"
            
    # CHAT INPUT
    user_input = st.chat_input("Reply to the advisor's question or ask anything...")
    effective_query = quick_query or user_input
    
    if effective_query:
        # Detect User Intent
        detected_intent = detect_user_intent(effective_query)
        
        # Add User Message to History
        st.session_state["messages"].append({
            "role": "user",
            "content": effective_query
        })
        
        # Generate Advisor Response with user-selected model & temperature
        with st.spinner("🧠 Advisor analyzing context & preparing practical guidance..."):
            resp = call_conversational_llm(
                st.session_state["messages"],
                st.session_state["business_profile"],
                st.session_state["financials"],
                detected_intent,
                effective_query,
                selected_model=ai_backend_model,
                temperature=llm_temperature
            )
            
            st.session_state["messages"].append({
                "role": "assistant",
                "content": resp["business_guidance"],
                "next_step": resp.get("immediate_next_step"),
                "next_question": resp.get("next_question"),
                "intent": detected_intent
            })
            
        st.rerun()

    # Expandable Full Business Summary (On Demand, NOT Cluttered)
    with st.expander("📋 View Comprehensive Business Summary & Financial Plan (On Demand)", expanded=False):
        tab_sum, tab_fin, tab_comp, tab_risk = st.tabs(["📊 Business Profile", "💰 Capital & Finance", "⚔️ Competitors", "⚠️ Top Risks"])
        
        with tab_sum:
            st.markdown(f"""
            - **Business Idea:** {prof.get('business_idea', '')}
            - **Industry:** {prof.get('industry')} | **Type:** {prof.get('business_type')}
            - **Target Customers:** {prof.get('target_customer')}
            - **Problem Solved:** {prof.get('problem')}
            - **Product / Line:** {prof.get('product_service')}
            - **Market Scope:** {prof.get('market')}
            """)
            
        with tab_fin:
            st.markdown("#### Deterministic Financial Structure")
            c_f1, c_f2, c_f3 = st.columns(3)
            c_f1.metric("Initial Investment", f"₹{fin['initial_investment']:,.0f}", help=fin['initial_investment_tag'])
            c_f2.metric("Monthly Revenue", f"₹{fin['monthly_revenue']:,.0f}", help=fin['monthly_revenue_tag'])
            c_f3.metric("Monthly Surplus", f"₹{fin['monthly_surplus']:,.0f}", help=fin['monthly_surplus_tag'])
            
            st.markdown(f"**Operating Runway:** {fin['runway_text']} `[PYTHON CALCULATED]`")
            st.markdown(f"**Break-Even Timeline:** {fin['break_even_text']} `[PYTHON CALCULATED]`")
            
            st.markdown("##### Recommended Capital Allocation")
            cap_items = get_capital_structure_breakdown(fin)
            for c in cap_items:
                st.markdown(f"**{c['source']} ({c['share']} ~ {c['amount']})**")
                st.markdown(f"*{c['purpose']}* — **Advantage:** {c['advantage']} | **Trade-off:** {c['tradeoff']}")
                
        with tab_comp:
            st.markdown("#### AI-Suggested Competitor & Alternative Analysis")
            comps = get_competitor_analysis(prof)
            for item in comps:
                st.markdown(f"**{item['category']}: {item['entity']}**")
                st.markdown(f"- *Alternative Provided:* {item['alternative']}")
                st.markdown(f"- *Their Strength:* {item['strength']}")
                st.markdown(f"- *Where Startup Differentiates:* **{item['differentiation']}**")
                st.markdown("---")
                
        with tab_risk:
            st.markdown("#### Top 3-5 Critical Business Risks")
            risks = get_top_3_5_risks(prof)
            for r in risks:
                st.markdown(f"**⚠️ {r['risk']}** ({r['category']} | Likelihood: `{r['likelihood']}` | Impact: `{r['impact']}`)")
                st.markdown(f"*{r['why']}*")
                st.markdown(f"**Mitigation:** {r['mitigation']}")
                st.markdown("---")