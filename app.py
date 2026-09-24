"""
StartupSense AI – Intelligent Startup Analysis & Business Advisory Platform
Refactored and upgraded from AI-Powered Startup Validator.
Integrates:
- Modern Streamlit Dashboard UI
- Natural Language Processing (NLP) Entity & Concept Extraction
- Hugging Face Large Language Models (Qwen 2.5 / Fallbacks)
- Scikit-Learn Random Forest Historical Pattern Analysis
- Dynamic Financial Modeling & Break-Even Visualizations
- Deep Risk Matrix & SWOT Benchmarking
- Multi-Stage Action Roadmap & Report Generation
"""

import streamlit as st
from huggingface_hub import InferenceClient
import pandas as pd
import numpy as np
import joblib
import json
import re
from datetime import datetime
import altair as alt

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="StartupSense AI – Intelligent Startup Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# MODERN SAAS / AI DASHBOARD CSS
# ==========================================================

def apply_custom_css():
    st.markdown("""
    <style>
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .block-container {
        max-width: 1240px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }
    
    /* Header Gradient & Badges */
    .brand-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
        text-align: center;
    }
    
    .brand-gradient {
        background: linear-gradient(120deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .brand-subtitle {
        font-size: 16px;
        color: #64748b;
        text-align: center;
        max-width: 820px;
        margin: 0 auto 20px auto;
        line-height: 1.5;
    }
    
    /* Architecture Flow Banner */
    .flow-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 12px 18px;
        background: rgba(59, 130, 246, 0.05);
        border: 1px solid rgba(59, 130, 246, 0.18);
        border-radius: 14px;
        margin: 10px auto 26px auto;
        font-size: 13px;
        font-weight: 600;
        color: #3b82f6;
    }
    
    .flow-step {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border: 1px solid rgba(59, 130, 246, 0.15);
    }
    
    .flow-arrow {
        color: #94a3b8;
        font-weight: 700;
    }
    
    /* Section Headers & Badges */
    .section-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15));
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #3b82f6;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        margin-right: 8px;
    }
    
    .section-title {
        font-size: 20px;
        font-weight: 700;
        display: inline-block;
        vertical-align: middle;
        margin-bottom: 4px;
    }
    
    .section-desc {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 14px;
    }
    
    /* Modern Content Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
    }
    
    .metric-hero {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(139, 92, 246, 0.08));
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .metric-value-large {
        font-size: 54px;
        font-weight: 800;
        line-height: 1.1;
        margin: 6px 0;
    }
    
    .tag-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 3px;
    }
    
    .tag-blue { background: rgba(59, 130, 246, 0.15); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); }
    .tag-green { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .tag-amber { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .tag-rose { background: rgba(244, 63, 94, 0.15); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.3); }
    .tag-purple { background: rgba(139, 92, 246, 0.15); color: #8b5cf6; border: 1px solid rgba(139, 92, 246, 0.3); }

    /* Custom Form & Button Polish */
    div.stButton > button {
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.2s ease;
    }
    
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea, div[data-testid="stSelectbox"] > div {
        border-radius: 10px;
    }
    
    /* Code/JSON Prettification */
    code {
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Analysis Block Cards */
    .analysis-block {
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        background: rgba(59, 130, 246, 0.03);
        border-radius: 0 12px 12px 0;
        margin-bottom: 16px;
    }
    
    .analysis-meta {
        font-size: 12px;
        color: #64748b;
        margin-top: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# ==========================================================
# ML MODEL LOADING (RANDOM FOREST)
# ==========================================================

@st.cache_resource
def load_ml_model():
    """Loads historical startup Random Forest classifier and feature names."""
    try:
        model = joblib.load("startup_model.pkl")
        features = joblib.load("startup_features.pkl")
        return model, features, True
    except Exception as e:
        return None, None, False

ml_model, ml_features, ml_available = load_ml_model()

# ==========================================================
# DEMO DATASET PROVIDER
# ==========================================================

def get_demo_data(mode="Existing"):
    """Returns complete real-world startup demo (VillageFresh) for live demonstrations."""
    is_ext = (mode == "Existing")
    return {
        "analysis_mode": "Existing" if is_ext else "New",
        "startup_name": "VillageFresh",
        "founder_name": "Arun Sharma & Team",
        "industry": "AgriTech",
        "startup_stage": "Growth" if is_ext else "Early Customers",
        "target_country": "India",
        "target_state": "Maharashtra",
        "target_city": "Pune & Western Maharashtra",
        "target_customer": "Urban middle-to-upper-income families seeking chemical-free farm produce and perishable staples.",
        
        "startup_idea": (
            "A farm-to-table direct-to-consumer agri-logistics company linking 420 verified partner farmers across 3 FPOs with 45 urban housing societies. We provide daily next-morning harvest delivery through temperature-controlled micro-hubs, currently serving 4,500 monthly recurring household orders."
            if is_ext else
            "A farm-to-table direct-to-consumer agri-logistics platform that links smallholder fruit and vegetable farmers with urban housing societies, providing guaranteed next-morning harvest delivery through temperature-controlled micro-hubs."
        ),
        "problem": "Traditional agricultural distribution relies on 4-6 intermediaries, causing 30-35% harvest spoilage, 48-hour delays, and leaving farmers with only 22-28% of retail prices while consumers receive stale produce laced with preservation chemicals.",
        "who_experiences": "Over 40 million tier-1/tier-2 urban apartment dwellers and 12 million smallholder farmers across western and southern India.",
        "frequency": "Daily perishable staple consumption (vegetables, greens, and fruits).",
        "current_alternatives": "Local wet vegetable mandis, roadside street vendors, and dark-store grocery quick-commerce apps (Blinkit, Zepto, BigBasket).",
        "why_insufficient": "Quick-commerce delivers stored warehouse stock with high plastic packaging overheads and premium pricing. Mandis lack cold-chain hygiene and quality assurance. Farmers still receive low wholesale rates.",
        
        "proposed_solution": "VillageFresh aggregates demand directly through neighborhood housing society drop-points, dispatches harvest schedules to verified partner farmer clusters at 4 PM, collects produce by 8 PM, and executes zero-warehouse consolidated deliveries by 7 AM.",
        "core_features": "Society group-buying portal, farmer fair-price transparent escrow, QR-code harvest traceability, IoT temperature-tracked transit crates, and dynamic demand forecasting.",
        "uvp": "Guaranteed harvest-to-kitchen delivery within 14 hours at mandi-equivalent prices, while paying farmers 35% above APMC market rates.",
        "competitive_advantage": "Society cluster delivery eliminates dark-store real estate overheads and reduces last-mile unit delivery cost from ₹65 to ₹12 per basket.",
        "tech_used": "Mobile App (Flutter), Demand Prediction Model (Python/LightGBM), Route Optimization Engine, IoT LoRaWAN temperature trackers.",
        "ip_differentiation": "Proprietary predictive demand clustering and society-level batch logistics dispatch algorithm.",
        
        "primary_customer": "Health-conscious suburban families, working couples, and parents managing family diets.",
        "secondary_customer": "Neighborhood organic cafes, cloud kitchens, and gated community cooperatives.",
        "customer_location": "Pune, Pimpri-Chinchwad, and expansion into Navi Mumbai.",
        "estimated_segment": "Approximately 450,000 households across 2,400 gated residential complexes.",
        "market_type": "Regional",
        "expected_volume": "12,000 monthly active household subscribers within 12 months.",
        "known_competitors": "BigBasket, Otipy, Country Delight, Fraazo, Local APMC vendors",
        "competition_known": True,
        
        "revenue_model": "Direct Sales",
        "pricing": "Average basket value of ₹450 with 28% gross margin, plus ₹199 monthly premium subscription for priority morning slots.",
        "expected_monthly_customers": 4500,
        "expected_monthly_revenue": 2250000.0,
        "cac": 380.0,
        "monthly_operating_cost": 1650000.0,
        "initial_investment": 3500000.0,
        "working_capital": 1200000.0,
        "funding_required": 5000000.0,
        "currency_symbol": "₹",
        
        "team_size": 8,
        "founder_experience": "6 years combined experience in supply chain operations at Flipkart and agronomical research at MPKV Rahuri.",
        "technical_skills": "Full-stack development, route planning algorithms, IoT hardware integration, SQL & analytics.",
        "business_skills": "Vendor negotiation, B2C growth marketing, farmer cooperative engagement, regulatory compliance.",
        "available_infra": "2 leased aggregation centers near Narayangaon and 4 refrigerated transit vans.",
        "partnerships": "Partnered with 3 Farmer Producer Organizations (FPOs) representing 420 smallholder farmers.",
        "mentors_advisors": "Retired NABARD General Manager and ex-VP of Logistics at Delhivery.",
        "available_resources": "Initial bootstrap capital, verified farmer FPO agreements, pilot operations in 18 housing societies.",
        
        # Operational ML signals (for Existing Startup Mode)
        "founded_year": 2023,
        "total_funding": 150000.0,
        "funding_rounds": 1,
        "first_funding_year": 2024,
        "latest_funding_year": 2024,
        "connections": 12,
        "has_vc": False,
        "has_angel": True,
        "has_round_a": False,
        "has_round_b": False,
        "has_round_c": False,
        "has_round_d": False
    }

# ==========================================================
# NLP PROCESSING & FEATURE EXTRACTION LAYER
# ==========================================================

def extract_nlp_insights(text_corpus, industry_hint, stage_hint, revenue_hint):
    """
    Simulates / extracts transparent NLP entities, semantic concepts,
    intent classification, and key customer pain points from raw text.
    """
    corpus_lower = text_corpus.lower()
    
    # Keyword extraction using regex frequency & business stop-words
    words = re.findall(r'\b[a-zA-Z]{4,}\b', corpus_lower)
    stopwords = {
        'this', 'that', 'with', 'from', 'have', 'were', 'which', 'their', 'there',
        'about', 'would', 'could', 'these', 'other', 'into', 'first', 'after',
        'while', 'where', 'startup', 'business', 'product', 'service', 'customer',
        'customers', 'users', 'market', 'solution', 'system', 'platform'
    }
    freq = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
            
    top_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    keyword_list = [k[0].capitalize() for k in top_keywords]
    
    # Entity & Concept Recognition
    detected_entities = []
    if any(k in corpus_lower for k in ['farmer', 'agriculture', 'harvest', 'crop', 'produce', 'soil']):
        detected_entities.append("Agri-Supply Chain")
    if any(k in corpus_lower for k in ['society', 'urban', 'apartment', 'family', 'consumer', 'household']):
        detected_entities.append("Urban Residential Consumer")
    if any(k in corpus_lower for k in ['cold-chain', 'delivery', 'transit', 'logistics', 'vehicle', 'route']):
        detected_entities.append("Micro-Logistics & Delivery")
    if any(k in corpus_lower for k in ['iot', 'ai', 'algorithm', 'app', 'software', 'platform']):
        detected_entities.append("Digital Orchestration Tech")
    if any(k in corpus_lower for k in ['margin', 'subscription', 'commission', 'sales', 'pricing', 'revenue']):
        detected_entities.append("Unit Margin Monetization")
        
    if not detected_entities:
        detected_entities = ["Digital Platform", "Consumer Engagement", "Operational Logistics"]

    # Customer Pain Points Extracted
    pain_points = []
    if any(k in corpus_lower for k in ['spoilage', 'waste', 'perish', 'loss']):
        pain_points.append("High Perishable Harvest Spoilage & Food Waste")
    if any(k in corpus_lower for k in ['middlemen', 'intermediar', 'commission', 'broker']):
        pain_points.append("Excessive Intermediary Margin Depletion")
    if any(k in corpus_lower for k in ['cost', 'expensive', 'high price', 'premium']):
        pain_points.append("Inflated Retail Costs for End Consumers")
    if any(k in corpus_lower for k in ['delay', 'slow', 'stale', 'time']):
        pain_points.append("Prolonged Transit Latency Leading to Stale Goods")
    if any(k in corpus_lower for k in ['quality', 'chemical', 'hygiene', 'safety']):
        pain_points.append("Inconsistent Quality and Preservation Chemical Risks")
        
    if not pain_points:
        pain_points.append("Lack of Transparent and Reliable Direct Market Access")
        pain_points.append("High Operational Inefficiencies in Existing Channels")

    # Business Opportunities Extracted
    opportunities = [
        f"Consolidated neighborhood drop-offs to compress last-mile delivery costs",
        f"Direct price transparency unlocking higher producer retention and brand loyalty",
        f"Value-added subscription or pre-scheduled baskets for recurring monthly predictability"
    ]

    return {
        "detected_industry": industry_hint,
        "detected_customer": detected_entities[0] if detected_entities else "Target Demographics",
        "detected_problem": pain_points[0] if pain_points else "Inefficient Distribution",
        "detected_business_model": revenue_hint,
        "detected_key_concepts": detected_entities,
        "extracted_keywords": keyword_list if keyword_list else ["AgriTech", "Delivery", "Supply", "Direct"],
        "customer_pain_points": pain_points,
        "business_opportunities": opportunities
    }

# ==========================================================
# FINANCIAL MODELING ENGINE
# ==========================================================

def calculate_financials(monthly_revenue, monthly_expenses, initial_investment, working_capital, funding_required):
    """
    Computes unit economics, annual run-rates, and break-even timeline.
    All formulas are deterministic and Python-derived.
    """
    monthly_rev = float(max(0.0, monthly_revenue))
    monthly_exp = float(max(0.0, monthly_expenses))
    init_inv = float(max(0.0, initial_investment))
    work_cap = float(max(0.0, working_capital))
    fund_req = float(max(0.0, funding_required))
    
    monthly_profit = monthly_rev - monthly_exp
    annual_revenue = monthly_rev * 12.0
    annual_operating_cost = monthly_exp * 12.0
    annual_profit = monthly_profit * 12.0
    
    # Net burn or net generation
    if monthly_profit > 0:
        # Profitable, calculate break-even period on initial investment
        break_even_months = round(init_inv / monthly_profit, 1) if monthly_profit > 0 else 0.0
        break_even_desc = f"{break_even_months} months based on current monthly net profit"
        financial_status = "Profitable Operations"
    elif monthly_profit < 0:
        monthly_burn = abs(monthly_profit)
        runway_months = round((work_cap + init_inv) / monthly_burn, 1) if monthly_burn > 0 else 0.0
        break_even_desc = f"Cash runway: ~{runway_months} months under current net burn (₹{monthly_burn:,.0f}/mo)"
        financial_status = "Capital Intensive / Burn Mode"
    else:
        break_even_desc = "Operating at exact break-even (zero net monthly margin)"
        financial_status = "Zero Net Margin"
        
    return {
        "monthly_revenue": monthly_rev,
        "monthly_expenses": monthly_exp,
        "monthly_profit": monthly_profit,
        "annual_revenue": annual_revenue,
        "annual_operating_cost": annual_operating_cost,
        "annual_profit": annual_profit,
        "break_even_months": break_even_desc,
        "financial_status": financial_status,
        "initial_investment": init_inv,
        "working_capital": work_cap,
        "funding_required": fund_req
    }

# ==========================================================
# TRANSPARENT WEIGHTED SCORING ENGINE
# ==========================================================

SCORE_WEIGHTS = {
    "problem_validation": 0.15,
    "customer_need": 0.15,
    "market_opportunity": 0.15,
    "solution_relevance": 0.15,
    "feasibility": 0.15,
    "revenue_potential": 0.10,
    "competitive_advantage": 0.05,
    "risk_management": 0.05,
    "scalability": 0.05
}

def calculate_overall_score(scores_dict, analysis_mode, ml_probability=None):
    """
    Transparently calculates weighted validation score in Python.
    Weights:
      Problem Validation: 15%
      Customer Need: 15%
      Market Opportunity: 15%
      Solution Relevance: 15%
      Feasibility: 15%
      Revenue Potential: 10%
      Competitive Advantage: 5%
      Risk Management: 5%
      Scalability: 5%
    In Existing Startup mode: 70% Qualitative AI Score + 30% Historical ML Signal.
    """
    raw_idea_score = 0.0
    for key, weight in SCORE_WEIGHTS.items():
        val = float(scores_dict.get(key, {}).get("score", 65))
        val = max(0.0, min(100.0, val))
        raw_idea_score += val * weight
        
    idea_score = round(raw_idea_score, 1)
    
    if analysis_mode == "Existing Startup" and ml_probability is not None:
        final_score = round(idea_score * 0.70 + ml_probability * 0.30, 1)
        score_blend_note = (
            f"Calculated as 70% Qualitative Business Validation ({idea_score}/100) + "
            f"30% Historical ML Pattern Match ({ml_probability:.1f}%)."
        )
    else:
        final_score = idea_score
        score_blend_note = "100% Qualitative Multi-Dimensional Business Model Analysis."
        
    return final_score, idea_score, score_blend_note

# ==========================================================
# HUGGING FACE INFERENCE & PARSER
# ==========================================================

def build_llm_prompt(startup_profile, nlp_data, financials, ml_info=None):
    """Constructs a rigorous analytical prompt requesting deep structured JSON analysis."""
    
    comp_context = (
        f"Known Competitors entered by user: {startup_profile['known_competitors']}"
        if startup_profile.get("competition_known")
        else "Competitors not specified by founder. Identify 3 to 4 realistic market competitors or legacy alternatives."
    )
    
    ml_context = ""
    if ml_info and ml_info.get("success_probability") is not None:
        ml_context = f"""
Historical ML Benchmark:
A Random Forest model trained on historical Crunchbase startup investment patterns evaluated this business profile with a {ml_info['success_probability']:.1f}% historical pattern match.
Treat this strictly as supplementary historical benchmark data and explain its context analytically.
"""

    prompt = f"""
You are a Senior Startup Analyst and Business Strategy Advisor evaluating a startup submission.
Analyze this venture with extreme rigor, analytical depth, realistic skepticism, and actionable business logic.
Avoid generic motivational fluff. Distinguish clearly between user-provided data, assumptions, and market realities.

=== STARTUP SUBMISSION ===
Startup Name: {startup_profile['startup_name']}
Founder / Team: {startup_profile.get('founder_name', 'Not provided')}
Industry: {startup_profile['industry']}
Stage: {startup_profile['startup_stage']}
Geography: {startup_profile.get('target_city', '')}, {startup_profile.get('target_state', '')}, {startup_profile.get('target_country', '')}
Target Customer: {startup_profile['target_customer']}

Startup Idea:
{startup_profile['startup_idea']}

Problem Being Solved:
{startup_profile['problem']}
Who Experiences: {startup_profile.get('who_experiences', '')}
Problem Frequency: {startup_profile.get('frequency', '')}
Current Alternatives: {startup_profile.get('current_alternatives', '')}
Why Alternatives Insufficient: {startup_profile.get('why_insufficient', '')}

Solution & Value Proposition:
{startup_profile['proposed_solution']}
Core Features: {startup_profile.get('core_features', '')}
Unique Value Proposition: {startup_profile.get('uvp', '')}
Competitive Advantage: {startup_profile.get('competitive_advantage', '')}
Technology Employed: {startup_profile.get('tech_used', '')}
IP / Differentiation: {startup_profile.get('ip_differentiation', '')}

Customer & Market:
Primary Segment: {startup_profile.get('primary_customer', '')}
Secondary Segment: {startup_profile.get('secondary_customer', '')}
Market Scope: {startup_profile.get('market_type', 'Regional')}
Expected Volume: {startup_profile.get('expected_volume', '')}
{comp_context}

Business Model & Economics:
Revenue Model: {startup_profile['revenue_model']}
Pricing Strategy: {startup_profile['pricing']}
Monthly Revenue (Estimated): {startup_profile['currency_symbol']}{financials['monthly_revenue']:,.0f}
Monthly Expenses: {startup_profile['currency_symbol']}{financials['monthly_expenses']:,.0f}
Initial Investment: {startup_profile['currency_symbol']}{financials['initial_investment']:,.0f}
Funding Required: {startup_profile['currency_symbol']}{financials['funding_required']:,.0f}
Break-even Horizon: {financials['break_even_months']}

Team & Resources:
Team Size: {startup_profile.get('team_size', '')}
Founder Background: {startup_profile.get('founder_experience', '')}
Technical Capabilities: {startup_profile.get('technical_skills', '')}
Business Capabilities: {startup_profile.get('business_skills', '')}
Infrastructure & Partnerships: {startup_profile.get('available_infra', '')} | {startup_profile.get('partnerships', '')}
Advisors: {startup_profile.get('mentors_advisors', '')}

{ml_context}

=== INSTRUCTIONS & REQUIRED OUTPUT FORMAT ===
Produce a structured, rigorous JSON report.
For EACH of the major analysis sections (problem_analysis, customer_analysis, solution_analysis, market_analysis, business_model_analysis, feasibility_analysis, scalability_analysis), provide:
1. "conclusion": Executive summary in 1 sharp sentence.
2. "detailed_explanation": A concise, critical evaluation paragraph (approx. 70-130 words).
3. "evidence": Direct reference to facts in the user's submission.
4. "assumptions": Core operational assumptions.
5. "implication": Business and strategic consequence.
6. "recommendation": High-leverage actionable next step.

Return ONLY a single valid JSON object with the exact keys below:
{{
  "executive_summary": "120-180 word executive brief evaluating overall strategic posture, investment readiness, and core execution bottlenecks.",
  "scores": {{
    "problem_validation": {{"score": 0-100, "interpretation": "text", "reason": "text", "evidence": "text"}},
    "customer_need": {{"score": 0-100, "interpretation": "text", "reason": "text", "evidence": "text"}},
    "market_opportunity": {{"score": 0-100, "interpretation": "text", "reason": "text", "evidence": "text"}},
    "solution_relevance": {{"score": 0-100, "interpretation": "text", "reason": "text", "evidence": "text"}},
    "feasibility": {{"score": 0-100, "interpretation": "text", "reason": "text", "evidence": "text"}},
    "revenue_potential": {{"score": 0-100, "interpretation": "text", "reason": "text", "evidence": "text"}},
    "competitive_advantage": {{"score": 0-100, "interpretation": "text", "reason": "text", "evidence": "text"}},
    "risk_management": {{"score": 0-100, "interpretation": "text", "reason": "text", "evidence": "text"}},
    "scalability": {{"score": 0-100, "interpretation": "text", "reason": "text", "evidence": "text"}}
  }},
  "problem_analysis": {{"conclusion": "", "detailed_explanation": "", "evidence": "", "assumptions": "", "implication": "", "recommendation": ""}},
  "customer_analysis": {{"conclusion": "", "detailed_explanation": "", "evidence": "", "assumptions": "", "implication": "", "recommendation": ""}},
  "solution_analysis": {{"conclusion": "", "detailed_explanation": "", "evidence": "", "assumptions": "", "implication": "", "recommendation": ""}},
  "market_analysis": {{"conclusion": "", "detailed_explanation": "", "evidence": "", "assumptions": "", "implication": "", "recommendation": ""}},
  "business_model_analysis": {{"conclusion": "", "detailed_explanation": "", "evidence": "", "assumptions": "", "implication": "", "recommendation": ""}},
  "feasibility_analysis": {{"conclusion": "", "detailed_explanation": "", "evidence": "", "assumptions": "", "implication": "", "recommendation": ""}},
  "scalability_analysis": {{"conclusion": "", "detailed_explanation": "", "evidence": "", "assumptions": "", "implication": "", "recommendation": ""}},
  "strengths": ["3 to 5 specific analytical points"],
  "weaknesses": ["3 to 5 critical internal vulnerabilities"],
  "opportunities": ["3 to 5 external strategic tailwinds"],
  "threats": ["3 to 5 competitive or macroeconomic risks"],
  "competitors": [
    {{
      "name": "Competitor name",
      "type": "Direct / Indirect / Legacy Alternative",
      "origin": "User-Provided or AI-Suggested",
      "target_customer": "",
      "main_strength": "",
      "main_weakness": "",
      "how_startup_differs": "",
      "differentiation_opportunity": ""
    }}
  ],
  "risks": [
    {{
      "risk": "Name of risk",
      "category": "Market / Financial / Operational / Technology / Competition / Regulatory / Execution / Scalability",
      "likelihood": "Low / Medium / High",
      "impact": "Low / Medium / High",
      "severity": "Low / Moderate / Critical",
      "why_it_matters": "",
      "early_warning_indicator": "",
      "mitigation_strategy": "",
      "priority": "High / Medium / Low"
    }}
  ],
  "team_resource_gap_analysis": "Detailed review of team skills vs project requirements, highlighting unaddressed gaps.",
  "mvp_recommendation": {{"scope": "", "deliverables": ["3-4 items"], "success_metrics": ["2-3 metrics"]}},
  "go_to_market_strategy": "120-180 words describing acquisition channel hierarchy, CAC containment, and early customer conversion.",
  "action_plan": {{
    "next_7_days": [{{"action": "", "reason": "", "outcome": "", "priority": "High"}}],
    "next_30_days": [{{"action": "", "reason": "", "outcome": "", "priority": "High"}}],
    "next_90_days": [{{"action": "", "reason": "", "outcome": "", "priority": "Medium"}}],
    "next_6_months": [{{"action": "", "reason": "", "outcome": "", "priority": "Medium"}}]
  }},
  "final_verdict": "Conclusive balanced advisory summary emphasizing risk boundaries and key milestone prerequisites."
}}
"""
    return prompt

def call_huggingface_llm(prompt, selected_model="Qwen/Qwen2.5-Coder-32B-Instruct", temperature=0.25):
    """
    Invokes Hugging Face router using prioritized reliable models.
    Supports user-selected backend models and graceful fallbacks.
    """
    token = st.secrets.get("HF_TOKEN")
    if not token:
        return None, "Missing Hugging Face token (HF_TOKEN) in Streamlit secrets."
        
    client = InferenceClient(api_key=token, timeout=40.0)
    
    # Prioritize user-selected model followed by fallbacks
    models_to_attempt = [selected_model] if selected_model else []
    defaults = [
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ]
    for m in defaults:
        if m not in models_to_attempt:
            models_to_attempt.append(m)
            
    last_err = None
    for model_name in models_to_attempt:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior startup analyst and business strategy advisor. Always respond in valid, parseable JSON conforming precisely to user schema."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=2500,
                temperature=temperature
            )
            raw_content = response.choices[0].message.content
            if raw_content:
                parsed = parse_llm_json(raw_content)
                if parsed:
                    return parsed, f"Analysis conducted with {model_name}"
        except Exception as e:
            last_err = str(e)
            continue
            
    return None, f"AI analysis service temporarily unavailable. Error detail: {last_err}"

def ask_ai_copilot(query, startup_profile, llm_analysis, model_name="Qwen/Qwen2.5-Coder-32B-Instruct", temperature=0.3):
    """
    Direct conversational advisory response from the chosen LLM backend.
    """
    token = st.secrets.get("HF_TOKEN")
    if not token:
        return "⚠️ Hugging Face token (HF_TOKEN) missing in secrets.toml."
        
    client = InferenceClient(api_key=token, timeout=35.0)
    
    system_prompt = f"""You are StartupSense AI Copilot, a senior venture strategist, financial analyst, and government funding advisor.
You are advising the founder of '{startup_profile.get('startup_name', 'the startup')}'.
Venture Profile:
- Industry: {startup_profile.get('industry', 'Technology')}
- Stage: {startup_profile.get('startup_stage', 'Early')}
- Target Customer: {startup_profile.get('target_customer', '')}
- Core Solution: {startup_profile.get('proposed_solution', '')}
- Value Proposition: {startup_profile.get('uvp', '')}
- Problem Solved: {startup_profile.get('problem', '')}
- Revenue Model & Pricing: {startup_profile.get('revenue_model', '')} | {startup_profile.get('pricing', '')}

Provide sharp, practical, step-by-step guidance tailored specifically to this business.
When asked about funding, government grants, subsidies, or financial health:
- Provide exact schemes relevant to India and global founders (e.g. Startup India Seed Fund Scheme SISFS up to ₹20L grant / ₹50L debt, CGTMSE collateral-free credit guarantee up to ₹5 Cr, Stand-Up India, MUDRA loans, MeitY SAMRIDH, BIRAC for bio/agri, State Startup policies like StartupTN, Karnataka ELEVATE).
- Include clear eligibility conditions, application portal links/steps, documents required, and financial benchmarks (runway, burn rate, LTV:CAC, gross margins).
Avoid generic fluff; give exact numbers, scripts, timelines, operational benchmarks, and tactical playbooks."""

    models = [model_name, "Qwen/Qwen2.5-Coder-32B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"]
    for m in models:
        try:
            res = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=850,
                temperature=temperature
            )
            return res.choices[0].message.content
        except Exception:
            continue
            
    return "The AI Copilot service is temporarily busy. Please try asking your question again in a moment."

def parse_llm_json(raw_text):
    """Cleans and extracts JSON even if enclosed in markdown code fences or conversational text."""
    if not raw_text:
        return None
        
    text = raw_text.strip()
    
    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
        
    # Attempt standard parse
    try:
        return json.loads(text)
    except Exception:
        pass
        
    # Attempt regex extraction of first and last curly braces
    match = re.search(r'(\{[\s\S]*\})', text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
            
    return None

def generate_fallback_analysis(startup_profile, nlp_data, financials):
    """
    Robust analytical fallback generator that ensures zero crashes
    and provides a coherent default business analysis if the HF API is unreachable.
    """
    name = startup_profile['startup_name']
    ind = startup_profile['industry']
    
    return {
        "executive_summary": (
            f"{name} is an emerging initiative in the {ind} domain targeting {startup_profile['target_customer']}. "
            f"The business addresses critical inefficiencies in current market mechanisms through a tailored direct model. "
            f"While the value proposition is aligned with modern customer preferences, long-term viability hinges on "
            f"disciplined unit economics, retention velocity, and containing customer acquisition costs."
        ),
        "scores": {
            "problem_validation": {"score": 78, "interpretation": "Strong Pain Point", "reason": "Clearly articulated operational bottleneck affecting both supply and demand sides.", "evidence": startup_profile['problem'][:120]},
            "customer_need": {"score": 82, "interpretation": "High Daily Frequency", "reason": "Target customer experiences continuous or recurring dependency on this solution.", "evidence": startup_profile.get('target_customer', '')[:120]},
            "market_opportunity": {"score": 75, "interpretation": "Large Addressable Market", "reason": "Substantial regional population with expanding willingness to pay for quality.", "evidence": startup_profile.get('market_type', 'Regional')},
            "solution_relevance": {"score": 80, "interpretation": "Direct Fit", "reason": "Eliminates unnecessary intermediary layers to pass value to both ends.", "evidence": startup_profile['proposed_solution'][:120]},
            "feasibility": {"score": 70, "interpretation": "Operationally Demanding", "reason": "Requires dependable logistical execution and cold-chain compliance under margin constraints.", "evidence": startup_profile.get('tech_used', 'Digital Platform')},
            "revenue_potential": {"score": 74, "interpretation": "Sustainable Volume Play", "reason": "Volume-driven model with stable margins provided retention remains high.", "evidence": f"Revenue Model: {startup_profile['revenue_model']}"},
            "competitive_advantage": {"score": 68, "interpretation": "Moderate Moat", "reason": "First-mile relationships provide defensibility, though capital-rich incumbents can duplicate features.", "evidence": startup_profile.get('competitive_advantage', '')[:120]},
            "risk_management": {"score": 66, "interpretation": "Needs Contingencies", "reason": "Perishable goods and logistical friction represent operational vulnerabilities.", "evidence": "Perishable distribution logistics"},
            "scalability": {"score": 72, "interpretation": "City-by-City Expansion", "reason": "Expansion follows localized clustering rather than frictionless software viral growth.", "evidence": startup_profile.get('target_location', 'Urban clusters')}
        },
        "problem_analysis": {
            "conclusion": "The identified problem represents genuine economic waste and price distortion.",
            "detailed_explanation": (
                f"The core friction identified in {name}'s problem statement reflects systemic friction in traditional {ind} distribution. "
                f"Multiple intermediate checkpoints extract incremental commissions without preserving freshness or providing supply-chain visibility. "
                f"Because perishability or turnaround delay compounds quality deterioration, both suppliers and end-consumers bear the financial burden. "
                f"This creates a substantial economic surplus that a streamlined direct model can effectively capture."
            ),
            "evidence": startup_profile['problem'],
            "assumptions": "Assumes end consumers prioritize freshness and price stability over hyper-fast 10-minute convenience.",
            "implication": "Failure to address this allows incumbents with dark-store networks to capture market share through promotional subsidies.",
            "recommendation": "Quantify exact time-to-table metrics and display savings transparently on every consumer invoice."
        },
        "customer_analysis": {
            "conclusion": "High engagement cohort with strong retention incentives if quality is consistent.",
            "detailed_explanation": (
                f"The target demographic—{startup_profile['target_customer']}—is typically characterized by high repeat purchase frequency. "
                f"These users evaluate food and staple purchases through the lens of family wellness, reliability, and fair pricing. "
                f"However, this customer segment is sensitive to stockouts, late deliveries, and inconsistent sizing. "
                f"Winning this segment requires proactive customer success and society-level community advocacy."
            ),
            "evidence": startup_profile.get('who_experiences', 'Urban households'),
            "assumptions": "Assumes housing societies and clusters permit bulk delivery staging without friction.",
            "implication": "Word-of-mouth in residential clusters can drive organic CAC down significantly.",
            "recommendation": "Launch residential society champion programs offering tiered discounts for community coordinators."
        },
        "solution_analysis": {
            "conclusion": "Lean, direct-to-consumer logistics model that bypasses conventional wholesale bottlenecks.",
            "detailed_explanation": (
                f"{name}'s solution architecture replaces fragmented wholesale mandis with direct demand-scheduled harvesting. "
                f"By aggregating consumer orders before dispatching procurement requests to farmers, working capital commitments are minimized "
                f"and inventory holding risks are virtually eliminated. This zero-warehouse or micro-hub strategy ensures superior gross margins."
            ),
            "evidence": startup_profile['proposed_solution'],
            "assumptions": "Assumes partner producers adhere to strict harvest timing and sorting specifications.",
            "implication": "Enables positive unit contribution from early stages without heavy capital expenditure on centralized warehouses.",
            "recommendation": "Equip collection points with automated digital scales and moisture/quality testing sensors."
        },
        "market_analysis": {
            "conclusion": "Rapidly growing market propelled by health awareness and organized retail penetration.",
            "detailed_explanation": (
                f"The target market in {startup_profile.get('target_state', 'the region')} is experiencing accelerated modernization. "
                f"Rising disposable incomes and heightened consumer awareness regarding pesticide residue are shifting purchase behavior "
                f"away from unorganized street vendors toward traceable, certified supply chains. The total addressable market provides ample headroom."
            ),
            "evidence": f"Operating scope: {startup_profile.get('market_type', 'Regional')}",
            "assumptions": "Assumes macroeconomic stability and continuing consumer preference for direct-sourced staples.",
            "implication": "Attracts competitive attention from established e-grocery players if unit economics prove lucrative.",
            "recommendation": "Secure exclusive FPO agreements to lock in supply-side defensibility."
        },
        "business_model_analysis": {
            "conclusion": "Attractive gross margin structure offset by last-mile delivery and transit costs.",
            "detailed_explanation": (
                f"The chosen model ({startup_profile['revenue_model']}) generates revenue from unit transaction margins augmented by "
                f"potential recurring subscription fees. At {startup_profile['pricing']}, the business model provides healthy gross profit "
                f"buffers. However, operating expenses must be monitored closely to prevent last-mile fuel and driver costs from eroding margins."
            ),
            "evidence": f"Estimated Monthly Revenue: ₹{financials['monthly_revenue']:,.0f}, Expenses: ₹{financials['monthly_expenses']:,.0f}",
            "assumptions": "Assumes delivery density reaches at least 8-12 deliveries per society stop.",
            "implication": "Sub-scale drop densities will result in negative unit economics per vehicle run.",
            "recommendation": "Enforce minimum society order batch thresholds before activating new delivery routes."
        },
        "feasibility_analysis": {
            "conclusion": "Technically viable; operational excellence and supply reliability are the true hurdles.",
            "detailed_explanation": (
                f"Building the digital dispatch software and customer portal is straightforward using modern tech stacks. "
                f"The critical feasibility constraint lies in physical logistics: weather disruptions, agricultural harvest yields, "
                f"and morning transit congestion. Maintaining a 98%+ on-time fulfillment rate will determine survival."
            ),
            "evidence": startup_profile.get('tech_used', 'Mobile and analytics platform'),
            "assumptions": "Assumes dependable third-party fleet availability or cost-effective van leasing.",
            "implication": "Delivery failures during morning breakfast windows trigger rapid customer churn.",
            "recommendation": "Establish a buffer sourcing arrangement with secondary nearby grower networks."
        },
        "scalability_analysis": {
            "conclusion": "Scalable via modular geographic hub replication rather than instant software scaling.",
            "detailed_explanation": (
                f"Unlike pure-play software, scaling {name} requires replicating physical collection centers, farmer relationships, "
                f"and delivery routes across new districts. Each cluster operates as an independent profit center. Once a playbook is perfected "
                f"in the primary city, regional expansion can proceed city-by-city with predictable capital requirements."
            ),
            "evidence": f"Starting in {startup_profile.get('target_city', 'Tier 1/2 Cities')}",
            "assumptions": "Assumes the supply chain playbook is modular and documented for new territory managers.",
            "implication": "Growth will be steady and defensible, though capital demands will rise with geographic footprints.",
            "recommendation": "Establish a standardized 'Cluster Launch Kit' covering farmer onboarding, route mapping, and society approvals."
        },
        "strengths": [
            "Direct farm sourcing yields 10-15% higher gross margins than traditional wholesale distribution.",
            "Zero central warehouse model dramatically lowers fixed infrastructure overhead.",
            "Strong customer retention driven by noticeable produce freshness differentials.",
            "Experienced founding team with hands-on logistics and agricultural network background."
        ],
        "weaknesses": [
            "Vulnerability to seasonal weather shocks and harvest yield fluctuations.",
            "High early-stage reliance on manual quality sorting at farm-gate collection.",
            "Moderate customer acquisition cost before residential society network effects kick in.",
            "Restricted product catalog compared to full-basket supermarket quick-commerce."
        ],
        "opportunities": [
            "Introduce high-margin value-added organic preserves, cold-pressed oils, and artisanal grains.",
            "Expand B2B micro-contracts to local boutique cafes, bakeries, and cloud kitchens.",
            "Implement carbon offset and sustainable agriculture certifications to attract ESG-focused consumers.",
            "Monetize subscription micro-memberships for pre-scheduled daily deliveries."
        ],
        "threats": [
            "Aggressive discounting and rapid expansion from well-funded quick-commerce giants.",
            "Regulatory shifts in Agricultural Produce Market Committee (APMC) direct purchase bylaws.",
            "Fuel and last-mile transportation price spikes eroding net operating margins.",
            "Unfavorable climate events creating supply deficits in key crop categories."
        ],
        "competitors": [
            {
                "name": "Quick-Commerce Operators (Blinkit, Zepto, Swiggy Instamart)",
                "type": "Direct Category Competitor",
                "origin": "AI-Suggested",
                "target_customer": "Impulse urban shoppers prioritizing 10-minute speed over farm provenance.",
                "main_strength": "Massive capital reserves, ubiquitous dark-store footprint, and multi-category catalogs.",
                "main_weakness": "High wastage rates, plastic packaging overhead, and cold-storage produce taste degradation.",
                "how_startup_differs": "Delivers fresher morning harvest direct from farm with transparent farmer provenance.",
                "differentiation_opportunity": "Emphasize chemical-free health credentials and pre-scheduled society drop savings."
            },
            {
                "name": "Subscription Dairy & Staples (Country Delight, Otipy)",
                "type": "Model Competitor",
                "origin": "AI-Suggested",
                "target_customer": "Daily residential morning subscribers.",
                "main_strength": "Established morning doorstep delivery routes and strong brand recall.",
                "main_weakness": "Premium pricing and inconsistent regional vegetable quality outside primary milk lines.",
                "how_startup_differs": "Specialized farm-gate produce curation with localized farmer FPO partnerships.",
                "differentiation_opportunity": "Provide farm-level batch traceability via QR codes on harvest crates."
            },
            {
                "name": "Traditional Wet Markets (APMC Mandis & Street Vendors)",
                "type": "Legacy Alternative",
                "origin": "AI-Suggested",
                "target_customer": "Price-conscious local shoppers willing to travel and inspect produce manually.",
                "main_strength": "Zero digital barrier, immediate tactile inspection, and entrenched habits.",
                "main_weakness": "Lack of hygiene guarantees, fluctuating weights/rates, and inconvenient shopping experience.",
                "how_startup_differs": "Delivers sanitized, sorted, and weighed baskets directly to residential lobby lockers.",
                "differentiation_opportunity": "Offer competitive mandi-matched pricing while guaranteeing net weight and food safety."
            }
        ],
        "risks": [
            {
                "risk": "Morning Fulfillment Latency Risk",
                "category": "Operational Risk",
                "likelihood": "Medium",
                "impact": "High",
                "severity": "Critical",
                "why_it_matters": "Late deliveries miss family morning meal preparations, triggering instant customer churn.",
                "early_warning_indicator": "Collection departure delays exceeding 30 minutes at farm collection centers.",
                "mitigation_strategy": "Institute strict 8 PM dispatch cutoffs and maintain backup dispatch vehicles on high-density routes.",
                "priority": "High"
            },
            {
                "risk": "Harvest Yield & Quality Volatility",
                "category": "Market Risk",
                "likelihood": "High",
                "impact": "Medium",
                "severity": "Moderate",
                "why_it_matters": "Monsoons and pests can wipe out primary crop varieties, creating inventory stockouts.",
                "early_warning_indicator": "Weather forecast warnings and farmer distress reports 3-5 days in advance.",
                "mitigation_strategy": "Diversify farmer sourcing across three distinct micro-climatic agro-zones.",
                "priority": "High"
            },
            {
                "risk": "Unit Economics Margin Compression",
                "category": "Financial Risk",
                "likelihood": "Medium",
                "impact": "High",
                "severity": "Critical",
                "why_it_matters": "Fuel inflation and sub-optimal drop density can turn positive gross profit into net operating loss.",
                "early_warning_indicator": "Average basket value dipping below ₹350 or less than 6 orders per society run.",
                "mitigation_strategy": "Establish dynamic delivery fees for low-volume buildings and incentivize society group buys.",
                "priority": "High"
            },
            {
                "risk": "Regulatory Compliance & APMC Scrutiny",
                "category": "Regulatory Risk",
                "likelihood": "Low",
                "impact": "High",
                "severity": "Moderate",
                "why_it_matters": "Local market committees occasionally challenge direct farm purchasing without mandi cess.",
                "early_warning_indicator": "Notices from regional agricultural marketing boards or local mandi authorities.",
                "mitigation_strategy": "Partner directly through registered Farmer Producer Organizations (FPOs) compliant with state exemptions.",
                "priority": "Medium"
            }
        ],
        "team_resource_gap_analysis": (
            "The founding team possesses commendable operational grit in supply chains and local agricultural ties. "
            "However, there is an evident gap in high-scale performance digital marketing and automated inventory routing software. "
            "Reaching 10,000+ daily orders will necessitate hiring a dedicated Lead Logistics Algorithm Engineer and a Community Growth Head."
        ),
        "mvp_recommendation": {
            "scope": "Launch a 60-day closed beta focused strictly on 15 high-density housing societies and 25 partner farmers.",
            "deliverables": [
                "Simplified WhatsApp Business & Web ordering interface with automated cutoffs.",
                "Standardized reusable food-grade transit crates with tamper-evident seals.",
                "Single aggregation center with digital weight grading and QA checklists.",
                "Dedicated society coordinator reward portal."
            ],
            "success_metrics": [
                "85%+ customer weekly repeat order rate across the 15 societies.",
                "Fulfillment punctuality score of 98.5% before 7:30 AM.",
                "Net customer rating exceeding 4.6/5.0 on produce freshness."
            ]
        },
        "go_to_market_strategy": (
            "Execute a hyper-localized B2B2C residential society penetration model. Rather than spending capital on broad digital ads, "
            "partner with Housing Society Management Committees (RWA). Offer society-wide Sunday morning 'Farm-to-Kitchen' sampling pop-ups. "
            "Appoint and incentivize residential society ambassadors with monthly produce allowances to achieve 35%+ household penetration per complex."
        ),
        "action_plan": {
            "next_7_days": [
                {
                    "action": "Finalize formal written procurement MOUs with 2 verified local Farmer Producer Organizations (FPOs).",
                    "reason": "Lock in harvest pricing, quality benchmarks, and minimum volume commitments.",
                    "outcome": "Guaranteed supply line of 15 key vegetable staples at fixed seasonal rate bands.",
                    "priority": "High"
                },
                {
                    "action": "Map delivery route logistics and gate security clearance for the first 10 targeted housing societies.",
                    "reason": "Prevent morning security delays during delivery drop-offs.",
                    "outcome": "Pre-authorized access badges and allocated society lobby drop-point zones.",
                    "priority": "High"
                }
            ],
            "next_30_days": [
                {
                    "action": "Deploy streamlined order management portal with integrated UPI payment links and automated morning delivery SMS.",
                    "reason": "Eliminate manual spreadsheet order tracking and reduce checkout friction.",
                    "outcome": "Zero manual order errors and automated digital dispatch manifests.",
                    "priority": "High"
                },
                {
                    "action": "Conduct quality and sorting workshop for partner farmers at the Narayangaon aggregation center.",
                    "reason": "Enforce strict grading criteria at the farm gate before produce enters transit.",
                    "outcome": "Transit spoilage and grading rejection rate compressed below 4%.",
                    "priority": "Medium"
                }
            ],
            "next_90_days": [
                {
                    "action": "Expand active society clusters from 10 to 45 complexes across the western metro corridor.",
                    "reason": "Drive route density to push last-mile logistics cost below ₹15 per basket.",
                    "outcome": "Achieve monthly revenue of ₹15,00,000 with positive route-level contribution margin.",
                    "priority": "High"
                },
                {
                    "action": "Introduce curated weekly organic fruit baskets and cold-pressed oil add-ons.",
                    "reason": "Increase average order basket size from ₹350 to ₹520.",
                    "outcome": "Gross margin expansion from 24% to 29% across high-income societies.",
                    "priority": "Medium"
                }
            ],
            "next_6_months": [
                {
                    "action": "Implement IoT temperature logging and route optimization algorithm for refrigerated transit vans.",
                    "reason": "Prevent heat spoilage during summer months and reduce diesel consumption.",
                    "outcome": "Maintain 99% freshness compliance during seasonal heat spikes.",
                    "priority": "Medium"
                },
                {
                    "action": "Prepare comprehensive Series Seed investment memorandum based on verified cluster unit economics.",
                    "reason": "Raise ₹3.5-5.0 Cr growth capital for regional multi-city expansion.",
                    "outcome": "Term sheets from specialized agri-tech and consumer impact venture funds.",
                    "priority": "High"
                }
            ]
        },
        "final_verdict": (
            f"{name} possesses a compelling, socially impactful, and commercially viable premise. "
            f"By solving genuine supply-chain inefficiencies in the {ind} ecosystem, it delivers clear value to both producers and consumers. "
            f"The critical differentiator between success and stagnation will not be software features, but relentless physical logistics discipline, "
            f"flawless morning punctuality, and maintaining dense society delivery clusters to protect operating margins."
        )
    }

# ==========================================================
# HISTORICAL ML (RANDOM FOREST) FEATURE ENGINEERING
# ==========================================================

def run_historical_ml(startup_profile, current_year):
    """
    Computes features and runs Random Forest prediction on historical Crunchbase dataset patterns.
    """
    if ml_model is None or ml_features is None:
        return None
        
    try:
        founded = int(startup_profile.get("founded_year", current_year - 2))
        age = max(0, current_year - founded)
        first_fund = int(startup_profile.get("first_funding_year", founded + 1))
        last_fund = int(startup_profile.get("latest_funding_year", current_year))
        
        age_first_funding = max(0.0, float(first_fund - founded))
        age_last_funding = max(0.0, float(last_fund - founded))
        age_first_milestone = min(float(age), 1.0)
        age_last_milestone = float(age)
        milestones = max(1, min(5, int(age)))
        rounds = int(startup_profile.get("funding_rounds", 1))
        tot_fund = float(startup_profile.get("total_funding", 100000.0))
        connections = int(startup_profile.get("connections", 6))
        avg_participants = 2.0 if rounds > 0 else 0.0
        
        ind = startup_profile.get("industry", "Technology")
        is_software = 1 if ind in ["SaaS", "Software", "Technology", "Artificial Intelligence", "Cybersecurity"] else 0
        is_ecommerce = 1 if ind in ["E-Commerce", "Retail"] else 0
        is_biotech = 1 if ind in ["Healthcare", "Biotech"] else 0
        is_othercategory = 1 if not any([is_software, is_ecommerce, is_biotech]) else 0
        
        # State mapping (Crunchbase features)
        loc = (str(startup_profile.get("target_state", "")) + " " + str(startup_profile.get("target_country", ""))).lower()
        is_ca = 1 if "california" in loc or " ca" in loc else 0
        is_ny = 1 if "new york" in loc or " ny" in loc else 0
        is_ma = 1 if "massachusetts" in loc or " ma" in loc else 0
        is_tx = 1 if "texas" in loc or " tx" in loc else 0
        is_otherstate = 1 if not any([is_ca, is_ny, is_ma, is_tx]) else 0
        
        feat_dict = {
            "age_first_funding_year": age_first_funding,
            "age_last_funding_year": age_last_funding,
            "age_first_milestone_year": age_first_milestone,
            "age_last_milestone_year": age_last_milestone,
            "relationships": connections,
            "funding_rounds": rounds,
            "funding_total_usd": tot_fund,
            "milestones": milestones,
            "is_CA": is_ca,
            "is_NY": is_ny,
            "is_MA": is_ma,
            "is_TX": is_tx,
            "is_otherstate": is_otherstate,
            "is_software": is_software,
            "is_web": 0,
            "is_mobile": 0,
            "is_enterprise": 0,
            "is_advertising": 0,
            "is_gamesvideo": 0,
            "is_ecommerce": is_ecommerce,
            "is_biotech": is_biotech,
            "is_consulting": 0,
            "is_othercategory": is_othercategory,
            "has_VC": int(startup_profile.get("has_vc", False)),
            "has_angel": int(startup_profile.get("has_angel", False)),
            "has_roundA": int(startup_profile.get("has_round_a", False)),
            "has_roundB": int(startup_profile.get("has_round_b", False)),
            "has_roundC": int(startup_profile.get("has_round_c", False)),
            "has_roundD": int(startup_profile.get("has_round_d", False)),
            "avg_participants": avg_participants,
            "is_top500": 0
        }
        
        input_row = [feat_dict.get(f, 0) for f in ml_features]
        df_input = pd.DataFrame([input_row], columns=ml_features)
        
        pred = ml_model.predict(df_input)[0]
        prob = ml_model.predict_proba(df_input)[0]
        
        return {
            "prediction": int(pred),
            "failure_probability": float(prob[0] * 100.0),
            "success_probability": float(prob[1] * 100.0),
            "input_features": feat_dict
        }
    except Exception:
        return None

# ==========================================================
# REPORT GENERATORS (MARKDOWN & HTML)
# ==========================================================

def generate_markdown_report(profile, analysis, scores, financials, ml_res=None):
    """Creates a downloadable professional executive report in GitHub-flavored Markdown."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_s = scores["final_score"]
    
    md = f"""# StartupSense AI – Startup Intelligence & Validation Report
**Startup Name:** {profile['startup_name']}  
**Industry:** {profile['industry']} | **Stage:** {profile['startup_stage']}  
**Generated:** {timestamp}  
**Overall Validation Score:** {final_s}/100  

---

## 1. Executive Summary
{analysis.get('executive_summary', 'N/A')}

---

## 2. Multi-Dimensional Validation Scores
| Evaluation Factor | Score | Interpretation | Strategic Rationale |
| :--- | :--- | :--- | :--- |
"""
    for k, v in analysis.get('scores', {}).items():
        name = k.replace('_', ' ').title()
        md += f"| {name} | {v.get('score', 0)}/100 | {v.get('interpretation', '')} | {v.get('reason', '')} |\n"

    md += f"""
**Score Methodology:** {scores.get('score_blend_note', '')}

---

## 3. Financial Summary & Unit Economics
- **Initial Investment:** {profile['currency_symbol']}{financials['initial_investment']:,.0f}
- **Monthly Revenue:** {profile['currency_symbol']}{financials['monthly_revenue']:,.0f}
- **Monthly Operating Expenses:** {profile['currency_symbol']}{financials['monthly_expenses']:,.0f}
- **Monthly Net Profit / Burn:** {profile['currency_symbol']}{financials['monthly_profit']:,.0f}
- **Annualized Revenue:** {profile['currency_symbol']}{financials['annual_revenue']:,.0f}
- **Break-Even Horizon:** {financials['break_even_months']}
- **Capital Required:** {profile['currency_symbol']}{financials['funding_required']:,.0f}

---

## 4. Deep Strategic Analysis
### Problem Validation
- **Executive Conclusion:** {analysis.get('problem_analysis', {}).get('conclusion', '')}
- **Detailed Assessment:** {analysis.get('problem_analysis', {}).get('detailed_explanation', '')}
- **Key Assumptions:** {analysis.get('problem_analysis', {}).get('assumptions', '')}
- **Recommendation:** {analysis.get('problem_analysis', {}).get('recommendation', '')}

### Customer Need & Behavioral Alignment
- **Executive Conclusion:** {analysis.get('customer_analysis', {}).get('conclusion', '')}
- **Detailed Assessment:** {analysis.get('customer_analysis', {}).get('detailed_explanation', '')}
- **Recommendation:** {analysis.get('customer_analysis', {}).get('recommendation', '')}

### Solution & Value Proposition
- **Executive Conclusion:** {analysis.get('solution_analysis', {}).get('conclusion', '')}
- **Detailed Assessment:** {analysis.get('solution_analysis', {}).get('detailed_explanation', '')}
- **Recommendation:** {analysis.get('solution_analysis', {}).get('recommendation', '')}

### Market Opportunity & Economics
- **Executive Conclusion:** {analysis.get('market_analysis', {}).get('conclusion', '')}
- **Detailed Assessment:** {analysis.get('market_analysis', {}).get('detailed_explanation', '')}
- **Recommendation:** {analysis.get('market_analysis', {}).get('recommendation', '')}

---

## 5. SWOT Analysis Matrix
### Strengths
"""
    for s in analysis.get('strengths', []):
        md += f"- {s}\n"
        
    md += "\n### Weaknesses\n"
    for w in analysis.get('weaknesses', []):
        md += f"- {w}\n"
        
    md += "\n### Opportunities\n"
    for o in analysis.get('opportunities', []):
        md += f"- {o}\n"
        
    md += "\n### Threats\n"
    for t in analysis.get('threats', []):
        md += f"- {t}\n"

    md += "\n---\n\n## 6. Comprehensive Risk Assessment\n"
    md += "| Risk | Category | Likelihood | Impact | Severity | Mitigation Strategy |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for r in analysis.get('risks', []):
        md += f"| {r.get('risk', '')} | {r.get('category', '')} | {r.get('likelihood', '')} | {r.get('impact', '')} | {r.get('severity', '')} | {r.get('mitigation_strategy', '')} |\n"

    md += "\n---\n\n## 7. Competitor Intelligence & Differentiation\n"
    for c in analysis.get('competitors', []):
        md += f"### {c.get('name', 'Competitor')} ({c.get('type', '')} - {c.get('origin', 'Identified')})\n"
        md += f"- **Target Customer:** {c.get('target_customer', '')}\n"
        md += f"- **Primary Strength:** {c.get('main_strength', '')}\n"
        md += f"- **Primary Weakness:** {c.get('main_weakness', '')}\n"
        md += f"- **Differentiation Advantage:** {c.get('differentiation_opportunity', '')}\n\n"

    md += "---\n\n## 8. Strategic Action Roadmap\n"
    act = analysis.get('action_plan', {})
    for horizon, label in [('next_7_days', 'Next 7 Days'), ('next_30_days', 'Next 30 Days'), ('next_90_days', 'Next 90 Days'), ('next_6_months', 'Next 6 Months')]:
        md += f"### {label}\n"
        for item in act.get(horizon, []):
            md += f"- **{item.get('action', '')}** (Priority: {item.get('priority', 'Medium')})\n"
            md += f"  - *Rationale:* {item.get('reason', '')}\n"
            md += f"  - *Expected Outcome:* {item.get('outcome', '')}\n"

    if ml_res:
        md += f"""
---

## 9. Historical Machine Learning Pattern Analysis (Random Forest)
- **Historical Success Pattern Match:** {ml_res.get('success_probability', 0):.1f}%
- **Historical Failure Pattern Match:** {ml_res.get('failure_probability', 0):.1f}%
- **Dataset Context:** Supervised Random Forest trained on 31 historical venture attributes (Crunchbase startup dataset).
- **Analytical Disclaimer:** Machine learning outputs represent historical statistical similarity, not guaranteed predictions.
"""

    md += f"""
---

## 10. Concluding Advisory Verdict
{analysis.get('final_verdict', '')}

---
*Notice: StartupSense AI is an intelligent venture decision-support system. All evaluations, projections, and estimates are model-derived and do not constitute formal legal, accounting, or financial guarantees.*
"""
    return md

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.markdown("### 🧠 **StartupSense AI**")
    st.caption("Intelligent Decision-Support & Validation Platform")
    
    st.divider()
    
    # Engine Status Indicators
    st.markdown("##### ⚙️ **System Engines**")
    
    # HF LLM Status
    hf_token_present = bool(st.secrets.get("HF_TOKEN"))
    if hf_token_present:
        st.markdown(
            """<div style="padding: 8px 12px; background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); border-radius: 8px; margin-bottom: 8px;">
            <b style="color:#10b981;">● Hugging Face LLM</b><br>
            <span style="font-size:11px;color:#94a3b8;">Qwen 2.5 / DeepSeek / Llama 3.3 Active</span>
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div style="padding: 8px 12px; background: rgba(244,63,94,0.1); border: 1px solid rgba(244,63,94,0.3); border-radius: 8px; margin-bottom: 8px;">
            <b style="color:#f43f5e;">○ HF_TOKEN Missing</b><br>
            <span style="font-size:11px;color:#94a3b8;">Add HF_TOKEN in secrets.toml</span>
            </div>""",
            unsafe_allow_html=True
        )
        
    # ML Engine Status
    if ml_available:
        st.markdown(
            """<div style="padding: 8px 12px; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3); border-radius: 8px; margin-bottom: 8px;">
            <b style="color:#3b82f6;">● Random Forest ML</b><br>
            <span style="font-size:11px;color:#94a3b8;">31-Feature Historical Pattern Model Loaded</span>
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div style="padding: 8px 12px; background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); border-radius: 8px; margin-bottom: 8px;">
            <b style="color:#f59e0b;">○ ML Module Standby</b><br>
            <span style="font-size:11px;color:#94a3b8;">Historical ML module unavailable – AI analysis is still available.</span>
            </div>""",
            unsafe_allow_html=True
        )

    # NLP Status
    st.markdown(
        """<div style="padding: 8px 12px; background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.3); border-radius: 8px; margin-bottom: 8px;">
        <b style="color:#8b5cf6;">● NLP Entity & Semantic Parser</b><br>
        <span style="font-size:11px;color:#94a3b8;">Active (Entity, Intent & Keyword Extraction)</span>
        </div>""",
        unsafe_allow_html=True
    )

    st.divider()

    # AI Backend Engine Selection
    st.markdown("##### 🤖 **AI Backend Engine**")
    ai_backend_model = st.selectbox(
        "Active AI Model:",
        [
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        ],
        index=0,
        help="Select the underlying generative LLM backend used for strategic analysis and copilot advisory."
    )
    llm_temperature = st.slider("Model Temperature", min_value=0.0, max_value=0.8, value=0.25, step=0.05, help="Controls analytical variance.")
    st.caption(f"Connected: `{ai_backend_model.split('/')[-1]}`")

    st.divider()
    
    st.markdown("##### 📌 **Positioning & Disclaimers**")
    st.info(
        "StartupSense AI is an intelligent decision-support platform designed to analyze startup mechanics, evaluate risk vectors, and surface strategic priorities. "
        "It is not an automated investment guarantee or financial warranty."
    )
    
    st.caption("Developed for Startup Founders & Investment Advisory Intelligence.")

# ==========================================================
# MAIN SCREEN HEADER & PIPELINE
# ==========================================================

st.markdown("""
<div class="brand-title">
    <span class="brand-gradient">StartupSense AI</span>
</div>
<div class="brand-subtitle">
    AI-Powered Startup Intelligence & Validation Platform — Analyze your startup idea using Natural Language Processing, Large Language Models and Machine Learning to identify opportunities, risks, business potential and actionable next steps.
</div>
""", unsafe_allow_html=True)

# Architecture pipeline display
st.markdown("""
<div class="flow-container">
    <span class="flow-step">📝 Natural Language Input</span>
    <span class="flow-arrow">→</span>
    <span class="flow-step">🧠 NLP Processing</span>
    <span class="flow-arrow">→</span>
    <span class="flow-step">🤖 AI Strategic Analysis</span>
    <span class="flow-arrow">→</span>
    <span class="flow-step">🌲 ML Historical Validation</span>
    <span class="flow-arrow">→</span>
    <span class="flow-step">📊 Business Intelligence</span>
</div>
""", unsafe_allow_html=True)

# Quick Sample Loader Button Banner
col_sample_a, col_sample_b, col_sample_c = st.columns([2, 1, 1])
with col_sample_a:
    st.markdown("##### 🚀 Quick Demo Auto-Fill")
    st.caption("Auto-fill realistic startup details to test and demonstrate the platform instantly.")
with col_sample_b:
    load_existing_clicked = st.button("📊 Load Existing Startup Demo", use_container_width=True, type="primary")
with col_sample_c:
    load_new_clicked = st.button("💡 Load New Idea Demo", use_container_width=True)

# Auto-initialize with complete Existing Startup demo data by default
if "form_data" not in st.session_state or not st.session_state["form_data"]:
    st.session_state["form_data"] = get_demo_data(mode="Existing")

if load_existing_clicked:
    sample_data = get_demo_data(mode="Existing")
    for k, v in sample_data.items():
        st.session_state["form_data"][k] = v
    st.session_state["form_data"]["analysis_mode"] = "Existing"
    st.success("✅ Loaded 'VillageFresh' Existing Startup Profile with verified traction, unit economics & ML features.")
    st.rerun()

if load_new_clicked:
    sample_data = get_demo_data(mode="New")
    for k, v in sample_data.items():
        st.session_state["form_data"][k] = v
    st.session_state["form_data"]["analysis_mode"] = "New"
    st.success("✅ Loaded 'VillageFresh' Early Ideation Profile.")
    st.rerun()

fd = st.session_state["form_data"]

# ==========================================================
# ANALYSIS MODE SELECTION
# ==========================================================

st.markdown("### 🎯 1. Select Analysis Paradigm")
col_mode_1, col_mode_2 = st.columns(2)
with col_mode_1:
    default_mode_idx = 1 if fd.get("analysis_mode", "Existing") == "Existing" else 0
    analysis_mode_choice = st.radio(
        "Choose Mode:",
        ["💡 New Startup Idea", "📊 Existing Startup"],
        index=default_mode_idx,
        horizontal=True,
        label_visibility="collapsed"
    )

is_existing_mode = (analysis_mode_choice == "📊 Existing Startup")

if not is_existing_mode:
    st.info("💡 **New Startup Mode**: Focuses on problem validation, customer need discovery, market sizing, feasibility, initial business model, and MVP execution.")
else:
    st.info("📊 **Existing Startup Mode**: Focuses additionally on operational history, funding velocity, customer retention, growth scalability, and historical Random Forest pattern matching.")

st.divider()

# ==========================================================
# MULTI-SECTION STRUCTURED USER INPUT FORM
# ==========================================================

st.markdown("### 📋 2. Startup Operational Profile")
st.caption("Fill in the structured fields below. All inputs support long-form natural language.")

# Tabbed Form Layout to prevent cognitive overload
form_tabs = st.tabs([
    "01. Profile & Geography",
    "02. Idea & Problem",
    "03. Solution & Moat",
    "04. Market & Competitors",
    "05. Business Model & Financials",
    "06. Team & Infrastructure"
])

# ----------------------------------------------------------
# SECTION 01: STARTUP PROFILE
# ----------------------------------------------------------
with form_tabs[0]:
    st.markdown("""
    <span class="section-badge">SECTION 01</span>
    <span class="section-title">Startup Profile & Target Demographics</span>
    <div class="section-desc">Define the startup identity, current development stage, and geographical focus.</div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        val_name = st.text_input("Startup Name *", value=fd.get("startup_name", ""), placeholder="e.g. VillageFresh, FinLeap, OmniHealth")
        val_founder = st.text_input("Founder / Team Name (optional)", value=fd.get("founder_name", ""), placeholder="e.g. Aditi Rao & Co-Founders")
        
        industry_options = [
            "Agriculture", "AgriTech", "Education", "EdTech", "Healthcare", "FinTech",
            "E-Commerce", "Retail", "Food & Beverage", "Manufacturing", "Logistics",
            "Travel", "Technology", "Artificial Intelligence", "SaaS", "Cybersecurity",
            "Social Impact", "Other"
        ]
        curr_ind = fd.get("industry", "AgriTech")
        ind_idx = industry_options.index(curr_ind) if curr_ind in industry_options else 0
        val_industry = st.selectbox("Industry / Sector *", industry_options, index=ind_idx)

    with col2:
        stage_options = ["Idea", "Prototype", "MVP", "Early Customers", "Growth", "Established Business"]
        curr_stage = fd.get("startup_stage", "Early Customers")
        stg_idx = stage_options.index(curr_stage) if curr_stage in stage_options else 0
        val_stage = st.selectbox("Startup Stage *", stage_options, index=stg_idx)
        
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            val_country = st.text_input("Target Country *", value=fd.get("target_country", "India"))
        with col_g2:
            val_state = st.text_input("Target State", value=fd.get("target_state", "Maharashtra"))
        with col_g3:
            val_city = st.text_input("Target City / District", value=fd.get("target_city", "Pune"))

    val_target_cust = st.text_area(
        "Target Customer Segment *",
        value=fd.get("target_customer", ""),
        placeholder="Who is the primary buyer? e.g. Urban middle-to-upper-income residential apartment families consuming fresh produce daily.",
        height=80
    )

# ----------------------------------------------------------
# SECTION 02: IDEA & PROBLEM
# ----------------------------------------------------------
with form_tabs[1]:
    st.markdown("""
    <span class="section-badge">SECTION 02</span>
    <span class="section-title">Idea Mechanics & Problem Definition</span>
    <div class="section-desc">Describe the unmet market need, pain severity, and inadequacy of incumbent alternatives.</div>
    """, unsafe_allow_html=True)
    
    val_idea = st.text_area(
        "Startup Idea (Product or Service Mechanics) *",
        value=fd.get("startup_idea", ""),
        placeholder="Describe what your product/service does, who uses it and how it works in complete natural language.",
        height=130
    )
    
    val_problem = st.text_area(
        "Problem Being Solved (Customer Pain Point) *",
        value=fd.get("problem", ""),
        placeholder="Describe the specific customer pain point, difficulty, cost, delay or unmet need.",
        height=120
    )
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        val_who_experiences = st.text_input("Who experiences this problem?", value=fd.get("who_experiences", ""), placeholder="e.g. 40M urban residential families and smallholder farmers")
        val_frequency = st.text_input("How frequently does the problem occur?", value=fd.get("frequency", ""), placeholder="e.g. Daily perishable food consumption cycle")
    with col_p2:
        val_alternatives = st.text_input("Current alternatives / existing solutions", value=fd.get("current_alternatives", ""), placeholder="e.g. Local wet mandis, roadside pushcarts, quick-commerce dark stores")
        val_why_insufficient = st.text_input("Why are current solutions insufficient?", value=fd.get("why_insufficient", ""), placeholder="e.g. Stale produce, multiple middleman commissions, 35% harvest spoilage")

# ----------------------------------------------------------
# SECTION 03: SOLUTION & VALUE PROPOSITION
# ----------------------------------------------------------
with form_tabs[2]:
    st.markdown("""
    <span class="section-badge">SECTION 03</span>
    <span class="section-title">Solution Architecture & Competitive Moat</span>
    <div class="section-desc">Detail your value proposition, technology foundation, and defensible intellectual property.</div>
    """, unsafe_allow_html=True)
    
    val_solution = st.text_area(
        "Proposed Solution (Operational Approach) *",
        value=fd.get("proposed_solution", ""),
        placeholder="Explain how your solution works, eliminates the pain point, and delivers tangible value.",
        height=110
    )
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        val_core_features = st.text_area("Core Product Features", value=fd.get("core_features", ""), placeholder="e.g. Housing society drop points, direct farmer scheduling, QR-code harvest tracking.", height=85)
        val_uvp = st.text_area("Unique Value Proposition (UVP) *", value=fd.get("uvp", ""), placeholder="e.g. Harvest-to-kitchen in under 14 hours at mandi-equivalent prices.", height=85)
    with col_s2:
        val_adv = st.text_area("Competitive Advantage *", value=fd.get("competitive_advantage", ""), placeholder="Why would customers choose this solution over alternatives? e.g. Eliminates retail store rents, 70% cheaper delivery per basket.", height=85)
        val_tech = st.text_area("Technology & IP Differentiation", value=fd.get("tech_used", ""), placeholder="e.g. IoT temperature monitoring, predictive route aggregation algorithm, automated escrow.", height=85)
    
    val_ip = st.text_input("Intellectual Property / Proprietary Moat", value=fd.get("ip_differentiation", ""), placeholder="e.g. Proprietary society-cluster batch logistics dispatch algorithm and direct FPO supplier network")

# ----------------------------------------------------------
# SECTION 04: CUSTOMER & MARKET
# ----------------------------------------------------------
with form_tabs[3]:
    st.markdown("""
    <span class="section-badge">SECTION 04</span>
    <span class="section-title">Customer Segmentation & Competitor Landscape</span>
    <div class="section-desc">Define your addressable customer tiers, geographic scope, and known market competitors.</div>
    """, unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        val_primary_cust = st.text_input("Primary Customer", value=fd.get("primary_customer", ""), placeholder="e.g. Suburban health-conscious families and working couples")
        val_secondary_cust = st.text_input("Secondary Customer", value=fd.get("secondary_customer", ""), placeholder="e.g. Local cafes, cloud kitchens, cooperative housing societies")
        val_cust_loc = st.text_input("Customer Location", value=fd.get("customer_location", ""), placeholder="e.g. Pune, Navi Mumbai, and surrounding western industrial belt")
    with col_m2:
        val_market_type = st.selectbox("Market Scope", ["Local", "Regional", "National", "Global"], index=["Local", "Regional", "National", "Global"].index(fd.get("market_type", "Regional")))
        val_est_segment = st.text_input("Estimated Customer Segment Size", value=fd.get("estimated_segment", ""), placeholder="e.g. 450,000 households across 2,400 gated residential complexes")
        val_expected_vol = st.text_input("Expected Customer Volume", value=fd.get("expected_volume", ""), placeholder="e.g. 12,000 monthly active subscribers within 12 months")
        
    val_comp_known = st.toggle("Competition is already known?", value=fd.get("competition_known", True))
    if val_comp_known:
        val_known_competitors = st.text_area("Competitor Names & Existing Alternatives", value=fd.get("known_competitors", ""), placeholder="List primary competitors e.g. BigBasket, Otipy, Zepto, Local Mandi pushcarts", height=70)
    else:
        val_known_competitors = ""
        st.info("🤖 **AI-Assisted Discovery**: The AI model will identify realistic incumbent alternatives and clearly label them as AI-generated suggestions.")

# ----------------------------------------------------------
# SECTION 05: BUSINESS MODEL & FINANCIALS
# ----------------------------------------------------------
with form_tabs[4]:
    st.markdown("""
    <span class="section-badge">SECTION 05</span>
    <span class="section-title">Business Model & Unit Economics</span>
    <div class="section-desc">Outline revenue monetization, pricing tiers, and operating expenses. Indian Rupee (₹) is fully supported.</div>
    """, unsafe_allow_html=True)
    
    col_cur1, col_cur2 = st.columns([1, 3])
    with col_cur1:
        curr_sym = st.selectbox("Currency", ["₹ (INR)", "$ (USD)"], index=0 if fd.get("currency_symbol", "₹") == "₹" else 1)
        sym = "₹" if "₹" in curr_sym else "$"
    with col_cur2:
        rev_models = ["Direct Sales", "Subscription", "Freemium", "Commission", "Marketplace Fee", "Advertising", "B2B Contract", "Licensing", "Other"]
        curr_rm = fd.get("revenue_model", "Direct Sales")
        rm_idx = rev_models.index(curr_rm) if curr_rm in rev_models else 0
        val_revenue_model = st.selectbox("Revenue Model *", rev_models, index=rm_idx)
        
    val_pricing = st.text_input("Pricing Structure & Unit Margins *", value=fd.get("pricing", ""), placeholder="e.g. ₹450 average order value with 28% gross margin, ₹199 monthly premium club")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        val_monthly_cust = st.number_input("Expected Monthly Customers", min_value=0, value=int(fd.get("expected_monthly_customers", 4500)), step=100)
        val_monthly_rev = st.number_input(f"Expected Monthly Revenue ({sym}) *", min_value=0.0, value=float(fd.get("expected_monthly_revenue", 2250000.0)), step=50000.0)
        val_cac = st.number_input(f"Customer Acquisition Cost (CAC) ({sym}) [Optional]", min_value=0.0, value=float(fd.get("cac", 380.0)), step=25.0)
    with col_f2:
        val_monthly_exp = st.number_input(f"Monthly Operating Cost ({sym}) *", min_value=0.0, value=float(fd.get("monthly_operating_cost", 1650000.0)), step=50000.0)
        val_initial_inv = st.number_input(f"Initial Capital Investment ({sym}) *", min_value=0.0, value=float(fd.get("initial_investment", 3500000.0)), step=100000.0)
    with col_f3:
        val_working_cap = st.number_input(f"Working Capital Buffer ({sym})", min_value=0.0, value=float(fd.get("working_capital", 1200000.0)), step=50000.0)
        val_funding_req = st.number_input(f"External Funding Required ({sym})", min_value=0.0, value=float(fd.get("funding_required", 5000000.0)), step=250000.0)
        
    # Real-Time Python Financial Derivation Preview
    fin_preview = calculate_financials(val_monthly_rev, val_monthly_exp, val_initial_inv, val_working_cap, val_funding_req)
    
    st.markdown("##### ⚡ Automated Unit Economics Preview *(Python-Calculated)*")
    c_met1, c_met2, c_met3, c_met4 = st.columns(4)
    with c_met1:
        st.metric("Net Monthly Profit / Burn", f"{sym}{fin_preview['monthly_profit']:,.0f}")
    with c_met2:
        st.metric("Annualized Run-Rate", f"{sym}{fin_preview['annual_revenue']:,.0f}")
    with c_met3:
        st.metric("Annual Operating Cost", f"{sym}{fin_preview['annual_operating_cost']:,.0f}")
    with c_met4:
        st.metric("Break-Even Status", fin_preview['financial_status'], delta=fin_preview['break_even_months'])
    st.caption("⚠️ *All financial metrics shown above are purely Python-derived from user input figures and are explicitly labelled as user-provided/estimated.*")

# ----------------------------------------------------------
# SECTION 06: TEAM & RESOURCES
# ----------------------------------------------------------
with form_tabs[5]:
    st.markdown("""
    <span class="section-badge">SECTION 06</span>
    <span class="section-title">Team Capabilities & Available Resources</span>
    <div class="section-desc">Provide founder background, technical proficiencies, existing partnerships, and institutional mentors.</div>
    """, unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        val_team_size = st.number_input("Team Size (Full-Time Equivalent)", min_value=1, value=int(fd.get("team_size", 8)), step=1)
        val_founder_exp = st.text_area("Founder Experience & Track Record", value=fd.get("founder_experience", ""), placeholder="e.g. 6 years in e-commerce supply chains at Flipkart and agronomical research.", height=75)
        val_tech_skills = st.text_input("Technical Proficiencies", value=fd.get("technical_skills", ""), placeholder="e.g. Flutter mobile, IoT LoRaWAN hardware, Python predictive modeling, PostgreSQL")
        val_biz_skills = st.text_input("Commercial & Business Skills", value=fd.get("business_skills", ""), placeholder="e.g. Farmer negotiation, B2C growth marketing, unit economics management")
    with col_t2:
        val_infra = st.text_input("Available Infrastructure", value=fd.get("available_infra", ""), placeholder="e.g. 2 leased aggregation hubs in Narayangaon and 4 refrigerated transit vans")
        val_partnerships = st.text_input("Key Strategic Partnerships", value=fd.get("partnerships", ""), placeholder="e.g. 3 Farmer Producer Organizations (420 smallholder farmers)")
        val_advisors = st.text_input("Mentors / Advisors", value=fd.get("mentors_advisors", ""), placeholder="e.g. Ex-General Manager NABARD and ex-VP Logistics at Delhivery")
        val_resources = st.text_input("Other Available Resources", value=fd.get("available_resources", ""), placeholder="e.g. Pilot operations active in 18 gated housing societies")

# ----------------------------------------------------------
# EXISTING STARTUP OPERATIONAL PARAMETERS (MODE B)
# ----------------------------------------------------------
if is_existing_mode:
    st.markdown("### 🌲 Historical Operational & Investment Features *(For Random Forest ML)*")
    st.caption("These features feed the historical startup pattern analyzer trained on Crunchbase venture outcomes.")
    
    cur_yr = datetime.now().year
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        val_founded_year = st.number_input("Founded Year", min_value=1990, max_value=cur_yr, value=int(fd.get("founded_year", max(2018, cur_yr - 3))), step=1)
        val_total_fund = st.number_input("Total Cumulative Funding Received ($ USD)", min_value=0.0, value=float(fd.get("total_funding", 150000.0)), step=25000.0)
        val_rounds = st.number_input("Total Number of Funding Rounds", min_value=0, value=int(fd.get("funding_rounds", 1)), step=1)
    with col_e2:
        val_first_fund_yr = st.number_input("First Funding Year", min_value=1990, max_value=cur_yr, value=int(fd.get("first_funding_year", max(2019, cur_yr - 2))), step=1)
        val_last_fund_yr = st.number_input("Latest Funding Year", min_value=1990, max_value=cur_yr, value=int(fd.get("latest_funding_year", cur_yr)), step=1)
        val_connections = st.number_input("Professional Network Connections (Investors / Board / Mentors)", min_value=0, value=int(fd.get("connections", 12)), step=1)
    with col_e3:
        st.markdown("##### Historical Round Participations")
        val_has_vc = st.checkbox("Venture Capital (VC) Backed", value=bool(fd.get("has_vc", False)))
        val_has_angel = st.checkbox("Angel Investor Backed", value=bool(fd.get("has_angel", True)))
        val_has_round_a = st.checkbox("Series A Raised", value=bool(fd.get("has_round_a", False)))
        val_has_round_b = st.checkbox("Series B Raised", value=bool(fd.get("has_round_b", False)))
        val_has_round_c = st.checkbox("Series C Raised", value=bool(fd.get("has_round_c", False)))
        val_has_round_d = st.checkbox("Series D Raised", value=bool(fd.get("has_round_d", False)))
else:
    cur_yr = datetime.now().year
    val_founded_year = cur_yr
    val_total_fund = 0.0
    val_rounds = 0
    val_first_fund_yr = cur_yr
    val_last_fund_yr = cur_yr
    val_connections = 4
    val_has_vc = False
    val_has_angel = False
    val_has_round_a = False
    val_has_round_b = False
    val_has_round_c = False
    val_has_round_d = False

# ==========================================================
# VALIDATION & TRIGGER BUTTON
# ==========================================================

st.divider()

col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    st.caption("Pressing the button initiates NLP token extraction, Hugging Face LLM analysis, and historical Machine Learning checks.")
with col_btn2:
    start_analysis = st.button("🚀 Start Deep Analysis", type="primary", use_container_width=True)

# ==========================================================
# EXECUTION & RESULTS PRESENTATION
# ==========================================================

if start_analysis:
    # 1. Input Validation
    missing_fields = []
    if not val_name.strip(): missing_fields.append("Startup Name")
    if not val_idea.strip(): missing_fields.append("Startup Idea")
    if not val_problem.strip(): missing_fields.append("Problem Being Solved")
    if not val_solution.strip(): missing_fields.append("Proposed Solution")
    if not val_target_cust.strip(): missing_fields.append("Target Customer")
    if not val_uvp.strip(): missing_fields.append("Unique Value Proposition")
    
    if missing_fields:
        st.error(f"⚠️ Please complete the following required fields before analysis: **{', '.join(missing_fields)}**")
        st.stop()
        
    if val_monthly_rev < 0 or val_monthly_exp < 0 or val_initial_inv < 0:
        st.error("⚠️ Financial values cannot be negative. Please check Section 05.")
        st.stop()

    # Package profile data
    profile_data = {
        "startup_name": val_name,
        "founder_name": val_founder,
        "industry": val_industry,
        "startup_stage": val_stage,
        "target_country": val_country,
        "target_state": val_state,
        "target_city": val_city,
        "target_customer": val_target_cust,
        "startup_idea": val_idea,
        "problem": val_problem,
        "who_experiences": val_who_experiences,
        "frequency": val_frequency,
        "current_alternatives": val_alternatives,
        "why_insufficient": val_why_insufficient,
        "proposed_solution": val_solution,
        "core_features": val_core_features,
        "uvp": val_uvp,
        "competitive_advantage": val_adv,
        "tech_used": val_tech,
        "ip_differentiation": val_ip,
        "primary_customer": val_primary_cust,
        "secondary_customer": val_secondary_cust,
        "customer_location": val_cust_loc,
        "market_type": val_market_type,
        "estimated_segment": val_est_segment,
        "expected_volume": val_expected_vol,
        "competition_known": val_comp_known,
        "known_competitors": val_known_competitors,
        "revenue_model": val_revenue_model,
        "pricing": val_pricing,
        "currency_symbol": sym,
        "team_size": val_team_size,
        "founder_experience": val_founder_exp,
        "technical_skills": val_tech_skills,
        "business_skills": val_biz_skills,
        "available_infra": val_infra,
        "partnerships": val_partnerships,
        "mentors_advisors": val_advisors,
        "available_resources": val_resources,
        # ML fields
        "founded_year": val_founded_year,
        "total_funding": val_total_fund,
        "funding_rounds": val_rounds,
        "first_funding_year": val_first_fund_yr,
        "latest_funding_year": val_last_fund_yr,
        "connections": val_connections,
        "has_vc": val_has_vc,
        "has_angel": val_has_angel,
        "has_round_a": val_has_round_a,
        "has_round_b": val_has_round_b,
        "has_round_c": val_has_round_c,
        "has_round_d": val_has_round_d
    }

    # Step 1: NLP Processing Layer
    with st.spinner("🔍 Running Natural Language Processing extraction & semantic categorization..."):
        full_corpus = f"{val_idea} {val_problem} {val_solution} {val_uvp} {val_adv} {val_target_cust}"
        nlp_data = extract_nlp_insights(full_corpus, val_industry, val_stage, val_revenue_model)
        
    # Step 2: Financial Calculation Engine
    financials = calculate_financials(val_monthly_rev, val_monthly_exp, val_initial_inv, val_working_cap, val_funding_req)
    
    # Step 3: Historical Machine Learning (Random Forest)
    ml_result = None
    if is_existing_mode and ml_available:
        with st.spinner("🌲 Evaluating historical Crunchbase pattern similarity via Random Forest..."):
            ml_result = run_historical_ml(profile_data, cur_yr)
            
    # Step 4: Hugging Face LLM Strategic Analysis
    with st.spinner("🤖 Senior Startup Analyst AI evaluating venture viability & risk vectors..."):
        prompt = build_llm_prompt(profile_data, nlp_data, financials, ml_result)
        llm_analysis, engine_status = call_huggingface_llm(prompt, selected_model=ai_backend_model, temperature=llm_temperature)
        
        # If API failed or returned malformed JSON, activate robust fallback
        if not llm_analysis:
            st.warning(f"ℹ️ {engine_status} Falling back to internal heuristic analysis engine.")
            llm_analysis = generate_fallback_analysis(profile_data, nlp_data, financials)
            engine_status = "StartupSense Heuristic Intelligence Engine"

    # Step 5: Scoring Engine (Deterministic Python Weighted Calculation)
    ml_prob = ml_result.get("success_probability") if ml_result else None
    final_score, idea_score, score_blend_note = calculate_overall_score(
        llm_analysis.get("scores", {}),
        "Existing Startup" if is_existing_mode else "New Startup",
        ml_prob
    )
    score_pack = {
        "final_score": final_score,
        "idea_score": idea_score,
        "score_blend_note": score_blend_note
    }

    # Store results in session state for persistent interaction
    st.session_state["analysis_results"] = {
        "profile": profile_data,
        "nlp": nlp_data,
        "financials": financials,
        "ml": ml_result,
        "llm": llm_analysis,
        "scores": score_pack,
        "engine": engine_status
    }

# ==========================================================
# RENDER VALIDATION REPORT DASHBOARD
# ==========================================================

if "analysis_results" in st.session_state:
    res = st.session_state["analysis_results"]
    prof = res["profile"]
    nlp = res["nlp"]
    fin = res["financials"]
    ml = res["ml"]
    llm = res["llm"]
    sc = res["scores"]
    engine = res["engine"]
    
    st.divider()
    
    # ------------------------------------------------------
    # REPORT HEADER & QUICK AI UNDERSTANDING
    # ------------------------------------------------------
    st.markdown(f"## 📊 {prof['startup_name']} – Strategic Intelligence Report")
    st.caption(f"Engine: {engine} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # AI Understanding Section
    st.markdown("##### 🧠 AI Natural Language Understanding")
    u_c1, u_c2, u_c3, u_c4, u_c5 = st.columns(5)
    with u_c1:
        st.markdown(f"**Detected Industry**<br><span class='tag-badge tag-blue'>{nlp['detected_industry']}</span>", unsafe_allow_html=True)
    with u_c2:
        st.markdown(f"**Detected Customer**<br><span class='tag-badge tag-purple'>{nlp['detected_customer']}</span>", unsafe_allow_html=True)
    with u_c3:
        st.markdown(f"**Detected Problem**<br><span class='tag-badge tag-rose'>{nlp['detected_problem']}</span>", unsafe_allow_html=True)
    with u_c4:
        st.markdown(f"**Business Model**<br><span class='tag-badge tag-green'>{nlp['detected_business_model']}</span>", unsafe_allow_html=True)
    with u_c5:
        concepts = " ".join([f"<span class='tag-badge tag-amber'>{c}</span>" for c in nlp['detected_key_concepts'][:2]])
        st.markdown(f"**Key Concepts**<br>{concepts}", unsafe_allow_html=True)

    # ------------------------------------------------------
    # DASHBOARD TABS
    # ------------------------------------------------------
    report_tabs = st.tabs([
        "🏆 Executive Dashboard",
        "💡 Deep Business Analysis",
        "🧭 End-to-End Founder Playbook",
        "💬 AI Advisory Copilot",
        "⚖️ SWOT Analysis",
        "🚨 Deep Risk Matrix",
        "🏢 Competitor Intelligence",
        "💰 Financial Modeling",
        "🎯 Strategic Action Plan",
        "🧠 NLP Insights",
        "🌲 Historical ML Pattern",
        "📑 Download Report"
    ])

    # ------------------------------------------------------
    # TAB 1: EXECUTIVE DASHBOARD
    # ------------------------------------------------------
    with report_tabs[0]:
        st.markdown("### 🏆 Executive Validation Dashboard")
        
        # Big Score Card
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            score_color = "#10b981" if sc['final_score'] >= 75 else ("#f59e0b" if sc['final_score'] >= 55 else "#f43f5e")
            st.markdown(f"""
            <div class="metric-hero">
                <div style="font-size:12px;font-weight:700;letter-spacing:1px;color:#64748b;">COMPOSITE VALIDATION SCORE</div>
                <div class="metric-value-large" style="color:{score_color};">{sc['final_score']}<span style="font-size:24px;color:#94a3b8;">/100</span></div>
                <div style="font-size:13px;font-weight:600;color:#64748b;margin-bottom:8px;">{sc['score_blend_note']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(int(sc['final_score']))
            if sc['final_score'] >= 75:
                st.success("🟢 **High Strategic Potential**: Solid problem-solution alignment with clear monetization potential.")
            elif sc['final_score'] >= 55:
                st.warning("🟡 **Moderate Potential (Needs De-risking)**: Viable premise requiring validation of unit economics and distribution.")
            else:
                st.error("🔴 **Elevated Vulnerability**: Fundamental execution, customer acquisition, or margin challenges identified.")
                
            if ml and ml.get("success_probability") is not None:
                st.info(f"🌲 **Historical ML Signal**: {ml['success_probability']:.1f}% pattern match to historical venture success datasets.")

        with col_s2:
            st.markdown("#### 📑 Executive Conclusion")
            st.write(llm.get("executive_summary", "Executive summary unavailable."))
            
            st.markdown("#### 🎯 Strategic Verdict")
            st.write(llm.get("final_verdict", "Strategic verdict unavailable."))

        st.divider()

        # Multi-factor score breakdown chart
        st.markdown("#### 📊 Evaluation Factor Breakdown")
        scores_data = []
        raw_scores = llm.get("scores", {})
        for factor_key, factor_val in raw_scores.items():
            scores_data.append({
                "Dimension": factor_key.replace("_", " ").title(),
                "Score": factor_val.get("score", 65),
                "Interpretation": factor_val.get("interpretation", ""),
                "Weight": f"{int(SCORE_WEIGHTS.get(factor_key, 0.1)*100)}%"
            })
            
        df_scores = pd.DataFrame(scores_data)
        
        c_chart, c_tbl = st.columns([3, 2])
        with c_chart:
            bar_chart = alt.Chart(df_scores).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
                x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 100]), title="Dimension Score (0–100)"),
                y=alt.Y('Dimension:N', sort='-x', title=""),
                color=alt.Color('Score:Q', scale=alt.Scale(scheme='blues'), legend=None),
                tooltip=['Dimension', 'Score', 'Interpretation', 'Weight']
            ).properties(height=320)
            st.altair_chart(bar_chart, use_container_width=True)
            
        with c_tbl:
            st.dataframe(
                df_scores[['Dimension', 'Score', 'Weight', 'Interpretation']],
                hide_index=True,
                use_container_width=True
            )

    # ------------------------------------------------------
    # TAB 2: DEEP BUSINESS ANALYSIS
    # ------------------------------------------------------
    with report_tabs[1]:
        st.markdown("### 💡 Rigorous Business Analysis (120–220 Words per Domain)")
        st.caption("Every domain features an executive conclusion, detailed analytical explanation, user evidence citation, core assumptions, business implications, and actionable recommendations.")
        
        analysis_sections = [
            ("Problem Analysis", "problem_analysis", "🔥"),
            ("Customer Need & Behavioral Dynamics", "customer_analysis", "👥"),
            ("Solution Relevance & Product Architecture", "solution_analysis", "💡"),
            ("Market Opportunity & Dynamics", "market_analysis", "📈"),
            ("Business Model & Margin Viability", "business_model_analysis", "💰"),
            ("Operational Feasibility", "feasibility_analysis", "⚙️"),
            ("Scalability & Network Effects", "scalability_analysis", "🚀")
        ]
        
        for title, key, icon in analysis_sections:
            sec_data = llm.get(key, {})
            with st.expander(f"{icon} {title}", expanded=(key in ["problem_analysis", "solution_analysis"])):
                st.markdown(f"**Executive Conclusion:** *{sec_data.get('conclusion', 'No conclusion available.')}*")
                st.markdown("---")
                st.write(sec_data.get("detailed_explanation", "Detailed analysis not populated."))
                
                col_sub1, col_sub2 = st.columns(2)
                with col_sub1:
                    st.markdown("**Evidence from User Submission:**")
                    st.caption(sec_data.get("evidence", "Based on submitted operational overview."))
                    st.markdown("**Core Assumptions:**")
                    st.caption(sec_data.get("assumptions", "Standard market continuity assumed."))
                with col_sub2:
                    st.markdown("**Strategic Implication:**")
                    st.caption(sec_data.get("implication", "Directly impacts capital efficiency and market entry."))
                    st.markdown("**Actionable Recommendation:**")
                    st.caption(f"👉 {sec_data.get('recommendation', 'Execute pilot validation tests.')}")

    # ------------------------------------------------------
    # TAB 3: END-TO-END FOUNDER PLAYBOOK
    # ------------------------------------------------------
    with report_tabs[2]:
        st.markdown(f"### 🧭 {prof['startup_name']} – End-to-End Founder Execution Playbook")
        st.caption("A tailored, phase-by-phase operational roadmap to incorporate, build, launch, acquire customers, and scale your venture.")
        
        playbook_stage = st.radio(
            "Select Execution Phase / Advisory Module:",
            [
                "1. ⚖️ Legal Incorporation & Compliance",
                "2. 🏛️ Government Schemes & Subsidies Navigator",
                "3. 💵 Financial Assistant & Runway Planner",
                "4. 🛠️ Product & MVP 4-Week Rapid Sprint",
                "5. 🚀 Go-To-Market & First 100 Customers",
                "6. 💼 Investor Readiness & 10-Slide Pitch Deck",
                "7. 👥 Team Structuring & Co-Founder Vesting"
            ],
            horizontal=True
        )
        
        if "1. ⚖️" in playbook_stage:
            st.markdown("#### ⚖️ Phase 1: Legal Incorporation & Compliance Guide")
            
            c_leg1, c_leg2 = st.columns(2)
            with c_leg1:
                with st.container(border=True):
                    st.markdown("##### 🏛️ Entity Structure Selection")
                    st.write("""
                    - **Private Limited Company (Recommended):** Essential if raising external Angel/VC funds, issuing ESOPs, or capping founder liability.
                      - *Timeline:* 7–10 days via SPICe+ MCA portal.
                      - *Mandatory Steps:* Digital Signature Certificates (DSC), Director Identification Number (DIN), RUN Name Approval, MoA & AoA drafting.
                    - **Limited Liability Partnership (LLP):** Best for bootstrapped or service ventures with lower statutory audit burdens.
                    """)
                    
                with st.container(border=True):
                    st.markdown("##### 🇮🇳 DPIIT Startup India Recognition")
                    st.write("""
                    - **Eligibility:** Incorporated <10 years, turnover <₹100 Cr, original scalable business entity.
                    - **Key Benefits:**
                      - **Section 80-IAC 3-Year Income Tax Holiday:** 100% tax exemption for 3 consecutive financial years.
                      - **Angel Tax Exemption:** Exemption under Section 56(2)(viib) on capital raised above fair market value.
                      - **Fast-Track IP:** Up to 80% rebate on patent filings and 50% rebate on trademark applications.
                    """)

            with c_leg2:
                with st.container(border=True):
                    st.markdown("##### 📜 Statutory Tax & Sector Licenses")
                    st.write(f"""
                    - **GST Registration:** Mandatory for inter-state digital sales or turnover >₹20L (Services) / >₹40L (Goods).
                    - **MSME / Udyam Registration:** Free priority sector lending, collateral-free credit, and delayed payment protection.
                    - **Sectoral Licensing for {prof['industry']}:**
                      - *AgriTech / Food:* FSSAI State/Central License, Mandi trading license or FPO direct exempt certification.
                      - *FinTech:* RBI payment aggregator authorization or regulatory sandbox application.
                      - *HealthTech:* CDSCO medical device / telehealth data privacy compliance.
                      - *E-Commerce / Consumer:* Consumer Protection (E-Commerce) Rules 2020 & Data Protection (DPDP) compliance.
                    """)
                    
                with st.container(border=True):
                    st.markdown("##### 🛡️ Founder Legal Shield Checklist")
                    st.write("""
                    - **Co-Founder Agreement:** Equity split, roles, vesting triggers, non-compete.
                    - **Intellectual Property (IP) Assignment Agreement:** Ensures all code, domain, and designs belong to the company, not individual founders.
                    - **Standard Non-Disclosure Agreement (NDA):** For third-party vendors and early pilot partners.
                    """)

        elif "2. 🏛️" in playbook_stage:
            st.markdown(f"#### 🏛️ Government Schemes, Grants & Subsidies Navigator")
            st.caption(f"Curated national and regional funding schemes matched for **{prof['startup_name']}** ({prof['industry']} | {prof['startup_stage']})")
            
            c_gs1, c_gs2 = st.columns(2)
            with c_gs1:
                with st.container(border=True):
                    st.markdown("##### 🇮🇳 1. Startup India Seed Fund Scheme (SISFS)")
                    st.markdown("""
                    - **Financial Assistance:**
                      - **Up to ₹20 Lakhs (100% Grant):** For validation of Proof of Concept (PoC), prototype development, and product trials.
                      - **Up to ₹50 Lakhs (Debt / Convertible Debentures):** For market entry, commercialization, and scaling.
                    - **Eligibility Criteria:**
                      - DPIIT-recognized startup incorporated not more than 2 years prior to application.
                      - Must have a business idea to develop a product or service with market fit and scope of scaling.
                      - Must not have received more than ₹10 Lakhs in monetary support under any other Central/State scheme.
                    - **Application Portal:** Direct application via `seedfund.startupindia.gov.in` by choosing up to 3 authorized incubation centers.
                    """)
                    
                with st.container(border=True):
                    st.markdown("##### 🛡️ 2. Credit Guarantee Scheme for Startups (CGSS / CGTMSE)")
                    st.markdown("""
                    - **Financial Assistance:** Collateral-free debt/loans from **₹5 Crores up to ₹10 Crores** per borrower.
                    - **Mechanism:** Government of India provides 75% to 85% sovereign guarantee coverage to lending banks (SBI, HDFC, SIDBI, etc.).
                    - **Best For:** Working capital, machinery leasing, fulfillment hub expansion without mortgaging personal assets.
                    """)

                with st.container(border=True):
                    st.markdown("##### 🏦 3. Pradhan Mantri MUDRA Yojana (PMMY)")
                    st.markdown("""
                    - **Shishu:** Loans up to ₹50,000 for early validation tools.
                    - **Kishore:** Loans from ₹50,000 to ₹5,00,000 for initial inventory and pilot rollouts.
                    - **Tarun:** Loans from ₹5,00,000 to ₹20,00,000 for growth equipment and logistics.
                    - **No Collateral Required:** Available across all commercial and rural banks with subsidized interest rates.
                    """)

            with c_gs2:
                with st.container(border=True):
                    st.markdown(f"##### 🎯 4. Sector-Specific Grants ({prof['industry']})")
                    ind_lower = prof['industry'].lower()
                    if "agri" in ind_lower or "food" in ind_lower:
                        st.markdown("""
                        - **RKVY-RAFTAAR (Ministry of Agriculture):**
                          - *Idea Stage:* Grant-in-aid up to **₹5 Lakhs** (Agripreneurship Orientation).
                          - *Seed Stage:* Grant-in-aid up to **₹25 Lakhs** for commercial scale.
                        - **NABARD Rural Innovation Fund (RIF):** Financial support for farm-to-fork direct supply chain innovations.
                        - **FPO Equity Grant Scheme:** Matching grants up to ₹15 Lakhs for farmer collectives and procurement networks.
                        """)
                    elif "health" in ind_lower or "bio" in ind_lower:
                        st.markdown("""
                        - **BIRAC BIG (Biotechnology Ignition Grant):** Equity-free grant up to **₹50 Lakhs** for 18 months for medical/biotech PoC.
                        - **SPARSH Scheme:** Social innovation grants for healthcare affordability.
                        """)
                    elif "tech" in ind_lower or "saas" in ind_lower or "ai" in ind_lower or "software" in ind_lower:
                        st.markdown("""
                        - **MeitY SAMRIDH Scheme:** Matching grant-in-aid up to **₹40 Lakhs** through certified tech accelerators.
                        - **TIDE 2.0 (MeitY):** Financial incubation support across 51 premier tech institutes.
                        - **GENESIS (Gen-Next Support for Innovative Startups):** ₹490 Cr national scheme for Tier-2/Tier-3 tech ventures.
                        """)
                    else:
                        st.markdown("""
                        - **MeitY SAMRIDH Scheme:** Matching grant-in-aid up to **₹40 Lakhs** for scalable digital products.
                        - **Stand-Up India Scheme:** Bank loans from **₹10 Lakhs to ₹1 Crore** for SC/ST and Women entrepreneurs.
                        - **MSME Champions Scheme:** Subsidies on digital tech adoption and design certifications.
                        """)

                with st.container(border=True):
                    st.markdown("##### 🗺️ 5. Top State Government Startup Grants")
                    st.markdown("""
                    - **Karnataka (ELEVATE / Idea2PoC):** Up to **₹50 Lakhs** equity-free grant across tech, agri, and social innovation.
                    - **Tamil Nadu (StartupTN TANSEED):** **₹10 Lakhs** seed grant for early-stage scalable ventures.
                    - **Kerala (KSUM Seed Loan):** Soft loans up to **₹15 Lakhs** with 6% simple interest.
                    - **Telangana (T-Hub Seed Fund):** Seed capital and incubation grants through the state innovation ecosystem.
                    - **Maharashtra (MSInS Voucher Scheme):** Up to ₹5 Lakhs for cloud credits, IP testing, and incubator access.
                    """)
                    
                with st.container(border=True):
                    st.markdown("##### 📋 Grant Application Readiness Checklist")
                    st.markdown("""
                    - [ ] DPIIT Recognition Certificate (from `startupindia.gov.in`)
                    - [ ] Company Incorporation Certificate (SPICe+ MCA) & Company PAN
                    - [ ] 2-Page Executive Brief & 10-Slide Pitch Deck PDF
                    - [ ] 12-Month Fund Utilization Projection (CapEx vs OpEx table)
                    - [ ] Letter of recommendation or association from an approved Incubator
                    """)

        elif "3. 💵" in playbook_stage:
            st.markdown(f"#### 💵 Financial Assistant & Runway Planner")
            st.caption(f"Financial health diagnostics, burn rate analysis, and capital strategy for **{prof['startup_name']}**")
            
            # Interactive financial simulation
            rev = fin.get('monthly_revenue', 0.0)
            exp = fin.get('monthly_expenses', 0.0)
            net_burn = max(0.0, exp - rev)
            cur_sym = prof.get('currency_symbol', '₹')
            
            c_f1, c_f2 = st.columns([1, 1])
            with c_f1:
                with st.container(border=True):
                    st.markdown("##### 🧮 Live Runway & Burn Rate Calculator")
                    default_reserve = float(fin.get('funding_required', 0.0)) if fin.get('funding_required', 0) > 0 else (net_burn * 6 if net_burn > 0 else 500000.0)
                    user_reserve = st.number_input(
                        f"Current Cash in Bank / Seed Reserve ({cur_sym}):",
                        min_value=0.0,
                        max_value=100000000.0,
                        value=float(default_reserve),
                        step=50000.0,
                        help="Enter your current liquid funds to project your survival runway."
                    )
                    
                    if net_burn > 0:
                        runway_mo = user_reserve / net_burn
                        st.metric("Monthly Net Cash Burn", f"{cur_sym}{net_burn:,.0f}/mo")
                        st.metric("Estimated Runway", f"{runway_mo:.1f} Months")
                        
                        if runway_mo >= 12:
                            st.success(f"🟢 **Healthy Runway ({runway_mo:.1f} mo):** You have ample time to hit product-market fit before needing fresh capital.")
                        elif runway_mo >= 6:
                            st.warning(f"🟡 **Moderate Runway ({runway_mo:.1f} mo):** Begin investor outreach or grant applications now. Typical fundraise takes 3–6 months.")
                        else:
                            st.error(f"🔴 **Critical Runway Warning ({runway_mo:.1f} mo):** High risk of insolvency. Implement immediate burn reduction or apply for SISFS/MUDRA grants.")
                    else:
                        st.success(f"🟢 **Cash-Flow Positive / Zero Net Burn:** Monthly revenue ({cur_sym}{rev:,.0f}) covers or exceeds monthly expenses ({cur_sym}{exp:,.0f}).")
                        st.metric("Monthly Net Profit", f"{cur_sym}{(rev - exp):,.0f}/mo")

            with c_f2:
                with st.container(border=True):
                    st.markdown("##### 📈 Unit Economics Health Benchmarks")
                    st.markdown(f"""
                    - **Target LTV : CAC Ratio:** >= 3:1 *(Customer Lifetime Value must be at least 3x the cost to acquire them)*.
                    - **CAC Payback Period:** < 12 months for B2B; < 6 months for direct consumer.
                    - **Gross Margin Target for {prof['industry']}:**
                      - *SaaS / Software:* 70% – 85%
                      - *Marketplace / Platform:* 15% – 25% take rate
                      - *D2C / AgriTech / Physical Supply:* 30% – 50% gross margin
                    - **Operating Breakeven Gap:** Monthly revenue gap to reach 0 burn: **{cur_sym}{net_burn:,.0f}**.
                    """)

            st.divider()
            c_fp1, c_fp2 = st.columns(2)
            with c_fp1:
                with st.container(border=True):
                    st.markdown("##### 💡 Fundraising Round Sizing Blueprint")
                    suggested_raise = net_burn * 18 if net_burn > 0 else 2500000.0
                    st.markdown(f"""
                    - **Golden Rule:** Raise enough for **18–24 months of runway** plus a 20% contingency buffer.
                    - **Recommended Round Target:** Approximately **{cur_sym}{suggested_raise:,.0f}** for pre-seed.
                    - **Founder Dilution Boundary:** Do not dilute more than **10% – 18%** in your Pre-Seed or Angel round.
                    - **Instrument:** Use an **i-SAFE (India Simple Agreement for Future Equity)** note with a 20% valuation discount cap to avoid protracted equity pricing debates.
                    """)
            with c_fp2:
                with st.container(border=True):
                    st.markdown("##### 📉 4 Tactical Tactics to Reduce Monthly Burn")
                    st.markdown("""
                    1. **Leverage Founder Cloud Credits:** Claim AWS Activate ($5,000–$100,000 credits), Google for Startups, and Microsoft Founders Hub.
                    2. **Performance-Linked Contractor Model:** Replace fixed agency retainers with milestone-based contractor payouts.
                    3. **Upfront Annual Billing Incentive:** Offer a 20% discount to customers who pay annually upfront to fund operational working capital.
                    4. **Zero-Inventory Batch Sourcing:** Procure goods only after aggregating confirmed customer demand batches.
                    """)

        elif "4. 🛠️" in playbook_stage:
            st.markdown("#### 🛠️ Phase 4: Product & MVP 4-Week Rapid Sprint")
            
            c_mvp1, c_mvp2 = st.columns(2)
            with c_mvp1:
                with st.container(border=True):
                    st.markdown("##### 🗓️ 4-Week Rapid MVP Build Calendar")
                    st.markdown("""
                    - **Week 1 (Wireframing & Core Architecture):**
                      - Define single core user transaction flow (Happy Path).
                      - Figma interactive wireframe testing with 10 prospective users.
                      - Database schema and API endpoints mapped.
                    - **Week 2 (Core Feature Build):**
                      - Implement primary customer onboarding & authentication.
                      - Build primary functional engine (e.g. order basket, matching logic).
                      - Integrate payment gateway (Razorpay / Stripe) in test mode.
                    - **Week 3 (Logistics / Admin Portal & QA):**
                      - Build lightweight operational dashboard or admin portal for order tracking.
                      - Stress test order fulfillment flow under simulated latency.
                    - **Week 4 (Closed Alpha & Seed Customers):**
                      - Deploy closed beta to initial 20–50 waitlist users.
                      - Track daily active conversion and feedback loops.
                    """)
            with c_mvp2:
                with st.container(border=True):
                    st.markdown("##### 💻 Recommended Production Architecture")
                    st.markdown(f"""
                    - **Frontend:** Flutter (Cross-platform iOS/Android) or Next.js React for web.
                    - **Backend:** Python FastAPI or Node.js with PostgreSQL (Supabase / AWS RDS).
                    - **Operational Tooling:** WhatsApp Cloud API for transactional order alerts; Metabase / Retool for internal fulfillment dispatch.
                    """)
                    
                with st.container(border=True):
                    st.markdown("##### 🗣️ Customer Discovery Interview Script ('The Mom Test')")
                    st.markdown("""
                    1. *"What is the hardest part about dealing with [specific problem] today?"*
                    2. *"Can you talk me through the last time that happened?"*
                    3. *"Why was that hard or frustrating?"*
                    4. *"What, if anything, have you done to try to solve this problem?"*
                    5. *"What don't you love about the current solutions you've tried?"*
                    """)

        elif "5. 🚀" in playbook_stage:
            st.markdown("#### 🚀 Phase 5: Go-To-Market & First 100 Customers Playbook")
            
            gtm_col1, gtm_col2 = st.columns(2)
            with gtm_col1:
                with st.container(border=True):
                    st.markdown("##### 🎯 0-to-1 Organic Customer Acquisition Funnel")
                    st.markdown(f"""
                    - **Tactic 1: Micro-Cluster Penetration (B2B2C):**
                      - Target residential housing associations (RWAs) or office parks.
                      - Host physical weekend morning 'Fresh Harvest' pop-up kiosks.
                    - **Tactic 2: Influencer / Community Ambassador Incentive:**
                      - Recruit 1 active champion per residential building; provide free monthly baskets for order coordination.
                    - **Tactic 3: High-Density Group Drops:**
                      - Offer 15% discount when 5+ neighboring households order simultaneously.
                    """)
            with gtm_col2:
                with st.container(border=True):
                    st.markdown("##### ✉️ High-Converting Outreach Templates")
                    st.markdown(f"""
                    ```text
                    Subject: Fresh harvest directly to [Building/Community Name]
                    
                    Hi [Name],
                    
                    I noticed residents in [Community] often face issues with stale produce from delivery apps.
                    
                    We built {prof['startup_name']} to deliver chemical-free harvest directly from partner farms within 14 hours at wholesale mandi rates.
                    
                    Can we offer a free sample box for your committee members this Saturday morning?
                    
                    Best,
                    {prof.get('founder_name', 'Founder')}, {prof['startup_name']}
                    ```
                    """)

        elif "6. 💼" in playbook_stage:
            st.markdown("#### 💼 Phase 6: Investor Readiness & 10-Slide Pitch Deck Builder")
            
            c_inv1, c_inv2 = st.columns(2)
            with c_inv1:
                with st.container(border=True):
                    st.markdown("##### 📑 The Standard 10-Slide Investor Deck Structure")
                    st.markdown(f"""
                    1. **Title Slide:** {prof['startup_name']} – 1-line vision hook.
                    2. **The Problem:** The systemic pain point ({prof['problem'][:90]}...).
                    3. **The Solution:** What you do & why customers love it ({prof['proposed_solution'][:90]}...).
                    4. **Market Opportunity (TAM/SAM/SOM):** Addressable customer pool in {prof.get('target_state', 'region')}.
                    5. **Product & Tech Moat:** Core mechanics and proprietary algorithms.
                    6. **Business Model & Unit Economics:** Pricing, gross margin, CAC, and LTV.
                    7. **Traction & Validation:** Validation score ({sc['final_score']}/100), pilot numbers, MoM growth.
                    8. **Competitive Differentiation:** Why incumbents ({prof.get('known_competitors', 'legacy options')}) can't easily displace you.
                    9. **Founding Team:** Why you are uniquely qualified to execute this.
                    10. **The Ask:** Seeking {prof['currency_symbol']}{fin['funding_required']:,.0f} for 18-month runway milestones.
                    """)
            with c_inv2:
                with st.container(border=True):
                    st.markdown("##### ✉️ Cold Investor Outreach Email Template")
                    st.markdown(f"""
                    ```text
                    Subject: {prof['startup_name']} (Traction in {prof['industry']}) – Seed Opportunity
                    
                    Hi [Investor Name],
                    
                    I've been following your investments in {prof['industry']}.
                    
                    I'm building {prof['startup_name']} ({prof['uvp'][:80]}...).
                    
                    Key Traction Highlights:
                    • Unit Economics: {prof['currency_symbol']}{fin['monthly_revenue']:,.0f}/mo with positive contribution margin
                    • Retention: Strong repeat order density across our pilot clusters
                    • Moat: Direct producer sourcing eliminating middleman margins
                    
                    We are raising our seed round to scale into 3 new regional clusters. Would you have 10 minutes next Tuesday for a quick intro?
                    
                    Deck attached below.
                    
                    Best regards,
                    {prof.get('founder_name', 'Founder')}
                    ```
                    """)

        elif "7. 👥" in playbook_stage:
            st.markdown("#### 👥 Phase 7: Team Structuring & Co-Founder Equity Vesting")
            
            c_tm1, c_tm2 = st.columns(2)
            with c_tm1:
                with st.container(border=True):
                    st.markdown("##### ⚖️ The Standard 4-Year Vesting with 1-Year Cliff")
                    st.markdown("""
                    - **Year 0 to 1 (The Cliff):** 0% equity vests. If a co-founder leaves within 12 months, they depart with zero shares.
                    - **Month 12:** Exactly 25% of allotted equity vests in a single tranche.
                    - **Months 13 to 48:** Remaining 75% vests linearly each month (1/48th per month).
                    - **Why this protects you:** Prevents an early departing co-founder from walking away with 40% of your cap table, rendering the company un-investable.
                    """)
            with c_tm2:
                with st.container(border=True):
                    st.markdown("##### 👔 First 3 Critical Hires Checklist")
                    st.markdown("""
                    1. **Lead Operations / Logistics Coordinator:** Manages daily vendor dispatch, QA checkpoints, and route fulfillment.
                    2. **Full-Stack Growth Developer:** Optimizes customer checkout funnel, automated CRM messaging, and referral tracking.
                    3. **Community & B2B Growth Lead:** Drives on-ground housing society approvals, partnership activations, and local word-of-mouth campaigns.
                    """)

    # ------------------------------------------------------
    # TAB 4: AI FOUNDER ADVISORY COPILOT
    # ------------------------------------------------------
    with report_tabs[3]:
        st.markdown(f"### 💬 StartupSense AI – Interactive Advisory Copilot")
        st.caption(f"Powered by active backend: **{ai_backend_model}** | Context-aware advisor for **{prof['startup_name']}**")
        
        # Initialize session state for copilot chat
        if "copilot_history" not in st.session_state:
            st.session_state["copilot_history"] = [
                {
                    "role": "assistant",
                    "content": f"Hello! I am your AI Startup Advisor for **{prof['startup_name']}**. Based on your validation score of **{sc['final_score']}/100**, I have full context on your {prof['industry']} model, unit economics, and risk vectors. What strategic question or operational hurdle can I help you tackle today?"
                }
            ]
            
        # Quick Prompt Buttons
        st.markdown("##### ⚡ Strategic Execution Prompts")
        qp_col1, qp_col2, qp_col3, qp_col4 = st.columns(4)
        quick_query = None
        with qp_col1:
            if st.button("🚀 First 100 Customers Plan", use_container_width=True):
                quick_query = "Give me a step-by-step tactical playbook to acquire my first 100 paying customers for my startup with zero ad budget."
        with qp_col2:
            if st.button("📑 10-Slide Pitch Deck Script", use_container_width=True):
                quick_query = "Write a comprehensive 10-slide investor pitch deck structure tailored specifically to my startup with talking points for each slide."
        with qp_col3:
            if st.button("⚖️ Legal & Licensing Checklist", use_container_width=True):
                quick_query = "What specific legal entity, tax registrations (GST, PAN), DPIIT Startup India benefits, and licenses do I need for my venture?"
        with qp_col4:
            if st.button("🤝 Co-Founder Vesting & Equity", use_container_width=True):
                quick_query = "How should we structure our co-founder equity split and 4-year vesting agreement with a 1-year cliff to protect the startup?"

        st.markdown("##### 🏛️ Financial & Government Schemes Prompts")
        fq_col1, fq_col2, fq_col3, fq_col4 = st.columns(4)
        with fq_col1:
            if st.button("🏛️ Best Govt Scheme Match", use_container_width=True):
                quick_query = f"Which Indian and regional government startup schemes, subsidies, or grants (like SISFS, SAMRIDH, MUDRA, or CGTMSE) are best suited for my {prof['industry']} startup, and how do I qualify?"
        with fq_col2:
            if st.button("💵 Financial Runway Audit", use_container_width=True):
                quick_query = f"Perform an audit on my startup's unit economics (Monthly Revenue: {prof['currency_symbol']}{fin['monthly_revenue']}, Monthly Expenses: {prof['currency_symbol']}{fin['monthly_expenses']}). How can I optimize my net burn rate and extend my runway?"
        with fq_col3:
            if st.button("📜 Apply for Seed Fund (SISFS)", use_container_width=True):
                quick_query = "Provide a step-by-step application walkthrough to apply for the Startup India Seed Fund Scheme (SISFS) up to ₹20L grant, including sample proposal responses for an incubator review."
        with fq_col4:
            if st.button("💰 Pre-Seed Funding Strategy", use_container_width=True):
                quick_query = f"How much capital should I raise for my Pre-Seed round in {prof['industry']}, what valuation cap or i-SAFE structure should I offer, and what milestones should this round achieve?"

        # Display Chat History
        for msg in st.session_state["copilot_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Handle Quick Query or Manual Input
        user_input = st.chat_input("Ask StartupSense AI Copilot any question about your startup...")
        active_query = quick_query or user_input
        
        if active_query:
            st.session_state["copilot_history"].append({"role": "user", "content": active_query})
            with st.chat_message("user"):
                st.markdown(active_query)
                
            with st.chat_message("assistant"):
                with st.spinner(f"Consulting {ai_backend_model.split('/')[-1]}..."):
                    ai_reply = ask_ai_copilot(active_query, prof, llm, model_name=ai_backend_model, temperature=llm_temperature)
                    st.markdown(ai_reply)
                    st.session_state["copilot_history"].append({"role": "assistant", "content": ai_reply})

    # ------------------------------------------------------
    # TAB 5: SWOT ANALYSIS
    # ------------------------------------------------------
    with report_tabs[4]:
        st.markdown("### ⚖️ Multi-Vector SWOT Matrix")
        st.caption("Context-grounded internal competencies and external environmental vectors.")
        
        swot_c1, swot_c2 = st.columns(2)
        with swot_c1:
            with st.container(border=True):
                st.markdown("#### 🟢 Strengths (Internal Moat)")
                for item in llm.get("strengths", []):
                    st.markdown(f"• **{item}**")
                    
            with st.container(border=True):
                st.markdown("#### 🔵 Opportunities (External Growth)")
                for item in llm.get("opportunities", []):
                    st.markdown(f"• **{item}**")
                    
        with swot_c2:
            with st.container(border=True):
                st.markdown("#### 🔴 Weaknesses (Internal Constraints)")
                for item in llm.get("weaknesses", []):
                    st.markdown(f"• **{item}**")
                    
            with st.container(border=True):
                st.markdown("#### 🟠 Threats (External Headwinds)")
                for item in llm.get("threats", []):
                    st.markdown(f"• **{item}**")

    # ------------------------------------------------------
    # TAB 6: DEEP RISK MATRIX
    # ------------------------------------------------------
    with report_tabs[5]:
        st.markdown("### 🚨 Deep Risk Matrix & Early Warning Mitigation")
        st.caption("Multi-category exposure scoring evaluating Likelihood, Impact, Early Indicators, and Mitigations.")
        
        risks = llm.get("risks", [])
        if risks:
            risk_df_data = []
            for r in risks:
                risk_df_data.append({
                    "Risk Name": r.get("risk", "Unknown Risk"),
                    "Category": r.get("category", "Operational"),
                    "Likelihood": r.get("likelihood", "Medium"),
                    "Impact": r.get("impact", "Medium"),
                    "Severity": r.get("severity", "Moderate"),
                    "Priority": r.get("priority", "Medium")
                })
            st.dataframe(pd.DataFrame(risk_df_data), hide_index=True, use_container_width=True)
            
            st.markdown("#### 🛡️ Detailed Risk Mitigation Breakdown")
            for idx, r in enumerate(risks, start=1):
                sev = r.get("severity", "Moderate")
                badge_class = "tag-rose" if sev == "Critical" else ("tag-amber" if sev == "Moderate" else "tag-blue")
                with st.expander(f"{idx}. {r.get('risk', '')} [{r.get('category', '')}] — Priority: {r.get('priority', '')}"):
                    st.markdown(f"**Why it matters:** {r.get('why_it_matters', '')}")
                    st.markdown(f"**Early Warning Indicator:** `{r.get('early_warning_indicator', 'N/A')}`")
                    st.markdown(f"**Mitigation Playbook:** {r.get('mitigation_strategy', '')}")
                    
        # Team Gaps
        st.divider()
        st.markdown("#### 👥 Team Capabilities & Resource Gap Analysis")
        st.write(llm.get("team_resource_gap_analysis", "Team capabilities align with initial prototype requirements."))

    # ------------------------------------------------------
    # TAB 7: COMPETITOR BENCHMARKING
    # ------------------------------------------------------
    with report_tabs[6]:
        st.markdown("### 🏢 Competitor Benchmarking & Differentiation")
        st.caption("Distinguishes user-entered competitors from AI-suggested alternatives.")
        
        comps = llm.get("competitors", [])
        if comps:
            for idx, c in enumerate(comps, start=1):
                origin = c.get("origin", "Identified")
                badge_tag = "<span class='tag-badge tag-green'>User-Provided</span>" if "User" in origin else "<span class='tag-badge tag-purple'>AI-Suggested</span>"
                
                with st.container(border=True):
                    st.markdown(f"#### {idx}. {c.get('name', 'Competitor')} {badge_tag}", unsafe_allow_html=True)
                    st.caption(f"Type: {c.get('type', 'Alternative')} | Target Customer: {c.get('target_customer', 'General Market')}")
                    
                    c_col1, c_col2 = st.columns(2)
                    with c_col1:
                        st.markdown(f"**Main Strength:** {c.get('main_strength', '')}")
                        st.markdown(f"**Main Weakness:** {c.get('main_weakness', '')}")
                    with c_col2:
                        st.markdown(f"**How Your Startup Differs:** {c.get('how_startup_differs', '')}")
                        st.markdown(f"**Differentiation Opportunity:** {c.get('differentiation_opportunity', '')}")

    # ------------------------------------------------------
    # TAB 8: FINANCIAL MODELING
    # ------------------------------------------------------
    with report_tabs[7]:
        st.markdown("### 💰 Financial Model & Break-Even Projections")
        st.caption("Unit economics synthesized from founder inputs and calculated deterministically in Python.")
        
        f_m1, f_m2, f_m3, f_m4 = st.columns(4)
        with f_m1:
            st.metric("Monthly Revenue", f"{prof['currency_symbol']}{fin['monthly_revenue']:,.0f}", help="User-entered base")
            st.metric("Annualized Revenue", f"{prof['currency_symbol']}{fin['annual_revenue']:,.0f}", help="Calculated (12x)")
        with f_m2:
            st.metric("Monthly Expenses", f"{prof['currency_symbol']}{fin['monthly_expenses']:,.0f}", help="User-entered operational cost")
            st.metric("Annualized Expenses", f"{prof['currency_symbol']}{fin['annual_operating_cost']:,.0f}", help="Calculated (12x)")
        with f_m3:
            st.metric("Net Monthly Profit / Loss", f"{prof['currency_symbol']}{fin['monthly_profit']:,.0f}", delta=f"{fin['financial_status']}")
            st.metric("Annual Net Margin", f"{prof['currency_symbol']}{fin['annual_profit']:,.0f}")
        with f_m4:
            st.metric("Initial Investment", f"{prof['currency_symbol']}{fin['initial_investment']:,.0f}")
            st.metric("External Funding Need", f"{prof['currency_symbol']}{fin['funding_required']:,.0f}")
            
        st.divider()
        
        # Financial Charts: 12-Month Trajectory
        months = [f"M{i}" for i in range(1, 13)]
        chart_data = []
        cum_cash = fin['initial_investment']
        for m in months:
            chart_data.append({"Month": m, "Stream": "Revenue", "Amount": fin['monthly_revenue']})
            chart_data.append({"Month": m, "Stream": "Operating Expenses", "Amount": fin['monthly_expenses']})
            
        df_fin_chart = pd.DataFrame(chart_data)
        
        st.markdown("#### 📈 Revenue vs. Operating Expense Run-Rate")
        rev_chart = alt.Chart(df_fin_chart).mark_bar().encode(
            x=alt.X('Month:N', title="Projection Month"),
            y=alt.Y('Amount:Q', title=f"Amount ({prof['currency_symbol']})"),
            color=alt.Color('Stream:N', scale=alt.Scale(domain=['Revenue', 'Operating Expenses'], range=['#3b82f6', '#f43f5e'])),
            xOffset='Stream:N',
            tooltip=['Month', 'Stream', 'Amount']
        ).properties(height=300)
        st.altair_chart(rev_chart, use_container_width=True)
        
        st.info(f"⏳ **Break-Even Analysis Horizon:** {fin['break_even_months']}")

    # ------------------------------------------------------
    # TAB 9: STRATEGIC ACTION PLAN
    # ------------------------------------------------------
    with report_tabs[8]:
        st.markdown("### 🎯 Phased Strategic Action Roadmap")
        st.caption("Prioritized operational milestones designed to validate assumptions and achieve unit profitability.")
        
        act = llm.get("action_plan", {})
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("#### ⚡ Next 7 Days (Sprint Validation)")
            for item in act.get("next_7_days", []):
                with st.container(border=True):
                    st.markdown(f"**{item.get('action', '')}**")
                    st.caption(f"**Why:** {item.get('reason', '')}")
                    st.caption(f"**Target Outcome:** {item.get('outcome', '')}")
                    
            st.markdown("#### 📅 Next 30 Days (Operational Setup)")
            for item in act.get("next_30_days", []):
                with st.container(border=True):
                    st.markdown(f"**{item.get('action', '')}**")
                    st.caption(f"**Why:** {item.get('reason', '')}")
                    st.caption(f"**Target Outcome:** {item.get('outcome', '')}")

        with col_a2:
            st.markdown("#### 🚀 Next 90 Days (Product-Market Expansion)")
            for item in act.get("next_90_days", []):
                with st.container(border=True):
                    st.markdown(f"**{item.get('action', '')}**")
                    st.caption(f"**Why:** {item.get('reason', '')}")
                    st.caption(f"**Target Outcome:** {item.get('outcome', '')}")
                    
            st.markdown("#### 🌐 Next 6 Months (Scale & Defensibility)")
            for item in act.get("next_6_months", []):
                with st.container(border=True):
                    st.markdown(f"**{item.get('action', '')}**")
                    st.caption(f"**Why:** {item.get('reason', '')}")
                    st.caption(f"**Target Outcome:** {item.get('outcome', '')}")

    # ------------------------------------------------------
    # TAB 10: NLP INSIGHTS
    # ------------------------------------------------------
    with report_tabs[9]:
        st.markdown("### 🧠 Natural Language Processing (NLP) Educational Architecture")
        st.caption("Overview of how Natural Language Processing and computational linguistics operate inside StartupSense AI.")
        
        st.markdown("""
        <div class="analysis-block">
        <b>NLP Pipeline Overview:</b><br>
        Raw Natural Language Input → Tokenization & Lemmatization → Stopword Filtering & Frequency Analysis → 
        Entity / Domain Classification → Semantic Intent Matching → Prompt Context Assembly → LLM Strategic Synthesis
        </div>
        """, unsafe_allow_html=True)
        
        nlp_c1, nlp_c2 = st.columns(2)
        with nlp_c1:
            st.markdown("##### 🔤 1. Keyword Extraction & Token Frequency")
            st.write("Top significant domain tokens extracted from your submission:")
            st.write(", ".join([f"`{k}`" for k in nlp.get("extracted_keywords", [])]))
            
            st.markdown("##### 🏷️ 2. Business Entity & Category Classification")
            st.write("Identified conceptual business entities:")
            for e in nlp.get("detected_key_concepts", []):
                st.markdown(f"- **{e}**")

        with nlp_c2:
            st.markdown("##### 🎯 3. Customer Pain Point Parsing")
            st.write("Linguistically extracted customer friction points:")
            for p in nlp.get("customer_pain_points", []):
                st.markdown(f"- {p}")
                
            st.markdown("##### 💡 4. Business Opportunity Synthesis")
            st.write("Opportunity vectors parsed from solution description:")
            for o in nlp.get("business_opportunities", []):
                st.markdown(f"- {o}")

        st.divider()
        st.markdown("##### 📘 Methodological Transparency")
        st.info(
            "Natural Language Processing in this platform combines deterministic regular expression tokenization, "
            "syntactic entity matching, and zero-shot generative classification via modern Large Language Models. "
            "This ensures both transparency of reasoning and depth of real-world business analysis."
        )

    # ------------------------------------------------------
    # TAB 11: ML HISTORICAL ANALYSIS
    # ------------------------------------------------------
    with report_tabs[10]:
        st.markdown("### 🌲 Historical Startup Pattern Analysis (Random Forest)")
        st.caption("Supervised machine learning baseline evaluating historical venture pattern similarity.")
        
        if ml and ml.get("success_probability") is not None:
            col_ml1, col_ml2 = st.columns(2)
            with col_ml1:
                st.metric("Historical Success Pattern Match", f"{ml['success_probability']:.1f}%")
                st.metric("Historical Failure Pattern Match", f"{ml['failure_probability']:.1f}%")
                st.progress(int(ml['success_probability']))
                
                if ml['prediction'] == 1:
                    st.success("🟢 Profile aligns with historical cohorts characterized by positive exit or milestone realization.")
                else:
                    st.warning("🟡 Profile aligns with historically higher-risk early venture cohorts.")
                    
            with col_ml2:
                st.markdown("##### Model Feature Attribution")
                st.write("Extracted model input vector (31 features):")
                st.json(ml['input_features'])
                
            st.divider()
            with st.expander("ℹ️ How the ML Model Works & Statistical Limitations", expanded=True):
                st.markdown("""
                - **Algorithm:** Scikit-Learn `RandomForestClassifier` (200 decision estimators, stratified split).
                - **Training Dataset:** Historical Crunchbase venture dataset tracking milestones, funding rounds, investment geography, and team relationships.
                - **Role in Platform:** Serves as a **secondary historical benchmark**.
                - **Critical Statistical Limitation:** Machine learning trained on past venture data reflects historical venture capital biases. A lower percentage indicates divergence from historical venture-backed patterns, **not** that an unconventional or bootstrapped business will fail.
                """)
        else:
            st.info(
                "🌲 **Historical ML Module Standby**: The Random Forest pattern model is active for the 'Existing Startup' mode when funding and operational history is provided. "
                "AI strategic analysis remains fully operational."
            )

    # ------------------------------------------------------
    # TAB 12: DOWNLOAD REPORT
    # ------------------------------------------------------
    with report_tabs[11]:
        st.markdown("### 📑 Download Executive Intelligence Report")
        st.caption("Export your structured startup validation report for offline review, investor discussions, or pitch deck preparation.")
        
        report_md = generate_markdown_report(prof, llm, sc, fin, ml)
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📥 Download Report (.md / Markdown)",
                data=report_md,
                file_name=f"{prof['startup_name'].lower().replace(' ', '_')}_validation_report.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col_d2:
            st.download_button(
                label="📄 Download Raw JSON Analysis Data",
                data=json.dumps(llm, indent=2),
                file_name=f"{prof['startup_name'].lower().replace(' ', '_')}_analysis.json",
                mime="application/json",
                use_container_width=True
            )

        st.markdown("#### 👁️ Report Preview")
        st.text_area("Markdown Report Content", report_md, height=350)