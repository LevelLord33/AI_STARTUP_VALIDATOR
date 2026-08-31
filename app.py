import streamlit as st
from huggingface_hub import InferenceClient
import pandas as pd
import joblib
import json
from datetime import datetime


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Startup Validator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# MODERN UI CSS
# ==========================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(43, 97, 255, 0.08), transparent 28%),
        radial-gradient(circle at top right, rgba(255, 65, 108, 0.06), transparent 25%);
}

.block-container {
    max-width: 1180px;
    padding-top: 1.6rem;
    padding-bottom: 4rem;
}

.main-title {
    font-size: 46px;
    font-weight: 850;
    text-align: center;
    margin-bottom: 5px;
}

.gradient-text {
    background: linear-gradient(
        90deg,
        #4f8cff,
        #8b5cf6,
        #ff5c7c
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #9ba4b4;
    margin-bottom: 22px;
}

.hero-card {
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.025);
    border-radius: 20px;
    padding: 22px 26px;
    margin-bottom: 25px;
}

.section-card {
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.025);
    border-radius: 18px;
    padding: 20px 24px;
    margin-top: 14px;
    margin-bottom: 18px;
}

.section-number {
    display: inline-block;
    background: rgba(79,140,255,0.13);
    border: 1px solid rgba(79,140,255,0.25);
    padding: 4px 9px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
    margin-right: 8px;
}

.section-heading {
    font-size: 22px;
    font-weight: 750;
    margin-bottom: 6px;
}

.section-description {
    color: #9ba4b4;
    font-size: 13px;
    margin-bottom: 12px;
}

.score-box {
    padding: 36px;
    border-radius: 22px;
    border: 1px solid rgba(79,140,255,0.25);
    background:
        linear-gradient(
            135deg,
            rgba(79,140,255,0.08),
            rgba(139,92,246,0.06)
        );
    text-align: center;
    margin-top: 10px;
    margin-bottom: 25px;
}

.score-label {
    font-size: 13px;
    letter-spacing: 2px;
    color: #9ba4b4;
    font-weight: 700;
}

.score-number {
    font-size: 68px;
    font-weight: 850;
    margin-top: 5px;
    margin-bottom: 4px;
}

.result-card {
    border: 1px solid rgba(255,255,255,0.07);
    background: rgba(255,255,255,0.025);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 14px;
}

.sidebar-badge {
    padding: 9px 11px;
    background: rgba(79,140,255,0.08);
    border-radius: 10px;
    margin-bottom: 8px;
}

div[data-testid="stTextInput"] input {
    min-height: 48px;
    border-radius: 11px;
}

div[data-testid="stTextArea"] textarea {
    border-radius: 11px;
}

div[data-testid="stSelectbox"] > div {
    border-radius: 11px;
}

div[data-testid="stNumberInput"] input {
    border-radius: 11px;
}

div.stButton > button {
    border-radius: 13px;
    min-height: 54px;
    font-size: 17px;
    font-weight: 700;
}

button[data-baseweb="tab"] {
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD ML MODEL
# ==========================================================

@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load("startup_model.pkl")
        features = joblib.load("startup_features.pkl")
        return model, features
    except Exception:
        return None, None


ml_model, ml_features = load_ml_model()


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("🚀 Startup Validator")

    st.caption(
        "AI + ML decision-support system for startup validation."
    )

    st.divider()

    st.markdown("### 🧠 AI Engine")

    st.markdown(
        """
        <div class="sidebar-badge">
        <b>Hugging Face LLM</b><br>
        <span style="font-size:12px;color:#9ba4b4;">
        Startup reasoning and explanation
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🌲 ML Engine")

    if ml_model is not None:

        st.success("Random Forest loaded")

    else:

        st.warning("Random Forest unavailable")

    st.caption(
        "Used for historical analysis of existing startups."
    )

    st.divider()

    st.markdown("### 🛠 Tech Stack")

    st.write("🐍 Python")
    st.write("⚡ Streamlit")
    st.write("🤗 Hugging Face")
    st.write("🌲 Scikit-learn")
    st.write("📊 Pandas")


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <div class="main-title">
        <span class="gradient-text">
            AI-Powered Startup Validator
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Analyze your idea, understand business risks,
        discover opportunities and make better startup decisions.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-card">
        <b>How it works:</b>
        Enter your startup details → AI evaluates the business →
        existing startups can also use historical ML →
        receive a structured validation report.
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# MODE
# ==========================================================

st.markdown(
    """
    <div class="section-heading">
        🎯 Choose Analysis Mode
    </div>
    """,
    unsafe_allow_html=True
)

analysis_mode = st.radio(
    "Choose mode",
    [
        "💡 New Startup Idea",
        "📊 Existing Startup"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


if analysis_mode == "💡 New Startup Idea":

    st.info(
        "Best for founders who are still validating an idea, "
        "prototype or early MVP."
    )

else:

    st.info(
        "Best for startups that already have operational "
        "or investment history."
    )


# ==========================================================
# SECTION 1
# ==========================================================

st.divider()

st.markdown(
    """
    <span class="section-number">01</span>
    <span class="section-heading">Startup Identity</span>
    <div class="section-description">
        Tell us what the startup is and who it is for.
    </div>
    """,
    unsafe_allow_html=True
)


startup_name = st.text_input(
    "Startup Name",
    placeholder="Example: SkillBridge AI"
)


target_customer = st.text_input(
    "Target Customer",
    placeholder="Example: College students and fresh graduates"
)


# ==========================================================
# SECTION 2
# ==========================================================

st.markdown(
    """
    <span class="section-number">02</span>
    <span class="section-heading">Problem & Solution</span>
    <div class="section-description">
        Explain the customer problem and how your startup solves it.
    </div>
    """,
    unsafe_allow_html=True
)


startup_idea = st.text_area(
    "Startup Idea",
    placeholder=(
        "Describe your product or service and how it works..."
    ),
    height=130
)


problem = st.text_area(
    "Problem Being Solved",
    placeholder=(
        "What difficulty or pain point is the customer facing?"
    ),
    height=120
)


# ==========================================================
# SECTION 3
# ==========================================================

st.markdown(
    """
    <span class="section-number">03</span>
    <span class="section-heading">Business Concept</span>
    <div class="section-description">
        Provide simple business information for deeper analysis.
    </div>
    """,
    unsafe_allow_html=True
)


c1, c2 = st.columns(2)


with c1:

    industry = st.selectbox(
        "Industry",
        [
            "Education / EdTech",
            "Healthcare",
            "Agriculture / AgriTech",
            "Finance / FinTech",
            "Software / SaaS",
            "E-Commerce",
            "Artificial Intelligence",
            "Cybersecurity",
            "Transportation",
            "Food / FoodTech",
            "Real Estate",
            "Entertainment",
            "Other"
        ]
    )


    startup_stage = st.selectbox(
        "Current Startup Stage",
        [
            "Idea",
            "Prototype",
            "MVP",
            "Early Customers",
            "Operating Business"
        ]
    )


    target_location = st.text_input(
        "Target Market / Location",
        placeholder="Example: India"
    )


with c2:

    revenue_model = st.selectbox(
        "Revenue Model",
        [
            "Subscription",
            "Freemium",
            "Commission",
            "Direct Sales",
            "Advertising",
            "Marketplace Fee",
            "Licensing",
            "B2B Contract",
            "Not Decided Yet",
            "Other"
        ]
    )


    expected_price = st.text_input(
        "Expected Pricing",
        placeholder="Example: ₹199/month"
    )


    competition_level = st.selectbox(
        "Competition Level",
        [
            "Not Sure",
            "Low",
            "Medium",
            "High"
        ]
    )


unique_advantage = st.text_area(
    "Unique Advantage",
    placeholder=(
        "Why should customers choose your startup over alternatives?"
    ),
    height=100
)


# ==========================================================
# EXISTING STARTUP SECTION
# ==========================================================

use_ml = False


if analysis_mode == "📊 Existing Startup":

    st.divider()

    st.markdown(
        """
        <span class="section-number">04</span>
        <span class="section-heading">
            Existing Startup History
        </span>
        <div class="section-description">
            Only simple information is required.
            Technical ML features are generated internally.
        </div>
        """,
        unsafe_allow_html=True
    )


    current_year = datetime.now().year


    e1, e2 = st.columns(2)


    with e1:

        founded_year = st.number_input(
            "Founded Year",
            min_value=1990,
            max_value=current_year,
            value=max(2020, current_year - 3),
            step=1
        )


        total_funding = st.number_input(
            "Total Funding Received (USD)",
            min_value=0.0,
            value=100000.0,
            step=10000.0
        )


        funding_rounds = st.number_input(
            "Number of Funding Rounds",
            min_value=0,
            value=1,
            step=1
        )


    with e2:

        first_funding_year = st.number_input(
            "First Funding Year",
            min_value=1990,
            max_value=current_year,
            value=max(2020, current_year - 2),
            step=1
        )


        latest_funding_year = st.number_input(
            "Latest Funding Year",
            min_value=1990,
            max_value=current_year,
            value=current_year,
            step=1
        )


        professional_connections = st.number_input(
            "Approximate Professional Connections",
            min_value=0,
            value=5,
            step=1,
            help=(
                "Investors, advisors, founders or major "
                "professional connections associated with the startup."
            )
        )


    st.markdown("### 💰 Investment History")


    i1, i2, i3 = st.columns(3)


    with i1:
        has_vc = st.checkbox("VC Funding")
        has_angel = st.checkbox("Angel Funding")


    with i2:
        has_round_a = st.checkbox("Series A")
        has_round_b = st.checkbox("Series B")


    with i3:
        has_round_c = st.checkbox("Series C")
        has_round_d = st.checkbox("Series D")


    use_ml = True


# ==========================================================
# ANALYZE BUTTON
# ==========================================================

st.divider()

analyze = st.button(
    "✨ Analyze & Validate Startup",
    type="primary",
    use_container_width=True
)


# ==========================================================
# EXECUTION
# ==========================================================

if analyze:

    if (
        not startup_name
        or not startup_idea
        or not problem
        or not target_customer
        or not target_location
        or not unique_advantage
    ):

        st.warning(
            "⚠️ Please complete all required startup information."
        )

        st.stop()


    success_probability = None
    failure_probability = None
    ml_prediction = None


    # ======================================================
    # RANDOM FOREST
    # ======================================================

    if (
        use_ml
        and ml_model is not None
        and ml_features is not None
    ):

        startup_age = max(
            0,
            current_year - founded_year
        )


        age_first_funding_year = max(
            0,
            first_funding_year - founded_year
        )


        age_last_funding_year = max(
            0,
            latest_funding_year - founded_year
        )


        age_first_milestone_year = min(
            startup_age,
            1.0
        )


        age_last_milestone_year = startup_age


        milestones = max(
            1,
            min(
                5,
                int(startup_age)
            )
        )


        avg_participants = (
            2.0
            if funding_rounds > 0
            else 0.0
        )


        if industry in [
            "Software / SaaS",
            "Artificial Intelligence",
            "Cybersecurity"
        ]:

            ml_industry = "Software"

        elif industry == "E-Commerce":

            ml_industry = "E-Commerce"

        else:

            ml_industry = "Other"


        target_lower = (
            target_location
            .lower()
            .strip()
        )


        is_ca = int(
            "california" in target_lower
        )

        is_ny = int(
            "new york" in target_lower
        )

        is_ma = int(
            "massachusetts" in target_lower
        )

        is_tx = int(
            "texas" in target_lower
        )


        is_otherstate = int(
            not any(
                [
                    is_ca,
                    is_ny,
                    is_ma,
                    is_tx
                ]
            )
        )


        input_data = {

            "age_first_funding_year":
                age_first_funding_year,

            "age_last_funding_year":
                age_last_funding_year,

            "age_first_milestone_year":
                age_first_milestone_year,

            "age_last_milestone_year":
                age_last_milestone_year,

            "relationships":
                professional_connections,

            "funding_rounds":
                funding_rounds,

            "funding_total_usd":
                total_funding,

            "milestones":
                milestones,

            "is_CA":
                is_ca,

            "is_NY":
                is_ny,

            "is_MA":
                is_ma,

            "is_TX":
                is_tx,

            "is_otherstate":
                is_otherstate,

            "is_software":
                int(
                    ml_industry == "Software"
                ),

            "is_web": 0,

            "is_mobile": 0,

            "is_enterprise": 0,

            "is_advertising": 0,

            "is_gamesvideo": 0,

            "is_ecommerce":
                int(
                    ml_industry == "E-Commerce"
                ),

            "is_biotech": 0,

            "is_consulting": 0,

            "is_othercategory":
                int(
                    ml_industry == "Other"
                ),

            "has_VC":
                int(has_vc),

            "has_angel":
                int(has_angel),

            "has_roundA":
                int(has_round_a),

            "has_roundB":
                int(has_round_b),

            "has_roundC":
                int(has_round_c),

            "has_roundD":
                int(has_round_d),

            "avg_participants":
                avg_participants,

            "is_top500":
                0
        }


        ml_input = pd.DataFrame(
            [
                [
                    input_data[f]
                    for f in ml_features
                ]
            ],
            columns=ml_features
        )


        ml_prediction = (
            ml_model.predict(
                ml_input
            )[0]
        )


        probabilities = (
            ml_model.predict_proba(
                ml_input
            )[0]
        )


        failure_probability = (
            probabilities[0] * 100
        )


        success_probability = (
            probabilities[1] * 100
        )


    # ======================================================
    # LLM CONTEXT
    # ======================================================

    ml_context = ""


    if success_probability is not None:

        ml_context = f"""
A Random Forest model trained on historical startup data
produced a historical success-pattern match of
{success_probability:.1f}%.

Use this only as supporting historical evidence.
Do not directly copy this percentage into your AI score.
"""


    # ======================================================
    # PROMPT
    # ======================================================

    prompt = f"""
You are a professional startup analyst.

Analyze this startup critically and realistically.

Startup Name:
{startup_name}

Startup Idea:
{startup_idea}

Problem:
{problem}

Target Customer:
{target_customer}

Industry:
{industry}

Current Stage:
{startup_stage}

Target Market:
{target_location}

Revenue Model:
{revenue_model}

Expected Pricing:
{expected_price}

Competition Level:
{competition_level}

Unique Advantage:
{unique_advantage}

{ml_context}

Evaluate these factors from 0 to 100:

1. Problem Strength
2. Customer Need
3. Solution Relevance
4. Market Opportunity
5. Competitive Advantage
6. Feasibility
7. Revenue Potential
8. Risk Management

Return ONLY one valid JSON object.

Required keys:

problem_strength
customer_need
solution_relevance
market_opportunity_score
competitive_advantage
feasibility
revenue_potential
risk_management
problem_analysis
customer_analysis
solution_analysis
market_analysis
competition_analysis
revenue_analysis
strengths
weaknesses
risks
improvements
competitors
revenue_suggestions
next_steps
final_opinion

competitors must contain exactly 3 competitor objects.

Each competitor must contain:

name
advantage
your_opportunity
"""


    # ======================================================
    # HUGGING FACE
    # ======================================================

    try:

        client = InferenceClient(
            provider="auto",
            api_key=st.secrets["HF_TOKEN"]
        )


        models_to_try = [
            "Qwen/Qwen2.5-7B-Instruct-1M",
            "Qwen/Qwen3-4B-Thinking-2507"
        ]


        response = None
        used_model = None
        last_error = None


        with st.spinner(
            "🧠 AI is evaluating your startup..."
        ):

            for model_name in models_to_try:

                try:

                    response = (
                        client
                        .chat
                        .completions
                        .create(
                            model=model_name,

                            messages=[
                                {
                                    "role":
                                        "system",

                                    "content":
                                        "You are a professional "
                                        "startup analyst. "
                                        "Return only valid JSON."
                                },

                                {
                                    "role":
                                        "user",

                                    "content":
                                        prompt
                                }
                            ],

                            response_format={
                                "type":
                                    "json_object"
                            },

                            max_tokens=2500,

                            temperature=0.2
                        )
                    )


                    used_model = model_name

                    break


                except Exception as e:

                    last_error = e


        if response is None:

            st.error(
                "The AI service is currently unavailable."
            )

            st.code(
                str(last_error)
            )

            st.stop()


        result = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )


        try:

            data = json.loads(
                result
            )

        except json.JSONDecodeError:

            st.error(
                "The AI returned an invalid response."
            )

            with st.expander(
                "Show raw AI response"
            ):

                st.code(
                    result
                )

            st.stop()


        # ==================================================
        # SCORES
        # ==================================================

        problem_score = int(
            data["problem_strength"]
        )

        customer_score = int(
            data["customer_need"]
        )

        solution_score = int(
            data["solution_relevance"]
        )

        market_score = int(
            data["market_opportunity_score"]
        )

        competition_score = int(
            data["competitive_advantage"]
        )

        feasibility_score = int(
            data["feasibility"]
        )

        revenue_score = int(
            data["revenue_potential"]
        )

        risk_score = int(
            data["risk_management"]
        )


        scores = [
            problem_score,
            customer_score,
            solution_score,
            market_score,
            competition_score,
            feasibility_score,
            revenue_score,
            risk_score
        ]


        scores = [
            max(
                0,
                min(
                    100,
                    value
                )
            )
            for value in scores
        ]


        (
            problem_score,
            customer_score,
            solution_score,
            market_score,
            competition_score,
            feasibility_score,
            revenue_score,
            risk_score
        ) = scores


        idea_score = round(

            problem_score * 0.15
            + customer_score * 0.15
            + solution_score * 0.15
            + market_score * 0.15
            + competition_score * 0.10
            + feasibility_score * 0.10
            + revenue_score * 0.10
            + risk_score * 0.10
        )


        if (
            analysis_mode
            == "📊 Existing Startup"
            and success_probability
            is not None
        ):

            final_score = round(
                idea_score * 0.70
                + success_probability
                * 0.30
            )

        else:

            final_score = idea_score


        # ==================================================
        # RESULTS HEADER
        # ==================================================

        st.divider()

        st.markdown(
            f"""
            <div class="section-heading">
                📊 {startup_name} Validation Report
            </div>
            """,
            unsafe_allow_html=True
        )


        st.caption(
            f"AI model used: {used_model}"
        )


        # ==================================================
        # TABS
        # ==================================================

        if (
            analysis_mode
            == "📊 Existing Startup"
        ):

            tabs = st.tabs(
                [
                    "🏆 Overview",
                    "💡 Analysis",
                    "🌲 ML",
                    "👥 Competition",
                    "📈 Market",
                    "🎯 Action Plan"
                ]
            )

        else:

            tabs = st.tabs(
                [
                    "🏆 Overview",
                    "💡 Analysis",
                    "👥 Competition",
                    "📈 Market",
                    "🎯 Action Plan"
                ]
            )


        # ==================================================
        # OVERVIEW
        # ==================================================

        with tabs[0]:

            score_html = f"""
<div class="score-box">
<div class="score-label">
STARTUP VALIDATION SCORE
</div>

<div class="score-number">
{final_score}/100
</div>
</div>
"""

            st.markdown(
                score_html,
                unsafe_allow_html=True
            )


            st.progress(
                final_score
            )


            if final_score >= 80:

                st.success(
                    "🟢 Strong Potential"
                )

            elif final_score >= 65:

                st.success(
                    "🟢 Promising Startup"
                )

            elif final_score >= 50:

                st.warning(
                    "🟡 Needs Further Validation"
                )

            else:

                st.error(
                    "🔴 Needs Major Improvement"
                )


            if success_probability is not None:

                metric1, metric2 = (
                    st.columns(2)
                )


                with metric1:

                    st.metric(
                        "AI Idea Score",
                        f"{idea_score}/100"
                    )


                with metric2:

                    st.metric(
                        "Historical ML Match",
                        f"{success_probability:.1f}%"
                    )


            st.markdown(
                "### 📊 Validation Factors"
            )


            score_df = pd.DataFrame(
                {
                    "Factor": [
                        "Problem",
                        "Customer Need",
                        "Solution",
                        "Market",
                        "Competition",
                        "Feasibility",
                        "Revenue",
                        "Risk"
                    ],

                    "Score": [
                        problem_score,
                        customer_score,
                        solution_score,
                        market_score,
                        competition_score,
                        feasibility_score,
                        revenue_score,
                        risk_score
                    ]
                }
            )


            st.bar_chart(
                score_df.set_index(
                    "Factor"
                )
            )


            st.markdown(
                "### 📌 Final Opinion"
            )

            st.write(
                data["final_opinion"]
            )


        # ==================================================
        # ANALYSIS
        # ==================================================

        with tabs[1]:

            st.markdown(
                "## 💡 Startup Analysis"
            )


            st.markdown(
                "### 🔥 Problem"
            )

            st.write(
                data["problem_analysis"]
            )


            st.markdown(
                "### 👥 Customer"
            )

            st.write(
                data["customer_analysis"]
            )


            st.markdown(
                "### 💡 Solution"
            )

            st.write(
                data["solution_analysis"]
            )


            left, right = st.columns(2)


            with left:

                st.markdown(
                    "### ✅ Strengths"
                )

                for item in data[
                    "strengths"
                ]:

                    st.success(
                        item
                    )


            with right:

                st.markdown(
                    "### ⚠️ Weaknesses"
                )

                for item in data[
                    "weaknesses"
                ]:

                    st.warning(
                        item
                    )


            st.markdown(
                "### 🚨 Risks"
            )

            for item in data["risks"]:

                st.write(
                    "•",
                    item
                )


            st.markdown(
                "### 🛠 Improvements"
            )

            for item in data[
                "improvements"
            ]:

                st.write(
                    "•",
                    item
                )


        # ==================================================
        # TAB MAPPING
        # ==================================================

        if (
            analysis_mode
            == "📊 Existing Startup"
        ):

            ml_tab = tabs[2]
            competition_tab = tabs[3]
            market_tab = tabs[4]
            action_tab = tabs[5]

        else:

            competition_tab = tabs[2]
            market_tab = tabs[3]
            action_tab = tabs[4]


        # ==================================================
        # ML TAB
        # ==================================================

        if (
            analysis_mode
            == "📊 Existing Startup"
        ):

            with ml_tab:

                st.markdown(
                    "## 🌲 Historical ML Prediction"
                )


                if success_probability is not None:

                    m1, m2 = (
                        st.columns(2)
                    )


                    with m1:

                        st.metric(
                            "Success Pattern Match",
                            f"{success_probability:.1f}%"
                        )


                    with m2:

                        st.metric(
                            "Failure Pattern Match",
                            f"{failure_probability:.1f}%"
                        )


                    st.progress(
                        int(
                            success_probability
                        )
                    )


                    if ml_prediction == 1:

                        st.success(
                            "This startup is closer to "
                            "historically successful startup patterns."
                        )

                    else:

                        st.warning(
                            "This startup is closer to "
                            "historically higher-risk startup patterns."
                        )


                    st.info(
                        "This is historical ML evidence, "
                        "not a guarantee of future success."
                    )


        # ==================================================
        # COMPETITION
        # ==================================================

        with competition_tab:

            st.markdown(
                "## 👥 Competitor Analysis"
            )


            st.write(
                data[
                    "competition_analysis"
                ]
            )


            for number, competitor in enumerate(
                data[
                    "competitors"
                ],
                start=1
            ):

                with st.expander(
                    f"🏢 {number}. "
                    f"{competitor['name']}",
                    expanded=True
                ):

                    st.markdown(
                        "**Competitor Advantage**"
                    )

                    st.write(
                        competitor[
                            "advantage"
                        ]
                    )


                    st.markdown(
                        "**Your Opportunity**"
                    )

                    st.write(
                        competitor[
                            "your_opportunity"
                        ]
                    )


        # ==================================================
        # MARKET
        # ==================================================

        with market_tab:

            st.markdown(
                "## 📈 Market & Revenue"
            )


            st.markdown(
                "### 🌍 Market Opportunity"
            )

            st.write(
                data[
                    "market_analysis"
                ]
            )


            st.markdown(
                "### 💰 Revenue Analysis"
            )

            st.write(
                data[
                    "revenue_analysis"
                ]
            )


            st.markdown(
                "### 💡 Revenue Suggestions"
            )

            for item in data[
                "revenue_suggestions"
            ]:

                st.write(
                    "•",
                    item
                )


        # ==================================================
        # ACTION PLAN
        # ==================================================

        with action_tab:

            st.markdown(
                "## 🎯 Recommended Action Plan"
            )


            for index, step in enumerate(
                data[
                    "next_steps"
                ],
                start=1
            ):

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"### Step {index}"
                    )

                    st.write(
                        step
                    )


    except Exception as e:

        st.error(
            "Something went wrong during startup analysis."
        )

        st.code(
            str(e)
        )