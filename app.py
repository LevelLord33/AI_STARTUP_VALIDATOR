import streamlit as st
from huggingface_hub import InferenceClient
import pandas as pd
import joblib
import json
from datetime import datetime


st.set_page_config(
    page_title="AI Startup Validator",
    page_icon="🚀",
    layout="wide"
)


st.markdown("""
<style>

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.main-title {
    font-size: 46px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #9ca3af;
    margin-bottom: 30px;
}

.score-box {
    padding: 35px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
    text-align: center;
    margin-bottom: 20px;
}

.score-label {
    font-size: 14px;
    color: #9ca3af;
    letter-spacing: 2px;
}

.score-number {
    font-size: 60px;
    font-weight: 800;
}

div.stButton > button {
    border-radius: 12px;
    min-height: 52px;
    font-size: 17px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load("startup_model.pkl")
        features = joblib.load("startup_features.pkl")
        return model, features
    except Exception:
        return None, None


ml_model, ml_features = load_ml_model()


with st.sidebar:

    st.title("🚀 Startup Validator")

    st.write(
        "AI-powered startup idea validation "
        "and historical startup analysis."
    )

    st.divider()

    st.subheader("🧠 AI Analysis")
    st.write("Qwen / Hugging Face")

    st.caption(
        "Problem, market, competition, "
        "risks and recommendations."
    )

    st.divider()

    st.subheader("🌲 ML Prediction")

    if ml_model is not None:
        st.success("Random Forest Ready")
    else:
        st.warning("Random Forest unavailable")

    st.divider()

    st.caption("AI Platform")
    st.write("Hugging Face")

    st.caption("ML Library")
    st.write("Scikit-learn")

    st.caption("Frontend")
    st.write("Streamlit")


st.markdown(
    '<div class="main-title">🚀 AI Startup Validator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Validate your startup idea, understand risks '
    'and discover opportunities using AI.'
    '</div>',
    unsafe_allow_html=True
)


st.header("🎯 Choose Analysis Mode")

analysis_mode = st.radio(
    "Analysis Mode",
    [
        "💡 New Startup Idea",
        "📊 Existing Startup"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


if analysis_mode == "💡 New Startup Idea":

    st.info(
        "Use this mode for an idea, prototype "
        "or early-stage startup."
    )

else:

    st.info(
        "Use this mode for an operating startup "
        "with funding or investment history."
    )


st.divider()


# ==========================================================
# STARTUP INFORMATION
# ==========================================================

st.header("📝 Startup Information")


startup_name = st.text_input(
    "Startup Name",
    placeholder="Example: SkillBridge AI"
)


startup_idea = st.text_area(
    "Startup Idea",
    placeholder="Explain what your startup does...",
    height=120
)


problem = st.text_area(
    "Problem Being Solved",
    placeholder="Explain the main customer problem...",
    height=120
)


target_customer = st.text_input(
    "Target Customer",
    placeholder="Example: College students"
)


# ==========================================================
# BUSINESS CONCEPT
# ==========================================================

st.divider()

st.header("💡 Business Concept")


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
    "What is your main advantage?",
    placeholder="Why would customers choose your startup?",
    height=100
)


# ==========================================================
# EXISTING STARTUP INPUTS
# ==========================================================

use_ml = False


if analysis_mode == "📊 Existing Startup":

    st.divider()

    st.header("📊 Existing Startup Details")

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
            step=1
        )


    st.subheader("💰 Investment History")


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


st.divider()


analyze = st.button(
    "🚀 Analyze Startup",
    type="primary",
    use_container_width=True
)


if analyze:

    if (
        not startup_name
        or not startup_idea
        or not problem
        or not target_customer
        or not target_location
        or not unique_advantage
    ):
        st.warning("Please complete all required fields.")
        st.stop()


    # ======================================================
    # RANDOM FOREST
    # ======================================================

    success_probability = None
    failure_probability = None
    ml_prediction = None


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
            min(5, int(startup_age))
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


        target_lower = target_location.lower().strip()

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
                int(ml_industry == "Software"),

            "is_web":
                0,

            "is_mobile":
                0,

            "is_enterprise":
                0,

            "is_advertising":
                0,

            "is_gamesvideo":
                0,

            "is_ecommerce":
                int(ml_industry == "E-Commerce"),

            "is_biotech":
                0,

            "is_consulting":
                0,

            "is_othercategory":
                int(ml_industry == "Other"),

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


        ml_prediction = ml_model.predict(
            ml_input
        )[0]


        probabilities = ml_model.predict_proba(
            ml_input
        )[0]


        failure_probability = (
            probabilities[0] * 100
        )

        success_probability = (
            probabilities[1] * 100
        )


    # ======================================================
    # PROMPT
    # ======================================================

    ml_context = ""


    if success_probability is not None:

        ml_context = f"""
Historical ML pattern match:
{success_probability:.1f}% similarity to historically
successful startup patterns.

Use this only as supporting evidence.
"""


    prompt = f"""
You are a professional startup analyst.

Analyze this startup realistically.

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

Be critical and realistic.

Return one valid JSON object only.

Required JSON keys:

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

competitors must be a list of 3 objects.

Each competitor object must contain:
name
advantage
your_opportunity
"""


    # ======================================================
    # HUGGING FACE JSON MODE
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
            "🤖 AI is analyzing your startup..."
        ):

            for model_name in models_to_try:

                try:

                    response = client.chat.completions.create(
                        model=model_name,

                        messages=[
                            {
                                "role": "system",
                                "content":
                                    "You are a startup analyst. "
                                    "Return ONLY a valid JSON object. "
                                    "Do not include markdown. "
                                    "Do not include text outside JSON."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],

                        response_format={
                            "type": "json_object"
                        },

                        max_tokens=2500,

                        temperature=0.2
                    )

                    used_model = model_name

                    break

                except Exception as e:

                    last_error = e


        if response is None:

            st.error(
                "The AI service could not complete the request."
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
                "The AI response was still not valid JSON."
            )

            with st.expander(
                "🔍 Show raw AI response"
            ):

                st.code(
                    result
                )

            st.stop()


        required_keys = [
            "problem_strength",
            "customer_need",
            "solution_relevance",
            "market_opportunity_score",
            "competitive_advantage",
            "feasibility",
            "revenue_potential",
            "risk_management",
            "problem_analysis",
            "customer_analysis",
            "solution_analysis",
            "market_analysis",
            "competition_analysis",
            "revenue_analysis",
            "strengths",
            "weaknesses",
            "risks",
            "improvements",
            "competitors",
            "revenue_suggestions",
            "next_steps",
            "final_opinion"
        ]


        missing_keys = [
            key
            for key in required_keys
            if key not in data
        ]


        if missing_keys:

            st.error(
                "The AI response is missing required fields."
            )

            st.write(
                "Missing:",
                missing_keys
            )

            with st.expander(
                "Show AI response"
            ):

                st.json(
                    data
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
                    score
                )
            )
            for score in scores
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
            analysis_mode == "📊 Existing Startup"
            and success_probability is not None
        ):

            final_score = round(
                idea_score * 0.70
                + success_probability * 0.30
            )

        else:

            final_score = idea_score


        # ==================================================
        # REPORT
        # ==================================================

        st.divider()

        st.header(
            f"📊 {startup_name} Validation Report"
        )

        st.caption(
            f"AI model used: {used_model}"
        )


        if analysis_mode == "📊 Existing Startup":

            tabs = st.tabs(
                [
                    "🏆 Overview",
                    "💡 Idea Analysis",
                    "🌲 ML Prediction",
                    "👥 Competition",
                    "📈 Market",
                    "🎯 Action Plan"
                ]
            )

        else:

            tabs = st.tabs(
                [
                    "🏆 Overview",
                    "💡 Idea Analysis",
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
                    "🟢 Promising"
                )

            elif final_score >= 50:

                st.warning(
                    "🟡 Needs Validation"
                )

            else:

                st.error(
                    "🔴 Needs Major Improvement"
                )


            if success_probability is not None:

                m1, m2 = st.columns(2)

                with m1:

                    st.metric(
                        "AI Idea Score",
                        f"{idea_score}/100"
                    )

                with m2:

                    st.metric(
                        "Historical ML Match",
                        f"{success_probability:.1f}%"
                    )


            st.subheader(
                "📊 Validation Factors"
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
                        "Risk Management"
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


            st.subheader(
                "📌 Final Opinion"
            )

            st.write(
                data["final_opinion"]
            )


        # ==================================================
        # IDEA ANALYSIS
        # ==================================================

        with tabs[1]:

            st.header(
                "💡 Startup Idea Analysis"
            )


            st.subheader(
                "🔥 Problem"
            )

            st.write(
                data["problem_analysis"]
            )


            st.subheader(
                "👥 Customer"
            )

            st.write(
                data["customer_analysis"]
            )


            st.subheader(
                "💡 Solution"
            )

            st.write(
                data["solution_analysis"]
            )


            left, right = st.columns(2)


            with left:

                st.subheader(
                    "✅ Strengths"
                )

                for item in data["strengths"]:

                    st.success(
                        item
                    )


            with right:

                st.subheader(
                    "⚠️ Weaknesses"
                )

                for item in data["weaknesses"]:

                    st.warning(
                        item
                    )


            st.subheader(
                "🚨 Risks"
            )

            for item in data["risks"]:

                st.write(
                    "•",
                    item
                )


            st.subheader(
                "🛠 Improvements"
            )

            for item in data["improvements"]:

                st.write(
                    "•",
                    item
                )


        # ==================================================
        # TAB POSITIONS
        # ==================================================

        if analysis_mode == "📊 Existing Startup":

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

        if analysis_mode == "📊 Existing Startup":

            with ml_tab:

                st.header(
                    "🌲 Historical ML Prediction"
                )


                if success_probability is not None:

                    x1, x2 = st.columns(2)


                    with x1:

                        st.metric(
                            "Success Pattern Match",
                            f"{success_probability:.1f}%"
                        )


                    with x2:

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
                            "Closer to historically successful patterns."
                        )

                    else:

                        st.warning(
                            "Closer to historically higher-risk patterns."
                        )


                    st.info(
                        "This is based on historical data "
                        "and is not a guarantee of future success."
                    )


        # ==================================================
        # COMPETITION
        # ==================================================

        with competition_tab:

            st.header(
                "👥 Competitor Analysis"
            )


            st.write(
                data["competition_analysis"]
            )


            for number, competitor in enumerate(
                data["competitors"],
                start=1
            ):

                with st.expander(
                    f"🏢 {number}. "
                    f"{competitor['name']}",
                    expanded=True
                ):

                    st.markdown(
                        "**Their Advantage**"
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

            st.header(
                "📈 Market & Revenue"
            )


            st.subheader(
                "🌍 Market Opportunity"
            )

            st.write(
                data["market_analysis"]
            )


            st.subheader(
                "💰 Revenue Analysis"
            )

            st.write(
                data["revenue_analysis"]
            )


            st.subheader(
                "💡 Revenue Suggestions"
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

            st.header(
                "🎯 Recommended Action Plan"
            )


            for index, step in enumerate(
                data["next_steps"],
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