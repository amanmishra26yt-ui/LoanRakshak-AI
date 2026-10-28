from pathlib import Path
import sqlite3
import html

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "business_loan_data.csv"
MODEL_PATH = BASE_DIR / "models" / "loan_default_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.pkl"
DB_PATH = BASE_DIR / "loanrakshak.db"

st.set_page_config(
    page_title="LoanRakshak AI | Credit Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROFESSIONAL DARK FINTECH DESIGN
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--bg:#080d18;--panel:#101827;--panel2:#141f32;--line:#26364d;--text:#f3f7ff;--muted:#9aaac2;--cyan:#65e5ef;--violet:#9d8cff;--green:#52e0b2;--red:#ff718c;--amber:#ffc66e}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--text)}
.stApp{background:radial-gradient(ellipse at 90% 0%,rgba(81,91,210,.16),transparent 32%),radial-gradient(ellipse at 0% 45%,rgba(36,180,195,.06),transparent 35%),var(--bg)}
header[data-testid="stHeader"]{background:rgba(8,13,24,.86)}
.block-container{max-width:1500px;padding-top:1.25rem;padding-bottom:3rem}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em;color:var(--text)}
h2{font-size:1.55rem!important;margin-top:.8rem!important} h3{font-size:1.12rem!important}
p,li{color:#c4cee0}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0e1727 0%,#0a111e 100%);border-right:1px solid #1e2b40}
[data-testid="stSidebar"]>div:first-child{padding-top:1.2rem}
[data-testid="stSidebar"] h2{font-size:1.25rem!important;letter-spacing:-.04em}
[data-testid="stSidebar"] [data-testid="stRadio"] label{font-weight:600}
[data-testid="stSidebar"] [data-testid="stRadio"] > div{gap:.22rem}
[data-testid="stSidebar"] [data-testid="stRadio"] label p{font-size:.94rem}
[data-testid="stSidebar"] hr{border-color:#27354a;margin:1.1rem 0}
.hero{position:relative;overflow:hidden;display:flex;align-items:center;justify-content:space-between;gap:28px;padding:28px 32px;border:1px solid #2a3a55;border-radius:24px;background:linear-gradient(115deg,rgba(24,42,71,.98),rgba(17,25,47,.97) 60%,rgba(35,32,75,.96));margin:0 0 22px;box-shadow:0 18px 55px rgba(0,0,0,.17)}
.hero:after{content:"";position:absolute;width:260px;height:260px;right:6%;top:-135px;border-radius:50%;background:rgba(101,229,239,.13);filter:blur(4px);pointer-events:none}
.hero-copy{position:relative;z-index:1;max-width:790px}.eyebrow{color:var(--cyan);font-size:10px;font-weight:800;letter-spacing:3px;text-transform:uppercase;margin-bottom:12px}
.hero h1{font-size:clamp(27px,3vw,40px);line-height:1.1;margin:0 0 12px!important;font-weight:700}
.hero p{font-size:14px;line-height:1.7;color:#b9c7dc;margin:0;max-width:740px}
.hero-visual{position:relative;z-index:1;min-width:160px;text-align:center;border:1px solid rgba(130,161,209,.25);border-radius:18px;padding:17px 20px;background:rgba(8,14,29,.38)}
.hero-visual .visual-icon{font-size:28px;line-height:1.2}.hero-visual .visual-label{font-size:10px;letter-spacing:1.7px;color:#aebbd1;font-weight:700;margin-top:8px}.hero-visual .visual-value{font-family:'Space Grotesk';font-size:18px;font-weight:700;color:#fff;margin-top:4px}
.metric-card{position:relative;overflow:hidden;background:linear-gradient(145deg,rgba(22,33,52,.98),rgba(13,21,35,.98));border:1px solid #293951;border-radius:17px;padding:18px 19px;min-height:132px;margin-bottom:12px;box-shadow:0 10px 26px rgba(0,0,0,.11);transition:transform .18s,border-color .18s}
.metric-card:before{content:"";position:absolute;left:0;top:0;width:3px;height:100%;background:linear-gradient(180deg,var(--cyan),var(--violet))}
.metric-label{color:#a9b8cf;font-size:10px;font-weight:800;letter-spacing:1.2px;text-transform:uppercase}
.metric-value{color:#f8fbff;font-size:clamp(24px,2.2vw,32px);font-weight:700;font-family:'Space Grotesk',sans-serif;letter-spacing:-1.2px;margin-top:14px;line-height:1.15}
.metric-help{color:#8293ad;font-size:11px;margin-top:8px}
.panel{background:linear-gradient(145deg,#121d2e,#0f1725);border:1px solid #27364c;border-radius:18px;padding:20px;margin-bottom:18px}
.info-box{background:rgba(101,229,239,.07);border:1px solid rgba(101,229,239,.23);border-left:3px solid var(--cyan);border-radius:12px;padding:14px;color:#d7e5f6;margin:10px 0}
.risk-high{background:rgba(255,87,115,.1);border:1px solid rgba(255,113,140,.45);border-radius:16px;padding:20px}.risk-medium{background:rgba(255,198,110,.09);border:1px solid rgba(255,198,110,.4);border-radius:16px;padding:20px}.risk-low{background:rgba(82,224,178,.09);border:1px solid rgba(82,224,178,.4);border-radius:16px;padding:20px}
div.stButton>button,div.stFormSubmitButton>button{background:linear-gradient(100deg,#65e5ef,#9d8cff);color:#07111e;font-weight:800;border:0;border-radius:10px;min-height:43px;transition:transform .15s,filter .15s}
div.stButton>button:hover,div.stFormSubmitButton>button:hover{filter:brightness(1.08);transform:translateY(-1px);color:#07111e;border:0}
[data-testid="stMetric"]{background:linear-gradient(145deg,#142033,#0f1726);padding:15px 17px;border:1px solid #293950;border-radius:14px}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid #26364d;border-radius:12px;overflow:hidden}
[data-testid="stPlotlyChart"]{background:rgba(16,24,39,.58);border:1px solid #25344a;border-radius:16px;padding:8px}
[data-testid="stForm"]{background:linear-gradient(145deg,rgba(18,29,46,.9),rgba(13,21,35,.9));border:1px solid #27364c;border-radius:18px;padding:20px}
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"],.stTextArea textarea{background:#0c1524!important;border-color:#2b3b53!important;border-radius:10px!important}
.stSlider [data-baseweb="slider"]{padding-top:8px}
hr{border-color:#27364b!important}.stCaption,[data-testid="stCaptionContainer"]{color:#8798b2}
[data-testid="stAlert"]{border-radius:12px}
@media(max-width:850px){.hero{padding:23px;display:block}.hero-visual{display:none}.hero h1{font-size:29px}.metric-card{padding:15px;min-height:120px}}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. DATABASE FUNCTIONS
# ============================================================

def initialize_database():
    """Create the prediction database and table if necessary."""

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                applicant_name TEXT NOT NULL,
                annual_revenue REAL,
                annual_profit REAL,
                loan_amount REAL NOT NULL,
                credit_score INTEGER NOT NULL,
                business_age_years REAL,
                existing_debt REAL,
                late_payments INTEGER,
                default_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_prediction(
    applicant_name,
    borrower_data,
    probability,
    risk_level
):
    """Save an assessment in the local SQLite database."""

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            INSERT INTO predictions (
                applicant_name,
                annual_revenue,
                annual_profit,
                loan_amount,
                credit_score,
                business_age_years,
                existing_debt,
                late_payments,
                default_probability,
                risk_level
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            applicant_name,
            borrower_data["annual_revenue"],
            borrower_data["annual_profit"],
            borrower_data["loan_amount"],
            borrower_data["credit_score"],
            borrower_data["business_age_years"],
            borrower_data["existing_debt"],
            borrower_data["late_payments"],
            probability,
            risk_level
        ))


def get_prediction_history():
    """Return all saved assessments, newest first."""

    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query("""
            SELECT *
            FROM predictions
            ORDER BY id DESC
        """, connection)


def get_database_summary():
    """Return assessment count and high-risk count."""

    history = get_prediction_history()

    if history.empty:
        return 0, 0

    total = len(history)
    high_risk = int((history["risk_level"] == "High").sum())

    return total, high_risk


initialize_database()


# ============================================================
# 4. LOAD DATASET, MODEL AND METADATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset(path, modified_time):
    """Load the project's CSV dataset."""

    if not Path(path).exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model(path, modified_time):
    """Load the trained model."""

    if not Path(path).exists():
        return None

    try:
        return joblib.load(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_metadata(path, modified_time):
    """Load model metadata without crashing if it is unavailable."""

    if not Path(path).exists():
        return {}

    try:
        result = joblib.load(path)

        if isinstance(result, dict):
            return result

        return {}

    except Exception:
        return {}


data = load_dataset(
    str(DATA_PATH),
    DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0
)

model = load_model(
    str(MODEL_PATH),
    MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0
)

metadata = load_metadata(
    str(METADATA_PATH),
    METADATA_PATH.stat().st_mtime if METADATA_PATH.exists() else 0
)

history = get_prediction_history()


# ============================================================
# 5. FEATURE PREPARATION AND PREDICTION
# ============================================================

# These names must match the features used when training the model.
DEFAULT_FEATURES = [
    "annual_revenue",
    "annual_profit",
    "loan_amount",
    "credit_score",
    "business_age_years",
    "existing_debt",
    "late_payments",
]


def get_model_features():
    """Read feature names from the model or metadata."""

    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    stored = metadata.get("numeric_features")

    if isinstance(stored, list) and stored:
        return stored

    return DEFAULT_FEATURES


def make_model_input(borrower_data):
    """Build a DataFrame in the exact order expected by the model."""

    feature_names = get_model_features()

    missing = [
        name for name in feature_names
        if name not in borrower_data
    ]

    if missing:
        raise ValueError(
            "The model requires input features not provided by "
            "the form: " + ", ".join(missing)
        )

    return pd.DataFrame(
        [[borrower_data[name] for name in feature_names]],
        columns=feature_names
    )


def predict_default_probability(borrower_data):
    """Predict the probability of the default class."""

    if model is None:
        raise RuntimeError(
            "The trained model could not be loaded. "
            "Check models/loan_default_model.pkl and run train_model.py."
        )

    model_input = make_model_input(borrower_data)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "This model does not support probability prediction. "
            "Train a classifier that supports predict_proba()."
        )

    probabilities = model.predict_proba(model_input)[0]

    classes = list(model.classes_)

    # This project uses 1 to represent loan default.
    if 1 not in classes:
        raise ValueError(
            "The model does not have a class labelled 1. "
            "Verify that the training target uses 1 = default."
        )

    default_index = classes.index(1)

    return float(probabilities[default_index])


def classify_risk(probability):
    """Assign the project's demonstration risk category."""

    if probability >= 0.70:
        return "High"

    if probability >= 0.40:
        return "Medium"

    return "Low"


# ============================================================
# 6. REUSABLE DASHBOARD COMPONENTS
# ============================================================

def metric_card(label, value, help_text=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-help">{html.escape(str(help_text))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_hero(title, description, eyebrow="LOANRAKSHAK AI"):
    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-copy">
            <div class="eyebrow">{html.escape(eyebrow)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(description)}</p>
          </div>
          <div class="hero-visual">
            <div class="visual-icon">◈</div>
            <div class="visual-label">CREDIT INTELLIGENCE</div>
            <div class="visual-value">AI • Risk • Insight</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True
    )


def get_accuracy():
    """Read the saved evaluation accuracy if available."""

    value = metadata.get("accuracy")

    if value is None:
        return None

    try:
        value = float(value)

        # Metadata may store accuracy as either 0.637 or 63.7.
        if value > 1:
            value = value / 100

        if 0 <= value <= 1:
            return value

    except (TypeError, ValueError):
        pass

    return None


# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================

def navigate_to_page(page_name):
    st.session_state["main_page_navigation"] = page_name


with st.sidebar:
    st.markdown("## 🛡️ LoanRakshak AI")
    st.caption("Business Credit Intelligence")
    st.divider()

    pages = [
        "Overview",
        "AI Risk Assessment",
        "Data Explorer",
        "Prediction History",
        "Model Intelligence",
        "About Platform",
    ]
    if "main_page_navigation" not in st.session_state:
        st.session_state["main_page_navigation"] = "Overview"
    page = st.radio(
        "NAVIGATION", pages,
        label_visibility="visible", key="main_page_navigation"
    )

    st.divider()

    st.markdown("### System Status")

    st.write(
        "🟢 Dataset loaded"
        if not data.empty
        else "🔴 Dataset missing"
    )

    st.write(
        "🟢 Model loaded"
        if model is not None
        else "🔴 Model unavailable"
    )

    st.write("🟢 Prediction database ready")

    st.caption("Machine learning demonstration platform")


# ============================================================
# 8. OVERVIEW PAGE
# ============================================================

if page == "Overview":

    show_hero(
        "Credit decisions, with more clarity.",
        "A single workspace to assess borrower risk, explore the training data, and review saved predictions.",
        "LOANRAKSHAK  /  CREDIT INTELLIGENCE"
    )

    total_assessments, high_risk_cases = get_database_summary()
    accuracy = get_accuracy()
    history = get_prediction_history()

    st.markdown("<div class='eyebrow'>PORTFOLIO SNAPSHOT</div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    with col1:
        metric_card("Training records", f"{len(data):,}", "Historical records available to the model")
    with col2:
        metric_card("Assessments run", f"{total_assessments:,}", "Saved borrower evaluations")
    with col3:
        metric_card("High-risk cases", f"{high_risk_cases:,}", "Saved predictions at 70% risk or above")
    with col4:
        metric_card("Test accuracy", f"{accuracy:.1%}" if accuracy is not None else "N/A", "Model evaluation metric · not a guarantee")

    st.markdown("## Risk & data overview")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown("### Recent assessment activity")
        if not history.empty:
            recent = history.head(6).copy()
            if "default_probability" in recent:
                recent["default_probability"] = (recent["default_probability"] * 100).round(1).astype(str) + "%"
            display_cols = [c for c in ["applicant_name", "default_probability", "risk_level", "created_at"] if c in recent.columns]
            recent = recent[display_cols].rename(columns={
                "applicant_name": "Applicant", "default_probability": "Default risk", "risk_level": "Risk tier", "created_at": "Assessed at"
            })
            st.dataframe(recent, use_container_width=True, hide_index=True)
            risk_counts = history["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
            fig = go.Figure(data=[go.Bar(
                x=risk_counts.index.tolist(), y=risk_counts.values.tolist(),
                marker_color=["#52e0b2", "#ffc66e", "#ff718c"],
                text=risk_counts.values.tolist(), textposition="outside",
                hovertemplate="%{x}: %{y} assessment(s)<extra></extra>"
            )])
            fig.update_layout(template="plotly_dark", height=235, margin=dict(l=10,r=10,t=20,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, xaxis_title="Risk tier", yaxis_title="Saved assessments")
            fig.update_yaxes(gridcolor="rgba(150,170,200,.12)", rangemode="tozero")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown("""
            <div class="panel"><div class="eyebrow">READY WHEN YOU ARE</div>
            <h3>No assessments saved yet</h3><p>Run your first borrower assessment to start building a risk portfolio and see the distribution here.</p></div>
            """, unsafe_allow_html=True)
            st.info("Choose **AI Risk Assessment** in the sidebar to get started.")
    with right:
        st.markdown("### Dataset health")
        if not data.empty:
            completeness = (data.notna().mean() * 100).sort_values()
            complete_df = completeness.rename_axis("Feature").reset_index(name="Completeness")
            fig = px.bar(complete_df, x="Completeness", y="Feature", orientation="h",
                template="plotly_dark", color="Completeness", range_x=[0, 100],
                color_continuous_scale=["#6d7cff", "#65e5ef"])
            fig.update_layout(height=330, margin=dict(l=4,r=8,t=12,b=8),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False, xaxis_title="Non-empty values (%)", yaxis_title="")
            fig.update_xaxes(gridcolor="rgba(150,170,200,.12)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            missing_cells = int(data.isna().sum().sum())
            total_cells = int(data.shape[0] * data.shape[1])
            coverage = (1 - missing_cells / total_cells) * 100 if total_cells else 0
            a, b = st.columns(2)
            with a:
                st.metric("Overall completeness", f"{coverage:.1f}%")
            with b:
                st.metric("Features", f"{data.shape[1]:,}")
        else:
            st.warning("Dataset not found. Check data/business_loan_data.csv.")

    st.markdown("## Jump back in")
    q1, q2, q3 = st.columns(3, gap="medium")
    with q1:
        st.markdown("<div class='panel'><div class='eyebrow'>01 / ANALYSE</div><h3>Assess a borrower</h3><p>Enter business and credit details to estimate default probability.</p></div>", unsafe_allow_html=True)
        st.button("Open risk assessment →", use_container_width=True,
                  on_click=navigate_to_page, args=("AI Risk Assessment",))
    with q2:
        st.markdown("<div class='panel'><div class='eyebrow'>02 / EXPLORE</div><h3>Inspect the data</h3><p>Review the historical dataset, features, and summary statistics.</p></div>", unsafe_allow_html=True)
        st.button("Open data explorer →", use_container_width=True,
                  on_click=navigate_to_page, args=("Data Explorer",))
    with q3:
        st.markdown("<div class='panel'><div class='eyebrow'>03 / REVIEW</div><h3>View saved cases</h3><p>Track previous assessments and export the saved results.</p></div>", unsafe_allow_html=True)
        st.button("Open prediction history →", use_container_width=True,
                  on_click=navigate_to_page, args=("Prediction History",))


# ============================================================
# 9. AI RISK ASSESSMENT PAGE
# ============================================================

elif page == "AI Risk Assessment":

    show_hero(
        "AI Risk Assessment",
        "Enter the borrower's financial profile to estimate loan-default probability.",
        "INTELLIGENT CREDIT ANALYSIS"
    )

    if model is None:
        st.error(
            "The trained model is unavailable. "
            "Check models/loan_default_model.pkl and train_model.py."
        )

    st.markdown("### Borrower information")

    with st.form("loan_assessment_form"):

        applicant_name = st.text_input(
            "Applicant / Business Name",
            placeholder="Enter applicant name"
        )

        col1, col2 = st.columns(2)

        with col1:
            annual_revenue = st.number_input(
                "Annual Revenue (₹)",
                min_value=0.0,
                value=1000000.0,
                step=50000.0
            )

            annual_profit = st.number_input(
                "Annual Profit (₹)",
                value=150000.0,
                step=10000.0
            )

            loan_amount = st.number_input(
                "Requested Loan Amount (₹)",
                min_value=1.0,
                value=500000.0,
                step=25000.0
            )

            credit_score = st.slider(
                "Credit Score",
                min_value=300,
                max_value=900,
                value=700
            )

        with col2:
            business_age_years = st.number_input(
                "Business Age (Years)",
                min_value=0.0,
                value=5.0,
                step=1.0
            )

            existing_debt = st.number_input(
                "Existing Debt (₹)",
                min_value=0.0,
                value=100000.0,
                step=10000.0
            )

            late_payments = st.number_input(
                "Previous Late Payments",
                min_value=0,
                value=0,
                step=1
            )

        submitted = st.form_submit_button(
            "🛡️ Predict Default Risk",
            use_container_width=True
        )

    if submitted:

        if not applicant_name.strip():
            st.error("Please enter the applicant's name.")

        elif model is None:
            st.error(
                "Cannot predict because the trained model is unavailable."
            )

        elif annual_profit > annual_revenue:
            st.error(
                "Please check the financial inputs: annual profit "
                "is greater than annual revenue."
            )

        else:

            borrower_data = {
                "annual_revenue": annual_revenue,
                "annual_profit": annual_profit,
                "loan_amount": loan_amount,
                "credit_score": credit_score,
                "business_age_years": business_age_years,
                "existing_debt": existing_debt,
                "late_payments": late_payments,
            }

            try:

                probability = predict_default_probability(
                    borrower_data
                )

                if not 0 <= probability <= 1:
                    raise ValueError(
                        "The model returned an invalid probability."
                    )

                risk_level = classify_risk(probability)

                st.markdown("---")
                st.markdown("## Assessment results")

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.metric(
                        "Estimated Default Probability",
                        f"{probability:.1%}"
                    )

                    st.progress(probability)

                    st.caption(
                        "Model-estimated probability for the supplied inputs."
                    )

                with result_col2:

                    css_class = {
                        "High": "risk-high",
                        "Medium": "risk-medium",
                        "Low": "risk-low",
                    }[risk_level]

                    st.markdown(
                        f"""
                        <div class="{css_class}">
                            <h3>{risk_level} Risk</h3>
                            <p>
                                Estimated default probability:
                                <b>{probability:.1%}</b>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Save the result to the database.
                save_prediction(
                    applicant_name.strip(),
                    borrower_data,
                    probability,
                    risk_level
                )

                st.success(
                    "Assessment completed and saved to Prediction History."
                )

                st.caption(
                    "This is a model estimate, not a guarantee of loan default "
                    "or a standalone lending decision."
                )

            except Exception as error:

                st.error(
                    f"Assessment failed: {error}"
                )

                st.info(
                    "Check whether the model's training features match "
                    "the input fields in this form."
                )


# ============================================================
# 10. DATA EXPLORER PAGE
# ============================================================

elif page == "Data Explorer":

    show_hero(
        "Data Explorer",
        "Explore the historical financial dataset used by the project.",
        "FINANCIAL DATA LAB"
    )

    if data.empty:

        st.error(
            "Dataset not found. Make sure "
            "data/business_loan_data.csv exists."
        )

    else:

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Total Records", f"{len(data):,}")

        with c2:
            st.metric("Total Features", len(data.columns))

        with c3:
            st.metric(
                "Missing Values",
                int(data.isna().sum().sum())
            )

        st.markdown("### Filter records")

        search_column = st.selectbox(
            "Choose a column to filter",
            ["All columns"] + list(data.columns)
        )

        search_text = st.text_input(
            "Search",
            placeholder="Enter a value to find"
        )

        filtered = data.copy()

        if search_text.strip():

            if search_column == "All columns":

                mask = filtered.astype(str).apply(
                    lambda column: column.str.contains(
                        search_text,
                        case=False,
                        na=False,
                        regex=False
                    )
                ).any(axis=1)

                filtered = filtered[mask]

            else:

                filtered = filtered[
                    filtered[search_column]
                    .astype(str)
                    .str.contains(
                        search_text,
                        case=False,
                        na=False,
                        regex=False
                    )
                ]

        st.write(f"Showing {len(filtered):,} matching records.")

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        csv_data = filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download filtered dataset",
            data=csv_data,
            file_name="loanrakshak_filtered_data.csv",
            mime="text/csv"
        )

        st.markdown("### Numerical feature analysis")

        numeric_columns = list(
            data.select_dtypes(include="number").columns
        )

        if numeric_columns:

            selected_column = st.selectbox(
                "Select a numerical feature",
                numeric_columns
            )

            fig = px.histogram(
                data,
                x=selected_column,
                nbins=30,
                template="plotly_dark",
                title=f"Distribution of {selected_column}"
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                data[numeric_columns].describe().T,
                use_container_width=True
            )

        else:

            st.info("No numerical columns were found.")


# ============================================================
# 11. PREDICTION HISTORY PAGE
# ============================================================

elif page == "Prediction History":

    show_hero(
        "Prediction History",
        "Review saved borrower assessments and their estimated default probabilities.",
        "ASSESSMENT RECORDS"
    )

    history = get_prediction_history()

    if history.empty:

        st.info(
            "No saved assessments yet. Open AI Risk Assessment "
            "and submit a borrower profile first."
        )

    else:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Assessments", len(history))

        with col2:
            st.metric(
                "High Risk",
                int((history["risk_level"] == "High").sum())
            )

        with col3:
            st.metric(
                "Average Estimated Probability",
                f"{history['default_probability'].mean():.1%}"
            )

        st.markdown("### Saved assessments")

        display_history = history.copy()

        display_history["default_probability"] = (
            display_history["default_probability"] * 100
        ).round(2)

        display_history = display_history.rename(columns={
            "id": "ID",
            "applicant_name": "Applicant",
            "annual_revenue": "Annual Revenue",
            "annual_profit": "Annual Profit",
            "loan_amount": "Loan Amount",
            "credit_score": "Credit Score",
            "business_age_years": "Business Age",
            "existing_debt": "Existing Debt",
            "late_payments": "Late Payments",
            "default_probability": "Default Probability (%)",
            "risk_level": "Risk Level",
            "created_at": "Assessment Date"
        })

        st.dataframe(
            display_history,
            use_container_width=True,
            hide_index=True
        )

        csv_data = display_history.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Export prediction history",
            data=csv_data,
            file_name="loanrakshak_prediction_history.csv",
            mime="text/csv"
        )

        st.markdown("### Risk distribution")

        risk_counts = history["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]

        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Count",
            hole=0.65,
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 12. MODEL INTELLIGENCE PAGE
# ============================================================

elif page == "Model Intelligence":

    show_hero(
        "Model Intelligence",
        "Inspect the saved model information and its evaluation metrics.",
        "MACHINE LEARNING INSIGHTS"
    )

    if model is None:

        st.error(
            "The trained model is unavailable. "
            "Check models/loan_default_model.pkl."
        )

    else:

        st.success(
            f"Model loaded successfully: {type(model).__name__}"
        )

    accuracy = get_accuracy()

    col1, col2, col3 = st.columns(3)

    with col1:

        metric_card(
            "MODEL TYPE",
            type(model).__name__ if model is not None else "Unavailable",
            "Loaded estimator"
        )

    with col2:

        metric_card(
            "TEST ACCURACY",
            f"{accuracy:.1%}" if accuracy is not None else "N/A",
            "Read from saved metadata"
        )

    with col3:

        metric_card(
            "TRAINING RECORDS",
            metadata.get("rows", len(data)),
            "Reported dataset size"
        )

    st.markdown("### Saved model metadata")

    if metadata:

        metadata_df = pd.DataFrame(
            [
                {"Property": key, "Value": str(value)}
                for key, value in metadata.items()
            ]
        )

        st.dataframe(
            metadata_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Model metadata is unavailable. "
            "Check models/model_metadata.pkl."
        )

    st.markdown("### Important evaluation metrics")

    st.info(
        "Accuracy alone is not enough to evaluate a loan-default model. "
        "Precision, recall, F1-score, ROC-AUC, class balance, and "
        "performance on unseen data should also be checked."
    )

    if data.empty:

        st.warning(
            "The dataset is unavailable, so its target labels cannot be inspected."
        )

    else:

        possible_targets = [
            column for column in data.columns
            if column.lower() in {
                "default",
                "loan_default",
                "defaulted",
                "default_status",
                "target",
                "label"
            }
        ]

        if possible_targets:

            target = possible_targets[0]

            st.markdown(f"#### Target distribution: `{target}`")

            target_counts = (
                data[target]
                .astype(str)
                .value_counts()
                .rename_axis("Class")
                .reset_index(name="Records")
            )

            fig = px.bar(
                target_counts,
                x="Class",
                y="Records",
                template="plotly_dark",
                title="Distribution of target classes"
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:

            st.warning(
                "Could not identify the default target column automatically. "
                "Check your dataset's column names."
            )


# ============================================================
# 13. ABOUT PLATFORM PAGE
# ============================================================

elif page == "About Platform":

    show_hero(
        "About LoanRakshak AI",
        "A machine learning demonstration platform for estimating "
        "business loan-default probability from financial information.",
        "PROJECT OVERVIEW"
    )

    st.markdown("## Project objective")

    st.markdown("""
    <div class="panel">
        <h3>Predict default probability using historical financial data.</h3>
        <p>
            The system uses historical borrower information to train a
            machine learning model, estimate the probability of loan default,
            and organize assessments for review.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Technology stack")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="panel">
            <h3>Machine Learning</h3>
            <p>Scikit-learn and the trained classification model.</p>
            <h3>Data Processing</h3>
            <p>Pandas and CSV financial data.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="panel">
            <h3>Application</h3>
            <p>Python and Streamlit.</p>
            <h3>Storage and Visualization</h3>
            <p>SQLite and Plotly.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Risk categories")

    st.write(
        "Low Risk: estimated probability below 40%."
    )

    st.write(
        "Medium Risk: estimated probability from 40% to below 70%."
    )

    st.write(
        "High Risk: estimated probability of 70% or above."
    )

    st.warning(
        "These are demonstration thresholds. Model outputs are estimates, "
        "not guarantees. A real lending system requires representative "
        "historical data, appropriate validation, fairness checks, "
        "and human review."
    )


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "LoanRakshak AI • Business Loan Default Prediction System • "
    "Machine Learning Demonstration"
)