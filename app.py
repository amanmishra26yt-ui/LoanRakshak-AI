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
@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap'
);

:root {
    --background: #080b14;
    --panel: #111827;
    --panel-light: #182237;
    --cyan: #55e6ff;
    --purple: #a78bfa;
    --green: #42e6ad;
    --red: #ff718c;
    --text: #f5f7ff;
    --muted: #9ba8c1;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            ellipse at 85% 0%,
            rgba(111, 76, 255, 0.13),
            transparent 35%
        ),
        var(--background);
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: rgba(8, 11, 20, 0.95);
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: #0d1220;
    border-right: 1px solid #202a40;
}

.hero {
    padding: 38px;
    border-radius: 26px;
    border: 1px solid #2a3654;
    background:
        radial-gradient(
            circle at 80% 10%,
            rgba(138, 92, 246, 0.23),
            transparent 38%
        ),
        linear-gradient(120deg, #14213d, #11172b);
    margin-bottom: 26px;
}

.eyebrow {
    color: var(--cyan);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
}

.hero h1 {
    font-size: clamp(30px, 4vw, 48px);
    line-height: 1.15;
    margin: 18px 0;
}

.hero p {
    color: #b7c4db;
    font-size: 16px;
    line-height: 1.8;
}

.metric-card {
    background: linear-gradient(145deg, #141c2e, #0e1422);
    border: 1px solid #283249;
    border-radius: 20px;
    padding: 22px;
    min-height: 145px;
    margin-bottom: 14px;
}

.metric-label {
    color: #a8b4cb;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #ffffff;
    font-size: 31px;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    margin-top: 15px;
}

.metric-help {
    color: #8391ab;
    font-size: 12px;
    margin-top: 8px;
}

.panel {
    background: #111827;
    border: 1px solid #263149;
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 20px;
}

.info-box {
    background: #121e31;
    border-left: 4px solid var(--cyan);
    border-radius: 10px;
    padding: 16px;
    color: #d7e3f7;
    margin: 12px 0;
}

.risk-high {
    background: rgba(255, 80, 110, 0.12);
    border: 1px solid #913c50;
    border-radius: 16px;
    padding: 22px;
}

.risk-medium {
    background: rgba(255, 186, 73, 0.10);
    border: 1px solid #806438;
    border-radius: 16px;
    padding: 22px;
}

.risk-low {
    background: rgba(53, 220, 158, 0.10);
    border: 1px solid #276b59;
    border-radius: 16px;
    padding: 22px;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(100deg, #43dff6, #8a7cff);
    color: #08101c;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    min-height: 44px;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    color: #ffffff;
    border: 1px solid var(--cyan);
}

[data-testid="stMetric"] {
    background: #111827;
    padding: 15px;
    border: 1px solid #263149;
    border-radius: 14px;
}

hr {
    border-color: #263149;
}

.stCaption {
    color: #91a0b9;
}
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
    """Load the dataset and preserve useful error details."""

    if not Path(path).exists():
        return pd.DataFrame(), f"Dataset file not found: {path}"

    try:
        df = pd.read_csv(path)

        if df.empty:
            return df, "The dataset exists but contains no records."

        return df, None

    except Exception as error:
        return pd.DataFrame(), f"{type(error).__name__}: {error}"


@st.cache_resource(show_spinner=False)
def load_model(path, modified_time):
    """Load the trained machine learning model."""

    if not Path(path).exists():
        return None, f"Model file not found: {path}"

    try:
        loaded_model = joblib.load(path)

        if not hasattr(loaded_model, "predict_proba"):
            return None, "Loaded model does not support predict_proba()."

        return loaded_model, None

    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


@st.cache_data(show_spinner=False)
def load_metadata(path, modified_time):
    """Load model metadata and report errors."""

    if not Path(path).exists():
        return {}, f"Metadata file not found: {path}"

    try:
        result = joblib.load(path)

        if not isinstance(result, dict):
            return {}, "Model metadata must be a dictionary."

        return result, None

    except Exception as error:
        return {}, f"{type(error).__name__}: {error}"


data, dataset_error = load_dataset(
    str(DATA_PATH),
    DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0
)

model, model_error = load_model(
    str(MODEL_PATH),
    MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0
)

metadata, metadata_error = load_metadata(
    str(METADATA_PATH),
    METADATA_PATH.stat().st_mtime if METADATA_PATH.exists() else 0
)

if dataset_error:
    st.error(f"Dataset loading error: {dataset_error}")

if model_error:
    st.error(f"Model loading error: {model_error}")

if metadata_error:
    st.warning(f"Metadata loading warning: {metadata_error}")

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
        <div class="hero">
            <div class="eyebrow">{html.escape(eyebrow)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(description)}</p>
        </div>
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

with st.sidebar:
    st.markdown("## 🛡️ LoanRakshak AI")
    st.caption("Business Credit Intelligence")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "AI Risk Assessment",
            "Data Explorer",
            "Prediction History",
            "Model Intelligence",
            "About Platform",
        ],
        label_visibility="visible"
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
        "See risk. Make smarter decisions.",
        "Explore business loan risk indicators, model insights, "
        "and saved assessments in one modern financial analytics workspace.",
        "YOUR INTELLIGENT LENDING WORKSPACE"
    )

    total_assessments, high_risk_cases = get_database_summary()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "TRAINING RECORDS",
            f"{len(data):,}",
            "Rows in the loaded dataset"
        )

    with col2:
        metric_card(
            "ASSESSMENTS",
            f"{total_assessments:,}",
            "Saved predictions"
        )

    with col3:
        metric_card(
            "HIGH-RISK CASES",
            f"{high_risk_cases:,}",
            "Saved assessments with risk ≥ 70%"
        )

    with col4:
        accuracy = get_accuracy()

        metric_card(
            "MODEL ACCURACY",
            f"{accuracy:.1%}" if accuracy is not None else "N/A",
            "Saved test-set accuracy"
        )

    st.markdown("## Portfolio intelligence")

    if not data.empty:

        left, right = st.columns(2)

        with left:
            st.markdown("### Dataset preview")
            st.dataframe(
                data.head(8),
                use_container_width=True,
                hide_index=True
            )

        with right:
            st.markdown("### Dataset completeness")

            completeness = (
                data.notna().mean() * 100
            ).sort_values()

            completeness_df = completeness.reset_index()
            completeness_df.columns = ["Feature", "Completeness (%)"]

            fig = px.bar(
                completeness_df,
                x="Completeness (%)",
                y="Feature",
                orientation="h",
                template="plotly_dark",
                color="Completeness (%)",
                color_continuous_scale=["#8a7cff", "#55e6ff"]
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_showscale=False
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(
            "Dataset not found. Check data/business_loan_data.csv."
        )

    st.markdown("### Quick actions")

    quick1, quick2 = st.columns(2)

    with quick1:
        st.info(
            "**Assess a borrower**\n\n"
            "Open AI Risk Assessment from the sidebar to calculate "
            "a default probability."
        )

    with quick2:
        st.info(
            "**Review saved cases**\n\n"
            "Open Prediction History to review previous assessments."
        )


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