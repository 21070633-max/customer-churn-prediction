from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from config import load_css, set_netflix_config

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_DIR = PROJECT_ROOT / "model"
LOGO_PATH = PROJECT_ROOT / "images.png"

# Load CSS and page configuration
load_css()
set_netflix_config()

# Application title
st.markdown('<div class="logo-banner">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 6])

with col1:
    st.image(str(LOGO_PATH), width=220)

with col2:
    st.markdown("""
    <h1 class="logo-title">Customer Churn Prediction System</h1>
    <p class="logo-subtitle">Predict whether customers will churn from the streaming platform</p>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load trained model"""
    model_path = MODEL_DIR / "best_model.pkl"
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_preprocessor():
    """Load preprocessor if needed"""
    preprocessor_path = MODEL_DIR / "preprocessor.pkl"
    try:
        with open(preprocessor_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.warning(f"Error loading preprocessor: {e}")
        return None


@st.cache_resource
def load_threshold():
    """Load optimal threshold"""
    threshold_path = MODEL_DIR / "optimal_threshold.pkl"
    try:
        with open(threshold_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return 0.5
    except Exception as e:
        st.warning(f"Error loading threshold: {e}. Using default threshold = 0.5")
        return 0.5


def calculate_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived features from raw input data"""

    # 1. Engagement score
    data["engagement_score"] = np.round(
        (data["watch_time_hours"] * 0.4)
        + (data["login_frequency"] * 0.3)
        + (data["recommendation_click_rate"] * 100 * 0.3),
        2,
    )

    # 2. Inactive risk score
    data["inactive_risk_score"] = np.round(
        (data["days_since_last_login"] * 0.7)
        + ((1 - data["recommendation_click_rate"]) * 10 * 0.3),
        2,
    )

    # 3. Fee usage ratio
    data["fee_usage_ratio"] = np.round(
        data["monthly_fee"] / (data["watch_time_hours"] + 1),
        2,
    )

    # 4. Content interest score
    movies_max_reference = 40
    data["content_interest_score"] = np.round(
        (data["recommendation_click_rate"] * 0.7)
        + ((data["movies_watched"] / (movies_max_reference + 1)) * 0.3),
        2,
    )

    # 5. Total amount spent
    data["total_amount_spent"] = np.round(
        data["monthly_fee"] * data["subscription_length"],
        2,
    )

    # 6. Estimated login count in 3 months
    data["est_login_count_3m"] = np.round(
        data["login_frequency"] * 3,
        2,
    )

    # 7. Estimated watch time in 3 months
    data["est_watch_time_3m"] = np.round(
        data["watch_time_hours"] * 3,
        2,
    )

    return data

# Prepare prediction data function
def prepare_prediction_data(form_data: dict) -> pd.DataFrame:
    """Prepare data for model prediction"""
    df = pd.DataFrame([form_data])
    df = calculate_derived_features(df)

    # Add missing columns if needed
    if 'user_id' not in df.columns:
        df['user_id'] = 1

    return df


# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Customer Information Form")

with col2:
    st.subheader("📊 Prediction Results")
    result_placeholder = st.empty()


# Input form
with st.form("customer_form"):
    st.markdown("### 👤 Usage Behavior")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        user_id = st.number_input(
            "User ID",
            min_value=1,
            value=1,
            step=1,
        )

    with col2:
        watch_time_hours = st.number_input(
            "Watch Time (hours/month)",
            min_value=0,
            max_value=120,
            value=50,
            step=1,
            help="Average watch time in one month",
        )

    with col3:
        login_frequency = st.number_input(
            "Login Frequency (times/month)",
            min_value=1,
            max_value=31,
            value=15,
            step=1,
            help="Average login frequency in one month",
        )

    with col4:
        movies_watched = st.number_input(
            "Movies Watched (completed)",
            min_value=0,
            max_value=40,
            value=20,
            step=1,
            help="Total completed movies on platform",
        )

    st.markdown("### 📱 Engagement & Time Information")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        first_login_date = st.date_input(
            "First Login Date",
            value=pd.to_datetime("2024-01-01"),
        )

    with col2:
        last_login_date = st.date_input(
            "Last Login Date",
            value=pd.to_datetime("2025-12-20"),
        )

    with col3:
        recommendation_click_rate = st.number_input(
            "Recommendation Click Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            format="%.2f",
        )

    with col4:
        subscription_type = st.selectbox(
            "Subscription Type",
            ["Basic", "Standard", "Premium"],
        )

    reference_date = pd.Timestamp("2025-12-31")
    first_login_ts = pd.to_datetime(first_login_date)
    last_login_ts = pd.to_datetime(last_login_date)

    subscription_length = max(round((reference_date - first_login_ts).days / 30), 1)
    days_since_last_login = max((reference_date - last_login_ts).days, 0)

    st.markdown("### 💰 Subscription Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.number_input(
            "Subscription Length (months)",
            min_value=1,
            max_value=120,
            value=int(subscription_length),
            step=1,
            disabled=True,
        )

    with col2:
        if subscription_type == "Basic":
            monthly_fee = st.number_input(
                "Monthly Fee ($)",
                min_value=5.0,
                max_value=9.0,
                value=7.0,
                step=0.5,
            )
        elif subscription_type == "Standard":
            monthly_fee = st.number_input(
                "Monthly Fee ($)",
                min_value=10.0,
                max_value=14.0,
                value=12.0,
                step=0.5,
            )
        else:
            monthly_fee = st.number_input(
                "Monthly Fee ($)",
                min_value=15.0,
                max_value=20.0,
                value=17.5,
                step=0.5,
            )

    with col3:
        max_fee = 20.0
        subscription_price_index = round(monthly_fee / max_fee, 2)

        st.number_input(
            "Subscription Price Index",
            min_value=0.0,
            max_value=1.0,
            value=float(subscription_price_index),
            step=0.01,
            format="%.2f",
            disabled=True,
            help="Automatically calculated from Monthly Fee",
        )

    st.markdown("### ⏱️ Activity Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.number_input(
            "Days Since Last Login",
            min_value=0,
            max_value=365,
            value=int(days_since_last_login),
            step=1,
            disabled=True,
        )

    with col2:
        st.caption(f"Reference Date: {reference_date.date()}")

    st.markdown("### 🎬 Market Factors")
    col1, col2 = st.columns(2)

    with col1:
        competitor_content_index = st.number_input(
            "Competitor Content Index",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            format="%.2f",
        )

    with col2:
        piracy_site_popularity = st.number_input(
            "Piracy Site Popularity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            format="%.2f",
        )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submitted = st.form_submit_button("🚀 Predict Churn", use_container_width=True)
        
# Process prediction
if submitted:
    model = load_model()
    preprocessor = load_preprocessor()
    threshold = load_threshold()

    if model is not None:
        try:
            if pd.to_datetime(first_login_date) > pd.to_datetime(last_login_date):
                st.error("First Login Date must be earlier than or equal to Last Login Date.")
                st.stop()

            if pd.to_datetime(last_login_date) > pd.Timestamp("2025-12-31"):
                st.error("Last Login Date must not be later than reference date 2025-12-31.")
                st.stop()

            form_data = {
                "user_id": int(user_id),
                "first_login_date": str(first_login_date),
                "last_login_date": str(last_login_date),
                "watch_time_hours": watch_time_hours,
                "login_frequency": login_frequency,
                "movies_watched": movies_watched,
                "days_since_last_login": days_since_last_login,
                "recommendation_click_rate": recommendation_click_rate,
                "subscription_type": subscription_type,
                "subscription_length": subscription_length,
                "monthly_fee": monthly_fee,
                "competitor_content_index": competitor_content_index,
                "piracy_site_popularity": piracy_site_popularity,
                "subscription_price_index": subscription_price_index,
            }
            
            # Prepare prediction data
            df_predict = prepare_prediction_data(form_data)
            # Make prediction
            churn_probability = model.predict_proba(df_predict)[:, 1][0]
            churn_prediction = 1 if churn_probability >= threshold else 0

            # Display results in result placeholder
            with result_placeholder.container():
                if churn_probability >= threshold:
                    st.error("🚨 HIGH RISK: Customer Likely to Churn", icon="⚠️")
                    st.metric(
                        "Churn Probability",
                        f"{churn_probability:.2%}",
                        delta="Action Required",
                    )
                else:
                    st.success("✅ LOW RISK: Customer Likely to Stay", icon="✔️")
                    st.metric(
                        "Churn Probability",
                        f"{churn_probability:.2%}",
                        delta="Stable",
                    )

            st.caption(f"Current decision threshold: {threshold:.2%}")

            # Detailed analysis
            st.markdown("---")
            st.subheader("📈 Detailed Analysis")

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.write("**Input Features:**")
                input_data = pd.DataFrame({
                    "Feature": [
                        "User ID",
                        "First Login Date",
                        "Last Login Date",
                        "Watch Time",
                        "Login Frequency",
                        "Movies Watched",
                        "Days Since Last Login",
                        "Recommendation Click Rate",
                        "Subscription Type",
                        "Monthly Fee",
                        "Subscription Length",
                        "Price Index",
                    ],
                    "Value": [
                        f"{user_id}",
                        str(first_login_date),
                        str(last_login_date),
                        f"{watch_time_hours} hours/month",
                        f"{login_frequency} times/month",
                        f"{movies_watched} completed movies",
                        f"{days_since_last_login} days",
                        f"{recommendation_click_rate:.2f}",
                        subscription_type,
                        f"${monthly_fee:.2f}",
                        f"{subscription_length} months",
                        f"{subscription_price_index:.2f}",
                    ],
                })
                st.dataframe(input_data, use_container_width=True, hide_index=True)

            with detail_col2:
                st.write("**Derived Features:**")
                derived_data = pd.DataFrame({
                    "Feature": [
                        "Engagement Score",
                        "Inactive Risk Score",
                        "Fee Usage Ratio",
                        "Content Interest Score",
                        "Total Amount Spent",
                        "Estimated Login Count (3M)",
                        "Estimated Watch Time (3M)",
                    ],
                    "Value": [
                        f"{df_predict['engagement_score'].values[0]:.2f}",
                        f"{df_predict['inactive_risk_score'].values[0]:.2f}",
                        f"{df_predict['fee_usage_ratio'].values[0]:.2f}",
                        f"{df_predict['content_interest_score'].values[0]:.2f}",
                        f"${df_predict['total_amount_spent'].values[0]:.2f}",
                        f"{df_predict['est_login_count_3m'].values[0]:.2f}",
                        f"{df_predict['est_watch_time_3m'].values[0]:.2f} hours",
                    ],
                })
                st.dataframe(derived_data, use_container_width=True, hide_index=True)

            # Insights
            st.markdown("---")
            st.subheader("💡 Insights & Recommendations")

            insights = []

            if watch_time_hours < 20:
                insights.append("⚠️ Low watch time - customer is using the service less than expected.")

            if login_frequency < 10:
                insights.append("⚠️ Low login frequency - customer is not returning regularly.")

            if days_since_last_login > 30:
                insights.append("🔴 Long time since last login - strong sign of churn risk.")

            if recommendation_click_rate < 0.3:
                insights.append("⚠️ Low recommendation engagement - content may not match customer interest.")

            if competitor_content_index > 0.7:
                insights.append("ℹ️ Strong competitor content pressure may increase churn risk.")

            if piracy_site_popularity > 0.6:
                insights.append("ℹ️ High piracy popularity may reduce perceived need for paid service.")

            if subscription_type == "Basic":
                insights.append("⚠️ Basic tier may have a slightly higher churn risk compared with higher tiers.")

            if watch_time_hours > 80 and login_frequency > 20 and recommendation_click_rate > 0.7:
                insights.append("✨ Highly engaged customer - focus on retention.")

            if insights:
                for insight in insights:
                    st.info(insight)
            else:
                st.success("✅ Customer indicators look healthy.")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("Unable to load model. Please check file paths.")

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #888;'>
        <p>🎬 Streaming Customer Churn Prediction System | Version 1.0 | {datetime.now().strftime('%d/%m/%Y')}</p>
        <p style='font-size: 0.85em;'>Built with ❤️ using Streamlit & Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)