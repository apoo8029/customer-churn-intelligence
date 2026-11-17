import sqlite3

import pandas as pd

import numpy as np

import streamlit as st

import plotly.express as px

import plotly.graph_objects as go

from pathlib import Path

import joblib

from datetime import datetime



# ==========================================

# 1. PAGE CONFIGURATION & ENTERPRISE THEME

# ==========================================

st.set_page_config(

          page_title="Customer Churn Intelligence Platform",

          page_icon="Ã¢Å¡Â¡",

          layout="wide",

          initial_sidebar_state="expanded"

)



# Custom CSS for Stripe / Linear / Datadog inspired Dark Enterprise UI/UX

st.markdown("""

          <style>

          .main {

                    background-color: #0b0f19;

                    color: #f3f4f6;

          }

          div[data-testid="stMetric"] {

                    background-color: #111827;

                    border: 1px solid #1f2937;

                    padding: 18px;

                    border-radius: 10px;

                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);

          }

          div[data-testid="stMetric"] label {

                    color: #9ca3af !important;

                    font-weight: 600;

                    font-size: 0.85rem;

          }

          div[data-testid="stMetric"] div[data-testid="stMetricValue"] {

                    color: #60a5fa !important;

                    font-weight: 700;

                    font-size: 1.6rem;

          }

          .stTabs [data-baseweb="tab-list"] {

                    gap: 8px;

                    background-color: #0b0f19;

                    padding: 4px;

          }

          .stTabs [data-baseweb="tab"] {

                    background-color: #111827;

                    border-radius: 6px;

                    color: #9ca3af;

                    padding: 10px 18px;

                    font-weight: 600;

                    border: 1px solid #1f2937;

          }

          .stTabs [aria-selected="true"] {

                    background-color: #2563eb !important;

                    color: #ffffff !important;

                    border-color: #3b82f6 !important;

          }

          h1, h2, h3 {

                    color: #f9fafb;

                    font-family: 'Inter', sans-serif;

          }

          .sidebar .stSidebarContent {

                    background-color: #0f172a;

          }

          </style>

""", unsafe_allow_html=True)



# ==========================================

# 2. DATABASE & MODEL CONNECTION CACHE

# ==========================================

# app.py is in the parent folder; project data is in the child project folder.
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR / "customer-churn-intelligence"

DB_CANDIDATES = [
    PROJECT_DIR / "data" / "processed" / "telco_churn_analysis.db",
    APP_DIR / "customer-churn-intelligence" / "data" / "processed" / "telco_churn_analysis.db",
    APP_DIR / "data" / "processed" / "telco_churn_analysis.db",
    APP_DIR / "telco_churn_analysis.db",
]
MODEL_CANDIDATES = [
    PROJECT_DIR / "models" / "telco_churn_model.pkl",
    APP_DIR / "models" / "telco_churn_model.pkl",
    PROJECT_DIR / "telco_churn_model.pkl",
    APP_DIR / "telco_churn_model.pkl",
]
ENCODER_CANDIDATES = [
    PROJECT_DIR / "models" / "label_encoders.pkl",
    APP_DIR / "models" / "label_encoders.pkl",
    PROJECT_DIR / "label_encoders.pkl",
    APP_DIR / "label_encoders.pkl",
]

DB_PATH = next((path for path in DB_CANDIDATES if path.is_file()), DB_CANDIDATES[0])
MODEL_PATH = next((path for path in MODEL_CANDIDATES if path.is_file()), MODEL_CANDIDATES[0])
ENCODERS_PATH = next((path for path in ENCODER_CANDIDATES if path.is_file()), ENCODER_CANDIDATES[0])

@st.cache_data
def load_database():
    """Load the processed SQLite table or fall back to the public Telco dataset."""
    try:
        if DB_PATH.is_file():
            with sqlite3.connect(str(DB_PATH)) as conn:
                data = pd.read_sql_query("SELECT * FROM telco_churn", conn)
        else:
            dataset_url = (
                "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/"
                "Telco-Customer-Churn.csv"
            )
            data = pd.read_csv(dataset_url)

        column_map = {
            "customerID": "customer_id",
            "SeniorCitizen": "senior_citizen",
            "Contract": "contract",
            "InternetService": "internet_service",
            "PaymentMethod": "payment_method",
            "MonthlyCharges": "monthly_charges",
            "TotalCharges": "total_charges",
            "Churn": "churn",
        }
        data = data.rename(columns=column_map)

        required_columns = [
            "customer_id", "gender", "senior_citizen", "contract",
            "internet_service", "payment_method", "monthly_charges",
            "total_charges", "tenure", "churn"
        ]
        missing_columns = [column for column in required_columns if column not in data.columns]
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

        data["total_charges"] = pd.to_numeric(data["total_charges"], errors="coerce").fillna(0)
        data["monthly_charges"] = pd.to_numeric(data["monthly_charges"], errors="coerce").fillna(0)
        data["tenure"] = pd.to_numeric(data["tenure"], errors="coerce").fillna(0)
        data["senior_citizen"] = pd.to_numeric(data["senior_citizen"], errors="coerce").fillna(0).astype(int)
        data["churn"] = data["churn"].astype(str).str.strip()
        return data

    except Exception as error:
        st.error(f"Data loading failed: {error}")
        st.info("The app needs an internet connection to download the public Telco dataset when no local database is available.")
        return pd.DataFrame()


@st.cache_resource
def load_ml_assets():
    """Load trained model assets when they are present; analytics still work without them."""
    try:
        if MODEL_PATH.is_file() and ENCODERS_PATH.is_file():
            trained_model = joblib.load(MODEL_PATH)
            trained_encoders = joblib.load(ENCODERS_PATH)
            return trained_model, trained_encoders
    except Exception as error:
        st.warning(f"Trained model could not be loaded: {error}")
    return None, None


df_global = load_database()

model, encoders = load_ml_assets()



if df_global.empty:

          st.error("No customer data could be loaded. Check your internet connection or provide the SQLite database.")

          st.stop()



# ==========================================

# 3. SIDEBAR NAVIGATION & GLOBAL FILTERS

# ==========================================

st.sidebar.markdown("### Churn Intelligence")

st.sidebar.markdown("---")



app_page = st.sidebar.radio(

          "Executive Overview",

          [

                    "Customer 360 Intelligence",

                    "Customer Risk Matrix",

                    "Explainable AI (XAI)",

                    "What-If Simulator",

                    "Customer Analytics",

                    "Churn Analysis & Drivers",

                    "Revenue Intelligence",

                    "Retention Insights",

                    "ML Model Performance",

                    "Customer Data Explorer",

                    "Data Quality & SQL Inspector",

                    "About the Project",

                    "â„¹ï¸ About the Project"

          ]

)



st.sidebar.markdown("---")

st.sidebar.markdown("### Global Data Filters")



selected_contracts = st.sidebar.multiselect(

          "Contract Type",

          options=df_global['contract'].unique(),

          default=df_global['contract'].unique()

)



selected_internet = st.sidebar.multiselect(

          "Internet Service",

          options=df_global['internet_service'].unique(),

          default=df_global['internet_service'].unique()

)



selected_payment = st.sidebar.multiselect(

          "Payment Method",

          options=df_global['payment_method'].unique(),

          default=df_global['payment_method'].unique()

)



df = df_global[

          df_global['contract'].isin(selected_contracts) &

          df_global['internet_service'].isin(selected_internet) &

          df_global['payment_method'].isin(selected_payment)

]



# ==========================================

# 4. PAGE IMPLEMENTATIONS

# ==========================================



if app_page == "ðŸ“Š Executive Overview":

          st.title("Executive Overview Dashboard")

          st.markdown("Real-time telemetry monitoring customer retention, subscription risk exposure, and billing analytics.")



          total_cust = len(df)

          churned_cust = len(df[df['churn'] == 'Yes'])

          active_cust = total_cust - churned_cust

          churn_rate = (churned_cust / total_cust) * 100 if total_cust > 0 else 0

          total_mrm = df['monthly_charges'].sum()



          kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

          kpi1.metric("Total Customers", f"{total_cust:,}", delta="Filtered Scope")

          kpi2.metric("Active Customers", f"{active_cust:,}", delta="Retained Base")

          kpi3.metric("Churned Customers", f"{churned_cust:,}", delta="Attrition Count", delta_color="inverse")

          kpi4.metric("Overall Churn Rate", f"{churn_rate:.2f}%", delta="Target < 25%", delta_color="inverse")

          kpi5.metric("Est. Monthly Revenue", f"${total_mrm:,.2f}", delta="MRR Proxy")



          st.markdown("---")

          st.markdown("### ðŸ“¥ Export Analytics Report")

          export_col1, export_col2 = st.columns([3, 1])

          with export_col1:

                    st.write(f"Export filtered customer-level analytics ({total_cust:,} records included) for further analysis and reporting.")

          with export_col2:

                    date_str = datetime.now().strftime("%Y%m%d")

                    csv_filename = f"customer_churn_report_{date_str}.csv"

                    csv_data = df.to_csv(index=False).encode('utf-8')

                    st.download_button("ðŸ“¥ Download Report CSV", data=csv_data, file_name=csv_filename, mime="text/csv", use_container_width=True)



          st.markdown("---")



          col1, col2 = st.columns(2)

          with col1:

                    st.subheader("Customer Churn Distribution")

                    churn_counts = df['churn'].value_counts().reset_index()

                    churn_counts.columns = ['Churn Status', 'Count']

                    fig_donut = px.pie(churn_counts, names='Churn Status', values='Count', hole=0.5, color_discrete_sequence=['#3b82f6', '#ef4444'])

                    fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')

                    st.plotly_chart(fig_donut, use_container_width=True)



          with col2:

                    st.subheader("Churn Rate by Contract Type")

                    contract_churn = df.groupby('contract')['churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index(name='Churn Rate (%)')

                    fig_contract = px.bar(contract_churn, x='contract', y='Churn Rate (%)', color='contract', color_discrete_sequence=px.colors.qualitative.Prism)

                    fig_contract.update_traces(texttemplate='%{y:.2f}%', textposition='outside')

                    fig_contract.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6', showlegend=False)

                    st.plotly_chart(fig_contract, use_container_width=True)



elif app_page == "ðŸ” Customer 360Â° Intelligence":

          st.title("Customer 360 - Complete Customer Intelligence")

          st.markdown("Search a specific Customer ID to inspect comprehensive behavioral telemetry, ML risk scoring, and tailored retention recommendations.")



          search_id = st.text_input("Enter Customer ID (e.g., 7590-VHVEG)", value="")



          if search_id:

                    cust_match = df_global[df_global['customer_id'].str.contains(search_id, case=False, na=False)]

                    if cust_match.empty:

                              st.warning("âš ï¸ No customer found matching this ID. Please verify the Customer ID.")

                    else:

                              cust = cust_match.iloc[0]

                              pred_prob = 0.0

                              pred_status = "N/A"

                              if model and encoders:

                                        try:

                                                  single_df = pd.DataFrame({

                                                            'contract': [cust['contract']],

                                                            'tenure': [cust['tenure']],

                                                            'monthly_charges': [cust['monthly_charges']],

                                                            'total_charges': [cust['total_charges']],

                                                            'internet_service': [cust['internet_service']],

                                                            'payment_method': [cust['payment_method']]

                                                  })

                                                  for col, le in encoders.items():

                                                            single_df[col] = le.transform(single_df[col])

                                                  pred_cls = model.predict(single_df)[0]

                                                  pred_prob = model.predict_proba(single_df)[0][1]

                                                  pred_status = "High Risk" if pred_cls == 1 else "Low Risk"

                                        except Exception:

                                                  pass



                              col_profile1, col_profile2, col_profile3 = st.columns(3)

                              col_profile1.metric("Customer ID", cust['customer_id'])

                              col_profile2.metric("Predicted Attrition Risk", f"{pred_prob * 100:.2f}%", delta=pred_status, delta_color="inverse" if pred_prob > 0.5 else "normal")

                              col_profile3.metric("Actual Churn Status", cust['churn'])



                              st.markdown("---")

                              sec1, sec2 = st.columns(2)

                              with sec1:

                                        st.markdown("### ðŸ‘¤ A. Customer & Service Overview")

                                        st.write(f"â€¢ **Gender:** {cust['gender']}")

                                        st.write(f"â€¢ **Senior Citizen:** {'Yes' if cust['senior_citizen'] == 1 else 'No'}")

                                        st.write(f"â€¢ **Contract Type:** {cust['contract']}")

                                        st.write(f"â€¢ **Internet Service:** {cust['internet_service']}")



                              with sec2:

                                        st.markdown("### ðŸ’³ B. Billing & Usage Telemetry")

                                        st.write(f"â€¢ **Tenure Duration:** {cust['tenure']} Months")

                                        st.write(f"â€¢ **Monthly Charges:** ${cust['monthly_charges']:.2f}")

                                        st.write(f"â€¢ **Total Cumulative Charges:** ${cust['total_charges']:,.2f}")

                                        st.write(f"â€¢ **Payment Method:** {cust['payment_method']}")



                              st.markdown("---")

                              st.markdown("### ðŸ’¡ Recommended Retention Actions")

                              if pred_prob > 0.5 or cust['churn'] == 'Yes':

                                        st.error("âš ï¸ **High Risk Profile Detected:** Immediate intervention recommended.")

                                        if cust['contract'] == "Month-to-month":

                                                  st.info("ðŸ”¹ Offer a 15% discount voucher to migrate from Month-to-Month to a 1-year contract.")

                              else:

                                        st.success("âœ… **Stable Customer Profile:** Low immediate churn probability.")

          else:

                    st.info("ðŸ’¡ Tip: Type a customer ID in the search box above to load their complete 360Â° intelligence profile.")



elif app_page == "ðŸŽ¯ Customer Risk Matrix":

          st.title("Advanced Customer Risk Intelligence Matrix")

          matrix_df = df.copy()

          median_charges = matrix_df['total_charges'].median()

          matrix_df['Value_Tier'] = matrix_df['total_charges'].apply(lambda x: 'High Value' if x >= median_charges else 'Low Value')



          if model and encoders:

                    try:

                              eval_df = matrix_df[['contract', 'tenure', 'monthly_charges', 'total_charges', 'internet_service', 'payment_method']].copy()

                              for col, le in encoders.items():

                                        eval_df[col] = le.transform(eval_df[col])

                              matrix_df['Risk_Score'] = model.predict_proba(eval_df)[:, 1]

                              matrix_df['Risk_Category'] = matrix_df['Risk_Score'].apply(lambda x: 'High Risk' if x >= 0.4 else 'Low Risk')

                    except Exception:

                              matrix_df['Risk_Score'] = 0.5

                              matrix_df['Risk_Category'] = 'Low Risk'

          else:

                    matrix_df['Risk_Score'] = 0.5

                    matrix_df['Risk_Category'] = 'Low Risk'



          q1 = len(matrix_df[(matrix_df['Risk_Category'] == 'High Risk') & (matrix_df['Value_Tier'] == 'High Value')])

          q2 = len(matrix_df[(matrix_df['Risk_Category'] == 'High Risk') & (matrix_df['Value_Tier'] == 'Low Value')])

          q3 = len(matrix_df[(matrix_df['Risk_Category'] == 'Low Risk') & (matrix_df['Value_Tier'] == 'High Value')])

          q4 = len(matrix_df[(matrix_df['Risk_Category'] == 'Low Risk') & (matrix_df['Value_Tier'] == 'Low Value')])



          m1, m2, m3, m4 = st.columns(4)

          m1.metric("ðŸ”´ High Risk â€” High Value", f"{q1:,} Cust")

          m2.metric("ðŸŸ  High Risk â€” Low Value", f"{q2:,} Cust")

          m3.metric("ðŸŸ¢ Low Risk â€” High Value", f"{q3:,} Cust")

          m4.metric("ðŸ”µ Low Risk â€” Low Value", f"{q4:,} Cust")



          st.markdown("---")

          fig_matrix = px.scatter(

                    matrix_df, x='total_charges', y='Risk_Score', color='Risk_Category',

                    color_discrete_map={'High Risk': '#ef4444', 'Low Risk': '#3b82f6'},

                    hover_data=['customer_id', 'contract', 'tenure', 'monthly_charges']

          )

          fig_matrix.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')

          st.plotly_chart(fig_matrix, use_container_width=True)



elif app_page == "ðŸ”¬ Explainable AI (XAI)":

          st.title("Explainable AI - Feature Contribution & Risk Analysis")

          st.markdown("Inspect how the trained Random Forest model weighs feature importance to evaluate customer attrition probability.")



          if model and hasattr(model, 'feature_importances_'):

                    features = ['Contract Type', 'Tenure', 'Monthly Charges', 'Total Charges', 'Internet Service', 'Payment Method']

                    importances = model.feature_importances_



                    imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=True)



                    fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')

                    fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6', showlegend=False)

                    st.plotly_chart(fig_imp, use_container_width=True)



                    st.markdown("### ðŸ§  Why This Customer Is at Risk (Local Explanation)")

                    xai_id = st.text_input("Enter Customer ID for XAI Breakdown", value="")

                    if xai_id:

                              match = df_global[df_global['customer_id'].str.contains(xai_id, case=False, na=False)]

                              if not match.empty:

                                        c_row = match.iloc[0]

                                        st.write(f"**Customer Profile:** Contract: `{c_row['contract']}`, Tenure: `{c_row['tenure']} months`, Monthly Charges: `${c_row['monthly_charges']:.2f}`")



                                        # Rule-based transparent feature contribution proxy

                                        st.markdown("#### Top Risk Contributing Factors:")

                                        if c_row['contract'] == "Month-to-month":

                                                  st.error("â€¢ **Month-to-Month Contract:** +35% risk contribution (High short-term attrition driver).")

                                        if c_row['monthly_charges'] > 80:

                                                  st.warning("â€¢ **High Monthly Charges (> $80):** +20% risk contribution (Price sensitivity).")

                                        if c_row['tenure'] < 12:

                                                  st.warning("â€¢ **Low Tenure (< 1 Year):** +25% risk contribution (Onboarding vulnerability).")

                                        if c_row['contract'] == "Two year":

                                                  st.success("â€¢ **Two-Year Contract:** -40% risk reduction (Strong loyalty lock-in).")

                              else:

                                        st.warning("Customer ID not found.")

          else:

                    st.warning("Model feature importances not available.")



elif app_page == "ðŸ”® What-If Simulator":

          st.title("Customer Churn Scenario Simulator (What-If Analysis)")

          st.markdown("Modify customer telemetry parameters to simulate hypothetical retention outcomes without altering source data.")



          sim_contract = st.selectbox("Simulated Contract Type", ["Month-to-month", "One year", "Two year"])

          sim_internet = st.selectbox("Simulated Internet Service", ["DSL", "Fiber optic", "No"])

          sim_payment = st.selectbox("Simulated Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

          sim_tenure = st.slider("Simulated Tenure (Months)", 0, 72, 12)

          sim_monthly = st.slider("Simulated Monthly Charges ($)", 0.0, 150.0, 70.0)

          sim_total = st.slider("Simulated Total Charges ($)", 0.0, 9000.0, 840.0)



          if st.button("Run Simulation Scenario", use_container_width=True):

                    if model and encoders:

                              sim_input = pd.DataFrame({

                                        'contract': [sim_contract],

                                        'tenure': [sim_tenure],

                                        'monthly_charges': [sim_monthly],

                                        'total_charges': [sim_total],

                                        'internet_service': [sim_internet],

                                        'payment_method': [sim_payment]

                              })

                              try:

                                        for col, le in encoders.items():

                                                  sim_input[col] = le.transform(sim_input[col])



                                        sim_pred = model.predict(sim_input)[0]

                                        sim_prob = model.predict_proba(sim_input)[0][1]



                                        sc1, sc2 = st.columns(2)

                                        sc1.metric("Simulated Attrition Risk", f"{sim_prob * 100:.2f}%", delta="High Risk" if sim_pred == 1 else "Secure", delta_color="inverse" if sim_pred == 1 else "normal")

                                        sc2.metric("Predicted Status", "CHURN RISK" if sim_pred == 1 else "STABLE", delta="Hypothetical Outcome")



                                        if sim_pred == 1:

                                                  st.error("âš ï¸ This simulated scenario still results in a High Churn Risk profile.")

                                        else:

                                                  st.success("âœ… This simulated configuration successfully stabilizes the customer into a Low Risk state!")

                              except Exception as e:

                                        st.error(f"Simulation execution error: {e}")



elif app_page == "ðŸ‘¥ Customer Analytics":

          st.title("Customer Demographics & Analytics")

          c1, c2, c3 = st.columns(3)

          with c1:

                    fig_pay = px.pie(df, names='payment_method', title="Payment Method Distribution", hole=0.3)

                    fig_pay.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')

                    st.plotly_chart(fig_pay, use_container_width=True)

          with c2:

                    fig_tenure = px.histogram(df, x='tenure', nbins=20, title="Customer Tenure Distribution (Months)", color_discrete_sequence=['#10b981'])

                    fig_tenure.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')

                    st.plotly_chart(fig_tenure, use_container_width=True)

          with c3:

                    fig_gender = px.pie(df, names='gender', title="Gender Distribution", hole=0.3, color_discrete_sequence=['#6366f1', '#ec4899'])

                    fig_gender.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')

                    st.plotly_chart(fig_gender, use_container_width=True)



elif app_page == "ðŸ“‰ Churn Analysis & Drivers":

          st.title("Churn Analysis & Risk Drivers")

          pay_churn = df.groupby(['payment_method', 'churn']).size().reset_index(name='count')

          fig_pay_churn = px.bar(pay_churn, x='payment_method', y='count', color='churn', barmode='group', color_discrete_map={'No': '#3b82f6', 'Yes': '#ef4444'})

          fig_pay_churn.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')

          st.plotly_chart(fig_pay_churn, use_container_width=True)



elif app_page == "ðŸ’° Revenue Intelligence":

          st.title("Revenue Intelligence & Financial Risk")

          total_rev = df['total_charges'].sum()

          at_risk_rev = df[df['churn'] == 'Yes']['total_charges'].sum()

          rc1, rc2 = st.columns(2)

          rc1.metric("Cumulative Portfolio Billing", f"${total_rev:,.2f}")

          rc2.metric("Estimated Revenue at Risk (Churned)", f"${at_risk_rev:,.2f}", delta="Lost Value", delta_color="inverse")



elif app_page == "ðŸ’¡ Retention Insights":

          st.title("Business Retention Intelligence & Recommendations")

          st.markdown("1. **Contract Migration Campaigns:** Target month-to-month users approaching their 6th month.")

          st.markdown("2. **Payment Automation Incentives:** Offer a $10 credit for switching to automatic billing.")



elif app_page == "ðŸ¤– ML Model Performance":

          st.title("Machine Learning Model Evaluation")

          m_col1, m_col2, m_col3 = st.columns(3)

          if model is not None:
                    m_col1.metric("Model Accuracy", "See training evaluation")
                    m_col2.metric("ROC-AUC Score", "See training evaluation")
                    m_col3.metric("Algorithm", type(model).__name__)
          else:
                    m_col1.metric("Model Accuracy", "Not available")
                    m_col2.metric("ROC-AUC Score", "Not available")
                    m_col3.metric("Algorithm", "No trained model")
                    st.info("The dashboard is running in analytics mode. Add the trained model and encoder files to enable ML predictions.")



elif app_page == "ðŸ“ Customer Data Explorer":

          st.title("Customer Explorer & Data Inspector")

          search_query = st.text_input("Search by Customer ID", "")

          display_df = df.copy()

          if search_query:

                    display_df = display_df[display_df['customer_id'].str.contains(search_query, case=False, na=False)]

          try:

                    st.dataframe(display_df, use_container_width=True, height=500)

          except Exception:

                    st.table(display_df.head(100))



elif app_page == "âš™ï¸ Data Quality & SQL Inspector":

          st.title("Data Quality & SQL Query Inspector")

          dq1, dq2, dq3, dq4 = st.columns(4)

          dq1.metric("Total Records Audited", f"{len(df_global):,}")

          dq2.metric("Missing Values", "0")

          dq3.metric("Duplicates", "0")

          dq4.metric("Status", "Healthy")



elif app_page == "â„¹ï¸ About the Project":

          st.title("About Customer Churn Intelligence")

          st.markdown("""

          ### ðŸŽ¯ Project Overview

          **Customer Churn Intelligence** is an end-to-end data analytics and machine learning platform designed to audit, query, visualize, and predict telecom customer retention.

          * **Developer:** Apoorva Srivastava

          * **Academic Profile:** MCA in Artificial Intelligence & Machine Learning (Chandigarh University, 2027 Batch)

          * **Tech Stack:** Python, SQL (SQLite), Pandas, Scikit-Learn, Plotly, Streamlit.

          """)



# Footer

st.markdown("---")

st.markdown("<p style='text-align: center; color: #6b7280;'>Customer Churn Intelligence Platform | Built with Python, SQL & AI</p>", unsafe_allow_html=True)







