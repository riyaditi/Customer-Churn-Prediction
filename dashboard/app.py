"""
Professional Customer Churn Prediction Dashboard
Built with Streamlit for real-time predictions and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Page configuration
st.set_page_config(
    page_title="Churn Predictor Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .high-risk {
        background: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%);
    }
    .medium-risk {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    .low-risk {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
</style>
""", unsafe_allow_html=True)

# Define paths
DATA_PATH = os.path.join(project_root, 'data')
PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')
RAW_PATH = os.path.join(DATA_PATH, 'raw')
MODELS_PATH = os.path.join(project_root, 'models')
REPORTS_PATH = os.path.join(project_root, 'reports')
FIGURES_PATH = os.path.join(REPORTS_PATH, 'figures')

# Load data and models
@st.cache_data
def load_data():
    """Load processed datasets"""
    try:
        X_test = pd.read_csv(os.path.join(PROCESSED_PATH, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(PROCESSED_PATH, 'y_test.csv')).squeeze()
        feature_names = pd.read_csv(os.path.join(PROCESSED_PATH, 'feature_names.csv')).squeeze().tolist()
        
        # Load sample of original data for EDA
        original_data = pd.read_csv(os.path.join(RAW_PATH, 'Telco-Customer-Churn.csv'))
        
        return X_test, y_test, feature_names, original_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Find the latest model
        model_files = [f for f in os.listdir(MODELS_PATH) if f.endswith('.pkl') and 'scaler' not in f]
        if not model_files:
            return None, None, None
        
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(MODELS_PATH, x)))
        model = joblib.load(os.path.join(MODELS_PATH, latest_model))
        
        # Load metadata
        metadata_path = os.path.join(MODELS_PATH, 'model_metadata_latest.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = None
        
        # Load scaler
        scaler_files = [f for f in os.listdir(MODELS_PATH) if 'scaler' in f]
        if scaler_files:
            latest_scaler = max(scaler_files, key=lambda x: os.path.getctime(os.path.join(MODELS_PATH, x)))
            scaler = joblib.load(os.path.join(MODELS_PATH, latest_scaler))
        else:
            scaler = None
        
        return model, metadata, scaler
    except Exception as e:
        st.warning(f"Model not found: {e}")
        return None, None, None

# Load all data
X_test, y_test, feature_names, original_data = load_data()
model, metadata, scaler = load_model()

# Header
st.markdown("<h1 class='main-header'>📊 Customer Churn Prediction Pro</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    Enterprise-grade predictive analytics for customer retention
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/analytics.png", width=100)
    st.markdown("<h2 style='text-align: center;'>Navigation</h2>", unsafe_allow_html=True)
    
    page = st.radio(
        "Select Page",
        ["🏠 Dashboard Home", "📈 Exploratory Analysis", "🤖 Predict Customer", "📊 Model Performance"]
    )
    
    st.markdown("---")
    
    if model is not None and metadata is not None:
        st.success(f"✅ Model Loaded: {metadata['model_info']['model_name']}")
        st.info(f"🎯 F1-Score: {metadata['performance']['f1_score']:.3f}")
        st.info(f"📊 ROC-AUC: {metadata['performance']['roc_auc']:.3f}")
    elif model is not None:
        st.success("✅ Model Loaded")
    else:
        st.warning("⚠️ No model found. Train a model first.")
    
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("This dashboard predicts customer churn using machine learning to help businesses retain valuable customers.")

# Dashboard Home Page
if page == "🏠 Dashboard Home":
    st.markdown("<h2 class='sub-header'>📊 Key Metrics</h2>", unsafe_allow_html=True)
    
    if original_data is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(original_data)
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Total Customers</h3>
                <h2>{total_customers:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            churn_rate = (original_data['Churn'] == 'Yes').mean() * 100
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Churn Rate</h3>
                <h2>{churn_rate:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_tenure = original_data['tenure'].mean()
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Avg Tenure</h3>
                <h2>{avg_tenure:.1f} months</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_charges = original_data['MonthlyCharges'].mean()
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Avg Monthly Charges</h3>
                <h2>${avg_charges:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Key Insights
        st.markdown("<h2 class='sub-header'>🔍 Key Insights</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.markdown("### 📈 Churn by Contract Type")
            contract_churn = pd.crosstab(original_data['Contract'], original_data['Churn'], normalize='index') * 100
            contract_churn = contract_churn.reset_index()
            
            fig = px.bar(
                contract_churn,
                x='Contract',
                y='Yes',
                title='Churn Rate by Contract Type',
                color='Yes',
                color_continuous_scale='Reds',
                labels={'Yes': 'Churn Rate (%)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Extract insights
            month_to_month = contract_churn[contract_churn['Contract'] == 'Month-to-month']['Yes'].values[0]
            two_year = contract_churn[contract_churn['Contract'] == 'Two year']['Yes'].values[0] if len(contract_churn[contract_churn['Contract'] == 'Two year']) > 0 else 0
            
            st.markdown(f"""
            **Business Insight:** 
            - Month-to-month contracts: **{month_to_month:.1f}%** churn rate
            - Two-year contracts: **{two_year:.1f}%** churn rate
            - Customers on month-to-month are **{month_to_month/two_year:.1f}x** more likely to churn
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.markdown("### 💳 Churn by Payment Method")
            payment_churn = pd.crosstab(original_data['PaymentMethod'], original_data['Churn'], normalize='index') * 100
            payment_churn = payment_churn.reset_index().sort_values('Yes', ascending=False)
            
            fig = px.bar(
                payment_churn,
                x='PaymentMethod',
                y='Yes',
                title='Churn Rate by Payment Method',
                color='Yes',
                color_continuous_scale='Reds',
                labels={'Yes': 'Churn Rate (%)'}
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Extract insights
            electronic_check = payment_churn[payment_churn['PaymentMethod'] == 'Electronic check']['Yes'].values[0] if len(payment_churn[payment_churn['PaymentMethod'] == 'Electronic check']) > 0 else 0
            
            st.markdown(f"""
            **Business Insight:** 
            - Electronic check users: **{electronic_check:.1f}%** churn rate
            - Highest risk payment method - target for retention campaigns!
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        #        with col1:
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.markdown("### 👥 Churn by Senior Citizen Status")
            
            # Fix for the Senior Citizen chart
            senior_data = original_data.copy()
            senior_data['Customer Type'] = senior_data['SeniorCitizen'].map({0: 'Non-Senior', 1: 'Senior'})
            senior_churn = pd.crosstab(senior_data['Customer Type'], senior_data['Churn'], normalize='index') * 100
            senior_churn = senior_churn.reset_index()
            
            fig = px.bar(
                senior_churn,
                x='Customer Type',
                y='Yes',
                title='Churn Rate by Customer Type',
                color='Yes',
                color_continuous_scale='Reds',
                labels={'Yes': 'Churn Rate (%)', 'Customer Type': ''}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights
            senior_rate = senior_churn[senior_churn['Customer Type'] == 'Senior']['Yes'].values[0] if len(senior_churn[senior_churn['Customer Type'] == 'Senior']) > 0 else 0
            non_senior_rate = senior_churn[senior_churn['Customer Type'] == 'Non-Senior']['Yes'].values[0] if len(senior_churn[senior_churn['Customer Type'] == 'Non-Senior']) > 0 else 0
            
            st.markdown(f"""
            **Business Insight:** 
            - Senior citizens: **{senior_rate:.1f}%** churn rate
            - Non-seniors: **{non_senior_rate:.1f}%** churn rate
            - Seniors are **{senior_rate/non_senior_rate:.1f}x** more likely to churn
            """)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.markdown("### 📱 Internet Service Impact")
            
            # Fix for Internet Service chart
            internet_churn = pd.crosstab(original_data['InternetService'], original_data['Churn'], normalize='index') * 100
            internet_churn = internet_churn.reset_index()
            internet_churn.columns = ['InternetService', 'No', 'Yes']  # Rename columns for clarity
            
            fig = px.bar(
                internet_churn,
                x='InternetService',
                y='Yes',
                title='Churn Rate by Internet Service',
                color='Yes',
                color_continuous_scale='Reds',
                labels={'Yes': 'Churn Rate (%)', 'InternetService': 'Internet Service Type'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights
            fiber_rate = internet_churn[internet_churn['InternetService'] == 'Fiber optic']['Yes'].values[0] if len(internet_churn[internet_churn['InternetService'] == 'Fiber optic']) > 0 else 0
            dsl_rate = internet_churn[internet_churn['InternetService'] == 'DSL']['Yes'].values[0] if len(internet_churn[internet_churn['InternetService'] == 'DSL']) > 0 else 0
            
            st.markdown(f"""
            **Business Insight:** 
            - Fiber optic: **{fiber_rate:.1f}%** churn rate
            - DSL: **{dsl_rate:.1f}%** churn rate
            - Fiber optic customers need additional support/services
            """)
            st.markdown("</div>", unsafe_allow_html=True)

# Exploratory Analysis Page
elif page == "📈 Exploratory Analysis":
    st.markdown("<h2 class='sub-header'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    
    if original_data is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution Analysis", "📈 Correlation", "🔍 Service Analysis", "📉 Customer Segments"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    original_data,
                    x='tenure',
                    color='Churn',
                    nbins=50,
                    title='Tenure Distribution by Churn Status',
                    color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'},
                    labels={'tenure': 'Tenure (months)', 'count': 'Number of Customers'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.markdown("**Tenure Statistics:**")
                churn_yes = original_data[original_data['Churn'] == 'Yes']['tenure']
                churn_no = original_data[original_data['Churn'] == 'No']['tenure']
                
                col1a, col1b = st.columns(2)
                with col1a:
                    st.metric("Avg Tenure (Churn)", f"{churn_yes.mean():.1f} months")
                    st.metric("Median Tenure (Churn)", f"{churn_yes.median():.1f} months")
                with col1b:
                    st.metric("Avg Tenure (No Churn)", f"{churn_no.mean():.1f} months")
                    st.metric("Median Tenure (No Churn)", f"{churn_no.median():.1f} months")
            
            with col2:
                fig = px.histogram(
                    original_data,
                    x='MonthlyCharges',
                    color='Churn',
                    nbins=50,
                    title='Monthly Charges Distribution by Churn',
                    color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'},
                    labels={'MonthlyCharges': 'Monthly Charges ($)', 'count': 'Number of Customers'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.markdown("**Monthly Charges Statistics:**")
                charges_yes = original_data[original_data['Churn'] == 'Yes']['MonthlyCharges']
                charges_no = original_data[original_data['Churn'] == 'No']['MonthlyCharges']
                
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("Avg Charges (Churn)", f"${charges_yes.mean():.2f}")
                    st.metric("Median Charges (Churn)", f"${charges_yes.median():.2f}")
                with col2b:
                    st.metric("Avg Charges (No Churn)", f"${charges_no.mean():.2f}")
                    st.metric("Median Charges (No Churn)", f"${charges_no.median():.2f}")
        
        with tab2:
            # Correlation matrix
            st.markdown("### Feature Correlation Matrix")
            
            # Prepare data for correlation
            df_corr = original_data.copy()
            for col in df_corr.select_dtypes(['object']).columns:
                if col != 'customerID':
                    df_corr[col] = pd.Categorical(df_corr[col]).codes
            
            # Select numerical columns
            numeric_cols = df_corr.select_dtypes(include=[np.number]).columns
            corr_matrix = df_corr[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title='Feature Correlation Matrix',
                height=800
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations with churn
            st.markdown("### Top Correlations with Churn")
            churn_corr = corr_matrix['Churn'].sort_values(ascending=False)
            churn_corr_df = pd.DataFrame({
                'Feature': churn_corr.index,
                'Correlation': churn_corr.values
            }).iloc[1:11]  # Skip Churn itself
            
            fig = px.bar(
                churn_corr_df,
                x='Correlation',
                y='Feature',
                orientation='h',
                title='Top 10 Features Correlated with Churn',
                color='Correlation',
                color_continuous_scale='RdBu_r',
                range_color=[-1, 1]
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Service analysis
            service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies']
            
            service_analysis = []
            for col in service_cols:
                if col in original_data.columns:
                    for val in original_data[col].unique():
                        if pd.notna(val):
                            subset = original_data[original_data[col] == val]
                            churn_rate = (subset['Churn'] == 'Yes').mean() * 100
                            count = len(subset)
                            service_analysis.append({
                                'Service': f"{col} - {val}",
                                'Churn Rate': churn_rate,
                                'Count': count,
                                'Category': col
                            })
            
            service_df = pd.DataFrame(service_analysis)
            service_df = service_df.sort_values('Churn Rate', ascending=False)
            
            # Top 15 highest churn service categories
            fig = px.bar(
                service_df.head(15),
                x='Churn Rate',
                y='Service',
                color='Churn Rate',
                color_continuous_scale='Reds',
                title='Top 15 Service Categories with Highest Churn',
                hover_data=['Count']
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Service impact table
            st.markdown("### Service Impact Summary")
            service_summary = service_df.groupby('Category').agg({
                'Churn Rate': ['mean', 'max', 'min'],
                'Count': 'sum'
            }).round(2)
            service_summary.columns = ['Avg Churn %', 'Max Churn %', 'Min Churn %', 'Total Customers']
            st.dataframe(service_summary, use_container_width=True)
        
        with tab4:
            # Create customer segments
            original_data['Tenure Segment'] = pd.cut(
                original_data['tenure'],
                bins=[-1, 6, 12, 24, 48, 100],
                labels=['0-6 months', '6-12 months', '1-2 years', '2-4 years', '4+ years']
            )
            
            original_data['Charge Segment'] = pd.cut(
                original_data['MonthlyCharges'],
                bins=[-1, 30, 60, 90, 120],
                labels=['Low (<$30)', 'Medium ($30-60)', 'High ($60-90)', 'Very High (>$90)']
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                segment_analysis = original_data.groupby('Tenure Segment', observed=True).agg({
                    'Churn': lambda x: (x == 'Yes').mean() * 100,
                    'customerID': 'count',
                    'MonthlyCharges': 'mean'
                }).round(2).reset_index()
                segment_analysis.columns = ['Segment', 'Churn Rate %', 'Count', 'Avg Monthly Charges']
                
                fig = px.bar(
                    segment_analysis,
                    x='Segment',
                    y='Churn Rate %',
                    color='Churn Rate %',
                    color_continuous_scale='Reds',
                    title='Churn Rate by Tenure Segment',
                    text='Churn Rate %'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(segment_analysis, use_container_width=True)
            
            with col2:
                charge_analysis = original_data.groupby('Charge Segment', observed=True).agg({
                    'Churn': lambda x: (x == 'Yes').mean() * 100,
                    'customerID': 'count'
                }).round(2).reset_index()
                charge_analysis.columns = ['Segment', 'Churn Rate %', 'Count']
                
                fig = px.bar(
                    charge_analysis,
                    x='Segment',
                    y='Churn Rate %',
                    color='Churn Rate %',
                    color_continuous_scale='Reds',
                    title='Churn Rate by Monthly Charge Segment',
                    text='Churn Rate %'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(charge_analysis, use_container_width=True)

# Predict Customer Page
elif page == "🤖 Predict Customer":
    st.markdown("<h2 class='sub-header'>🤖 Predict Customer Churn</h2>", unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ No trained model found. Please train a model first.")
        st.info("Run the model training notebook to create and save a model.")
    else:
        st.markdown("### Enter Customer Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📝 Account Information")
            
            tenure = st.slider("Tenure (months)", 0, 72, 12, help="How long the customer has been with the company")
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0)
            total_charges = monthly_charges * tenure
            
            contract = st.selectbox("Contract Type", 
                                   ["Month-to-month", "One year", "Two year"],
                                   help="Type of contract the customer has")
            
            payment_method = st.selectbox("Payment Method", 
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            
            paperless_billing = st.checkbox("Paperless Billing", True)
            
        with col2:
            st.markdown("#### 📱 Services Subscribed")
            
            phone_service = st.checkbox("Phone Service", True)
            if phone_service:
                multiple_lines = st.checkbox("Multiple Lines")
            else:
                multiple_lines = False
            
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            
            if internet_service != "No":
                st.markdown("**Internet Services:**")
                online_security = st.checkbox("Online Security")
                online_backup = st.checkbox("Online Backup")
                device_protection = st.checkbox("Device Protection")
                tech_support = st.checkbox("Tech Support")
                streaming_tv = st.checkbox("Streaming TV")
                streaming_movies = st.checkbox("Streaming Movies")
            else:
                online_security = online_backup = device_protection = tech_support = streaming_tv = streaming_movies = False
            
            st.markdown("#### 👤 Demographics")
            senior_citizen = st.checkbox("Senior Citizen")
            partner = st.checkbox("Has Partner")
            dependents = st.checkbox("Has Dependents")
        
        # Predict button
        if st.button("🔮 Predict Churn Probability", type="primary"):
            # Create input dataframe
            input_data = {
                'tenure': tenure,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
                'SeniorCitizen': 1 if senior_citizen else 0,
                'Partner_Yes': 1 if partner else 0,
                'Dependents_Yes': 1 if dependents else 0,
                'PhoneService_Yes': 1 if phone_service else 0,
                'MultipleLines_Yes': 1 if multiple_lines else 0,
                'InternetService_DSL': 1 if internet_service == "DSL" else 0,
                'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
                'InternetService_No': 1 if internet_service == "No" else 0,
                'OnlineSecurity_Yes': 1 if online_security else 0,
                'OnlineBackup_Yes': 1 if online_backup else 0,
                'DeviceProtection_Yes': 1 if device_protection else 0,
                'TechSupport_Yes': 1 if tech_support else 0,
                'StreamingTV_Yes': 1 if streaming_tv else 0,
                'StreamingMovies_Yes': 1 if streaming_movies else 0,
                'Contract_Month-to-month': 1 if contract == "Month-to-month" else 0,
                'Contract_One year': 1 if contract == "One year" else 0,
                'Contract_Two year': 1 if contract == "Two year" else 0,
                'PaperlessBilling_Yes': 1 if paperless_billing else 0,
                'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
                'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
                'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == "Bank transfer (automatic)" else 0,
                'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure all columns match training data
            if feature_names is not None:
                for col in feature_names:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                # Reorder columns
                input_df = input_df[feature_names]
            
            # Make prediction
            try:
                proba = model.predict_proba(input_df)[0][1]
                prediction = model.predict(input_df)[0]
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    proba_percent = proba * 100
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Churn Probability</h3>
                        <h2>{proba_percent:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if proba > 0.7:
                        risk_level = "High"
                        risk_class = "high-risk"
                        risk_color = "#f43f5e"
                    elif proba > 0.3:
                        risk_level = "Medium"
                        risk_class = "medium-risk"
                        risk_color = "#f59e0b"
                    else:
                        risk_level = "Low"
                        risk_class = "low-risk"
                        risk_color = "#10b981"
                    
                    st.markdown(f"""
                    <div class='metric-card' style='background: {risk_color};'>
                        <h3>Risk Level</h3>
                        <h2>{risk_level}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    if proba > 0.7:
                        action = "🚨 Immediate Retention Needed"
                    elif proba > 0.3:
                        action = "📊 Monitor Closely"
                    else:
                        action = "✅ Low Priority"
                    
                    st.markdown(f"""
                    <div class='metric-card' style='background: #3B82F6;'>
                        <h3>Recommended Action</h3>
                        <h4>{action}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Prediction gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = proba * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Risk Score", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 30], 'color': '#10b981'},
                            {'range': [30, 70], 'color': '#f59e0b'},
                            {'range': [70, 100], 'color': '#f43f5e'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': proba * 100
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations based on risk factors
                st.markdown("### 💡 Retention Recommendations")
                
                recommendations = []
                if tenure < 12:
                    recommendations.append("• **New customer**: Offer loyalty incentives and onboarding support")
                if contract == "Month-to-month":
                    recommendations.append("• **Month-to-month contract**: Promote annual contract with discount")
                if payment_method == "Electronic check":
                    recommendations.append("• **Electronic check user**: Incentivize switching to automatic payment")
                if internet_service == "Fiber optic" and not tech_support:
                    recommendations.append("• **Fiber optic without tech support**: Bundle tech support trial")
                if not online_security and internet_service != "No":
                    recommendations.append("• **No online security**: Highlight security benefits and offer free trial")
                if monthly_charges > 80:
                    recommendations.append("• **High monthly charges**: Review plan suitability and offer optimization")
                
                if recommendations:
                    for rec in recommendations:
                        st.markdown(rec)
                else:
                    st.markdown("• Customer shows low risk factors. Maintain regular engagement.")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("Please check that the model and features are compatible.")

# Model Performance Page
elif page == "📊 Model Performance":
    st.markdown("<h2 class='sub-header'>📊 Model Performance Dashboard</h2>", unsafe_allow_html=True)
    
    if metadata is None:
        st.warning("No model metadata found. Train a model first or check the models folder.")
        
        # Try to load from latest model directly
        if model is not None:
            st.info("Model loaded but no metadata available. Displaying basic information.")
            st.json({
                "model_type": str(type(model).__name__),
                "features": len(feature_names) if feature_names else "Unknown"
            })
    else:
        # Model metrics
        st.markdown("### 📈 Performance Metrics")
        
        metrics = metadata['performance']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}", 
                     delta=None, delta_color="normal")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
        with col5:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MCC", f"{metrics.get('mcc', 0):.3f}")
        with col2:
            st.metric("Kappa", f"{metrics.get('kappa', 0):.3f}")
        with col3:
            st.metric("Log Loss", f"{metrics.get('log_loss', 0):.3f}")
        
        st.markdown("---")
        
        # Confusion Matrix
        st.markdown("### 📊 Confusion Matrix")
        
        # Try to load confusion matrix from figures
        cm_path = os.path.join(FIGURES_PATH, 'confusion_matrices.png')
        if os.path.exists(cm_path):
            st.image(cm_path, use_column_width=True)
        else:
            # Display sample confusion matrix
            st.info("Confusion matrix image not found. Run model training to generate it.")
        
        # Feature Importance
        st.markdown("### 🔑 Feature Importance")
        
        # Try to load feature importance from metadata
        if 'feature_importance' in metadata and metadata['feature_importance']:
            feature_df = pd.DataFrame(metadata['feature_importance']).head(15)
            
            fig = px.bar(
                feature_df,
                y='feature',
                x='importance',
                orientation='h',
                title='Top 15 Feature Importances',
                color='importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.markdown("### 📋 Feature Importance Table")
            st.dataframe(feature_df, use_container_width=True)
        else:
            # Try to load from CSV
            csv_path = os.path.join(FIGURES_PATH, 'shap_feature_importance.csv')
            if os.path.exists(csv_path):
                feature_df = pd.read_csv(csv_path).head(15)
                fig = px.bar(
                    feature_df,
                    y='feature',
                    x='importance',
                    orientation='h',
                    title='Top 15 Feature Importances (from SHAP)',
                    color='importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(feature_df, use_container_width=True)
            else:
                st.info("Feature importance data not available. Run training with SHAP analysis.")
        
        # Model Information
        st.markdown("### ℹ️ Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Details:**")
            model_info = metadata.get('model_info', {})
            st.json({
                "Model Name": model_info.get('model_name', 'Unknown'),
                "Model Type": model_info.get('model_type', 'Unknown'),
                "Training Date": model_info.get('timestamp', 'Unknown'),
            })
        
        with col2:
            st.markdown("**Data Information:**")
            data_info = metadata.get('data_info', {})
            st.json({
                "Number of Features": data_info.get('n_features', 'Unknown'),
                "Training Samples": data_info.get('n_train_samples', 'Unknown'),
                "Test Samples": data_info.get('n_test_samples', 'Unknown'),
                "Churn Rate (Train)": f"{data_info.get('churn_rate_train', 0)*100:.1f}%",
                "Churn Rate (Test)": f"{data_info.get('churn_rate_test', 0)*100:.1f}%"
            })

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>"
    "Customer Churn Prediction Pro | Built with Streamlit | Data Analytics Internship Project | Codec Technologies"
    "</div>",
    unsafe_allow_html=True
)