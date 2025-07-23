# ====================================================================================================
# 📊 PRODUCTION-READY STREAMLIT FORECASTING APP – ERP SPEND PREDICTOR
# ----------------------------------------------------------------------------------------------------
# Integrates with trained ML models from training pipeline
# Features: Real-time PPI, confidence intervals, model performance tracking
# ====================================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
from fredapi import Fred
from scipy import stats
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================
# 🔧 CONFIGURATION & SETUP
# ============================================================
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

# PPI mapping from data preparation
PPI_SERIES_MAP = {
    'Office Supplies': 'WPU0911',
    'Packaging': 'WPU091', 
    'MRO': 'WPU114',
    'Raw Materials': 'WPU061',
    'Electronics': 'WPU117',
    'Chemicals': 'WPU065',
    'Services': 'WPU381',
    'Metals': 'WPU101',
    'Manufacturing': 'WPU114',
    'Food Products': 'WPU012'
}

# Enhanced business intelligence based on procurement analytics
CATEGORY_INSIGHTS = {
    'Raw Materials': {'volatility': 'High', 'lead_time_sensitivity': 'High', 'seasonal_impact': 'Medium'},
    'Chemicals': {'volatility': 'High', 'lead_time_sensitivity': 'Medium', 'seasonal_impact': 'Low'},
    'Metals': {'volatility': 'Very High', 'lead_time_sensitivity': 'High', 'seasonal_impact': 'Medium'},
    'Electronics': {'volatility': 'Medium', 'lead_time_sensitivity': 'High', 'seasonal_impact': 'High'},
    'Food Products': {'volatility': 'High', 'lead_time_sensitivity': 'Medium', 'seasonal_impact': 'High'},
    'Office Supplies': {'volatility': 'Low', 'lead_time_sensitivity': 'Low', 'seasonal_impact': 'Medium'},
    'Packaging': {'volatility': 'Medium', 'lead_time_sensitivity': 'Medium', 'seasonal_impact': 'Medium'},
    'MRO': {'volatility': 'Medium', 'lead_time_sensitivity': 'Medium', 'seasonal_impact': 'Low'},
    'Manufacturing': {'volatility': 'Medium', 'lead_time_sensitivity': 'High', 'seasonal_impact': 'Medium'},
    'Services': {'volatility': 'Low', 'lead_time_sensitivity': 'Low', 'seasonal_impact': 'Low'}
}
# ============================================================
# 🔄 MODEL LOADING WITH ERROR HANDLING
# ============================================================
@st.cache_resource
def load_trained_model():
    """Load trained model artifacts with comprehensive error handling"""
    try:
        # Load model artifacts
        model = joblib.load("models/best_model.joblib")
        scaler = joblib.load("models/feature_scaler.joblib")
        features = joblib.load("models/feature_names.joblib")
        metadata = joblib.load("models/model_metadata.joblib")
        
        return {
            'model': model,
            'scaler': scaler,
            'features': features,
            'metadata': metadata,
            'status': 'success'
        }
    except FileNotFoundError as e:
        return {
            'status': 'error',
            'message': f"Model files not found: {str(e)}. Please run the training script first."
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error loading model: {str(e)}"
        }

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_category_ppi(category, months_back=6):
    """Fetch category-specific PPI data using exact mapping"""
    if not FRED_API_KEY:
        st.warning("⚠️ FRED API key not configured. Using estimated PPI values.")
        return pd.DataFrame()
    
    series_id = PPI_SERIES_MAP.get(category)
    if not series_id:
        st.warning(f"⚠️ No PPI series found for {category}")
        return pd.DataFrame()
    
    try:
        fred = Fred(api_key=FRED_API_KEY)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back*30)
        
        # Fetch data with proper date range
        ppi_data = fred.get_series(
            series_id, 
            start=start_date.strftime('%Y-%m-%d'), 
            end=end_date.strftime('%Y-%m-%d')
        )
        
        if ppi_data.empty:
            return pd.DataFrame()
            
        df = ppi_data.reset_index()
        df.columns = ['date', 'ppi_value']
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna().sort_values('date')
        
        # Additional filter to ensure the last 6 months
        cutoff_date = end_date - timedelta(days=210)
        df = df[df['date'] >= cutoff_date]
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error fetching PPI for {category}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_historical_benchmarks():
    """Load cleaned historical data for benchmarking"""
    try:
        benchmark_path = "data/processed/cleaned_procurement_with_ppi_extended.csv"
        if os.path.exists(benchmark_path):
            df = pd.read_csv(benchmark_path)
            return df
        else:
            st.warning("⚠️ Historical benchmark data not found. Using model predictions only.")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"⚠️ Could not load historical data: {str(e)}")
        return pd.DataFrame()

# ============================================================
# 🧠 ENHANCED PREDICTION ENGINE
# ============================================================
class ProductionForecastEngine:
    def __init__(self, model_artifacts, historical_data=None):
        self.model = model_artifacts['model']
        self.scaler = model_artifacts['scaler']
        self.features = model_artifacts['features']
        self.metadata = model_artifacts['metadata']
        self.historical_data = historical_data
        self.model_name = self.model.__class__.__name__
        
    def predict_spend(self, quantity, negotiated_price, lead_time, category, current_ppi=None):
        """Generate predictions using trained model"""
        
        # Handle missing PPI
        if current_ppi is None:
            current_ppi = self._estimate_ppi(category)
            
        # Create interaction feature (as per the training script)
        qty_leadtime_interaction = quantity * lead_time
        
        # Prepare input data with exact feature names and order
        input_data = pd.DataFrame({
            'Quantity': [quantity],
            'Negotiated_Price': [negotiated_price],
            'Lead Time (Days)': [lead_time],
            'PPI': [current_ppi],
            'Qty_LeadTime_Interaction': [qty_leadtime_interaction]
        })
        
        # Ensure feature order matches training
        input_data = input_data[self.features]
        
        # Apply scaling for linear models (as per the training logic)
        if self.model_name in ['LinearRegression', 'Ridge', 'Lasso']:
            input_scaled = self.scaler.transform(input_data)
            predicted_spend = self.model.predict(input_scaled)[0]
            use_scaling = True
        else:
            predicted_spend = self.model.predict(input_data)[0]
            use_scaling = False
        
        # Get logical breakdown
        breakdown = self._get_model_breakdown(predicted_spend, quantity, negotiated_price)
        
        # Calculate prediction intervals using model uncertainty
        prediction_intervals = self._calculate_prediction_intervals(
            predicted_spend, quantity, lead_time, category
        )
        
        # Business intelligence analysis
        business_analysis = self._analyze_business_factors(
            quantity, negotiated_price, lead_time, category, predicted_spend
        )
        
        return {
            'predicted_spend': predicted_spend,
            'base_spend': quantity * negotiated_price,
            'cost_per_unit': predicted_spend / quantity,
            'prediction_intervals': prediction_intervals,
            'business_analysis': business_analysis,
            'model_breakdown': breakdown,
            'model_info': {
                'model_name': self.model_name,
                'used_scaling': use_scaling,
                'features_used': self.features,
                'ppi_value': current_ppi
            }
        }
    
    def _estimate_ppi(self, category):
        """Estimate PPI when real-time data unavailable"""
        category_ppi_estimates = {
            'Raw Materials': 280, 'Chemicals': 275, 'Metals': 290,
            'Electronics': 105, 'Food Products': 230, 'Office Supplies': 140,
            'Packaging': 200, 'MRO': 180, 'Manufacturing': 195, 'Services': 160
        }
        return category_ppi_estimates.get(category, 200)
    
    def _get_model_breakdown(self, predicted_spend, quantity, negotiated_price):
        """Get logical model breakdown showing base cost and AI adjustment"""
        
        # Simple, logical breakdown that makes business sense
        base_cost = quantity * negotiated_price
        total_adjustment = predicted_spend - base_cost
        
        breakdown = {
            'type': 'logical',
            'base_value': base_cost,
            'feature_contributions': {
                'Base_Cost': base_cost,
                'AI_Model_Adjustment': total_adjustment
            },
            'total_prediction': predicted_spend
        }
        
        return breakdown
    
    def _calculate_prediction_intervals(self, predicted_spend, quantity, lead_time, category):
        """Calculate realistic prediction intervals"""
        
        # Base uncertainty from model performance
        model_rmse = self.metadata.get('rmse', predicted_spend * 0.15)
        model_mae = self.metadata.get('mae', predicted_spend * 0.10)
        
        # Adjust uncertainty based on business factors
        uncertainty_multiplier = 1.0
        
        # Category-based uncertainty
        category_volatility = CATEGORY_INSIGHTS.get(category, {}).get('volatility', 'Medium')
        if category_volatility == 'Very High':
            uncertainty_multiplier *= 1.8
        elif category_volatility == 'High':
            uncertainty_multiplier *= 1.5
        elif category_volatility == 'Medium':
            uncertainty_multiplier *= 1.2
        
        # Lead time uncertainty
        if lead_time > 90:
            uncertainty_multiplier *= 1.4
        elif lead_time > 45:
            uncertainty_multiplier *= 1.2
        
        # Order size uncertainty
        if quantity > 1000:
            uncertainty_multiplier *= 1.3
        elif quantity < 10:
            uncertainty_multiplier *= 1.2
        
        # Calculate intervals
        adjusted_std = model_rmse * uncertainty_multiplier
        
        return {
            'std_dev': adjusted_std,
            'lower_68': predicted_spend - adjusted_std,
            'upper_68': predicted_spend + adjusted_std,
            'lower_95': predicted_spend - 1.96 * adjusted_std,
            'upper_95': predicted_spend + 1.96 * adjusted_std,
            'uncertainty_factors': {
                'model_rmse': model_rmse,
                'category_volatility': category_volatility,
                'uncertainty_multiplier': uncertainty_multiplier
            }
        }
    
    def _analyze_business_factors(self, quantity, negotiated_price, lead_time, category, predicted_spend):
        """Provide business intelligence insights"""
        base_spend = quantity * negotiated_price
        premium = predicted_spend - base_spend
        premium_pct = (premium / base_spend) * 100 if base_spend > 0 else 0
        
        insights = []
        # Premium analysis
        if premium_pct > 20:
            insights.append(f"🔴 High premium detected: {premium_pct:.1f}% above negotiated price")
        elif premium_pct > 10:
            insights.append(f"🟡 Moderate premium: {premium_pct:.1f}% above negotiated price")
        else:
            insights.append(f"🟢 <b>Reasonable premium</b>: {premium_pct:.1f}% above negotiated price")
        
        # Lead time analysis
        if lead_time > 90:
            insights.append("⚠️ Extended lead time may increase costs and risks")
        elif lead_time < 7:
            insights.append("⚡ Rush order - potential expedite fees")
        
        # Quantity analysis
        if quantity > 1000:
            insights.append("📦 Large order - consider bulk discount negotiations")
        elif quantity < 10:
            insights.append("📦 Small order - limited economies of scale")
        
        # Category-specific insights
        category_info = CATEGORY_INSIGHTS.get(category, {})
        if category_info.get('volatility') in ['High', 'Very High']:
            insights.append(f"🌪️ {category} is a volatile category - monitor market conditions")
        
        return {
            'premium_amount': premium,
            'premium_percentage': premium_pct,
            'insights': insights,
            'category_profile': category_info
        }

# ============================================================
# 🖥️ STREAMLIT APP INTERFACE
# ============================================================
def main():
    st.set_page_config(
        page_title="ERP Spend Forecasting", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS to reduce sidebar spacing
    st.markdown("""
    <style>
    /* Reduce spacing between sidebar elements */
    .css-1d391kg {
        gap: 0.2rem;
    }
    
    /* Reduce spacing for selectbox, number_input, etc. */
    .stSelectbox > div > div {
        margin-bottom: -15px;
    }
    
    .stNumberInput > div > div {
        margin-bottom: -15px;
    }
    
    .stCheckbox > div > div {
        margin-bottom: -15px;
    }
    
    /* Reduce spacing in sidebar markdown */
    .css-1544g2n {
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
    }
    
    /* Reduce spacing for expander */
    .streamlit-expanderHeader {
        margin-bottom: -15px;
    }
    
    /* General sidebar spacing reduction */
    .css-1d391kg .element-container {
        margin-bottom: 0.2rem !important;
    }
    
    /* Reduce spacing for markdown elements in sidebar */
    .css-1d391kg .markdown-text-container {
        margin-bottom: 0.1rem !important;
    }
    
    /* Reduce spacing around headers and subheaders */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }
    
    /* Reduce spacing around buttons */
    .css-1d391kg .stButton {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }
    
    /* Compact spacing for all sidebar content */
    .css-1d391kg > div {
        margin-bottom: 0.3rem !important;
    }
    
    /* Reduce padding in sidebar container */
    .css-1d391kg {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Set default header styling (no sidebar controls)
    header_height = 100
    header_width = 100
    bg_color = "#1f77b4"
    text_color = "#ffffff"
    border_radius = 15
    shadow_enabled = True
    gradient_enabled = True
    gradient_color = "#4dabf7"
    animation_enabled = True
    
    # Create the styled header variables
    gradient_bg = f"linear-gradient(135deg, {bg_color}, {gradient_color})" if gradient_enabled else bg_color
    box_shadow = "0 8px 25px rgba(0,0,0,0.15)" if shadow_enabled else "none"
    hover_transform = "transform: translateY(-2px); box-shadow: 0 12px 35px rgba(0,0,0,0.2);" if animation_enabled else ""
    
    # FIXED: Use st.markdown directly instead of the problematic f-string
    st.markdown("""
    <div style="position: relative; padding: 30px; background: #2b2b3a; border-radius: 10px;">

    <!-- Author Credentials - Top Right Corner -->
    <div style="position: absolute; top: 15px; right: 20px; text-align: right; font-size: 0.8em; opacity: 0.85;">
        <div style="font-weight: bold; color: white;">Apu Datta</div>
        <div style="color: white;">MS in Business Analytics</div>
        <div style="color: white;">Baruch College, CUNY</div>
    </div>

    <!-- Main Title -->
    <h1 style="margin: 0px 0 -5px 0; font-size: 2.6em; font-weight: bold; color: #ffffff; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        📊 ERP Spend Forecasting
    </h1>

    <!-- Subtitle -->
    <p style="margin: 5px 0 0 0; font-size: 1.05em; font-weight: 300; color: #eaeaea; opacity: 0.9;">
        AI-powered predictions for procurement spend using trained machine learning models
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


    # Load model artifacts
    with st.spinner("🔄 Loading trained models..."):
        model_artifacts = load_trained_model()
    
    if model_artifacts['status'] == 'error':
        st.error(f"❌ {model_artifacts['message']}")
        st.info("Please ensure to run the model training script and the model files exist in the `models/` directory.")
        return
    
    # Load historical data
    historical_data = load_historical_benchmarks()
    
    # Initialize forecast engine
    forecast_engine = ProductionForecastEngine(model_artifacts, historical_data)
    
    # Sidebar inputs
    
    # Category selection
    category = st.sidebar.selectbox(
        "📦 Item Category",
        options=list(PPI_SERIES_MAP.keys()),
        index=0,
        help="Select the procurement category for PPI-based pricing"
    )
    
    # Display category insights
    category_info = CATEGORY_INSIGHTS.get(category, {})
    ppi_series = PPI_SERIES_MAP.get(category, 'Unknown')
    
    if category_info:
        st.sidebar.markdown(f"""
        **Order Specification:**
        - Volatility: {category_info.get('volatility', 'Unknown')}
        - Lead Time Sensitivity: {category_info.get('lead_time_sensitivity', 'Unknown')}
        - Seasonal Impact: {category_info.get('seasonal_impact', 'Unknown')}
        """)
    else:
        st.sidebar.markdown(f"""
        **Order Specification:**
        """)
    
    # Input parameters
    quantity = st.sidebar.number_input(
        "📊 Order Quantity", 
        min_value=1, 
        value=100,
        help="Total units to procure"
    )
    
    negotiated_price = st.sidebar.number_input(
        "💰 Negotiated Unit Price ($)", 
        min_value=0.01, 
        value=50.0,
        format="%.2f",
        help="Price per unit as negotiated with supplier"
    )
    
    lead_time = st.sidebar.number_input(
        "⏱️ Lead Time (Days)", 
        min_value=1, 
        value=30,
        help="Expected delivery time in days"
    )
    
    # Advanced options
    with st.sidebar.expander("🔧 Advanced Options"):
        use_realtime_ppi = st.checkbox("📈 Use Real-time PPI", value=True)
        show_intervals = st.checkbox("📊 Show Prediction Intervals", value=True)
        show_benchmarks = st.checkbox("📈 Historical Benchmarks", value=True)
        manual_ppi = st.number_input("Manual PPI Override", value=0.0, help="Leave 0 to use real-time/estimated PPI")
    
    # PPI Information Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📊 Producer Price Index (PPI)")
    
    # Get PPI series for selected category
    ppi_series = PPI_SERIES_MAP.get(category, 'Unknown')
    
    # Fetch PPI data to show latest value
    ppi_data = pd.DataFrame()
    current_ppi = None
    
    if use_realtime_ppi and manual_ppi == 0:
        with st.spinner(f"🔄 Fetching {category} PPI data..."):
            ppi_data = fetch_category_ppi(category,months_back=6)
            if not ppi_data.empty:
                current_ppi = ppi_data['ppi_value'].iloc[-1]
                latest_date = ppi_data['date'].iloc[-1].strftime('%Y-%m-%d')
                
                # Display PPI info in compact list format with no gaps
                st.sidebar.markdown(f"""
                <div style="line-height: 1.2; margin: 0; padding: 0;">
                🔗 <b>Data Source:</b> <a href="https://fred.stlouisfed.org/series/{ppi_series}" target="_blank">View on FRED</a><br>
                📊 <b>Latest {category} PPI:</b> {current_ppi:.2f}<br>
                📅 <b>As of:</b> {latest_date}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.sidebar.warning("⚠️ Could not fetch latest PPI data")
                st.sidebar.markdown(f"""
                <div style="line-height: 1.2; margin: 0; padding: 0;">
                🔗 <b>Data Source:</b> <a href="https://fred.stlouisfed.org/series/{ppi_series}" target="_blank">View on FRED</a>
                </div>
                """, unsafe_allow_html=True)
    elif manual_ppi > 0:
        current_ppi = manual_ppi
        st.sidebar.markdown(f"""
        <div style="line-height: 1.2; margin: 0; padding: 0;">
        🔗 <b>Data Source:</b> <a href="https://fred.stlouisfed.org/series/{ppi_series}" target="_blank">View on FRED</a><br>
        📊 <b>Manual PPI Override:</b> {current_ppi:.2f}<br>
        📅 <b>As of:</b> Manual Entry
        </div>
        """, unsafe_allow_html=True)
    else:
        # Use estimated PPI
        estimated_ppi = ProductionForecastEngine({
            'model': None, 'scaler': None, 'features': None, 'metadata': None
        }, None)._estimate_ppi(category)
        current_ppi = estimated_ppi
        st.sidebar.markdown(f"""
        <div style="line-height: 1.2; margin: 0; padding: 0;">
        🔗 <b>Data Source:</b> <a href="https://fred.stlouisfed.org/series/{ppi_series}" target="_blank">View on FRED</a><br>
        📊 <b>Estimated {category} PPI:</b> {current_ppi:.3f}<br>
        📅 <b>As of:</b> Estimated Value
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Generate forecast
    if st.sidebar.button("🚀 Generate Forecast", type="primary"):
        
        with st.spinner("🧠 Generating AI prediction..."):
            forecast_result = forecast_engine.predict_spend(
                quantity, negotiated_price, lead_time, category, current_ppi
            )
        
        # Display main results in compact side-by-side layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Create metrics in a container box
            with st.container():
                st.markdown("""
                <div style="border: 2px solid #1f77b4; border-radius: 5px; padding: 5px; background-color: #f0f8ff;">
                <h4 style="margin-top: 0; color: #1f77b4;">🎯 Forecast Results</h4>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)  # This adds the gap
                
                # Compact metrics in 2x2 grid
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    delta_amount = forecast_result['predicted_spend'] - forecast_result['base_spend']
                    st.metric(
                        "💵 Total Spend",
                        f"${forecast_result['predicted_spend']:,.0f}",
                        delta=f"${delta_amount:,.0f}"
                    )
                    
                    premium_pct = forecast_result['business_analysis']['premium_percentage']
                    st.metric(
                        "📈 Premium",
                        f"{premium_pct:.1f}%"
                    )
                
                with metric_col2:
                    delta_unit = forecast_result['cost_per_unit'] - negotiated_price
                    st.metric(
                        "💲 Cost/Unit",
                        f"${forecast_result['cost_per_unit']:.2f}",
                        delta=f"${delta_unit:.2f}"
                    )
                    
                    r2_score = model_artifacts['metadata']['r2']
                    st.metric(
                        "🎯 Confidence",
                        f"{r2_score:.1%}"
                    )
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Prepare breakdown data
            breakdown = forecast_result.get('model_breakdown', {})
            base_cost = quantity * negotiated_price
            predicted_spend = forecast_result['predicted_spend']
            
            breakdown_data = {
                'Cost Component': [
                    '💵 Base Cost',
                    '🤖 AI Adjustment',
                    '🎯 Total Predicted'
                ],
                'Amount ($)': [
                    float(base_cost),
                    float(predicted_spend - base_cost),
                    float(predicted_spend)
                ],
                'Impact (%)': [
                    100.0,
                    float(((predicted_spend - base_cost) / base_cost * 100)) if base_cost > 0 else 0.0,
                    float((predicted_spend / base_cost * 100)) if base_cost > 0 else 0.0
                ]
            }
            
            breakdown_df = pd.DataFrame(breakdown_data)
            
            # Convert to proper types
            for col in breakdown_df.columns:
                if breakdown_df[col].dtype == 'object':
                    continue
                breakdown_df[col] = breakdown_df[col].astype(float)
            
            # Style the dataframe
            def highlight_rows(row):
                if 'Total Predicted' in str(row['Cost Component']):
                    return ['background-color: #e8f4fd; font-weight: bold; border-top: 2px solid #1f77b4'] * len(row)
                elif 'Base Cost' in str(row['Cost Component']):
                    return ['background-color: #f0f8ff; font-weight: bold'] * len(row)
                elif float(row['Amount ($)']) < 0:
                    return ['background-color: #ffe6e6; color: #d63384'] * len(row)
                elif float(row['Amount ($)']) > 0 and 'AI Adjustment' in str(row['Cost Component']):
                    return ['background-color: #fff8e1; color: #f57c00'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_breakdown = breakdown_df.style.apply(highlight_rows, axis=1).format({
                'Amount ($)': '${:,.0f}',
                'Impact (%)': '{:+.1f}%'
            })
            
            # Create breakdown table in a container box
            with st.container():
                st.markdown("""
                <div style="border: 2px solid #ff7f0e; border-radius: 5px; padding: 5px; background-color: #fff8f0;">
                <h4 style="margin-top: 0; color: #ff7f0e;">💰 Spend Breakdown Analysis</h4>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)  # This adds the gap
                 
                st.dataframe(styled_breakdown, use_container_width=True, hide_index=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Add detailed note about what's included in AI Model Adjustment
            with st.expander("📋 What's Included in AI Adjustment?", expanded=False):
                st.markdown("""
                **🔍 Factors Considered:**
                - **📊 Market/PPI Impact**: Current vs historical averages
                - **⏱️ Lead Time Effects**: Delivery timeline impact
                - **📦 Quantity Dynamics**: Volume-based pricing effects
                - **🔗 Interaction Effects**: Qty×LeadTime relationships
                - **📈 Category Patterns**: Historical pricing patterns
                - **🌪️ Market Volatility**: Category-based risk premiums
                
                **📊 Model**: {model_name} | **Accuracy**: R² = {r2:.1%}
                """.format(
                    model_name=forecast_result['model_info']['model_name'],
                    r2=model_artifacts['metadata']['r2']
                ))
        
        # Business Intelligence - Combined insights in one colored box
        st.subheader("💡 Business Intelligence")
        business_analysis = forecast_result['business_analysis']
        
        # Build combined message with HTML formatting
        combined_message = ""
        
        # Show adjustment explanation for logical breakdown
        if breakdown.get('type') == 'logical':
            adjustment = predicted_spend - base_cost
            if abs(adjustment) > 1:
                adjustment_pct = (adjustment / base_cost * 100) if base_cost > 0 else 0
                if adjustment > 0:
                    combined_message += f"💡 <b>Model Insight</b>: AI predicts a <b>${adjustment:,.2f} ({adjustment_pct:+.1f}%) premium</b> based on market conditions, lead time, and category factors learned from historical data.<br><br>"
                else:
                    combined_message += f"💡 <b>Model Insight</b>: AI predicts <b>${abs(adjustment):,.2f} ({abs(adjustment_pct):.1f}%) savings</b> based on favorable conditions learned from historical data.<br><br>"
            else:
                combined_message += f"💡 <b>Model Insight</b>: AI predicts spending very close to negotiated price - good market alignment.<br><br>"
        
        # Add business insights
        for insight in business_analysis['insights']:
            combined_message += f"{insight}<br><br>"
        
        # Display in colored markdown box with small font - determine color based on premium level
        premium_pct = business_analysis['premium_percentage']
        if premium_pct > 20:
            # Red box
            st.markdown(f"""
            <div style="padding: 1px 1px 1px 1px; border-radius: 5px; background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; font-size: 1.0em;">
            {combined_message}
            </div>
            """, unsafe_allow_html=True)
        elif premium_pct > 10:
            # Yellow box
            st.markdown(f"""
            <div style="padding: 1px 1px 1px 1px; border-radius: 5px; background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; font-size: 1.0em;">
            {combined_message}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Green box
            st.markdown(f"""
            <div style="padding: 1px 1px 1px 1px; border-radius: 5px; background-color: #d1edff; border: 1px solid #bee5eb; color: #0c5460; font-size: 1.0em;">
            {combined_message}
            </div>
            """, unsafe_allow_html=True)
        
            st.markdown("<br>", unsafe_allow_html=True)  # This adds the gap

        # Prediction intervals
        if show_intervals:
            st.subheader("📊 Prediction Uncertainty")
            
            intervals = forecast_result['prediction_intervals']
            predicted = forecast_result['predicted_spend']

            # Create two columns for better layout
            col1, col2 = st.columns([1.2, 1])
            
            with col1:
                # Create a styled container for confidence intervals
                st.markdown("""
                <div style="border: 2px solid #28a745; border-radius: 10px; padding: 15px; background-color: #f8fff9; margin-bottom: 10px;">
                <h5 style="margin-top: 0; color: #28a745; text-align: center;">🎯 Confidence Intervals</h5>
                """, unsafe_allow_html=True)
                
                # Createing metrics for confidence intervals
                ci_col1, ci_col2 = st.columns(2)
                
                with ci_col1:
                    # 68% Confidence Interval
                    st.markdown("""
                    <div style="text-align: center; padding: 10px; background-color: #e8f5e8; border-radius: 8px; margin: 5px;">
                        <h6 style="margin: 0; color: #155724;">68% Confidence</h6>
                        <p style="margin: 5px 0; font-size: 0.9em; color: #6c757d;">±1 Standard Deviation</p>
                        <div style="font-weight: bold; color: #28a745; font-size: 1.1em;">
                            ${:,.0f} - ${:,.0f}
                        </div>
                        <p style="margin: 5px 0; font-size: 0.8em; color: #6c757d;">Range: ${:,.0f}</p>
                    </div>
                    """.format(
                        intervals['lower_68'], 
                        intervals['upper_68'],
                        intervals['upper_68'] - intervals['lower_68']
                    ), unsafe_allow_html=True)
                
                with ci_col2:
                    # 95% Confidence Interval
                    st.markdown("""
                    <div style="text-align: center; padding: 10px; background-color: #fff3cd; border-radius: 8px; margin: 5px;">
                        <h6 style="margin: 0; color: #856404;">95% Confidence</h6>
                        <p style="margin: 5px 0; font-size: 0.9em; color: #6c757d;">±2 Standard Deviations</p>
                        <div style="font-weight: bold; color: #d39e00; font-size: 1.1em;">
                            ${:,.0f} - ${:,.0f}
                        </div>
                        <p style="margin: 5px 0; font-size: 0.8em; color: #6c757d;">Range: ${:,.0f}</p>
                    </div>
                    """.format(
                        max(0, intervals['lower_95']),  # Ensure no negative values
                        intervals['upper_95'],
                        intervals['upper_95'] - max(0, intervals['lower_95'])
                    ), unsafe_allow_html=True)
                
                # Add explanation
                st.markdown("""
                <div style="margin-top: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff;">
                    <small style="color: #495057;">
                        <strong>💡 Interpretation:</strong><br>
                        • 68% chance actual spend falls within the narrower range<br>
                        • 95% chance actual spend falls within the wider range<br>
                        • Higher uncertainty reflects market volatility and lead time risks
                    </small>
                </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Enhanced uncertainty visualization
                fig = go.Figure()
                
                # Create a more informative uncertainty chart
                categories = ['Lower 95%', 'Lower 68%', 'Prediction', 'Upper 68%', 'Upper 95%']
                values = [
                    max(0, intervals['lower_95']),
                    intervals['lower_68'], 
                    predicted,
                    intervals['upper_68'],
                    intervals['upper_95']
                ]
                colors = ['#ff4444', '#ffaa00', '#00aa00', '#ffaa00', '#ff4444']
                
                # Create horizontal bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        y=categories,
                        x=values,
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f'${v:,.0f}' for v in values],
                        textposition='outside',
                        textfont=dict(size=10)
                    )
                ])
                
                # Adding vertical line for prediction
                fig.add_vline(
                    x=predicted,
                    line_dash="solid",
                    line_color="green",
                    line_width=3,
                    annotation_text=f"Prediction: ${predicted:,.0f}",
                    annotation_position="top"
                )
                
                fig.update_layout(
                    title={
                        'text': "📊 Uncertainty Visualization",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16}
                    },
                    xaxis_title="Spend ($)",
                    yaxis_title="",
                    height=350,
                    margin=dict(l=80, r=80, t=60, b=40),
                    plot_bgcolor='rgba(248, 249, 250, 0.8)',
                    paper_bgcolor='white',
                    showlegend=False
                )
                
                # Format x-axis to show currency
                fig.update_xaxes(
                    tickformat='$,.0f',
                    showgrid=True,
                    gridcolor='lightgray'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Add uncertainty factors explanation below
            st.markdown("---")
            
            # Display uncertainty factors in a clean format
            uncertainty_factors = intervals.get('uncertainty_factors', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div style="text-align: center; padding: 10px; background-color: #f0f8ff; border-radius: 8px;">
                    <h6 style="margin: 0; color: #1f77b4;">📊 Model Performance</h6>
                    <div style="font-weight: bold; color: #1f77b4; font-size: 1.1em;">
                        RMSE: ${:,.0f}
                    </div>
                </div>
                """.format(uncertainty_factors.get('model_rmse', 0)), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="text-align: center; padding: 10px; background-color: #fff8f0; border-radius: 8px;">
                    <h6 style="margin: 0; color: #ff7f0e;">🌪️ Category Volatility</h6>
                    <div style="font-weight: bold; color: #ff7f0e; font-size: 1.1em;">
                        {}
                    </div>
                </div>
                """.format(uncertainty_factors.get('category_volatility', 'Medium')), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div style="text-align: center; padding: 10px; background-color: #f8f0ff; border-radius: 8px;">
                    <h6 style="margin: 0; color: #9467bd;">⚡ Risk Multiplier</h6>
                    <div style="font-weight: bold; color: #9467bd; font-size: 1.1em;">
                        {:.2f}x
                    </div>
                </div>
                """.format(uncertainty_factors.get('uncertainty_multiplier', 1.0)), unsafe_allow_html=True)

        # PPI visualization
        if not ppi_data.empty:
            st.subheader(f"📈 {category} Producer Price Index - Last 6 Months")
            
            # Create the chart with a container/box
            with st.container():
                fig = px.line(
                    ppi_data, 
                    x='date', 
                    y='ppi_value',
                    title=f"Recent PPI Trend - {category} (6 Month Period)",
                    markers=True
                )
                
                # Highlighting current value
                if current_ppi:
                    fig.add_annotation(
                        x=ppi_data['date'].iloc[-1],
                        y=current_ppi,
                        text=f"Current: {current_ppi:.1f}",
                        showarrow=True,
                        arrowhead=2,
                        bgcolor="yellow",
                        bordercolor="black"
                    )
                
                # Adding source information at bottom of chart
                fig.add_annotation(
                    text="Data Source: Federal Reserve Economic Data (FRED) - <a href='https://fred.stlouisfed.org/docs/api/fred/' target='_blank'>Free API Access</a>",
                    xref="paper", yref="paper",
                    x=0, y=-0.25,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=10, color="gray"),
                    showarrow=False
                )
                
                # Format x-axis to show all months
                fig.update_xaxes(
                    dtick="M1",  # Show every month
                    tickformat="%b %Y",  # Format as "Jan 2024"
                    tickangle=45,  # Rotate labels for better readability
                    title_text="Date"
                )
                
                # Adding box styling and layout improvements
                fig.update_layout(
                    height=450,
                    margin=dict(b=100, l=50, r=50, t=50),  # Increased bottom margin from 80 to 100
                    plot_bgcolor='rgba(248, 249, 250, 0.8)',  # Light background
                    paper_bgcolor='white',
                    showlegend=False,
                    # Add border-like styling
                    shapes=[
                        dict(
                            type="rect",
                            xref="paper", yref="paper",
                            x0=0, y0=0, x1=1, y1=1,
                            line=dict(color="lightgray", width=2)
                        )
                    ]
                )
                
                # Style the chart container with CSS
                st.markdown("""
                <style>
                .stPlotlyChart > div {
                    border: 2px solid #e6e6e6;
                    border-radius: 2px;
                    padding: 2px;
                    background-color: #f8f9fa;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Historical benchmarking
        if show_benchmarks and not historical_data.empty:
            st.subheader("📊 Historical Benchmarking")
            
            # Filter historical data for same category
            category_hist = historical_data[
                historical_data['Item_Category'] == category
            ] if 'Item_Category' in historical_data.columns else historical_data
            
            if not category_hist.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Spend distribution
                    if 'Spend' in category_hist.columns:
                        fig = px.histogram(
                            category_hist, 
                            x='Spend',
                            title=f"Historical Spend Distribution - {category}",
                            nbins=30
                        )
                        
                        # Add current prediction line
                        fig.add_vline(
                            x=forecast_result['predicted_spend'],
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Prediction"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Lead time vs spend scatter
                    if 'Lead Time (Days)' in category_hist.columns and 'Spend' in category_hist.columns:
                        fig = px.scatter(
                            category_hist,
                            x='Lead Time (Days)',
                            y='Spend',
                            title="Lead Time vs Spend Relationship",
                            opacity=0.6
                        )
                        
                        # Add current prediction point
                        fig.add_trace(go.Scatter(
                            x=[lead_time],
                            y=[forecast_result['predicted_spend']],
                            mode='markers',
                            marker=dict(size=15, color='red', symbol='star'),
                            name='Prediction'
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        # Create two columns - Model Training Summary on left, Export Results on right
        model_col, export_col = st.columns([1, 1])
        
        # LEFT COLUMN - Model Training Summary
        with model_col:
            st.markdown("### 🤖 Model Training Summary")
            
            try:
                # Load data from training script
                metadata = joblib.load("models/model_metadata.joblib")
                features = joblib.load("models/feature_names.joblib")
                best_model = joblib.load("models/best_model.joblib")
                
                # Try to load dataset for shape info
                try:
                    df = pd.read_csv("data/processed/cleaned_procurement_with_ppi_extended.csv")
                    dataset_shape = df.shape
                    total_features = df.shape[1]
                except:
                    dataset_shape = "Data not available"
                    total_features = "N/A"
                
                # 📊 Model Performance Overview
                st.markdown("""
                <div style="background-color: #e8f4fd; padding: 15px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid #1f77b4;">
                    <h4 style="margin: 0; color: #1f77b4;">🏆 Best Model Performance</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Create performance metrics in a clean layout
                perf_col1, perf_col2 = st.columns(2)
                
                with perf_col1:
                    st.metric(
                        "🎯 Model Accuracy (R²)", 
                        f"{metadata['r2']:.1%}",
                        help="Coefficient of determination - higher is better"
                    )
                    st.metric(
                        "📉 Mean Absolute Error", 
                        f"${metadata['mae']:,.0f}",
                        help="Average prediction error in dollars"
                    )
                
                with perf_col2:
                    st.metric(
                        "📊 Root Mean Square Error", 
                        f"${metadata['rmse']:,.0f}",
                        help="Standard deviation of prediction errors"
                    )
                    st.metric(
                        "🔁 Cross-Validation Score", 
                        f"{metadata['cv_r2_mean']:.1%}",
                        help="Average R² across 5-fold cross-validation"
                    )
                
                # Model Selection Justification
                st.markdown(f"""
                <div style="background-color: #d4edda; padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #28a745;">
                    <strong>🏆 Selected Model:</strong> {metadata['model'].__class__.__name__}<br>
                    <strong>📈 Justification:</strong> Highest R² score ({metadata['r2']:.1%}) on test data
                </div>
                """, unsafe_allow_html=True)
                
                # 📂 Dataset Information - Expandable
                with st.expander("📂 Dataset & Training Configuration"):
                    config_col1, config_col2 = st.columns(2)
                    
                    with config_col1:
                        st.markdown(f"""
                        **📊 Dataset Information:**
                        - **Total Records:** {dataset_shape[0] if isinstance(dataset_shape, tuple) else 'N/A'}
                        - **Total Columns:** {total_features}
                        - **Selected Features:** {len(features)}
                        - **Train/Test Split:** 80% / 20%
                        """)
                    
                    with config_col2:
                        st.markdown(f"""
                        **🔧 Training Configuration:**
                        - **Models Tested:** 5 algorithms
                        - **Cross-Validation:** 5-fold
                        - **Scaling:** Applied to linear models
                        - **Random State:** 42
                        """)
                
                # 📊 Feature Importance - Expandable
                with st.expander("📊 Feature Importance Analysis"):
                    if hasattr(best_model, 'feature_importances_'):
                        # Create feature importance dataframe
                        importance_dict = dict(zip(features, best_model.feature_importances_))
                        importance_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                        
                        # Display top features with visual bars
                        st.markdown("**🔝 Most Important Features:**")
                        
                        for i, (feature, importance) in enumerate(importance_items, 1):
                            # Create a visual bar using HTML/CSS
                            bar_width = int(importance * 100)
                            color = "#1f77b4" if i == 1 else "#ff7f0e" if i == 2 else "#2ca02c" if i == 3 else "#d62728" if i == 4 else "#9467bd"
                            
                            st.markdown(f"""
                            <div style="margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                                    <span style="font-weight: bold;">{i}. {feature}</span>
                                    <span style="font-size: 0.9em; color: #666;">{importance:.1%}</span>
                                </div>
                                <div style="background-color: #f0f0f0; border-radius: 10px; height: 8px;">
                                    <div style="background-color: {color}; height: 8px; border-radius: 10px; width: {bar_width}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Feature insights
                        st.markdown("""
                        <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 10px; border-left: 4px solid #ffc107;">
                            <small><strong>💡 Key Insights:</strong><br>
                            • Quantity and Negotiated Price are the primary cost drivers<br>
                            • Lead time interactions show moderate impact<br>
                            • PPI provides market context for pricing decisions</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Feature importance not available for this model type.")
                
                # 🔄 Model Comparison - Expandable
                with st.expander("🔄 All Models Comparison"):
                    # Real training results from model_testing_training.py output
                    model_comparison = {
                        'Model': ['LinearRegression', 'Ridge', 'Lasso', 'RandomForest', 'XGBoost'],
                        'R² Score': [0.9164, 0.9162, 0.9164, 0.9963, 0.9915],
                        'MAE ($)': [9498, 9506, 9498, 1854, 2996],
                        'RMSE ($)': [13554, 13567, 13554, 2833, 4310],
                        'CV R²': [0.8422, 0.8424, 0.8422, 0.9689, 0.9756]
                    }
                    
                    comparison_df = pd.DataFrame(model_comparison)
                    
                    # Style the dataframe to highlight the best model
                    def highlight_best_model(row):
                        if row['Model'] == metadata['model'].__class__.__name__:
                            return ['background-color: #d4edda; font-weight: bold'] * len(row)
                        else:
                            return [''] * len(row)
                    
                    styled_comparison = comparison_df.style.apply(highlight_best_model, axis=1).format({
                        'R² Score': '{:.1%}',
                        'MAE ($)': '${:,.0f}',
                        'RMSE ($)': '${:,.0f}',
                        'CV R²': '{:.1%}'
                    })
                    
                    st.dataframe(styled_comparison, use_container_width=True, hide_index=True)
                    
                    st.markdown("""
                    <div style="background-color: #f8f9fa; padding: 8px; border-radius: 5px; margin-top: 8px;">
                        <small><strong>📈 Selection Criteria:</strong> RandomForest selected for highest test R² (99.6%) 
                        and lowest error rates, indicating excellent predictive performance.</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 📈 Training Process - Expandable
                with st.expander("📈 Training Process Details"):
                    st.markdown("""
                    **🔧 Model Training Pipeline:**
                    
                    1. **Data Preparation** ✅
                       - Feature selection and engineering
                       - Train/test split (80/20)
                       - Feature scaling for linear models
                    
                    2. **Model Training** ✅
                       - 5 algorithms tested simultaneously
                       - Cross-validation for robust evaluation
                       - Hyperparameter optimization
                    
                    3. **Model Selection** ✅
                       - Performance comparison across metrics
                       - Best model selection based on R²
                       - Model artifacts saved for deployment
                    
                    4. **Validation & Export** ✅
                       - Final model evaluation
                       - Performance plots generated
                       - Production-ready artifacts created
                    """)
                
                # Success indicator
                r2_percentage = metadata['r2'] * 100
                if r2_percentage > 95:
                    status_color = "#28a745"
                    status_icon = "🌟"
                    status_text = "Excellent"
                elif r2_percentage > 90:
                    status_color = "#17a2b8"
                    status_icon = "⭐"
                    status_text = "Very Good"
                else:
                    status_color = "#ffc107"
                    status_icon = "👍"
                    status_text = "Good"
                
                st.markdown(f"""
                <div style="background-color: {status_color}15; padding: 15px; border-radius: 10px; margin-top: 15px; border: 2px solid {status_color};">
                    <h4 style="margin: 0; color: {status_color}; text-align: center;">
                        {status_icon} Model Status: {status_text} ({r2_percentage:.1f}% Accuracy)
                    </h4>
                    <p style="margin: 5px 3px 2px 1px; text-align: center; color: {status_color}; font-weight: bold;">
                        Ready for Deployment
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Could not load model data: {str(e)}")
                st.info("Please ensure model training has been completed and files exist in the models/ directory.")
        
        # Right Column - Export Results
        with export_col:
            st.markdown("### 📥 Export Results")
            
            # Prepare to export data
            export_data = {
                'forecast_timestamp': datetime.now().isoformat(),
                'inputs': {
                    'category': category,
                    'quantity': quantity,
                    'negotiated_price': negotiated_price,
                    'lead_time': lead_time,
                    'ppi_used': current_ppi
                },
                'results': forecast_result,
                'model_metadata': {
                    'model_name': model_artifacts['metadata']['model'].__class__.__name__,
                    'r2_score': model_artifacts['metadata']['r2'],
                    'mae': model_artifacts['metadata']['mae'],
                    'rmse': model_artifacts['metadata']['rmse']
                }
            }
            summary_df = pd.DataFrame([{
                'Category': category,
                'Quantity': quantity,
                'Negotiated_Price': negotiated_price,
                'Lead_Time_Days': lead_time,
                'Predicted_Spend': forecast_result['predicted_spend'],
                'Cost_Per_Unit': forecast_result['cost_per_unit'],
                'Premium_Percentage': business_analysis['premium_percentage'],
                'Model_Used': model_artifacts['metadata']['model'].__class__.__name__,
                'Prediction_Date': datetime.now()
            }])

            chart_data = {
                'prediction_intervals': forecast_result['prediction_intervals'],
                'breakdown': forecast_result['model_breakdown'],
                'ppi_data': ppi_data.to_dict('records') if not ppi_data.empty else []
            }

            # Working download buttons
            st.download_button(
                "📊 Full Report (JSON)",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"spend_forecast_{category}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                key="download_full_report"
            )

            st.download_button(
                "📈 Chart Data (JSON)",
                data=json.dumps(chart_data, indent=2, default=str),
                file_name=f"chart_data_{category}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                key="download_chart_data"
            )

            st.download_button(
                "📄 Summary (CSV)",
                data=summary_df.to_csv(index=False),
                file_name=f"forecast_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="download_summary_csv"
            )

if __name__ == "__main__":
    main()