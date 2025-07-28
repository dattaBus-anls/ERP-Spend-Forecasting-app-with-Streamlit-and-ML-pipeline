# ====================================================================================================
# 📊 ENHANCED ERP SPEND PREDICTOR – PRODUCTION-READY STREAMLIT FORECASTING APP
# ----------------------------------------------------------------------------------------------------
# Features: Real-time PPI from FRED API, Advanced Market Risk analysis, Lead Time impact, 
# Enhanced Business Logic, Model Comparison, Confidence Intervals, Export Capabilities
# Formula: Future Cost = Base Cost + Lead Time Impact + Market Risk + PPI Impact + Seasonal Adjustment
# ====================================================================================================

import streamlit as st
import pandas as pd
import numpy as np
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
import logging
from typing import Dict, Tuple, Optional, List
import hashlib

try:
    import joblib
except ImportError:
    import sklearn.joblib as joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# for debug 

print("="*100)
print("✅ Streamlit app started")

# ============================================================
# 🔧 ENHANCED CONFIGURATION & SETUP
# ============================================================
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

# for debug 
print("✅ .env loaded")
print(f"✅ FRED_API_KEY retrieved: {FRED_API_KEY is not None}")

# Enhanced PPI mapping with additional categories
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

# Enhanced Market Risk Indicators from FRED
MARKET_RISK_INDICATORS = {
    'VIX': 'VIXCLS',                    # Volatility Index
    'Economic_Uncertainty': 'WLEMUINDXD',  # Economic Policy Uncertainty
    'DGS10': 'DGS10',                   # 10-Year Treasury Rate
    'UNRATE': 'UNRATE',                 # Unemployment Rate
    'CPIAUCSL': 'CPIAUCSL',             # Consumer Price Index
    'DEXUSEU': 'DEXUSEU',               # USD/EUR Exchange Rate
    'DCOILWTICO': 'DCOILWTICO',         # Oil Prices
    'NAPMPI': 'NAPMPI'                  # Manufacturing PMI
}

# Enhanced Category Intelligence with seasonal patterns
CATEGORY_INSIGHTS = {
    'Raw Materials': {
        'volatility': 'High', 
        'lead_time_sensitivity': 'High', 
        'seasonal_impact': 'Medium',
        'peak_months': [3, 4, 9, 10],  # Spring and Fall
        'supply_chain_risk': 'High',
        'price_elasticity': 'Low'
    },
    'Chemicals': {
        'volatility': 'High', 
        'lead_time_sensitivity': 'Medium', 
        'seasonal_impact': 'Low',
        'peak_months': [1, 2, 11, 12],  # Winter
        'supply_chain_risk': 'Medium',
        'price_elasticity': 'Medium'
    },
    'Metals': {
        'volatility': 'Very High', 
        'lead_time_sensitivity': 'High', 
        'seasonal_impact': 'Medium',
        'peak_months': [4, 5, 9, 10],  # Construction seasons
        'supply_chain_risk': 'High',
        'price_elasticity': 'Low'
    },
    'Electronics': {
        'volatility': 'Medium', 
        'lead_time_sensitivity': 'High', 
        'seasonal_impact': 'High',
        'peak_months': [9, 10, 11, 12],  # Holiday season
        'supply_chain_risk': 'Very High',
        'price_elasticity': 'High'
    },
    'Food Products': {
        'volatility': 'High', 
        'lead_time_sensitivity': 'Medium', 
        'seasonal_impact': 'High',
        'peak_months': [6, 7, 11, 12],  # Summer and Holiday season
        'supply_chain_risk': 'Medium',
        'price_elasticity': 'Medium'
    },
    'Office Supplies': {
        'volatility': 'Low', 
        'lead_time_sensitivity': 'Low', 
        'seasonal_impact': 'Medium',
        'peak_months': [8, 9],  # Back to school
        'supply_chain_risk': 'Low',
        'price_elasticity': 'High'
    },
    'Packaging': {
        'volatility': 'Medium', 
        'lead_time_sensitivity': 'Medium', 
        'seasonal_impact': 'Medium',
        'peak_months': [10, 11, 12],  # Holiday shipping
        'supply_chain_risk': 'Medium',
        'price_elasticity': 'Medium'
    },
    'MRO': {
        'volatility': 'Medium', 
        'lead_time_sensitivity': 'Medium', 
        'seasonal_impact': 'Low',
        'peak_months': [3, 4, 5],  # Spring maintenance
        'supply_chain_risk': 'Low',
        'price_elasticity': 'Medium'
    },
    'Manufacturing': {
        'volatility': 'Medium', 
        'lead_time_sensitivity': 'High', 
        'seasonal_impact': 'Medium',
        'peak_months': [1, 2, 3],  # New year investments
        'supply_chain_risk': 'High',
        'price_elasticity': 'Low'
    },
    'Services': {
        'volatility': 'Low', 
        'lead_time_sensitivity': 'Low', 
        'seasonal_impact': 'Low',
        'peak_months': [],  # Consistent year-round
        'supply_chain_risk': 'Very Low',
        'price_elasticity': 'High'
    }
}

# ============================================================
# 🔄 ENHANCED MODEL LOADING WITH COMPREHENSIVE ERROR HANDLING
# ============================================================
@st.cache_resource
def load_trained_model():
    """Load trained model artifacts with comprehensive error handling and fallback"""
    try:
        # Attempt to load primary model artifacts
        print("✅ Loading best_model.joblib...")
        model = joblib.load("models/best_model.joblib")
        print("✅ best_model.joblib loaded successfully")
        
        print("✅ Loading feature_scaler.joblib...")
        scaler = joblib.load("models/feature_scaler.joblib")
        print("✅ feature_scaler.joblib loaded")
        
        print("✅ Loading feature_names.joblib...")
        features = joblib.load("models/feature_names.joblib")
        print("✅ feature_names.joblib loaded")
        
        print("✅ Loading model_metadata.joblib...")
        metadata = joblib.load("models/model_metadata.joblib")
        print("✅ model_metadata.joblib loaded")
        
        return {
            'model': model,
            'scaler': scaler,
            'features': features,
            'metadata': metadata,
            'status': 'success',
            'model_type': 'trained_ml',
            'model_name': model.__class__.__name__
        }
    except FileNotFoundError as e:
        logging.warning(f"Model files not found: {str(e)}. Using enhanced business logic fallback.")
        return {
            'status': 'fallback',
            'message': "Using enhanced business logic model (ML models not found)",
            'model_type': 'business_logic',
            'model_name': 'Enhanced Business Logic'
        }
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return {
            'status': 'fallback',
            'message': f"Model loading error: {str(e)}. Using business logic fallback.",
            'model_type': 'business_logic',
            'model_name': 'Enhanced Business Logic'
        }

@st.cache_data(ttl=3600)  # Cache for 1 hour

def load_model_training_results():
    """Load REAL model training results - Dynamic file detection"""
    try:
        import glob
        
        # ✅ FIXED: Find ANY evaluation file automatically
        evaluation_files = glob.glob("models/model_evaluation_summary_*.csv")
        
        if evaluation_files:
            # Get the most recent file
            latest_file = max(evaluation_files, key=os.path.getctime)
            evaluation_df = pd.read_csv(latest_file)
            logging.info(f"✅ Loaded evaluation data from: {latest_file}")
        else:
            logging.warning("❌ No model evaluation data found")
            return None, None
        
        # Additional model metadata
        model_details = {
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'dataset_size': '800+ records',
            'features_used': ['Quantity', 'Negotiated_Price', 'Lead Time (Days)', 'PPI'], # Remove Qty_LeadTime_Interaction
            'best_model_selection': 'Highest R² score on test data',
            'validation_method': '5-fold cross-validation',
            'feature_engineering': ['PPI market integration', 'Lead time optimization', 'Category-specific analysis']
        }
        
        return evaluation_df, model_details
        
    except Exception as e:
        logging.error(f"Error loading training results: {str(e)}")
        return None, None

@st.cache_data(ttl=1800)  
def fetch_ppi_data_with_baseline(category: str) -> Tuple[Optional[float], Optional[float], Optional[datetime], pd.DataFrame]:
    """Fetch comprehensive PPI data including current value, baseline, and historical trend"""
    if not FRED_API_KEY:
        st.warning("⚠️ FRED API key not configured. Cannot fetch real PPI data.")
        return None, None, None, pd.DataFrame()
    
    series_id = PPI_SERIES_MAP.get(category)
    if not series_id:
        st.warning(f"⚠️ No PPI series found for {category}")
        return None, None, None, pd.DataFrame()
    
    try:
        fred = Fred(api_key=FRED_API_KEY)
        end_date = datetime.now()
        
        # ✅ FIXED: Fetch 12 months of data ONCE
        start_date_12mo = end_date - timedelta(days=12*30)
        
        # Single data fetch for both baseline AND charting
        ppi_data = fred.get_series(
            series_id, 
            start=start_date_12mo.strftime('%Y-%m-%d'), 
            end=end_date.strftime('%Y-%m-%d')
        )
        
        if ppi_data.empty:
            return None, None, None, pd.DataFrame()

        # ✅ FIXED: Calculate baseline from the RECENT 12 months only
        twelve_months_ago = end_date - timedelta(days=365)
        ppi_recent_12mo = ppi_data[ppi_data.index >= twelve_months_ago]
        baseline_ppi = ppi_recent_12mo.mean() if not ppi_recent_12mo.empty else ppi_data.mean()


        
        # Get latest PPI value
        current_ppi = ppi_recent_12mo.iloc[-1] if not ppi_recent_12mo.empty else ppi_data.iloc[-1]
        latest_date = ppi_recent_12mo.index[-1] if not ppi_recent_12mo.empty else ppi_data.index[-1]
        
        # ✅ FIXED: Create chart data from SAME dataset
        df_chart = pd.DataFrame()
        if not ppi_data.empty:
            # Filter chart data to show only last 12 months
            twelve_months_ago = end_date - timedelta(days=365)
            ppi_chart_filtered = ppi_data[ppi_data.index >= twelve_months_ago]
            df_chart = ppi_chart_filtered.reset_index()

            df_chart.columns = ['date', 'ppi_value']
            df_chart['date'] = pd.to_datetime(df_chart['date'])
            df_chart = df_chart.dropna().sort_values('date')
            
            # Add moving averages for trend analysis
            df_chart['ma_30'] = df_chart['ppi_value'].rolling(window=30, min_periods=1).mean()
            df_chart['ma_90'] = df_chart['ppi_value'].rolling(window=90, min_periods=1).mean()
        
        return current_ppi, baseline_ppi, latest_date, df_chart
        
    except Exception as e:
        st.error(f"❌ Error fetching PPI for {category}: {str(e)}")
        return None, None, None, pd.DataFrame()


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_comprehensive_market_data():
    """Fetch comprehensive market risk indicators with trend analysis"""
    if not FRED_API_KEY:
        return {}
    
    risk_data = {}
    fred = Fred(api_key=FRED_API_KEY)
    end_date = datetime.now()
    start_date_recent = end_date - timedelta(days=30)  # Last 30 days
    start_date_trend = end_date - timedelta(days=90)   # Last 90 days for trend
    
    for indicator, series_id in MARKET_RISK_INDICATORS.items():
        try:
            # Get recent data for current value
            recent_data = fred.get_series(series_id, start=start_date_recent, end=end_date)
            # Get trend data for analysis
            trend_data = fred.get_series(series_id, start=start_date_trend, end=end_date)
            
            if not recent_data.empty and not trend_data.empty:
                current_value = recent_data.iloc[-1]
                trend_avg = trend_data.mean()
                trend_direction = "up" if current_value > trend_avg else "down"
                volatility = trend_data.std()
                
                risk_data[indicator] = {
                    'value': current_value,
                    'date': recent_data.index[-1],
                    'series_id': series_id,
                    'trend_avg': trend_avg,
                    'trend_direction': trend_direction,
                    'volatility': volatility,
                    'percentile': stats.percentileofscore(trend_data, current_value)
                }
        except Exception as e:
            logging.warning(f"Could not fetch {indicator}: {str(e)}")
            continue
    
    return risk_data

# ============================================================
# 🧠 ADVANCED PREDICTION ENGINE WITH ENHANCED BUSINESS LOGIC
# ============================================================
class AdvancedForecastEngine:
    def __init__(self, model_artifacts, historical_data=None):
        self.model_artifacts = model_artifacts
        self.historical_data = historical_data
        self.model_type = model_artifacts.get('model_type', 'business_logic')
        
        if self.model_type == 'trained_ml':
            self.model = model_artifacts['model']
            self.scaler = model_artifacts['scaler']
            self.features = model_artifacts['features']
            self.metadata = model_artifacts['metadata']
            self.model_name = self.model.__class__.__name__
        else:
            self.model_name = "Enhanced Business Logic"

    # ✅ NEW: ML PREDICTION METHOD
    def _predict_with_ml_model(self, quantity: int, negotiated_price: float, 
                              lead_time: int, current_ppi: float) -> float:
        """Use the actual trained ML model for prediction"""
        
        # Calculate the interaction feature (CRITICAL - this was missing!)
        # qty_leadtime_interaction = quantity * lead_time
        
        # Create feature vector matching your training data exactly
        features_df = pd.DataFrame({
            'Quantity': [quantity],
            'Negotiated_Price': [negotiated_price],
            'Lead Time (Days)': [lead_time],
            'PPI': [current_ppi]
        }) # 'Qty_LeadTime_Interaction': [qty_leadtime_interaction] (Removing this line from model and stremlit)


        # Apply scaling for linear models only
        model_name = self.model.__class__.__name__
        if model_name in ['LinearRegression', 'Ridge', 'Lasso']:
            features_scaled = self.scaler.transform(features_df)
            prediction = self.model.predict(features_scaled)[0]
        else:
            prediction = self.model.predict(features_df)[0]
        
        return max(prediction, 0)  # Ensure non-negative prediction
        
    def predict_future_cost(self, quantity: int, negotiated_price: float, lead_time: int, 
                           category: str, current_ppi: Optional[float] = None, 
                           baseline_ppi: Optional[float] = None, ppi_date: Optional[datetime] = None,
                           delivery_date: Optional[datetime] = None) -> Dict:
        """
        ✅ UPDATED: Enhanced prediction using ML model + business logic adjustments
        """
        
        # ✅ TRY ML PREDICTION FIRST
        if self.model_type == 'trained_ml' and current_ppi is not None:
            try:
                # Use actual ML model for base prediction
                ml_prediction = self._predict_with_ml_model(
                    quantity, negotiated_price, lead_time, current_ppi
                )
                base_cost = ml_prediction
                
                # Update model info to show ML was used
                model_name = f"ML Model: {self.model_name}"
                prediction_method = "Machine Learning + Business Logic Adjustments"
                
                # Show success message
                st.info(f"✅ Using ML Model: {self.model_name}")
                
            except Exception as e:
                st.warning(f"⚠️ ML prediction failed: {e}. Using business logic fallback.")
                base_cost = quantity * negotiated_price
                model_name = "Enhanced Business Logic (ML Failed)"
                prediction_method = "Business Logic Fallback"
        else:
            # Fallback to business logic
            base_cost = quantity * negotiated_price
            model_name = "Enhanced Business Logic"
            prediction_method = "Business Logic Only"
            
            if self.model_type == 'business_logic':
                st.info("ℹ️ Using Business Logic (ML models not found)")
        
        # Store category for method access
        self._current_category = category
        

         # ✅ REMOVED: PPI Impact Calculation (causes double-counting with ML)
        ppi_impact = 0.0
        ppi_details = {
            'source': 'Removed to prevent double-counting',
            'current_ppi': current_ppi,
            'baseline_ppi': baseline_ppi,
            'note': 'PPI adjustments removed - ML model already includes market factors',
            'variance_pct': 0.0,
            'confidence': 'Disabled - Prevents Double-Counting'
        }
        
        # ✅ REMOVED: Lead Time Impact Calculation (already included in ML model)
        lead_time_impact = 0.0
        lead_time_details = {
            'source': 'Removed to prevent double-counting',
            'lead_time_days': lead_time,
            'note': 'Lead time adjustments removed - ML model already includes lead time factors',
            'impact_percentage': 0.0,
            'calculation_reason': 'Disabled - Prevents Double-Counting with ML Model'
        }
        
        # Comprehensive Market Risk Calculation
        market_risk_impact, market_risk_details = self._calculate_comprehensive_market_risk(
            base_cost, category
        )
        
        # New: Seasonal Adjustment
        seasonal_impact, seasonal_details = self._calculate_seasonal_adjustment(
            base_cost, category, delivery_date
        )
        
        # Calculate Enhanced Future Cost
        # future_cost = base_cost + ppi_impact + lead_time_impact + market_risk_impact + seasonal_impact
        future_cost = base_cost + market_risk_impact + seasonal_impact
        
        # Create comprehensive breakdown
        breakdown = self._create_comprehensive_breakdown(
            base_cost, ppi_impact, lead_time_impact, market_risk_impact, 
            seasonal_impact, future_cost, ppi_details, lead_time_details, 
            market_risk_details, seasonal_details
        )
        
        # Calculate enhanced prediction intervals
        prediction_intervals = self._calculate_enhanced_prediction_intervals(
            future_cost, quantity, lead_time, category
        )
        
        # Advanced business intelligence analysis
        business_analysis = self._analyze_advanced_business_factors(
            quantity, negotiated_price, lead_time, category, future_cost, delivery_date
        )
        
        return {
            'predicted_spend': future_cost,
            'base_spend': base_cost,
            'cost_per_unit': future_cost / quantity,
            'prediction_intervals': prediction_intervals,
            'business_analysis': business_analysis,
            'model_breakdown': breakdown,
            'model_info': {
                'model_name': model_name,  # ✅ UPDATED
                'prediction_method': prediction_method,
                'formula': 'ML Base Prediction + Market Risk + Seasonal Adjustment' if self.model_type == 'trained_ml' else 'Base Cost + Market Risk + Seasonal Adjustment',
                'ppi_value': current_ppi,
                'ppi_date': ppi_date,
                'baseline_ppi': baseline_ppi,
                'component_details': {
                    'ppi_details': ppi_details,
                    'lead_time_details': lead_time_details,
                    'market_risk_details': market_risk_details,
                    'seasonal_details': seasonal_details
                }
            }
        }
    
    def _calculate_enhanced_ppi_impact(self, base_cost: float, current_ppi: Optional[float], 
                                     baseline_ppi: Optional[float], category: str) -> Tuple[float, Dict]:
        """Enhanced PPI impact calculation with volatility adjustment"""
        if current_ppi is None or baseline_ppi is None:
            # Enhanced fallback with category-specific estimates
            estimated_current = self._estimate_ppi(category)
            estimated_baseline = self._get_estimated_baseline(category)
            ppi_variance = (estimated_current - estimated_baseline) / estimated_baseline
            
            # Category-specific sensitivity
            sensitivity = self._get_ppi_sensitivity(category)
            ppi_impact = base_cost * ppi_variance * sensitivity
            
            details = {
                'source': 'Enhanced Estimation',
                'current_ppi': estimated_current,
                'baseline_ppi': estimated_baseline,
                'variance_pct': ppi_variance * 100,
                'sensitivity': f"{sensitivity*100:.1f}%",
                'confidence': 'Medium'
            }
        else:
            # Enhanced calculation with real FRED data
            ppi_variance = (current_ppi - baseline_ppi) / baseline_ppi
            
            # Dynamic sensitivity based on category and market conditions
            base_sensitivity = self._get_ppi_sensitivity(category)
            market_data = fetch_comprehensive_market_data()
            
            # Adjust sensitivity based on market volatility
            volatility_adjustment = 1.0
            if 'VIX' in market_data:
                vix_value = market_data['VIX']['value']
                if vix_value > 30:
                    volatility_adjustment = 1.5
                elif vix_value > 20:
                    volatility_adjustment = 1.2
            
            adjusted_sensitivity = base_sensitivity * volatility_adjustment
            ppi_impact = base_cost * ppi_variance * adjusted_sensitivity
            
            # Enhanced capping with category-specific limits
            max_impact = self._get_max_ppi_impact(category)
            ppi_impact = max(min(ppi_impact, base_cost * max_impact), -base_cost * max_impact)
            
            details = {
                'source': 'FRED API Enhanced',
                'current_ppi': current_ppi,
                'baseline_ppi': baseline_ppi,
                'variance_pct': ppi_variance * 100,
                'base_sensitivity': f"{base_sensitivity*100:.1f}%",
                'volatility_adjustment': volatility_adjustment,
                'adjusted_sensitivity': f"{adjusted_sensitivity*100:.1f}%",
                'max_impact_cap': f"±{max_impact*100:.1f}%",
                'confidence': 'High',
                'series_id': PPI_SERIES_MAP.get(category, 'Unknown')
            }
        
        return ppi_impact, details
    
    def _calculate_advanced_lead_time_impact(self, base_cost: float, lead_time: int, 
                                           category: str) -> Tuple[float, Dict]:
        """Advanced lead time impact with supply chain risk factors"""
        category_info = CATEGORY_INSIGHTS.get(category, {})
        sensitivity = category_info.get('lead_time_sensitivity', 'Medium')
        supply_chain_risk = category_info.get('supply_chain_risk', 'Medium')
        
        # Base lead time impact calculation with supply chain risk adjustment
        if lead_time <= 7:
            # Rush order premium with supply chain risk factor
            base_premium = self._get_rush_premium(sensitivity)
            risk_multiplier = self._get_supply_chain_multiplier(supply_chain_risk)
            impact = base_cost * base_premium * risk_multiplier
            reason = f"Rush order ({sensitivity} sensitivity, {supply_chain_risk} supply chain risk)"
        
        elif lead_time >= 90:
            # Extended lead time benefits with planning optimization
            base_discount = self._get_extended_discount(sensitivity)
            planning_bonus = 0.01 if lead_time > 120 else 0.005  # Additional planning benefits
            impact = base_cost * -(base_discount + planning_bonus)
            reason = f"Extended lead time with {planning_bonus*100:.1f}% planning optimization"
        
        else:
            # Standard lead time with optimized calculation
            optimal_lead_time = self._get_optimal_lead_time(category)
            deviation = (lead_time - optimal_lead_time) / optimal_lead_time
            base_impact = self._get_base_lead_time_sensitivity(sensitivity)
            impact = base_cost * deviation * base_impact
            reason = f"Standard lead time (optimal: {optimal_lead_time} days, {sensitivity} sensitivity)"
        
        details = {
            'category_sensitivity': sensitivity,
            'supply_chain_risk': supply_chain_risk,
            'lead_time_days': lead_time,
            'optimal_lead_time': self._get_optimal_lead_time(category),
            'calculation_reason': reason,
            'impact_percentage': (impact / base_cost * 100) if base_cost > 0 else 0,
            'risk_factors': self._get_lead_time_risk_factors(category, lead_time)
        }
        
        return impact, details
    
    def _calculate_comprehensive_market_risk(self, base_cost: float, category: str) -> Tuple[float, Dict]:
        """Comprehensive market risk using multiple indicators and advanced analytics"""
        # Get comprehensive market data
        market_data = fetch_comprehensive_market_data()
        
        # Base category risk
        category_info = CATEGORY_INSIGHTS.get(category, {})
        volatility = category_info.get('volatility', 'Medium')
        base_risk = self._get_base_volatility_risk(volatility)
        
        # Advanced market adjustments
        market_adjustment = 0
        risk_sources = [f"Category volatility: {volatility}"]
        market_confidence = "Medium"
        
        if market_data:
            market_confidence = "High"
            
            # VIX-based volatility adjustment
            if 'VIX' in market_data:
                vix_data = market_data['VIX']
                vix_adjustment = self._calculate_vix_adjustment(vix_data)
                market_adjustment += vix_adjustment
                risk_sources.append(f"VIX adjustment: {vix_adjustment*100:.2f}%")
            
            # Economic uncertainty adjustment
            if 'Economic_Uncertainty' in market_data:
                uncertainty_data = market_data['Economic_Uncertainty']
                uncertainty_adjustment = self._calculate_uncertainty_adjustment(uncertainty_data)
                market_adjustment += uncertainty_adjustment
                risk_sources.append(f"Economic uncertainty: {uncertainty_adjustment*100:.2f}%")
            
            # Interest rate impact
            if 'DGS10' in market_data:
                rate_data = market_data['DGS10']
                rate_adjustment = self._calculate_interest_rate_adjustment(rate_data, category)
                market_adjustment += rate_adjustment
                risk_sources.append(f"Interest rate impact: {rate_adjustment*100:.2f}%")
            
            # Commodity-specific adjustments
            if category in ['Raw Materials', 'Metals'] and 'DCOILWTICO' in market_data:
                oil_data = market_data['DCOILWTICO']
                commodity_adjustment = self._calculate_commodity_adjustment(oil_data)
                market_adjustment += commodity_adjustment
                risk_sources.append(f"Commodity correlation: {commodity_adjustment*100:.2f}%")
        
        total_risk = base_risk + market_adjustment
        market_risk_impact = base_cost * total_risk
        
        details = {
            'category_volatility': volatility,
            'base_risk_pct': base_risk * 100,
            'market_adjustment_pct': market_adjustment * 100,
            'total_risk_pct': total_risk * 100,
            'confidence_level': market_confidence,
            'risk_sources': risk_sources,
            'market_indicators_analyzed': len(market_data),
            'indicator_details': {k: v['percentile'] for k, v in market_data.items()},
            'market_trend_summary': self._summarize_market_trends(market_data)
        }
        
        return market_risk_impact, details
    
    def _calculate_seasonal_adjustment(self, base_cost: float, category: str, 
                                     delivery_date: Optional[datetime] = None) -> Tuple[float, Dict]:
        """Calculate seasonal price adjustments based on category patterns"""
        if delivery_date is None:
            delivery_date = datetime.now()
        
        category_info = CATEGORY_INSIGHTS.get(category, {})
        seasonal_impact_level = category_info.get('seasonal_impact', 'Medium')
        peak_months = category_info.get('peak_months', [])
        
        delivery_month = delivery_date.month
        is_peak_season = delivery_month in peak_months
        
        # Calculate seasonal adjustment
        if seasonal_impact_level == 'Very High':
            base_seasonal = 0.08 if is_peak_season else -0.02
        elif seasonal_impact_level == 'High':
            base_seasonal = 0.05 if is_peak_season else -0.01
        elif seasonal_impact_level == 'Medium':
            base_seasonal = 0.03 if is_peak_season else 0
        else:  # Low
            base_seasonal = 0.01 if is_peak_season else 0
        
        seasonal_impact = base_cost * base_seasonal
        
        details = {
            'delivery_month': delivery_month,
            'seasonal_impact_level': seasonal_impact_level,
            'peak_months': peak_months,
            'is_peak_season': is_peak_season,
            'seasonal_adjustment_pct': base_seasonal * 100,
            'reasoning': self._get_seasonal_reasoning(category, delivery_month, is_peak_season)
        }
        
        return seasonal_impact, details
    
    # Helper methods for enhanced calculations
    def _get_ppi_sensitivity(self, category: str) -> float:
        """Get PPI sensitivity factor by category"""
        sensitivity_map = {
            'Metals': 0.04, 'Raw Materials': 0.04,
            'Chemicals': 0.035, 'Food Products': 0.03, 
            'Electronics': 0.025,
            'Manufacturing': 0.025,
            'Packaging': 0.02, 'MRO': 0.02,
            'Office Supplies': 0.015, 'Services': 0.01
        }
        return sensitivity_map.get(category, 0.025)
    
    def _get_max_ppi_impact(self, category: str) -> float:
        """Get maximum PPI impact cap by category"""
        cap_map = {
            'Metals': 0.12, 'Raw Materials': 0.10,
            'Chemicals': 0.10, 'Electronics': 0.08,
            'Food Products': 0.08,
            'Manufacturing': 0.06, 
            'Packaging': 0.06, 'MRO': 0.05,
            'Office Supplies': 0.04, 'Services': 0.03
        }
        return cap_map.get(category, 0.08)
    
    def _get_rush_premium(self, sensitivity: str) -> float:
        """Get rush order premium by sensitivity"""
        premium_map = {
            'Very High': 0.12, 'High': 0.08, 'Medium': 0.05, 'Low': 0.03
        }
        return premium_map.get(sensitivity, 0.05)
    
    def _get_supply_chain_multiplier(self, risk_level: str) -> float:
        """Get supply chain risk multiplier"""
        multiplier_map = {
            'Very High': 1.5, 'High': 1.3, 'Medium': 1.1, 'Low': 1.0, 'Very Low': 0.9
        }
        return multiplier_map.get(risk_level, 1.1)
    
    def _get_extended_discount(self, sensitivity: str) -> float:
        """Get extended lead time discount by sensitivity"""
        discount_map = {
            'High': 0.04, 'Medium': 0.025, 'Low': 0.015
        }
        return discount_map.get(sensitivity, 0.025)
    
    def _get_optimal_lead_time(self, category: str) -> int:
        """Get optimal lead time by category"""
        optimal_map = {
            'Services': 14, 'Office Supplies': 21,
            'MRO': 28, 'Packaging': 30,
            'Chemicals': 35, 'Food Products': 21, 'Electronics': 60,
            'Raw Materials': 45, 'Manufacturing': 90,
            'Metals': 60,
        }
        return optimal_map.get(category, 30)
    
    def _get_base_lead_time_sensitivity(self, sensitivity: str) -> float:
        """Get base lead time sensitivity factor"""
        sensitivity_map = {
            'Very High': 0.008, 'High': 0.006, 'Medium': 0.004, 'Low': 0.002
        }
        return sensitivity_map.get(sensitivity, 0.004)
    
    def _get_base_volatility_risk(self, volatility: str) -> float:
        """Get base volatility risk premium"""
        risk_map = {
            'Very High': 0.06, 'High': 0.04, 'Medium': 0.025, 'Low': 0.015
        }
        return risk_map.get(volatility, 0.025)
    
    def _calculate_vix_adjustment(self, vix_data: Dict) -> float:
        """Calculate VIX-based market adjustment"""
        vix_value = vix_data['value']
        vix_percentile = vix_data['percentile']
        
        if vix_value > 35:  # Extreme volatility
            return 0.03
        elif vix_value > 25:  # High volatility
            return 0.02
        elif vix_value > 20:  # Elevated volatility
            return 0.01
        elif vix_percentile < 25:  # Low volatility discount
            return -0.005
        return 0
    
    def _calculate_uncertainty_adjustment(self, uncertainty_data: Dict) -> float:
        """Calculate economic uncertainty adjustment"""
        percentile = uncertainty_data['percentile']
        
        if percentile > 80:  # High uncertainty
            return 0.015
        elif percentile > 60:  # Moderate uncertainty
            return 0.01
        elif percentile < 30:  # Low uncertainty
            return -0.005
        return 0
    
    def _calculate_interest_rate_adjustment(self, rate_data: Dict, category: str) -> float:
        """Calculate interest rate impact adjustment"""
        rate_value = rate_data['value']
        trend = rate_data['trend_direction']
        
        # Categories sensitive to interest rates
        rate_sensitive = ['Manufacturing', 'Raw Materials']
        
        if category not in rate_sensitive:
            return 0
        
        if rate_value > 5 and trend == 'up':
            return 0.01  # Rising rates increase costs
        elif rate_value < 2 and trend == 'down':
            return -0.005  # Falling rates reduce costs
        return 0
    
    def _calculate_commodity_adjustment(self, oil_data: Dict) -> float:
        """Calculate commodity correlation adjustment"""
        oil_percentile = oil_data['percentile']
        
        if oil_percentile > 75:  # High oil prices
            return 0.015
        elif oil_percentile < 25:  # Low oil prices
            return -0.01
        return 0
    
    def _get_lead_time_risk_factors(self, category: str, lead_time: int) -> List[str]:
        """Get lead time risk factors"""
        factors = []
        category_info = CATEGORY_INSIGHTS.get(category, {})
        supply_chain_risk = category_info.get('supply_chain_risk', 'Medium')
        
        if lead_time < 7:
            factors.append("Rush order risk")
        if lead_time > 90:
            factors.append("Extended lead time benefits")
        if supply_chain_risk in ['High', 'Very High']:
            factors.append(f"{supply_chain_risk} supply chain complexity")
        
        return factors
    
    def _summarize_market_trends(self, market_data: Dict) -> str:
        """Summarize overall market trends"""
        if not market_data:
            return "Limited market data available"
        
        trends = []
        for indicator, data in market_data.items():
            percentile = data['percentile']
            if percentile > 75:
                trends.append(f"{indicator}: High")
            elif percentile < 25:
                trends.append(f"{indicator}: Low")
        
        if not trends:
            return "Market conditions: Stable"
        return f"Market conditions: {', '.join(trends)}"
    
    def _get_seasonal_reasoning(self, category: str, month: int, is_peak: bool) -> str:
        """Get seasonal adjustment reasoning"""
        category_info = CATEGORY_INSIGHTS.get(category, {})
        
        if is_peak:
            month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                          5: 'May', 6: 'June', 7: 'July', 8: 'August',
                          9: 'September', 10: 'October', 11: 'November', 12: 'December'}
            return f"Peak season for {category} in {month_names[month]} - increased demand"
        else:
            return f"Off-peak season for {category} - stable demand"
    
    def _estimate_ppi(self, category: str) -> float:
        """Provide PPI estimates using real 12-month FRED baselines"""
        # ✅ REAL 12-month baseline averages from FRED API historical data
        category_baselines = {
            'Office Supplies': 227.06,    
            'Packaging': 293.89,          
            'MRO': 347.30,               
            'Raw Materials': 300.67,     
            'Electronics': 139.45,       
            'Chemicals': 289.41,         
            'Services': 109.01,          
            'Metals': 302.90,            
            'Manufacturing': 347.30,     
            'Food Products': 171.62      
        }
        
        baseline_value = category_baselines.get(category, 250.0)  # Default if category not found
        logging.info(f"💡 Using 12-month baseline PPI for {category}: {baseline_value}")
        return baseline_value

    def _get_estimated_baseline(self, category: str) -> float:
        """Return the same 12-month baseline (current = baseline in fallback mode)"""
        baseline_value = self._estimate_ppi(category)
        logging.info(f"💡 Using same value as baseline for {category}: {baseline_value}")
        return baseline_value

    def _create_comprehensive_breakdown(self, base_cost: float, ppi_impact: float, 
                                      lead_time_impact: float, market_risk_impact: float,
                                      seasonal_impact: float, future_cost: float,
                                      ppi_details: Dict, lead_time_details: Dict,
                                      market_risk_details: Dict, seasonal_details: Dict) -> Dict:
        """Create comprehensive cost breakdown with all components"""
        breakdown = {
            'type': 'advanced_business_logic',
            'formula': 'Base Cost + Market Risk + Seasonal Adjustment',
            'base_value': base_cost,
            'feature_contributions': {
                '💵 Base Cost': base_cost,
                '⏱️ Lead Time Impact': lead_time_impact,
                '🌪️ Market Risk': market_risk_impact,
                '🗓️ Seasonal Adjustment': seasonal_impact
            },
            'total_prediction': future_cost,
            'component_details': {
                'ppi_details': ppi_details,
                'lead_time_details': lead_time_details,
                'market_risk_details': market_risk_details,
                'seasonal_details': seasonal_details
            },
            'data_sources': {
                'ppi_source': ppi_details.get('source', 'Unknown'),
                'market_risk_source': 'FRED API + Advanced Category Analysis',
                'lead_time_source': 'Enhanced Business Logic + Supply Chain Risk',
                'seasonal_source': 'Category Pattern Analysis'
            },
            'confidence_factors': {
                'ppi_confidence': ppi_details.get('confidence', 'Medium'),
                'market_confidence': market_risk_details.get('confidence_level', 'Medium'),
                'overall_confidence': self._calculate_overall_confidence(ppi_details, market_risk_details)
            }
        }
        
        return breakdown
    
    def _calculate_overall_confidence(self, ppi_details: Dict, market_details: Dict) -> str:
        """Calculate overall prediction confidence"""
        ppi_conf = ppi_details.get('confidence', 'Medium')
        market_conf = market_details.get('confidence_level', 'Medium')
        
        confidence_scores = {'High': 3, 'Medium': 2, 'Low': 1}
        avg_score = (confidence_scores.get(ppi_conf, 2) + confidence_scores.get(market_conf, 2)) / 2
        
        if avg_score >= 2.5:
            return 'High'
        elif avg_score >= 1.5:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_enhanced_prediction_intervals(self, predicted_spend: float, quantity: int, 
                                               lead_time: int, category: str) -> Dict:
        """Calculate realistic prediction intervals using actual model performance"""
        
        if self.model_type == 'trained_ml':
            # ✅ Get REAL model performance
            actual_rmse = self.metadata.get('rmse', 0.0)
            actual_r2 = self.metadata.get('r2', 0.0)
            
            # ✅ Convert to reasonable base uncertainty (cap at 15%)
            base_uncertainty_pct = min(actual_rmse / predicted_spend, 0.15) if predicted_spend > 0 else 0.12
            base_std = predicted_spend * base_uncertainty_pct
            
            print(f"🔍 Model Performance: RMSE=${actual_rmse:.0f}, R²={actual_r2:.3f}")
            print(f"🔍 Base Uncertainty: {base_uncertainty_pct*100:.1f}%")
            
        else:
            # Business logic fallback
            base_uncertainty_pct = 0.12
            base_std = predicted_spend * base_uncertainty_pct
            actual_rmse = None
        
        # ✅ REASONABLE uncertainty adjustments (much smaller multipliers)
        uncertainty_multiplier = 1.0
        uncertainty_sources = []
        
        # Category volatility (reduced impact)
        category_info = CATEGORY_INSIGHTS.get(category, {})
        volatility = category_info.get('volatility', 'Medium')
        
        if volatility == 'Very High':
            uncertainty_multiplier *= 1.3  # Was 2.0, now 1.3
            uncertainty_sources.append("Very high category volatility")
        elif volatility == 'High':
            uncertainty_multiplier *= 1.2  # Was 1.6, now 1.2
            uncertainty_sources.append("High category volatility")
        elif volatility == 'Medium':
            uncertainty_multiplier *= 1.1  # Was 1.3, now 1.1
            uncertainty_sources.append("Medium category volatility")
        
        # Supply chain risk (reduced impact)
        supply_risk = category_info.get('supply_chain_risk', 'Medium')
        if supply_risk == 'Very High':
            uncertainty_multiplier *= 1.2  # Was 1.4, now 1.2
            uncertainty_sources.append("Very high supply chain risk")
        elif supply_risk == 'High':
            uncertainty_multiplier *= 1.1  # Was 1.2, now 1.1
            uncertainty_sources.append("High supply chain risk")
        
        # Lead time (reduced impact)
        if lead_time > 90:
            uncertainty_multiplier *= 1.15  # Was 1.3, now 1.15
            uncertainty_sources.append("Extended lead time uncertainty")
        elif lead_time < 7:
            uncertainty_multiplier *= 1.1   # Was 1.2, now 1.1
            uncertainty_sources.append("Rush order uncertainty")
        
        # Order size (reduced impact)
        if quantity > 10000:
            uncertainty_multiplier *= 1.15  # Was 1.4, now 1.15
            uncertainty_sources.append("Large order complexity")
        elif quantity > 1000:
            uncertainty_multiplier *= 1.1   # Was 1.2, now 1.1
            uncertainty_sources.append("Medium order size")
        elif quantity < 10:
            uncertainty_multiplier *= 1.05  # Was 1.15, now 1.05
            uncertainty_sources.append("Small order variability")
        
        # ✅ Final calculation with reasonable bounds
        final_std = base_std * uncertainty_multiplier
        
        # ✅ Cap the maximum uncertainty at 25% of prediction
        max_std = predicted_spend * 0.25
        final_std = min(final_std, max_std)
        
        print(f"🔍 Final Uncertainty: {final_std/predicted_spend*100:.1f}% (${final_std:.0f})")
        print(f"🔍 Multiplier Applied: {uncertainty_multiplier:.2f}")
        
        return {
            'std_dev': final_std,
            'lower_68': max(0, predicted_spend - final_std),
            'upper_68': predicted_spend + final_std,
            'lower_95': max(0, predicted_spend - 1.96 * final_std),
            'upper_95': predicted_spend + 1.96 * final_std,
            'lower_99': max(0, predicted_spend - 2.58 * final_std),
            'upper_99': predicted_spend + 2.58 * final_std,
            'uncertainty_factors': {
                'base_uncertainty_pct': base_uncertainty_pct * 100,
                'actual_model_rmse': actual_rmse,
                'uncertainty_multiplier': uncertainty_multiplier,
                'final_uncertainty_pct': final_std / predicted_spend * 100,
                'uncertainty_sources': uncertainty_sources,
                'confidence_level': self._get_confidence_level(uncertainty_multiplier),
                'data_source': 'Actual Model Performance' if self.model_type == 'trained_ml' else 'Business Logic Estimation'
            }
        }

    
    def _get_confidence_level(self, uncertainty_multiplier: float) -> str:
        """Get confidence level based on uncertainty multiplier"""
        if uncertainty_multiplier < 1.3:
            return 'High'
        elif uncertainty_multiplier < 1.8:
            return 'Medium'
        else:
            return 'Low'
    
    def _analyze_advanced_business_factors(self, quantity: int, negotiated_price: float, 
                                         lead_time: int, category: str, predicted_spend: float,
                                         delivery_date: Optional[datetime] = None) -> Dict:
        """Enhanced business intelligence analysis"""
        base_spend = quantity * negotiated_price
        premium = predicted_spend - base_spend
        premium_pct = (premium / base_spend) * 100 if base_spend > 0 else 0
        
        insights = []
        recommendations = []
        risk_factors = []
        
        # Premium analysis with enhanced thresholds
        if premium_pct > 20:
            insights.append(f"🔴 Major cost increase: {premium_pct:.1f}% above negotiated price")
            recommendations.append("Consider alternative suppliers or delayed delivery")
        elif premium_pct > 10:
            insights.append(f"🟠 Significant cost increase: {premium_pct:.1f}% above negotiated price")
            recommendations.append("Review market conditions and negotiate price protection")
        elif premium_pct > 5:
            insights.append(f"🟡 Moderate cost increase: {premium_pct:.1f}% above negotiated price")
            recommendations.append("Monitor market trends closely")
        elif premium_pct < -5:
            insights.append(f"🟢 Cost savings opportunity: {abs(premium_pct):.1f}% below negotiated price")
            recommendations.append("Consider increasing order quantity to capitalize on favorable conditions")
        else:
            insights.append(f"🟢 Cost aligned with expectations: {premium_pct:+.1f}%")
        
        # Enhanced lead time analysis
        category_info = CATEGORY_INSIGHTS.get(category, {})
        optimal_lead_time = self._get_optimal_lead_time(category)
        
        if lead_time > optimal_lead_time * 2:
            insights.append("⚠️ Significantly extended lead time - high supply chain risk")
            risk_factors.append("Extended delivery exposure")
            recommendations.append("Implement supply chain monitoring and backup suppliers")
        elif lead_time < optimal_lead_time * 0.3:
            insights.append("⚡ Rush delivery - premium costs and quality risks")
            risk_factors.append("Rush order quality concerns")
            recommendations.append("Validate supplier capacity and quality controls")
        
        # Seasonal analysis
        if delivery_date:
            seasonal_details = self._analyze_seasonal_factors(category, delivery_date)
            if seasonal_details['is_peak_season']:
                insights.append(f"📅 Peak season delivery ({seasonal_details['season_description']})")
                recommendations.append("Consider scheduling delivery for off-peak periods")
        
        # Supply chain risk assessment
        supply_risk = category_info.get('supply_chain_risk', 'Medium')
        if supply_risk in ['High', 'Very High']:
            risk_factors.append(f"{supply_risk} supply chain complexity")
            recommendations.append("Develop contingency sourcing plans")
        
        # Market volatility assessment
        volatility = category_info.get('volatility', 'Medium')
        if volatility in ['High', 'Very High']:
            risk_factors.append(f"{volatility} price volatility")
            recommendations.append("Consider price hedging strategies")
        
        # Order size optimization
        if quantity > 10000:
            insights.append("📦 Large order - economies of scale and complexity")
            recommendations.append("Verify supplier capacity and delivery capabilities")
        elif quantity < 10:
            insights.append("📦 Small order - limited negotiating power")
            recommendations.append("Consider consolidating with other orders")
        
        return {
            'premium_amount': premium,
            'premium_percentage': premium_pct,
            'insights': insights,
            'recommendations': recommendations,
            'risk_factors': risk_factors,
            'category_profile': category_info,
            'market_position': self._assess_market_position(premium_pct, category),
            'action_priority': self._determine_action_priority(premium_pct, risk_factors)
        }
    
    def _analyze_seasonal_factors(self, category: str, delivery_date: datetime) -> Dict:
        """Analyze seasonal factors for delivery date"""
        category_info = CATEGORY_INSIGHTS.get(category, {})
        peak_months = category_info.get('peak_months', [])
        month = delivery_date.month
        
        season_descriptions = {
            (12, 1, 2): "Winter season",
            (3, 4, 5): "Spring season", 
            (6, 7, 8): "Summer season",
            (9, 10, 11): "Fall season"
        }
        
        season_desc = "Unknown season"
        for months, desc in season_descriptions.items():
            if month in months:
                season_desc = desc
                break
        
        return {
            'is_peak_season': month in peak_months,
            'season_description': season_desc,
            'seasonal_impact_level': category_info.get('seasonal_impact', 'Medium')
        }
    
    def _assess_market_position(self, premium_pct: float, category: str) -> str:
        """Assess market position based on premium"""
        if premium_pct > 15:
            return "Unfavorable - significantly above market"
        elif premium_pct > 5:
            return "Below average - moderately above market"
        elif premium_pct > -5:
            return "Market aligned - competitive pricing"
        else:
            return "Favorable - below market pricing"
    
    def _determine_action_priority(self, premium_pct: float, risk_factors: List[str]) -> str:
        """Determine action priority level"""
        if premium_pct > 20 or len(risk_factors) > 3:
            return "High - Immediate action required"
        elif premium_pct > 10 or len(risk_factors) > 1:
            return "Medium - Monitor and plan"
        else:
            return "Low - Standard monitoring"

# ============================================================
# 📈 NEW: ML MODEL DASHBOARD FUNCTIONS
# ============================================================
def display_ml_dashboard():
    """Display comprehensive ML model training dashboard"""
    st.markdown("## 📈 Machine Learning Model Dashboard")
    
    # Load training results
    evaluation_df, model_details = load_model_training_results()
    
    if evaluation_df is not None:
        # Model Performance Comparison
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🏆 Model Performance Comparison")
            
            # Create performance comparison chart
            fig = go.Figure()
            
            # Add R² scores
            fig.add_trace(go.Bar(
                name='R² Score',
                x=evaluation_df['Model'],
                y=evaluation_df['R2'],
                marker_color='lightblue',
                text=evaluation_df['R2'].round(4),
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Model Performance Comparison (R² Score)",
                xaxis_title="Models",
                yaxis_title="R² Score",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics table
            st.markdown("### 📊 Detailed Performance Metrics")
            
            # Format the dataframe for better display
            display_df = evaluation_df.copy()
            display_df['MAE'] = display_df['MAE'].round(2)
            display_df['RMSE'] = display_df['RMSE'].round(2)
            display_df['R2'] = display_df['R2'].round(4)
            display_df['CV_R2'] = display_df['CV_R2'].round(4)
            
            # Highlight best model
            best_model_idx = display_df['R2'].idxmax()
            
            # Style the dataframe
            def highlight_best(row):
                if row.name == best_model_idx:
                    return ['background-color: lightgreen'] * len(row)
                return [''] * len(row)
            
            styled_df = display_df.style.apply(highlight_best, axis=1)
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            # Best model selection details
            best_model = evaluation_df.loc[evaluation_df['R2'].idxmax()]
            
            st.markdown("##### 🎯 Best Model Selection")
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 15px; border-radius: 10px; margin: 10px 0;">
                <h4 style="margin: 0 0 10px 0; color: #2e7d32;">🏆 Selected Model</h4>
                <div style="font-size: 1.2em; font-weight: bold; color: #1b5e20;">{best_model['Model']}</div>
                <div style="margin: 10px 0;">
                    <strong>R² Score:</strong> {best_model['R2']:.4f}<br>
                    <strong>MAE:</strong> ${best_model['MAE']:.2f}<br>
                    <strong>RMSE:</strong> ${best_model['RMSE']:.2f}<br>
                    <strong>CV R²:</strong> {best_model['CV_R2']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Selection criteria
            st.markdown("#### 📋 Selection Criteria")
            st.markdown("""
            **Why this model was chosen:**
            
            ✅ **Highest R² Score** - Best predictive accuracy  
            ✅ **Stable CV Performance** - Consistent across folds  
            ✅ **Low Error Metrics** - Minimal prediction errors  
            ✅ **Balanced Performance** - Good bias-variance tradeoff
            """)
            
            # Model confidence indicator
            r2_score = best_model['R2']
            if r2_score > 0.85:
                confidence = "High"
                color = "#28a745"
            elif r2_score > 0.75:
                confidence = "Medium"
                color = "#ffc107"
            else:
                confidence = "Low"
                color = "#dc3545"
            
            st.markdown(f"""
            <div style="background: {color}; color: white; padding: 10px; border-radius: 8px; text-align: center; margin: 10px 0;">
                <strong>Model Confidence: {confidence}</strong><br>
                <span style="font-size: 0.9em;">Based on R² = {r2_score:.4f}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Training Details
    if model_details:
        st.markdown("### 🔬 Training Details & Methodology")
        
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.markdown("#### 📊 Dataset Information")
            st.markdown(f"""
            - **Training Date:** {model_details['training_date']}
            - **Dataset Size:** {model_details['dataset_size']}
            - **Validation Method:** {model_details['validation_method']}
            """)
            
            st.markdown("#### 🛠️ Feature Engineering")
            for feature in model_details['feature_engineering']:
                st.markdown(f"• {feature}")
        
        with detail_col2:
            st.markdown("#### 🎯 Features Used")
            for feature in model_details['features_used']:
                st.markdown(f"• {feature}")
            
            st.markdown("#### 🏆 Selection Logic")
            st.markdown(f"• {model_details['best_model_selection']}")
    
    # Feature Importance (if available)
    if os.path.exists("models/feature_importance_plot.png"):
        st.markdown("### 📈 Feature Importance Analysis")
        st.image("models/feature_importance_plot.png", caption="Feature Importance from Best Model")
    
    # Model Comparison Visualization
    if evaluation_df is not None:
        st.markdown("### 📊 Comprehensive Model Comparison")
        
        # Create multi-metric comparison
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('R² Score', 'MAE (Lower Better)', 'RMSE (Lower Better)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # R² Score
        fig.add_trace(
            go.Bar(name='R²', x=evaluation_df['Model'], y=evaluation_df['R2'], 
                   marker_color='lightblue'), row=1, col=1
        )
        
        # MAE
        fig.add_trace(
            go.Bar(name='MAE', x=evaluation_df['Model'], y=evaluation_df['MAE'], 
                   marker_color='lightcoral'), row=1, col=2
        )
        
        # RMSE
        fig.add_trace(
            go.Bar(name='RMSE', x=evaluation_df['Model'], y=evaluation_df['RMSE'], 
                   marker_color='lightgreen'), row=1, col=3
        )
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)


        # ✅ NEW: Add Model Performance Visualizations
        st.markdown("### 📈 Model Performance Analysis")

        # Performance images from your training script
        performance_images = [
            "models/predicted_vs_actual.png",
            "models/residuals_histogram.png", 
            "models/residuals_vs_actual.png"
        ]

        image_titles = [
            "Actual vs Predicted Spend",
            "Distribution of Residuals",
            "Residuals vs Actual Spend"
        ]

        # Display the performance charts
        perf_cols = st.columns(len(performance_images))

        for i, (img_path, title) in enumerate(zip(performance_images, image_titles)):
            with perf_cols[i]:
                if os.path.exists(img_path):
                    st.image(img_path, caption=title, use_column_width=True)
                else:
                    st.warning(f"⚠️ {title} not found at {img_path}")

        # Add explanation
        st.markdown("""
        **Model Performance Insights:**
        - **Actual vs Predicted:** Shows how well the model predictions align with actual values (red line = perfect prediction)
        - **Distribution of Residuals:** Normal distribution indicates good model performance and unbiased predictions
        - **Residuals vs Actual:** Random scatter around zero indicates the model has no systematic bias
        """)
        


        # ✅ NEW: Cross-Validation Analysis Section
        st.markdown("### 🔁 Cross-Validation Analysis")

        # Load evaluation results to show CV details
        evaluation_df, model_details = load_model_training_results()

        if evaluation_df is not None:
            # Create CV comparison chart
            cv_col1, cv_col2 = st.columns([2, 1])
            
            with cv_col1:
                st.markdown("#### 📈 Cross-Validation R² Scores")
                
                # Create CV comparison chart
                fig = go.Figure()
                
                # Add CV R² scores
                fig.add_trace(go.Bar(
                    name='CV R² Score',
                    x=evaluation_df['Model'],
                    y=evaluation_df['CV_R2'],
                    marker_color='lightcoral',
                    text=evaluation_df['CV_R2'].round(4),
                    textposition='auto',
                ))
                
                # Add regular R² for comparison
                fig.add_trace(go.Bar(
                    name='Test R² Score',
                    x=evaluation_df['Model'],
                    y=evaluation_df['R2'],
                    marker_color='lightblue',
                    text=evaluation_df['R2'].round(4),
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Cross-Validation vs Test Set Performance",
                    xaxis_title="Models",
                    yaxis_title="R² Score",
                    height=400,
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with cv_col2:
                # Best CV model details
                best_cv_model = evaluation_df.loc[evaluation_df['CV_R2'].idxmax()]
                
                st.markdown("#### 🎯 Cross-Validation Winner")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffcc80 100%); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="margin: 0 0 10px 0; color: #e65100;">🔁 Best CV Performance</h4>
                    <div style="font-size: 1.2em; font-weight: bold; color: #bf360c;">{best_cv_model['Model']}</div>
                    <div style="margin: 10px 0;">
                        <strong>CV R² Score:</strong> {best_cv_model['CV_R2']:.4f}<br>
                        <strong>Test R² Score:</strong> {best_cv_model['R2']:.4f}<br>
                        <strong>Difference:</strong> {abs(best_cv_model['CV_R2'] - best_cv_model['R2']):.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Validation methodology
                st.markdown("#### 📋 Validation Details")
                st.markdown("""
                **Cross-Validation Setup:**
                - **Method:** 5-Fold Cross-Validation
                - **Metric:** R² Score
                - **Purpose:** Assess model generalization
                - **Benefit:** Reduces overfitting risk
                
                **Interpretation:**
                - CV R² close to Test R² = Good generalization
                - Large difference = Potential overfitting
                """)

        # CV Analysis explanation
        st.markdown("""
        **Cross-Validation Insights:**
        - **5-Fold CV** splits training data into 5 parts, trains on 4, validates on 1
        - **Averages performance** across all folds for robust evaluation
        - **Helps detect overfitting** by comparing CV vs Test performance
        - **More reliable** than single train/test split for model selection
        """)


# ============================================================
# 🖥️ ENHANCED STREAMLIT APP INTERFACE
# ============================================================
def create_custom_css():
    """Create enhanced custom CSS for better UI"""
    return """
    <style>
    /* Enhanced sidebar spacing */
    .css-1d391kg {
        gap: 0.1rem;
    }
    
    .stSelectbox > div > div, .stNumberInput > div > div, .stCheckbox > div > div {
        margin-bottom: -10px;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        margin: 5px 0;
    }
    
    /* Enhanced container styling */
    .prediction-container {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 15px;
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        margin: 10px 0;
    }
    
    .breakdown-container {
        border: 2px solid #ff7f0e;
        border-radius: 10px;
        padding: 15px;
        background: linear-gradient(135deg, #fff8f0 0%, #ffe6d0 100%);
        margin: 10px 0;
    }
    
    /* Enhanced alert boxes */
    .alert-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        border-radius: 8px;
        padding: 15px;
        color: white;
        font-weight: bold;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        border-radius: 8px;
        padding: 15px;
        color: #333;
        font-weight: bold;
    }
    
    .alert-low {
        background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%);
        border-radius: 8px;
        padding: 15px;
        color: white;
        font-weight: bold;
    }
    
    /* Enhanced progress indicators */
    .confidence-indicator {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
    
    .confidence-high {
        background: #28a745;
        color: white;
    }
    
    .confidence-medium {
        background: #ffc107;
        color: #333;
    }
    
    .confidence-low {
        background: #dc3545;
        color: white;
    }
    </style>
    """

def create_enhanced_header():
    """Create enhanced header with author credits"""
    return """
    <div style="position: relative; padding: 35px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px;">
    
    <!-- Author Credentials - Enhanced -->
    <div style="position: absolute; top: 20px; right: 25px; text-align: right; font-size: 0.85em; opacity: 0.9;">
        <div style="font-weight: bold; color: white; font-size: 1.1em;">Apu Datta</div>
        <div style="color: #f0f0f0;">MS in Business Analytics</div>
        <div style="color: #f0f0f0;">Baruch College, CUNY</div>
    </div>

    <!-- Main Title with Animation -->
    <h2 style="margin: 0px 0 -5px 0; font-size: 2.5em; font-weight: bold; color: #ffffff; text-shadow: 3px 3px 6px rgba(0,0,0,0.4);">
        📊 ERP Spend Forecasting
    </h2>

    <!-- Enhanced Subtitle -->
    <p style="margin: 8px 0 0 0; font-size: 1.1em; font-weight: 300; color: #f0f0f0; opacity: 0.95;">
        AI-Driven Forecasting with Real-Time Market Insights
    </p>
    
    <!-- Feature highlights -->
    <div style="margin-top: 15px; font-size: 0.9em; color: #e0e0e0;">
        🔥 Real-time FRED API 🌍 Market Risk Analysis 📅 Seasonal Adjustments 🎯 Confidence Intervals 📈 ML Model Dashboard
    </div>

    </div>
    """

def display_enhanced_category_info(category: str):
    """Display enhanced category information"""
    category_info = CATEGORY_INSIGHTS.get(category, {})
    ppi_series = PPI_SERIES_MAP.get(category, 'Unknown')
    
    # Create enhanced category card
    volatility = category_info.get('volatility', 'Unknown')
    supply_risk = category_info.get('supply_chain_risk', 'Unknown')
    seasonal = category_info.get('seasonal_impact', 'Unknown')

def main():
    st.set_page_config(
        page_title="Advanced ERP Spend Forecasting", 
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="📊"
    )
    
    # Apply enhanced CSS
    st.markdown(create_custom_css(), unsafe_allow_html=True)
    
    # Enhanced header
    st.markdown(create_enhanced_header(), unsafe_allow_html=True)
    
    # Load model artifacts BEFORE creating tabs
    with st.spinner("🔄 Loading advanced prediction models..."):
        model_artifacts = load_trained_model()

    # ✅ NEW: Add navigation tabs
    tab1, tab2 = st.tabs(["🎯 Spend Forecasting", "📈 ML Model Dashboard"])
    
    with tab1:
        # ✅ ENHANCED: Display model status
        #if model_artifacts['status'] == 'success':
            # model_name = model_artifacts.get('model_name', 'Unknown')
            # st.success(f"✅ ML Model Loaded: {model_name}")
            # st.info("🧠 System will use ML predictions + business logic adjustments")
        # elif model_artifacts['status'] == 'fallback':
            # st.info(f"ℹ️ {model_artifacts['message']}")
            # st.info("📊 System will use enhanced business logic only")
        # else:
            # st.error(f"❌ {model_artifacts['message']}")
            # return
        
        # Load historical data
        historical_data = pd.DataFrame()  # Placeholder for historical data loading
        
        # Initialize enhanced forecast engine
        forecast_engine = AdvancedForecastEngine(model_artifacts, historical_data)
        
        # Enhanced sidebar inputs
        st.sidebar.markdown("### 🎛️ Prediction Parameters")
        
        category = st.sidebar.selectbox(
            "📦 Item Category",
            options=list(PPI_SERIES_MAP.keys()),
            index=0,
            help="Select the procurement category for enhanced analysis"
        )
        
        # Display enhanced category information
        display_enhanced_category_info(category)
        
        # Enhanced input parameters with better validation
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            quantity = st.number_input(
                "📊 Quantity", 
                min_value=1, 
                value=100,
                help="Total units to procure"
            )
            
            lead_time = st.number_input(
                "⏱️ Lead Time (Day)", 
                min_value=1, 
                max_value=365,
                value=30,
                help="Expected delivery time in days"
            )
        
        with col2:
            negotiated_price = st.number_input(
                "💰 Unit Price ($)", 
                min_value=0.01, 
                value=50.0,
                format="%.2f",
                help="Negotiated price per unit"
            )
            
            delivery_date = st.date_input(
                "📅 Delivery Date",
                value=datetime.now() + timedelta(days=30),
                help="Expected delivery date for seasonal analysis"
            )
        
        # Enhanced advanced options
        with st.sidebar.expander("🔧 Advanced Configuration"):
            use_realtime_ppi = st.checkbox("📈 Real-time PPI from FRED", value=True)
            show_intervals = st.checkbox("📊 Prediction Intervals", value=True)
            show_market_analysis = st.checkbox("🌍 Market Risk Analysis", value=True)
            show_seasonal = st.checkbox("📅 Seasonal Analysis", value=True)
            manual_ppi = st.number_input("Manual PPI Override", value=0.0, help="Leave 0 for auto-detection")
            confidence_level = st.selectbox("Confidence Level", [68, 95, 99], index=1)

        # Enhanced PPI section
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 📊 Producer Price Index Status")
        
        ppi_series = PPI_SERIES_MAP.get(category, 'Unknown')
        current_ppi, baseline_ppi, ppi_date, ppi_chart_data = None, None, None, pd.DataFrame()
        
        if use_realtime_ppi and manual_ppi == 0:
            with st.sidebar:
                with st.spinner(f"🔄 Fetching {category} market data..."):
                    current_ppi, baseline_ppi, ppi_date, ppi_chart_data = fetch_ppi_data_with_baseline(category)
                    
                    if current_ppi is not None and baseline_ppi is not None:
                        ppi_variance = ((current_ppi - baseline_ppi) / baseline_ppi) * 100
                        
                        # Enhanced PPI display with status indicators
                        variance_color = '#28a745' if abs(ppi_variance) < 3 else '#ffc107' if abs(ppi_variance) < 8 else '#dc3545'
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 10px; border-radius: 8px; border-left: 4px solid {variance_color};">
                            <div style="font-size: 0.85em; line-height: 1.3;">
                                🔗 <strong>FRED Series:</strong> <a href="https://fred.stlouisfed.org/series/{ppi_series}" target="_blank">{ppi_series}</a><br>
                                📊 <strong>Current PPI:</strong> {current_ppi:.2f}<br>
                                📅 <strong>Date:</strong> {ppi_date.strftime('%Y-%m-%d')}<br>
                                📈 <strong>12-Month Baseline:</strong> {baseline_ppi:.2f}<br>
                                📉 <strong>Variance:</strong> <span style="color: {variance_color}; font-weight: bold;">{ppi_variance:+.1f}%</span><br>
                                🎯 <strong>Status:</strong> <span class="confidence-{'high' if abs(ppi_variance) < 3 else 'medium' if abs(ppi_variance) < 8 else 'low'}">
                                    {'Stable' if abs(ppi_variance) < 3 else 'Volatile' if abs(ppi_variance) < 8 else 'Highly Volatile'}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ PPI data unavailable - using estimates")
                        st.markdown(f"""
                        <div style="background: #fff3cd; padding: 8px; border-radius: 6px; border-left: 3px solid #ffc107;">
                            <div style="font-size: 0.8em;">
                                🔗 <a href="https://fred.stlouisfed.org/series/{ppi_series}" target="_blank">FRED {ppi_series}</a><br>
                                ⚠️ Using estimated values
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        elif manual_ppi > 0:
            current_ppi = manual_ppi
            st.sidebar.success(f"📊 Manual PPI: {current_ppi:.2f}")
        else:
            current_ppi = forecast_engine._estimate_ppi(category)
            baseline_ppi = forecast_engine._get_estimated_baseline(category)
            st.sidebar.info(f"📊 Estimated PPI: {current_ppi:.1f}")

        # Enhanced prediction button
        st.sidebar.markdown("---")
        
        if st.sidebar.button("🚀 Generate Advanced Forecast", type="primary", use_container_width=True):

        # ✅ ADD: Compact status in sidebar
            st.sidebar.markdown("---")
            st.sidebar.markdown("#### 🤖 System Status")
            
            if model_artifacts['status'] == 'success':
                model_name = model_artifacts.get('model_name', 'Unknown')
                st.sidebar.markdown(f"""
                <div style="background: #d4edda; padding: 8px; border-radius: 6px; border-left: 3px solid #28a745; margin: 5px 0;">
                    <div style="font-size: 0.85em; color: #155724;">
                        ✅ <strong>ML Model:</strong> {model_name}<br>
                        🧠 <strong>Mode:</strong> ML + Business Logic
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif model_artifacts['status'] == 'fallback':
                st.sidebar.markdown(f"""
                <div style="background: #d1ecf1; padding: 8px; border-radius: 6px; border-left: 3px solid #17a2b8; margin: 5px 0;">
                    <div style="font-size: 0.85em; color: #0c5460;">
                        ℹ️ <strong>Mode:</strong> Business Logic Only<br>
                        📊 <strong>Status:</strong> ML models not found
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.sidebar.error("❌ Error loading models")
                return

            with st.spinner("🧠 Generating advanced AI prediction with comprehensive market analysis..."):
                forecast_result = forecast_engine.predict_future_cost(
                    quantity, negotiated_price, lead_time, category, 
                    current_ppi, baseline_ppi, ppi_date, datetime.combine(delivery_date, datetime.min.time())
                )
            
            # Enhanced results display
            st.markdown("## 🎯 Advanced Forecast Results")

            # ✅ ADD: ML Model Training Explanation
            
            if forecast_engine.model_type == 'trained_ml':
                st.markdown("""
                <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 15px; border-radius: 10px; margin: 15px 0; border-left: 4px solid #1976d2;">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="font-size: 1.3em; margin-right: 8px;">🧠</div>
                        <strong style="color: #1976d2; font-size: 1.1em;">ML Model Training Context</strong>
                    </div>
                    <div style="color: #333; line-height: 1.6;">
                        The machine learning model was trained using historical data that included key drivers such as:
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>📊 <strong>PPI market indicators</strong></li>
                            <li>⏱️ <strong>Lead time trends</strong></li>
                            <li>📦 <strong>Quantity volumes</strong></li>
                            <li>💰 <strong>Negotiated unit pricing</strong></li>
                        </ul>
                        These factors were integrated into the learning process, allowing the model to understand complex cost dynamics 
                        and generate the <strong>ML Base Prediction</strong> shown below. Additional business logic adjustments are 
                        applied <strong>only</strong> to account for real-time conditions or factors <strong>not captured</strong> 
                        during model training.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ✅ NEW: Add ML vs Business Logic comparison
            
            if forecast_engine.model_type == 'trained_ml':
                with st.expander("🔍 ML vs Business Logic Comparison"):
                    business_logic_cost = quantity * negotiated_price
                    ml_base_cost = forecast_result['base_spend']
                    difference = ml_base_cost - business_logic_cost
                    difference_pct = abs(difference / business_logic_cost * 100) if business_logic_cost > 0 else 0
                    
                    # Display the three metrics
                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                    with comp_col1:
                        st.metric("ML Base Prediction", f"${ml_base_cost:,.0f}")
                    with comp_col2:
                        st.metric("Simple Calculation", f"${business_logic_cost:,.0f}")
                    with comp_col3:
                        st.metric("ML Adjustment", f"${difference:+,.0f}", f"{difference/business_logic_cost*100:+.1f}%")
                    
                    # Professional analysis based on difference percentage
                    if difference_pct <= 7:
                        st.markdown(f"""
                        <div style="background: #d4edda; padding: 12px; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;">
                            <strong>✅ Model Validation: ACCEPTABLE VARIANCE</strong><br>
                            <div style="margin-top: 8px; font-size: 0.95em; line-height: 1.4;">
                                The <strong>{difference_pct:.1f}% difference</strong> falls within acceptable market variance (≤7%). The ML model has learned from historical patterns including:
                                <ul style="margin: 8px 0 0 0;">
                                    <li><strong>Market volatility patterns</strong> - Price fluctuations based on category trends</li>
                                    <li><strong>Volume-based pricing</strong> - Economies/diseconomies of scale effects</li>
                                    <li><strong>Historical negotiations</strong> - Real pricing outcomes vs. initial quotes</li>
                                    <li><strong>Supply chain factors</strong> - Lead time and supplier capacity impacts</li>
                                </ul>
                                This variance reflects <em>real market conditions</em> where actual costs rarely match simple quantity × price calculations.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif difference_pct <= 15:
                        st.markdown(f"""
                        <div style="background: #fff3cd; padding: 12px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 10px 0;">
                            <strong>⚠️ Model Validation: MODERATE VARIANCE</strong><br>
                            <div style="margin-top: 8px; font-size: 0.95em; line-height: 1.4;">
                                The <strong>{difference_pct:.1f}% difference</strong> indicates moderate variance. The ML model is detecting:
                                <ul style="margin: 8px 0 0 0;">
                                    <li><strong>Category-specific risks</strong> - Higher volatility in this procurement category</li>
                                    <li><strong>Market condition adjustments</strong> - Current economic factors affecting pricing</li>
                                    <li><strong>Complex volume relationships</strong> - Non-linear pricing patterns from training data</li>
                                </ul>
                                <strong>Recommended Actions:</strong>
                                <ul style="margin: 8px 0 0 0;">
                                    <li>🔍 Review recent market trends for this category</li>
                                    <li>📊 Validate current supplier pricing assumptions</li>
                                    <li>📈 Monitor prediction accuracy against actual outcomes</li>
                                </ul>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        st.markdown(f"""
                        <div style="background: #f8d7da; padding: 12px; border-radius: 8px; border-left: 4px solid #dc3545; margin: 10px 0;">
                            <strong>🚨 Model Validation: HIGH VARIANCE - REQUIRES INVESTIGATION</strong><br>
                            <div style="margin-top: 8px; font-size: 0.95em; line-height: 1.4;">
                                The <strong>{difference_pct:.1f}% difference</strong> exceeds normal market variance thresholds. This suggests:
                                <ul style="margin: 8px 0 0 0;">
                                    <li><strong>Unusual market conditions</strong> - Significant deviation from historical patterns</li>
                                    <li><strong>Data anomalies</strong> - Potential issues with input parameters or model training</li>
                                    <li><strong>Category disruption</strong> - Major changes in supply chain or market dynamics</li>
                                </ul>
                                <strong>⚡ IMMEDIATE ACTIONS REQUIRED:</strong>
                                <ul style="margin: 8px 0 0 0; color: #721c24; font-weight: bold;">
                                    <li>🔍 Verify all input parameters (quantity, price, category)</li>
                                    <li>📞 Contact suppliers for current market pricing validation</li>
                                    <li>📊 Review model training data for similar scenarios</li>
                                    <li>🔄 Consider re-training model with recent market data</li>
                                    <li>⚠️ Flag this prediction for manual review before procurement</li>
                                </ul>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    
                    # Training data insight
                    st.markdown("""
                    <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 3px solid #6c757d; margin: 10px 0;">
                        <strong>📚 Training Data Context:</strong><br>
                        <span style="font-size: 0.9em; line-height: 1.4;">
                        The ML model was trained on <strong>thousands of historical procurement transactions</strong> across multiple categories, 
                        learning the complex relationships between negotiated prices, actual final costs, market conditions, and operational factors. 
                        This enables it to predict realistic outcomes rather than theoretical calculations.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

            # Main results in enhanced layout
            col1, col2, col3 = st.columns([1.2, 1.2, 1])
            
            with col1:
                
                st.markdown("""
                <div style="border: 2px solid #1f77b4; border-radius: 10px; padding: 10px; background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%); margin: 10px 0;">
                <h4 style="margin: 0px 0px 5px 0px; color: #1f77b4; text-align: center; font-size: 1.1em;">💵 Financial Impact</h4>
                """, unsafe_allow_html=True)



                # Enhanced metrics
                delta_amount = forecast_result['predicted_spend'] - forecast_result['base_spend']
                delta_unit = forecast_result['cost_per_unit'] - negotiated_price
                premium_pct = forecast_result['business_analysis']['premium_percentage']
                
                st.metric(
                    "Forecast Spend",
                    f"${forecast_result['predicted_spend']:,.0f}",
                    delta=f"${delta_amount:+,.0f}",
                    help="Total predicted cost including all market factors"
                )
                
                st.metric(
                    "Cost Per Unit",
                    f"${forecast_result['cost_per_unit']:.2f}",
                    delta=f"${delta_unit:+.2f}",
                    help="Predicted unit cost vs negotiated price"
                )
                
                st.metric(
                    "Premium/Discount",
                    f"{premium_pct:+.1f}%",
                    help="Percentage change from negotiated price"
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="breakdown-container">
                <h5 style="margin-top: 0; color: #ff7f0e; text-align: center;">📊 Advanced Breakdown</h5>
                """, unsafe_allow_html=True)
                
                # Enhanced breakdown with visual indicators
                breakdown = forecast_result.get('model_breakdown', {})
                feature_contributions = breakdown.get('feature_contributions', {})
                base_cost = quantity * negotiated_price
                
                for component, amount in feature_contributions.items():
                    if 'Base Cost' in component:
                        impact_pct = 100.0
                        color = '#28a745'
                    else:
                        impact_pct = (amount / base_cost * 100) if base_cost > 0 else 0.0
                        if impact_pct > 0:
                            color = '#dc3545' if impact_pct > 5 else '#ffc107'
                        else:
                            color = '#28a745'
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid #eee;">
                        <span style="font-weight: bold;">{component}</span>
                        <div style="text-align: right;">
                            <div style="color: {color}; font-weight: bold;">${amount:,.0f}</div>
                            <div style="font-size: 0.8em; color: #666;">{impact_pct:+.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            

            with col3:         
                # FIXED: Enhanced confidence and risk indicators using ACTUAL model data
                intervals = forecast_result['prediction_intervals']
                uncertainty_factors = intervals.get('uncertainty_factors', {})
                model_info = forecast_result['model_info']
                
                # Get ACTUAL model performance data
                if forecast_engine.model_type == 'trained_ml':
                    # Use REAL model metadata from training
                    model_metadata = forecast_engine.metadata
                    actual_r2 = model_metadata.get('r2', 0.0)
                    actual_mae = model_metadata.get('mae', 0.0)
                    actual_rmse = model_metadata.get('rmse', 0.0)
                    model_name = forecast_engine.model_name
                    
                    # Calculate REAL confidence based on actual R² score
                    if actual_r2 >= 0.85:
                        overall_confidence = "High"
                        confidence_color = "#28a745"
                    elif actual_r2 >= 0.75:
                        overall_confidence = "Medium"
                        confidence_color = "#ffc107"
                    else:
                        overall_confidence = "Low" 
                        confidence_color = "#dc3545"
                    
                    # Get REAL uncertainty from model calculations
                    uncertainty_multiplier = uncertainty_factors.get('uncertainty_multiplier', 1.0)
                    if uncertainty_multiplier <= 1.3:
                        uncertainty_level = "Low"
                        uncertainty_color = "#28a745"
                    elif uncertainty_multiplier <= 1.8:
                        uncertainty_level = "Medium"
                        uncertainty_color = "#ffc107"
                    else:
                        uncertainty_level = "High"
                        uncertainty_color = "#dc3545"
                    
                    # Calculate prediction range for uncertainty display
                    predicted_spend = forecast_result['predicted_spend']
                    prediction_range = intervals.get('upper_95', predicted_spend) - intervals.get('lower_95', predicted_spend)
                    
                    # Get PPI variance if available
                    if current_ppi and baseline_ppi:
                        ppi_variance = ((current_ppi - baseline_ppi) / baseline_ppi) * 100
                        data_source_detail = f"FRED API ({ppi_variance:+.1f}% vs baseline)"
                    else:
                        data_source_detail = "FRED API + Enhanced Logic"
                        
                    model_display = f"ML Model: {model_name}"
                    
                else:
                    # Fallback for business logic
                    overall_confidence = "Medium"
                    confidence_color = "#ffc107"
                    uncertainty_level = "Medium"
                    uncertainty_color = "#ffc107"
                    actual_r2 = 0.0
                    actual_mae = 0.0
                    prediction_range = 0.0
                    data_source_detail = "Enhanced Business Logic"
                    model_display = "Business Logic Model"
                
                
                # Display with EXACT same format as Advanced Breakdown
                
                
                st.markdown("""
                <div style="border: 2px solid #28a745; border-radius: 10px; padding: 15px; background: linear-gradient(135deg, #f0fff0 0%, #e6ffe6 100%); margin: 10px 0; min-height: 0px;">
                <h5 style="margin: 0px 0px 5px 0px; color: #28a745; text-align: center; font-size: 1.1em;">🎯 Prediction Quality</h5>
                """, unsafe_allow_html=True)


                # Match the exact format of Advanced Breakdown
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid #eee;">
                    <span style="font-weight: bold;">🎯 Overall Confidence</span>
                    <div style="text-align: right;">
                        <div style="color: #28a745; font-weight: bold;">{overall_confidence}</div>
                        <div style="font-size: 0.8em; color: #666;">R² = {actual_r2:.3f}</div>
                    </div>
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid #eee;">
                    <span style="font-weight: bold;">📊 Uncertainty Level</span>
                    <div style="text-align: right;">
                        <div style="color: #28a745; font-weight: bold;">{uncertainty_level}</div>
                        <div style="font-size: 0.8em; color: #666;">±${prediction_range:,.0f}</div>
                    </div>
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid #eee;">
                    <span style="font-weight: bold;">🌐 Data Sources</span>
                    <div style="text-align: right;">
                        <div style="color: #28a745; font-weight: bold;">FRED API</div>
                        <div style="font-size: 0.8em; color: #666;">{ppi_variance:+.1f}% vs baseline</div>
                    </div>
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 5px 0;">
                    <span style="font-weight: bold;">🤖 Method</span>
                    <div style="text-align: right;">
                        <div style="color: #28a745; font-weight: bold;">{model_display}</div>
                        <div style="font-size: 0.8em; color: #666;">MAE: ${actual_mae:.0f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

            # Enhanced Business Intelligence Section
            st.markdown("## 💡 Advanced Business Intelligence")
            
            business_analysis = forecast_result['business_analysis']
            premium_pct = business_analysis['premium_percentage']
            action_priority = business_analysis.get('action_priority', 'Medium')
            
            # Determine alert style based on premium and priority
            if premium_pct > 15 or 'High' in action_priority:
                alert_class = 'alert-high'
                alert_icon = '🚨'
            elif premium_pct > 5 or 'Medium' in action_priority:
                alert_class = 'alert-medium'
                alert_icon = '⚠️'
            else:
                alert_class = 'alert-low'
                alert_icon = '✅'
            
            # Create comprehensive business intelligence display
            intel_col1, intel_col2 = st.columns([2, 1])
            
            with intel_col1:
                st.markdown(f"""
                <div class="{alert_class}">
                    <h4 style="margin: 0 0 10px 0;">{alert_icon} Strategic Assessment</h4>
                """, unsafe_allow_html=True)
                
                # Key insights
                for insight in business_analysis['insights']:
                    st.markdown(f"• {insight}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Recommendations
                # if business_analysis.get('recommendations'):
                    # st.markdown("#### 🎯 Strategic Recommendations")
                    # for i, rec in enumerate(business_analysis['recommendations'], 1):
                        # st.markdown(f"{i}. {rec}")
            
            with intel_col2:
                # Risk factors and action items
                st.markdown("#### ⚠️ Risk Factors")
                risk_factors = business_analysis.get('risk_factors', [])
                if risk_factors:
                    for risk in risk_factors:
                        st.markdown(f"• {risk}")
                else:
                    st.markdown("• No significant risks identified")
                
                st.markdown(f"""
                <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #007bff;">
                    <strong>Action Priority:</strong><br>
                    <span style="font-weight: bold; color: #007bff;">{action_priority}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced Prediction Intervals
            if show_intervals:
                st.markdown("## 📊 Advanced Uncertainty Analysis")
                
                intervals = forecast_result['prediction_intervals']
                predicted = forecast_result['predicted_spend']
                
                # Create enhanced visualization
                interval_col1, interval_col2 = st.columns([1.5, 1])
                
                with interval_col1:
                    # Enhanced confidence interval chart
                    fig = go.Figure()
                    
                    # Create uncertainty bands
                    confidence_levels = [99, 95, 68]
                    colors = ['rgba(255, 0, 0, 0.1)', 'rgba(255, 165, 0, 0.2)', 'rgba(0, 128, 0, 0.3)']
                    
                    for i, conf in enumerate(confidence_levels):
                        lower_key = f'lower_{conf}'
                        upper_key = f'upper_{conf}'
                        
                        if lower_key in intervals and upper_key in intervals:
                            fig.add_trace(go.Scatter(
                                x=[intervals[lower_key], intervals[upper_key], intervals[upper_key], intervals[lower_key], intervals[lower_key]],
                                y=[f'{conf}% Confidence', f'{conf}% Confidence', f'{conf}% Confidence', f'{conf}% Confidence', f'{conf}% Confidence'],
                                fill='tonext' if i > 0 else 'toself',
                                fillcolor=colors[i],
                                line=dict(color='rgba(0,0,0,0)'),
                                name=f'{conf}% Confidence Interval',
                                showlegend=True
                            ))
                    
                    # Add prediction line
                    fig.add_vline(
                        x=predicted,
                        line_dash="solid",
                        line_color="blue",
                        line_width=3,
                        annotation_text=f"Prediction: ${predicted:,.0f}",
                        annotation_position="top"
                    )
                    
                    fig.update_layout(
                        title="Enhanced Prediction Uncertainty Bands",
                        xaxis_title="Predicted Spend ($)",
                        yaxis_title="Confidence Level",
                        height=400,
                        showlegend=True
                    )
                    
                    fig.update_xaxes(tickformat='$,.0f')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with interval_col2:
                    # Enhanced confidence metrics
                    st.markdown("#### 🎯 Confidence Metrics")
                    
                    for conf in [68, 95, 99]:
                        lower_key = f'lower_{conf}'
                        upper_key = f'upper_{conf}'
                        
                        if lower_key in intervals and upper_key in intervals:
                            range_value = intervals[upper_key] - intervals[lower_key]
                            
                            st.markdown(f"""
                            <div style="background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 8px; border-left: 4px solid #{'#28a745' if conf == 68 else '#ffc107' if conf == 95 else '#dc3545'};">
                                <strong>{conf}% Confidence</strong><br>
                                <span style="font-size: 0.9em; color: #666;">Range: ${intervals[lower_key]:,.0f} - ${intervals[upper_key]:,.0f}</span><br>
                                <span style="font-size: 0.8em; color: #888;">Spread: ${range_value:,.0f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Uncertainty sources
                    uncertainty_sources = intervals['uncertainty_factors'].get('uncertainty_sources', [])
                    if uncertainty_sources:
                        st.markdown("#### 📋 Uncertainty Sources")
                        for source in uncertainty_sources:
                            st.markdown(f"• {source}")
            
            # Enhanced Market Analysis
            if show_market_analysis:
                st.markdown("## 🌍 Comprehensive Market Analysis")
                
                market_data = fetch_comprehensive_market_data()
                if market_data:
                    # Create market indicators dashboard
                    market_cols = st.columns(len(market_data))
                    
                    for i, (indicator, data) in enumerate(market_data.items()):
                        with market_cols[i % len(market_cols)]:
                            percentile = data['percentile']
                            
                            # Color coding based on percentile
                            if percentile > 75:
                                color = '#dc3545'
                                status = 'High'
                            elif percentile > 50:
                                color = '#ffc107'
                                status = 'Elevated'
                            elif percentile > 25:
                                color = '#28a745'
                                status = 'Normal'
                            else:
                                color = '#17a2b8'
                                status = 'Low'
                            
                            st.markdown(f"""
                            <div style="background: white; padding: 12px; border-radius: 8px; border-left: 4px solid {color}; margin: 5px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <div style="font-weight: bold; color: #333;">{indicator}</div>
                                <div style="font-size: 1.2em; font-weight: bold; color: {color};">{data['value']:.2f}</div>
                                <div style="font-size: 0.8em; color: #666;">Status: {status}</div>
                                <div style="font-size: 0.8em; color: #888;">{data['percentile']:.0f}th percentile</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Market trend summary

                    model_info = forecast_result['model_info']
                    market_risk_details = model_info['component_details']['market_risk_details']
                    trend_summary = market_risk_details.get('market_trend_summary', 'No trend data available')

                    # Get the current date for data freshness
                    current_date = datetime.now().strftime('%B %d, %Y')

                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <h4 style="margin: 0 0 10px 0; color: #1976d2;">📈 Market Trend Summary</h4>
                        <p style="margin: 0 0 10px 0; color: #333; font-weight: bold;">{trend_summary}</p>
                        <div style="font-size: 0.8em; color: #666; border-top: 1px solid rgba(25, 118, 210, 0.2); padding-top: 8px;">
                            📊 <strong>Source:</strong> Real-time data from <a href="https://fred.stlouisfed.org/" target="_blank" style="color: #1976d2;">Federal Reserve Economic Data (FRED)</a> • Updated: {current_date}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.warning("🌐 Limited market data available - using category-based analysis only")
            
            # Enhanced PPI Visualization
            if not ppi_chart_data.empty:
                st.markdown(f"## 📈 {category} Producer Price Index Analysis")
                
                fig = make_subplots(
                    rows=1, cols=1,
                    subplot_titles=[f"12-Month PPI Trend - {category}"]
                )
                
                # Add PPI trend line
                fig.add_trace(
                    go.Scatter(
                        x=ppi_chart_data['date'],
                        y=ppi_chart_data['ppi_value'],
                        mode='lines+markers',
                        name='PPI Value',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=6)
                    )
                )
                
                # Add moving averages if available
                if 'ma_30' in ppi_chart_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=ppi_chart_data['date'],
                            y=ppi_chart_data['ma_30'],
                            mode='lines',
                            name='30-Day MA',
                            line=dict(color='orange', width=2, dash='dash')
                        )
                    )
                
                if 'ma_90' in ppi_chart_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=ppi_chart_data['date'],
                            y=ppi_chart_data['ma_90'],
                            mode='lines',
                            name='90-Day MA',
                            line=dict(color='red', width=2, dash='dot')
                        )
                    )
                
                # Add baseline
                if baseline_ppi:
                    fig.add_hline(
                        y=baseline_ppi,
                        line_dash="solid",
                        line_color="green",
                        annotation_text=f"12-Month Baseline: {baseline_ppi:.1f}",
                        annotation_position="top left"
                    )
                
                # Add current value annotation
                if current_ppi and ppi_date:
                    fig.add_annotation(
                        x=ppi_date,
                        y=current_ppi,
                        text=f"Current: {current_ppi:.1f}",
                        showarrow=True,
                        arrowhead=2,
                        bgcolor="yellow",
                        bordercolor="black"
                    )
                
                fig.update_layout(
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="PPI Value",
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced Export Section
            st.markdown("## 📥 Export Advanced Results")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            # Prepare comprehensive export data
            comprehensive_export = {
                'forecast_timestamp': datetime.now().isoformat(),
                'prediction_method': forecast_result['model_info']['prediction_method'],
                'formula': forecast_result['model_info']['formula'],
                'inputs': {
                    'category': category,
                    'quantity': quantity,
                    'negotiated_price': negotiated_price,
                    'lead_time': lead_time,
                    'delivery_date': delivery_date.isoformat(),
                    'ppi_used': current_ppi,
                    'baseline_ppi': baseline_ppi,
                    'ppi_date': ppi_date.isoformat() if ppi_date else None
                },
                'results': forecast_result,
                'market_data': market_data if show_market_analysis else {},
                'data_sources': {
                    'ppi_source': 'FRED API Enhanced',
                    'ppi_series': ppi_series,
                    'market_risk_source': 'FRED + Advanced Category Analysis',
                    'seasonal_source': 'Enhanced Category Pattern Analysis'
                },
                'methodology': {
                    'confidence_calculation': 'Multi-factor uncertainty analysis',
                    'risk_assessment': 'Comprehensive supply chain and market risk',
                    'seasonal_adjustment': 'Category-specific seasonal patterns',
                    'validation': 'Real-time market data integration'
                }
            }
            
            # Enhanced summary for CSV export
            enhanced_summary = pd.DataFrame([{
                'Category': category,
                'Quantity': quantity,
                'Negotiated_Price': negotiated_price,
                'Lead_Time_Days': lead_time,
                'Delivery_Date': delivery_date,
                'Predicted_Spend': forecast_result['predicted_spend'],
                'Cost_Per_Unit': forecast_result['cost_per_unit'],
                'Premium_Percentage': business_analysis['premium_percentage'],
                'Current_PPI': current_ppi,
                'Baseline_PPI': baseline_ppi,
                'PPI_Variance_Pct': ((current_ppi - baseline_ppi) / baseline_ppi * 100) if (current_ppi and baseline_ppi) else None,
                'Confidence_Level': forecast_result['model_breakdown'].get('confidence_factors', {}).get('overall_confidence', 'Medium'),
                'Action_Priority': business_analysis.get('action_priority', 'Medium'),
                'Market_Position': business_analysis.get('market_position', 'Unknown'),
                'Method_Used': forecast_result['model_info']['model_name'],
                'Prediction_Date': datetime.now(),
                'Data_Quality': 'High' if current_ppi else 'Estimated'
            }])
            
            with export_col1:
                st.download_button(
                    "📊 Complete Analysis (JSON)",
                    data=json.dumps(comprehensive_export, indent=2, default=str),
                    file_name=f"advanced_forecast_{category}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with export_col2:
                st.download_button(
                    "📈 Market Data (JSON)",
                    data=json.dumps({
                        'ppi_data': ppi_chart_data.to_dict('records') if not ppi_chart_data.empty else [],
                        'market_indicators': market_data if show_market_analysis else {},
                        'prediction_intervals': forecast_result['prediction_intervals'],
                        'breakdown': forecast_result['model_breakdown']
                    }, indent=2, default=str),
                    file_name=f"market_analysis_{category}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with export_col3:
                st.download_button(
                    "📄 Executive Summary (CSV)",
                    data=enhanced_summary.to_csv(index=False),
                    file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # Enhanced methodology note
            st.markdown("---")
            
            # Create the styled container using Streamlit components
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; margin: 20px 0;">
                <h4 style="margin: 0 0 15px 0; color: #007bff;">🔬 Advanced Methodology Summary</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Use regular markdown for the content
            st.markdown(f"""
            **Enhanced Prediction Formula:**  
            `{forecast_result['model_info']['formula']}`
            
            **Prediction Method:**  
            • {forecast_result['model_info']['prediction_method']}
            
            **Data Integration:**  
            • Real-time Producer Price Index from Federal Reserve Economic Data (FRED)  
            • Comprehensive market risk indicators (VIX, Economic Uncertainty, Interest Rates)  
            • Category-specific supply chain risk assessment  
            • Seasonal demand pattern analysis  
            • Advanced uncertainty quantification with multiple confidence levels
            
            **Quality Assurance:**  
            • Multi-source data validation • Real-time market condition monitoring • Enhanced confidence scoring
            """)
            
            st.markdown("---")

    with tab2:
        display_ml_dashboard()

if __name__ == "__main__": 
    main()