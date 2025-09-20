import streamlit as st
import joblib
import pandas as pd
import requests
import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="🚲 Bike Rental Forecaster Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================
# Load Model & Feature Data
# ==============================
@st.cache_resource
def load_model_and_data():
    """Loads the pre-trained model and feature information."""
    try:
        # IMPORTANT: Replace these paths with the actual paths to your files.
        model = joblib.load("bike_rental_rf_model.pkl")
        feature_cols = joblib.load("feature_columns.pkl")
    except FileNotFoundError:
        st.error("🚨 Model or feature files not found! Please ensure the paths in the script are correct.")
        return None, None, None, None
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    friendly_feature_names = {
        'temp': 'Temperature (°C)', 'hum': 'Humidity (%)', 'windspeed': 'Wind Speed (km/h)',
        'season': 'Season', 'yr': 'Year', 'mnth': 'Month', 'weathersit': 'Weather Situation'
    }
    
    return model, feature_cols, feature_importance_df, friendly_feature_names

model, feature_cols, feature_importance_df, friendly_feature_names = load_model_and_data()

if model is None:
    st.stop()

# ==============================
# Helper Functions
# ==============================
def get_live_weather(city, api_key):
    """Fetches live weather data from OpenWeatherMap API."""
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {'q': city, 'appid': api_key, 'units': 'metric'}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return {
            "temp": data['main']['temp'], "hum": data['main']['humidity'],
            "windspeed": data['wind']['speed'] * 3.6,
            "weathersit_main": data['weather'][0]['main']
        }
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Weather API Error: Could not fetch weather for '{city}'. Please check the city name. (HTTP {http_err.response.status_code})")
        return None
    except Exception as e:
        st.error(f"An error occurred while fetching weather data: {e}")
        return None

def get_prediction_distribution(model, input_df):
    """Calculates prediction distribution from the Random Forest."""
    tree_predictions = [tree.predict(input_df)[0] for tree in model.estimators_]
    mean_prediction = np.mean(tree_predictions)
    std_dev = np.std(tree_predictions)
    lower_bound = max(0, mean_prediction - 1.96 * std_dev)
    upper_bound = mean_prediction + 1.96 * std_dev
    return int(mean_prediction), int(lower_bound), int(upper_bound), tree_predictions

# ==============================
# CSS Styling - Hardcore & Consistent
# ==============================
st.markdown("""
<style>
    /* 1. FONT IMPORT --- Ensure the font is always available */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* 2. CSS RESET --- Remove browser default styles */
    html, body, div, span, applet, object, iframe,
    h1, h2, h3, h4, h5, h6, p, blockquote, pre,
    a, abbr, acronym, address, big, cite, code,
    del, dfn, em, img, ins, kbd, q, s, samp,
    small, strike, strong, sub, sup, tt, var,
    b, u, i, center,
    dl, dt, dd, ol, ul, li,
    fieldset, form, label, legend,
    table, caption, tbody, tfoot, thead, tr, th, td,
    article, aside, canvas, details, embed, 
    figure, figcaption, footer, header, hgroup, 
    menu, nav, output, ruby, section, summary,
    time, mark, audio, video {
        margin: 0;
        padding: 0;
        border: 0;
        font-size: 100%;
        font: inherit;
        vertical-align: baseline;
        box-sizing: border-box; /* Force consistent box model */
    }

    /* 3. HARDCODE ROOT STYLES --- Enforce your styles everywhere! */
    :root {
        --primary-color: #00C0F2 !important;
        --background-color: #0f2027 !important;
        --secondary-background-color: #203a43 !important;
        --text-color: #FFFFFF !important;
        --font-family: 'Inter', sans-serif !important;
    }

    body, .stApp {
        font-family: var(--font-family);
        color: var(--text-color);
        background: var(--background-color);
        background: -webkit-linear-gradient(to right, #2c5364, var(--secondary-background-color), var(--background-color));
        background: linear-gradient(to right, #2c5364, var(--secondary-background-color), var(--background-color));
    }
    
    /* Force text color on all text elements */
    p, h1, h2, h3, h4, h5, h6, li, span, div, label, button, input {
        color: var(--text-color);
    }
    
    /* 4. WIDGET OVERRIDES --- Style all Streamlit components */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: var(--primary-color);
    }
    .stRadio [data-baseweb="radio"] span:first-child {
        background-color: var(--primary-color);
        border: 2px solid var(--primary-color) !important; /* Ensure border is also styled */
    }
    div.stButton > button {
        background-color: var(--primary-color);
        color: var(--text-color);
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: var(--primary-color);
    }
    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
    }
    
    /* 5. CUSTOM COMPONENT STYLING */
    .prediction-card {
        background: linear-gradient(135deg, rgba(0, 192, 242, 0.9) 0%, rgba(0, 130, 169, 0.9) 100%);
        padding: 2.5rem;
        border-radius: 20px;
    }
    .prediction-card * { /* Apply to all text inside the card */
         color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.markdown("## 🚲 **About the App**")
    st.markdown("This **Forecaster Pro** app uses a Random Forest model to predict daily bike rental demand. It leverages live weather data and provides advanced analytics to understand the factors driving the forecast.")
    st.markdown("---")
    st.info("💡 **Tip:** Use the 'What-If Analysis' tab to explore how changes in weather impact rental predictions!")
    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit")


# ==============================
# Main App UI
# ==============================
st.markdown("<h1 class='main-title'>📈 Bike Rental Forecaster Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Enter your parameters to generate a forecast, then explore the prediction insights.</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "**🚀 Generate Forecast**", 
    "**📊 Prediction Insights**", 
    "**🔬 What-If Analysis**"
])

# ---- TAB 1: GENERATE FORECAST ----
with tab1:
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.subheader("⚙️ Input Parameters")
        with st.expander("🌦️ **Live Weather Input (Optional)**", expanded=True):
            api_key = "4ccd08ecc324ec92d5b942c8ccb48bb3"
            city = st.text_input("Enter a city to fetch live weather", "Sonipat", help="Fetches current temperature, humidity, and wind speed.")
            
            if st.button("Fetch & Auto-Fill Weather"):
                with st.spinner("Fetching live weather data..."):
                    weather_data = get_live_weather(city, api_key)
                    if weather_data:
                        st.session_state.temp = float(weather_data['temp'])
                        st.session_state.hum = float(weather_data['hum'])
                        st.session_state.windspeed = float(weather_data['windspeed'])
                        
                        api_condition = weather_data['weathersit_main']
                        if api_condition in ["Rain", "Drizzle", "Snow", "Thunderstorm"]:
                            st.session_state.weather_idx = 2
                        elif api_condition in ["Clouds", "Mist", "Fog", "Haze"]:
                            st.session_state.weather_idx = 1
                        else:
                            st.session_state.weather_idx = 0
                        st.toast(f"✅ Weather for {city.title()} auto-filled!", icon="🌤️")

            c1, c2, c3 = st.columns(3)
            with c1:
                temp = st.slider("Temperature (°C)", -8.0, 39.0, value=st.session_state.get('temp', 25.0), step=0.1)
            with c2:
                hum = st.slider("Humidity (%)", 0.0, 100.0, value=st.session_state.get('hum', 60.0), step=0.1)
            with c3:
                windspeed = st.slider("Wind Speed (km/h)", 0.0, 67.0, value=st.session_state.get('windspeed', 15.0), step=0.1)

        c1, c2 = st.columns(2)
        with c1:
            with st.expander("🗓️ **Date & Season**", expanded=True):
                today = datetime.date.today()
                yr_map = {"2011": 0, "2012": 1}
                yr = st.selectbox("Year (for model)", list(yr_map.keys()), index=1)
                season_map = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}
                season_default_idx = (today.month % 12 // 3) 
                season = st.selectbox("Season", list(season_map.keys()), index=season_default_idx)
                mnth = st.slider("Month", 1, 12, today.month)
        
        with c2:
            with st.expander("📋 **Day Type & Weather Condition**", expanded=True):
                weather_map = {"Clear / Few clouds": 1, "Mist / Cloudy": 2, "Light Snow / Rain": 3}
                weathersit_key = st.selectbox("Weather Condition", list(weather_map.keys()), index=st.session_state.get('weather_idx', 0))
                weekday_map = {"Sun": 0, "Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6}
                weekday_default_idx = (today.weekday() + 1) % 7
                weekday = st.selectbox("Day of Week", list(weekday_map.keys()), index=weekday_default_idx)
                holiday_map = {"No Holiday": 0, "Holiday": 1}
                holiday = st.radio("Holiday?", list(holiday_map.keys()), horizontal=True, index=0)
                workingday_map = {"Not Working Day": 0, "Working Day": 1}
                workingday = st.radio("Working Day?", list(workingday_map.keys()), horizontal=True, index=1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔮 **Generate Forecast**", use_container_width=True):
            with st.spinner("🧠 Running advanced forecast model..."):
                user_input = {
                    'season': season_map[season], 'yr': yr_map[yr], 'mnth': mnth,
                    'holiday': holiday_map[holiday], 'weekday': weekday_map[weekday],
                    'workingday': workingday_map[workingday], 'weathersit': weather_map[weathersit_key],
                    'temp': temp / 39.0, 'atemp': ((temp * 0.95) + 3) / 48.0,
                    'hum': hum / 100.0, 'windspeed': windspeed / 67.0,
                }
                input_df = pd.DataFrame([user_input], columns=feature_cols)
                mean, lower, upper, dist = get_prediction_distribution(model, input_df)
                
                st.session_state.prediction_made = True
                st.session_state.prediction_mean = mean
                st.session_state.prediction_range = f"{lower} – {upper}"
                st.session_state.prediction_dist = dist
                st.session_state.last_user_input = user_input
            st.toast("Forecast generated successfully!", icon="✅")

    with col2:
        st.subheader("📊 Forecast Result")
        if not st.session_state.get("prediction_made", False):
            st.info("Click 'Generate Forecast' to see the result here.")
        else:
            st.markdown(f"""
            <div class="prediction-card">
                <h4>Estimated Daily Rentals</h4>
                <h2>🚲 {st.session_state.prediction_mean}</h2>
                <p>95% Confidence Range: <strong>{st.session_state.prediction_range}</strong></p>
            </div>
            """, unsafe_allow_html=True)

# ---- TAB 2: PREDICTION INSIGHTS ----
with tab2:
    st.subheader("💡 Understanding the Prediction")
    if not st.session_state.get("prediction_made", False):
        st.info("Please generate a forecast on the 'Generate Forecast' tab first to see insights here.")
    else:
        st.markdown("##### 🎲 **Prediction Distribution**")
        st.markdown("This histogram shows the predictions from individual decision trees in the model. A narrower distribution indicates higher model confidence.")
        fig_dist = px.histogram(x=st.session_state.prediction_dist, nbins=30)
        fig_dist.update_traces(marker_color='#00C0F2')
        fig_dist.add_vline(x=st.session_state.prediction_mean, line_width=3, line_dash="dash", line_color="red", annotation_text="Mean Prediction")
        fig_dist.update_layout(
            title_text='Distribution of Predictions Across All Decision Trees',
            xaxis_title_text='Predicted Rentals per Tree', yaxis_title_text='Frequency',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white'
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")
        
        st.markdown("##### 🌳 **Global Feature Importance**")
        st.markdown("This chart shows which factors generally have the biggest impact on bike rentals, according to the model.")
        fig_importance = px.bar(
            feature_importance_df.head(10), x='importance', y='feature', orientation='h',
            title='Top 10 Most Important Prediction Factors', text_auto='.3f'
        )
        fig_importance.update_traces(marker_color='#00C0F2', textposition='outside')
        fig_importance.update_layout(
            yaxis={'categoryorder':'total ascending'}, title_x=0.5,
            xaxis_title='Importance Score', yaxis_title='Factor',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white'
        )
        st.plotly_chart(fig_importance, use_container_width=True)

# ---- TAB 3: WHAT-IF ANALYSIS ----
with tab3:
    st.subheader("🔬 What-If Analysis")
    if not st.session_state.get("prediction_made", False):
        st.info("Please generate a forecast on the 'Generate Forecast' tab first to perform a what-if analysis.")
    else:
        st.markdown("Select a feature to vary and see how it impacts the rental prediction, holding all other inputs constant.")
        
        selectable_features = {
            friendly_name: tech_name 
            for tech_name, friendly_name in friendly_feature_names.items() 
            if tech_name in ['temp', 'hum', 'windspeed']
        }
        
        feature_to_vary_friendly = st.selectbox("Select a feature to analyze:", options=list(selectable_features.keys()))
        feature_to_vary = selectable_features[feature_to_vary_friendly]

        if feature_to_vary == 'temp': values_range = np.linspace(-8, 39, 50)
        elif feature_to_vary == 'hum': values_range = np.linspace(0, 100, 50)
        else: values_range = np.linspace(0, 67, 50)

        predictions = []
        base_input = st.session_state.last_user_input.copy()
        
        for val in values_range:
            current_input = base_input.copy()
            if feature_to_vary == 'temp':
                current_input['temp'] = val / 39.0
                current_input['atemp'] = ((val * 0.95) + 3) / 48.0
            elif feature_to_vary == 'hum': current_input['hum'] = val / 100.0
            elif feature_to_vary == 'windspeed': current_input['windspeed'] = val / 67.0
            
            input_df = pd.DataFrame([current_input], columns=feature_cols)
            prediction = model.predict(input_df)[0]
            predictions.append(prediction)
            
        fig_what_if = go.Figure()

        fig_what_if.add_trace(go.Scatter(x=values_range, y=predictions, mode='lines', name='Predicted Rentals', line=dict(color='#00C0F2')))
        
        if feature_to_vary == 'temp': current_val = st.session_state.last_user_input[feature_to_vary] * 39.0
        elif feature_to_vary == 'hum': current_val = st.session_state.last_user_input[feature_to_vary] * 100.0
        else: current_val = st.session_state.last_user_input[feature_to_vary] * 67.0
        
        fig_what_if.add_trace(go.Scatter(
            x=[current_val], y=[st.session_state.prediction_mean],
            mode='markers', name='Your Current Input',
            marker=dict(color='red', size=15, symbol='x')
        ))
        fig_what_if.update_layout(
            title_text=f"Impact of {feature_to_vary_friendly} on Bike Rentals",
            xaxis_title=feature_to_vary_friendly,
            yaxis_title="Predicted Number of Rentals",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white'
        )
        st.plotly_chart(fig_what_if, use_container_width=True)
