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
    
    # Calculate global feature importances
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # A dictionary to map technical feature names to user-friendly names
    friendly_feature_names = {
        'temp': 'Temperature (°C)', 'hum': 'Humidity (%)', 'windspeed': 'Wind Speed (km/h)',
        'season': 'Season', 'yr': 'Year', 'mnth': 'Month', 'weathersit': 'Weather Situation'
    }
    
    return model, feature_cols, feature_importance_df, friendly_feature_names

model, feature_cols, feature_importance_df, friendly_feature_names = load_model_and_data()

# Stop the app if model loading fails
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
        response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)
        data = response.json()
        return {
            "temp": data['main']['temp'], "hum": data['main']['humidity'],
            "windspeed": data['wind']['speed'] * 3.6, # Convert m/s to km/h
            "weathersit_main": data['weather'][0]['main']
        }
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Weather API Error: Could not fetch weather for '{city}'. Please check the city name. (HTTP {http_err.response.status_code})")
        return None
    except Exception as e:
        st.error(f"An error occurred while fetching weather data: {e}")
        return None

def get_prediction_distribution(model, input_df):
    """
    Calculates the prediction from each tree in the Random Forest to get a distribution,
    mean, and a 95% confidence interval.
    """
    tree_predictions = [tree.predict(input_df)[0] for tree in model.estimators_]
    mean_prediction = np.mean(tree_predictions)
    std_dev = np.std(tree_predictions)
    
    # Calculate 95% confidence interval
    lower_bound = max(0, mean_prediction - 1.96 * std_dev)
    upper_bound = mean_prediction + 1.96 * std_dev
    
    return int(mean_prediction), int(lower_bound), int(upper_bound), tree_predictions

# ==============================
# CSS Styling
# ==============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    body, .stApp {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: #0f2027;
        background: -webkit-linear-gradient(to right, #2c5364, #203a43, #0f2027);
        background: linear-gradient(to right, #2c5364, #203a43, #0f2027);
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        text-align: center;
        color: #a0a0a0;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00C0F2;
    }
    div.stButton > button {
        background-color: #00C0F2;
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 0.8rem 1.5rem;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: #00a8d6;
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0, 192, 242, 0.3);
    }
    .prediction-card {
        background: linear-gradient(135deg, rgba(0, 192, 242, 0.9) 0%, rgba(0, 130, 169, 0.9) 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    .prediction-card h4 {
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 400;
        opacity: 0.9;
    }
    .prediction-card h2 {
        margin: 0;
        font-size: 4rem;
        font-weight: 700;
        letter-spacing: -2px;
    }
    .prediction-card p {
        margin-top: 1rem;
        font-size: 1.1rem;
        opacity: 0.9;
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
    st.markdown("Developed with ❤️ by siddhant ")


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
    col1, col2 = st.columns([6, 4]) # Give more space to inputs
    
    with col1:
        st.subheader("⚙️ Input Parameters")

        # Live Weather Section
        with st.expander("🌦️ **Live Weather Input (Optional)**", expanded=True):
            api_key = "4ccd08ecc324ec92d5b942c8ccb48bb3" # Best practice: use st.secrets for API keys
            city = st.text_input("Enter a city to fetch live weather", "Sonipat", help="Fetches current temperature, humidity, and wind speed.")
            
            if st.button("Fetch & Auto-Fill Weather"):
                with st.spinner("Fetching live weather data..."):
                    weather_data = get_live_weather(city, api_key)
                    if weather_data:
                        st.session_state.temp = weather_data['temp']
                        st.session_state.hum = weather_data['hum']
                        st.session_state.windspeed = weather_data['windspeed']
                        
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

        # Date & Day Type Section
        c1, c2 = st.columns(2)
        with c1:
            with st.expander("🗓️ **Date & Season**", expanded=True):
                today = datetime.date.today()
                
                yr_map = {"2011": 0, "2012": 1}
                yr = st.selectbox("Year (for model)", list(yr_map.keys()), index=1)
                
                season_map = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}
                # Default season based on current month
                season_default_idx = (today.month % 12 // 3) 
                season = st.selectbox("Season", list(season_map.keys()), index=season_default_idx)
                
                mnth = st.slider("Month", 1, 12, today.month)
        
        with c2:
            with st.expander("📋 **Day Type & Weather Condition**", expanded=True):
                weather_map = {"Clear / Few clouds": 1, "Mist / Cloudy": 2, "Light Snow / Rain": 3}
                weathersit_key = st.selectbox("Weather Condition", list(weather_map.keys()), index=st.session_state.get('weather_idx', 0))

                # Default day of week based on today
                weekday_map = {"Sun": 0, "Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6}
                weekday_default_idx = (today.weekday() + 1) % 7
                weekday = st.selectbox("Day of Week", list(weekday_map.keys()), index=weekday_default_idx)
                
                holiday_map = {"No Holiday": 0, "Holiday": 1}
                holiday = st.radio("Holiday?", list(holiday_map.keys()), horizontal=True, index=0)
                
                workingday_map = {"Not Working Day": 0, "Working Day": 1}
                workingday = st.radio("Working Day?", list(workingday_map.keys()), horizontal=True, index=1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # The main prediction button
        if st.button("🔮 **Generate Forecast**", use_container_width=True):
            with st.spinner("🧠 Running advanced forecast model..."):
                # Prepare input data for the model
                # Normalization values are assumed from the original dataset's min/max
                user_input = {
                    'season': season_map[season], 'yr': yr_map[yr], 'mnth': mnth,
                    'holiday': holiday_map[holiday], 'weekday': weekday_map[weekday],
                    'workingday': workingday_map[workingday], 'weathersit': weather_map[weathersit_key],
                    'temp': temp / 39.0, # Normalizing based on max value
                    'atemp': ((temp * 0.95) + 3) / 48.0, # A more realistic atemp conversion and normalization
                    'hum': hum / 100.0,
                    'windspeed': windspeed / 67.0,
                }
                # Create a DataFrame with columns in the same order as the model was trained on
                input_df = pd.DataFrame([user_input], columns=feature_cols)

                mean, lower, upper, dist = get_prediction_distribution(model, input_df)
                
                # Store results in session state to use across tabs
                st.session_state.prediction_made = True
                st.session_state.prediction_mean = mean
                st.session_state.prediction_range = f"{lower} – {upper}"
                st.session_state.prediction_dist = dist
                st.session_state.last_input_df = input_df
                st.session_state.last_user_input = user_input # Store for What-If analysis
            
            st.toast("Forecast generated successfully!", icon="✅")

    # This column displays the prediction result
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
        st.markdown("This histogram shows the predictions from each of the individual decision trees in the Random Forest model. A narrower distribution indicates higher model confidence in the forecast.")
        
        fig_dist = px.histogram(
            x=st.session_state.prediction_dist, 
            nbins=30,
            labels={'x': 'Predicted Rentals per Tree', 'y': 'Frequency'},
            title='Distribution of Predictions Across All Decision Trees'
        )
        fig_dist.add_vline(x=st.session_state.prediction_mean, line_width=3, line_dash="dash", line_color="red", annotation_text="Mean Prediction")
        fig_dist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#FFFFFF")
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")
        
        st.markdown("##### 🌳 **Global Feature Importance**")
        st.markdown("The chart below shows which factors generally have the biggest impact on the number of bike rentals, according to the model.")
        
        fig_importance = px.bar(
            feature_importance_df.head(10), x='importance', y='feature', orientation='h',
            title='Top 10 Most Important Prediction Factors',
            labels={'importance': 'Importance Score', 'feature': 'Factor'}, text='importance'
        )
        fig_importance.update_layout(
            yaxis={'categoryorder':'total ascending'}, title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#FFFFFF")
        )
        fig_importance.update_traces(texttemplate='%{text:.3f}', textposition='outside', marker_color='#00C0F2')
        st.plotly_chart(fig_importance, use_container_width=True)

# ---- TAB 3: WHAT-IF ANALYSIS ----
with tab3:
    st.subheader("🔬 What-If Analysis")
    if not st.session_state.get("prediction_made", False):
        st.info("Please generate a forecast on the 'Generate Forecast' tab first to perform a what-if analysis.")
    else:
        st.markdown("Select a feature to vary and see how it impacts the rental prediction, holding all other inputs constant.")
        
        # Get user-friendly names for selectable features
        selectable_features = {
            friendly_name: tech_name 
            for tech_name, friendly_name in friendly_feature_names.items() 
            if tech_name in ['temp', 'hum', 'windspeed']
        }
        
        feature_to_vary_friendly = st.selectbox(
            "Select a feature to analyze:",
            options=list(selectable_features.keys())
        )
        feature_to_vary = selectable_features[feature_to_vary_friendly]

        # Define ranges for analysis
        if feature_to_vary == 'temp':
            values_range = np.linspace(-8, 39, 50)
        elif feature_to_vary == 'hum':
            values_range = np.linspace(0, 100, 50)
        else: # windspeed
            values_range = np.linspace(0, 67, 50)

        # Run predictions for the range
        predictions = []
        base_input = st.session_state.last_user_input.copy()
        
        for val in values_range:
            # Create a copy of the base input
            current_input = base_input.copy()
            
            # Update the varying feature with its normalized value
            if feature_to_vary == 'temp':
                current_input['temp'] = val / 39.0
                current_input['atemp'] = ((val * 0.95) + 3) / 48.0
            elif feature_to_vary == 'hum':
                current_input['hum'] = val / 100.0
            elif feature_to_vary == 'windspeed':
                current_input['windspeed'] = val / 67.0
            
            # Make prediction
            input_df = pd.DataFrame([current_input], columns=feature_cols)
            prediction = model.predict(input_df)[0]
            predictions.append(prediction)
            
        # Create the plot
        fig_what_if = go.Figure()

        # Add the line for the what-if analysis
        fig_what_if.add_trace(go.Scatter(
            x=values_range, y=predictions, mode='lines',
            name='Predicted Rentals', line=dict(color='#00C0F2', width=3)
        ))
        
        # Add a marker for the current user input
        current_val_raw = st.session_state.last_user_input[feature_to_vary]
        # De-normalize current value to plot it
        if feature_to_vary == 'temp': current_val = current_val_raw * 39.0
        elif feature_to_vary == 'hum': current_val = current_val_raw * 100.0
        else: current_val = current_val_raw * 67.0
        
        fig_what_if.add_trace(go.Scatter(
            x=[current_val], y=[st.session_state.prediction_mean],
            mode='markers', name='Your Current Input',
            marker=dict(color='red', size=15, symbol='x')
        ))

        fig_what_if.update_layout(
            title=f"Impact of {feature_to_vary_friendly} on Bike Rentals",
            xaxis_title=feature_to_vary_friendly,
            yaxis_title="Predicted Number of Rentals",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#FFFFFF")
        )
        st.plotly_chart(fig_what_if, use_container_width=True)
