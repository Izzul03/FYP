import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import calendar

# -------------------------
# Configuration
# -------------------------
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="🌾 Climate Impact & Food Security Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E7D32;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #4CAF50;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #388E3C;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .highlight-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .risk-low {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">Climate Impact on Malaysian Food Availability Dashboard</div>',
            unsafe_allow_html=True)


# -------------------------
# Helper Functions
# -------------------------
def clean_numeric(series: pd.Series) -> pd.Series:
    """Clean numeric columns with various formatting issues"""
    return pd.to_numeric(
        series.astype(str).str.replace("\u00A0", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", np.nan),
        errors="coerce"
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and preprocess the crop data"""
    try:
        # Try multiple possible file locations
        possible_paths = [
            "/Users/izzulfidaey/Desktop/FYP/crops_state.csv",
            "crops_state.csv",
            "./crops_state.csv",
            "../crops_state.csv",
            "./data/crops_state.csv"
        ]

        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break

        if file_path is None:
            st.sidebar.warning("CSV file not found in default locations.")
            uploaded_file = st.sidebar.file_uploader("Upload crops_state.csv", type="csv", key="file_uploader")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, encoding="latin1")
                st.sidebar.success("File uploaded successfully!")
            else:
                # Create sample data if no file is found
                st.sidebar.info("Using sample data for demonstration.")
                return create_sample_data()
        else:
            df = pd.read_csv(file_path, encoding="latin1")

        # Basic data cleaning
        df.columns = [c.strip() for c in df.columns]

        # Clean state names
        if "state" in df.columns:
            df["state"] = df["state"].astype(str).str.strip().str.title()

        # Clean crop types
        if "crop_type" in df.columns:
            df["crop_type"] = df["crop_type"].astype(str).str.strip().str.title()
            # Standardize crop names
            crop_mapping = {
                "Cash_Crops": "Corn",
                "Industrial_Crops": "Palm Oil",
                "cash_crops": "Corn",
                "industrial_crops": "Palm Oil",
                "Cash Crops": "Corn",
                "Industrial Crops": "Palm Oil"
            }
            df["crop_type"] = df["crop_type"].replace(crop_mapping)

        # Clean numeric columns
        numeric_cols = ["temperature", "humidity", "production", "planted_area"]
        for col in numeric_cols:
            if col in df.columns:
                # Remove special characters
                df[col] = df[col].astype(str).str.replace("°C", "", regex=False)
                df[col] = df[col].astype(str).str.replace("%", "", regex=False)
                df[col] = df[col].astype(str).str.replace("。", "", regex=False)
                df[col] = df[col].astype(str).str.replace("吽C", "", regex=False)
                df[col] = clean_numeric(df[col])

        # Parse dates
        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
            except:
                try:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                except:
                    df["date"] = pd.NaT

            df["year"] = df["date"].dt.year.fillna(2017).astype(int)
            df["month"] = df["date"].dt.month.fillna(1).astype(int)

        # Calculate yield efficiency
        if {"production", "planted_area"}.issubset(df.columns):
            df["yield_efficiency"] = np.where(
                df["planted_area"] > 0,
                df["production"] / df["planted_area"],
                np.nan
            )

        # State coordinates
        STATE_COORDS = {
            "Johor": (1.4927, 103.7414),
            "Kedah": (6.1254, 100.3678),
            "Kelantan": (6.1252, 102.2382),
            "Melaka": (2.1896, 102.2501),
            "Negeri Sembilan": (2.7254, 101.9420),
            "Pahang": (3.8079, 102.5485),
            "Penang": (5.4164, 100.3327),
            "Perak": (4.5975, 101.0901),
            "Perlis": (6.4425, 100.2083),
            "Sabah": (5.9804, 116.0735),
            "Sarawak": (1.5533, 110.3593),
            "Selangor": (3.0738, 101.5183),
            "Terengganu": (5.3333, 103.1333),
            "W.P. Kuala Lumpur": (3.1390, 101.6869),
            "W.P. Labuan": (5.2830, 115.2340),
            "W.P. Putrajaya": (2.9264, 101.6963),
            "Malaysia": (4.2105, 101.9758)
        }

        df["lat"] = df["state"].map(lambda s: STATE_COORDS.get(s, (np.nan, np.nan))[0])
        df["lon"] = df["state"].map(lambda s: STATE_COORDS.get(s, (np.nan, np.nan))[1])

        # Fill missing years if needed
        if df["year"].isna().any():
            df["year"] = df["year"].fillna(2017)

        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return sample data as fallback
        return create_sample_data()


def create_sample_data():
    """Create sample data for demonstration"""
    states = ["Johor", "Kedah", "Kelantan", "Melaka", "Selangor"]
    crops = ["Corn", "Palm Oil", "Vegetables", "Fruits", "Paddy"]
    years = [2017, 2018, 2019, 2020, 2021, 2022]

    data = []
    for state in states:
        for crop in crops:
            for year in years:
                base_production = np.random.randint(10000, 100000)
                base_area = np.random.randint(1000, 10000)
                temp = np.random.uniform(25, 32)
                humidity = np.random.uniform(70, 90)

                data.append({
                    "state": state,
                    "crop_type": crop,
                    "year": year,
                    "production": base_production * (1 + 0.05 * (year - 2017)),
                    "planted_area": base_area,
                    "temperature": temp + (year - 2017) * 0.1,  # Slight warming trend
                    "humidity": humidity,
                    "lat": 4.0 + np.random.uniform(-1, 1),
                    "lon": 102.0 + np.random.uniform(-2, 2)
                })

    df = pd.DataFrame(data)
    df["yield_efficiency"] = df["production"] / df["planted_area"]
    return df


# Load data
df = load_data()

if df.empty:
    st.error("No data available. Please check your data file.")
    st.stop()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header(" Dashboard Controls")

# Multi-tab selector
tab_selection = st.sidebar.radio(
    "Select Dashboard View:",
    ["📊 Summary & Overview", "🔍 Data Exploration", "🎯 Climate Simulation"]
)

# Common filters
state_options = sorted(df["state"].dropna().unique())
state_options = [s for s in state_options if s.lower() != "malaysia"]

crop_options = sorted(df["crop_type"].dropna().unique())


# Default selections
default_states = state_options
default_crops = crop_options

states_selected = st.sidebar.multiselect(
    "Select State(s):",
    options=state_options,
    default=default_states
)

crops_selected = st.sidebar.multiselect(
    "Select Crop Type(s):",
    options=crop_options,
    default=default_crops
)

# Year range
min_year = int(df["year"].min()) if df["year"].notnull().any() else 2017
max_year = int(df["year"].max()) if df["year"].notnull().any() else 2022
year_range = st.sidebar.slider(
    "Year Range:",
    min_year, max_year, (min_year, max_year)
)

# Apply filters
filtered = df[
    (df["state"].isin(states_selected)) &
    (df["crop_type"].isin(crops_selected)) &
    (df["year"].between(year_range[0], year_range[1]))
    ].copy()

if filtered.empty:
    st.warning("No data after applying filters. Please adjust your selections.")
    # Show available data info
    st.info(f"Available data: {len(df)} records, {len(state_options)} states, {len(crop_options)} crops")
    st.stop()

# Calculate additional metrics
if "temperature" in filtered.columns and "humidity" in filtered.columns:
    filtered["climate_stress_index"] = (
            (filtered["temperature"] - filtered["temperature"].mean()) +
            (filtered["humidity"] - filtered["humidity"].mean())
    )

if "yield_efficiency" not in filtered.columns and "planted_area" in filtered.columns and "production" in filtered.columns:
    filtered["yield_efficiency"] = np.where(
        filtered["planted_area"] > 0,
        filtered["production"] / filtered["planted_area"],
        np.nan
    )

# Calculate YoY changes
filtered = filtered.sort_values(["state", "crop_type", "year"])
filtered["yoy_growth"] = filtered.groupby(["state", "crop_type"])["production"].pct_change() * 100
filtered["yoy_temp_change"] = filtered.groupby(["state", "crop_type"])["temperature"].diff()
filtered["yoy_humidity_change"] = filtered.groupby(["state", "crop_type"])["humidity"].diff()

# -------------------------
# TAB 1: SUMMARY & OVERVIEW PAGE
# -------------------------
if tab_selection == "📊 Summary & Overview":
    st.markdown('<div class="sub-header">National Overview & Key Insights</div>', unsafe_allow_html=True)

    # --- [NEW] 1. CALCULATE YoY METRICS ---
    available_years = sorted(filtered['year'].unique())
    if len(available_years) >= 2:
        latest_year = available_years[-1]
        prev_year = available_years[-2]

        # Get data for calculation
        latest_data = filtered[filtered['year'] == latest_year]
        prev_data = filtered[filtered['year'] == prev_year]

        # Calculate Deltas
        prod_delta = ((latest_data['production'].sum() - prev_data['production'].sum()) / prev_data[
            'production'].sum()) * 100 if prev_data['production'].sum() != 0 else 0
        yield_delta = ((latest_data['yield_efficiency'].mean() - prev_data['yield_efficiency'].mean()) / prev_data[
            'yield_efficiency'].mean()) * 100 if prev_data['yield_efficiency'].mean() != 0 else 0
        temp_delta = latest_data['temperature'].mean() - prev_data['temperature'].mean()
    else:
        latest_year = available_years[0] if available_years else "N/A"
        prev_year = "N/A"
        prod_delta = yield_delta = temp_delta = 0

    # Display data info
    st.sidebar.info(f"Showing {len(filtered)} records")

    # --- [NEW] 2. UPDATED KPI CARDS ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Production", f"{filtered['production'].sum():,.0f} MT", f"{prod_delta:+.1f}% (Year-over-Year)")
    with kpi2:
        st.metric("Avg Yield Efficiency", f"{filtered['yield_efficiency'].mean():.2f} T/Ha",
                  f"{yield_delta:+.1f}% (Year-over-Year)")
    with kpi3:
        st.metric("Avg Temperature", f"{filtered['temperature'].mean():.1f}°C", f"{temp_delta:+.1f}°C",
                  delta_color="inverse")
    with kpi4:
        st.metric("Total Planted Area", f"{filtered['planted_area'].sum():,.0f} Ha", "Cumulative")

    # Additional KPIs
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        avg_humidity = filtered["humidity"].mean(skipna=True)
        st.metric("Avg Humidity", f"{avg_humidity:.1f}%")

    with col6:
        avg_yield = filtered["yield_efficiency"].mean(skipna=True)
        st.metric("Avg Yield", f"{avg_yield:.2f} MT/Ha")

    with col7:
        num_states = filtered["state"].nunique()
        st.metric("States Covered", f"{num_states}")

    with col8:
        num_crops = filtered["crop_type"].nunique()
        st.metric("Crop Types", f"{num_crops}")

    # National Trends Row
    st.markdown("---")
    st.markdown('<div class="sub-header">National Trends Analysis</div>', unsafe_allow_html=True)

    # Prepare national data (exclude Malaysia aggregate if present)
    national_data = filtered[~filtered["state"].str.contains("Malaysia", case=False, na=False)]

    if not national_data.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Production trend over time
            prod_trend = national_data.groupby("year").agg({
                "production": "sum",
                "temperature": "mean"
            }).reset_index()

            # Create dual axis chart
            base = alt.Chart(prod_trend).encode(
                x=alt.X("year:O", title="Year")
            )

            production_line = base.mark_line(color="#2E7D32", strokeWidth=3).encode(
                y=alt.Y("production:Q", title="Production (MT)", scale=alt.Scale(zero=False)),
                tooltip=["year", "production"]
            ).properties(
                title="Annual Production Trend"
            )

            st.altair_chart(production_line, use_container_width=True)

            # Temperature trend
            temp_line = base.mark_line(color="#FF6B6B", strokeWidth=2, strokeDash=[5, 5]).encode(
                y=alt.Y("temperature:Q", title="Temperature (°C)", scale=alt.Scale(zero=False)),
                tooltip=["year", "temperature"]
            ).properties(
                title="Average Temperature Trend"
            )

            st.altair_chart(temp_line, use_container_width=True)

        with col2:
            # Crop composition
            crop_composition = national_data.groupby("crop_type").agg({
                "production": "sum",
                "planted_area": "sum"
            }).reset_index()

            crop_composition["yield"] = crop_composition["production"] / crop_composition["planted_area"]
            crop_composition = crop_composition.sort_values("production", ascending=False)

            # Production by crop
            bars = alt.Chart(crop_composition).mark_bar().encode(
                x=alt.X("production:Q", title="Production (MT)"),
                y=alt.Y("crop_type:N", sort="-x", title="Crop Type"),
                color=alt.Color("crop_type:N", legend=None),
                tooltip=["crop_type", "production", "yield"]
            ).properties(
                title="Production by Crop Type",
                height=350
            )

            st.altair_chart(bars, use_container_width=True)

            # Yield by crop
            yield_chart = alt.Chart(crop_composition).mark_bar(color="#FF9800").encode(
                x=alt.X("crop_type:N", title="Crop Type", sort="-y"),
                y=alt.Y("yield:Q", title="Yield (MT/Ha)"),
                tooltip=["crop_type", "yield"]
            ).properties(
                title="Yield Efficiency by Crop",
                height=350
            )

            st.altair_chart(yield_chart, use_container_width=True)

    # Replace the entire Heat Sensitivity Analysis section with this corrected version:

    # Heat Sensitivity Analysis
    st.markdown("---")
    st.markdown('<div class="sub-header">Heat Sensitivity Analysis</div>', unsafe_allow_html=True)

    # Calculate heat sensitivity
    sensitivity_data = filtered.dropna(subset=["temperature", "production", "planted_area"])

    if len(sensitivity_data) > 1:
        sensitivity_results = []

        for crop in sensitivity_data["crop_type"].unique():
            crop_data = sensitivity_data[sensitivity_data["crop_type"] == crop]

            if len(crop_data) > 2:
                # Group by year
                yearly_agg = crop_data.groupby("year").agg({
                    "temperature": "mean",
                    "production": "sum",
                    "planted_area": "sum"
                }).reset_index()

                yearly_agg = yearly_agg.sort_values("year")
                yearly_agg["yield"] = yearly_agg["production"] / yearly_agg["planted_area"]

                if len(yearly_agg) > 1:
                    # Calculate correlations
                    temp_yield_corr = yearly_agg["temperature"].corr(yearly_agg["yield"])

                    # Classify sensitivity
                    if temp_yield_corr < -0.5:
                        risk_level = "🔴 High Risk"
                        risk_color = "red"
                    elif temp_yield_corr < -0.2:
                        risk_level = "🟡 Medium Risk"
                        risk_color = "orange"
                    else:
                        risk_level = "🟢 Low Risk"
                        risk_color = "green"

                    sensitivity_results.append({
                        "Crop": crop,
                        "Temperature-Yield Correlation": temp_yield_corr,
                        "Risk Level": risk_level,
                        "Risk Color": risk_color,
                        "Avg Temp (°C)": round(yearly_agg['temperature'].mean(), 1),
                        "Avg Yield (MT/Ha)": round(yearly_agg['yield'].mean(), 2)
                    })

        if sensitivity_results:
            # Create a DataFrame
            sensitivity_df = pd.DataFrame(sensitivity_results)

            # Sort by correlation (most negative first)
            sensitivity_df = sensitivity_df.sort_values("Temperature-Yield Correlation")

            # Display as a styled dataframe
            st.dataframe(
                sensitivity_df[["Crop", "Temperature-Yield Correlation", "Risk Level", "Avg Temp (°C)",
                                "Avg Yield (MT/Ha)"]].style.format({
                    "Temperature-Yield Correlation": "{:.3f}",
                    "Avg Temp (°C)": "{:.1f}",
                    "Avg Yield (MT/Ha)": "{:.2f}"
                }).apply(
                    lambda x: [f'color: {row["Risk Color"]}' for _, row in sensitivity_df.iterrows()]
                    if x.name == "Risk Level" else [''] * len(x),
                    axis=0
                ),
                use_container_width=True
            )

            # Alternative: Create a bar chart for visualization
            st.markdown("#### Visual Correlation Analysis")

            # Create bar chart showing correlations
            corr_chart = alt.Chart(sensitivity_df).mark_bar().encode(
                x=alt.X("Crop:N", sort="-y", title="Crop Type"),
                y=alt.Y("Temperature-Yield Correlation:Q", title="Correlation Coefficient"),
                color=alt.Color("Risk Level:N",
                                scale=alt.Scale(
                                    domain=["🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"],
                                    range=["#ff4444", "#ffaa44", "#44aa44"]
                                )),
                tooltip=["Crop", "Temperature-Yield Correlation", "Risk Level", "Avg Temp (°C)", "Avg Yield (MT/Ha)"]
            ).properties(
                title="Temperature-Yield Correlation by Crop",
                height=400
            )

            # Add a zero line for reference
            zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', strokeDash=[5, 5]).encode(y='y:Q')

            st.altair_chart(corr_chart + zero_line, use_container_width=True)

            st.markdown("""
            <div class="highlight-box" style="background-color: #1a472a; padding: 15px; border-radius: 10px; border-left: 5px solid #2e8b57; color: white;">
            <h4 style="color: white; margin-top: 0;">📝 How to interpret these results:</h4>

            <div style="display: flex; align-items: center; margin: 10px 0;">
                <span style="color: #90ee90; font-size: 20px; margin-right: 10px;">🟢</span>
                <span style="color: white;"><b>Low Risk (Correlation > -0.2):</b> Temperature has minimal impact on yield</span>
            </div>

            <div style="display: flex; align-items: center; margin: 10px 0;">
                <span style="color: #ffd700; font-size: 20px; margin-right: 10px;">🟡</span>
                <span style="color: white;"><b>Medium Risk (-0.5 to -0.2):</b> Moderate temperature sensitivity</span>
            </div>

            <div style="display: flex; align-items: center; margin: 10px 0;">
                <span style="color: #ff6b6b; font-size: 20px; margin-right: 10px;">🔴</span>
                <span style="color: white;"><b>High Risk (Correlation < -0.5):</b> Strong negative impact of temperature on yield</span>
            </div>

            <p style="margin-top: 15px; color: #e0e0e0;">
            <b>Key Insight:</b> Negative correlation means higher temperatures are associated with lower yields. 
            Positive correlation means higher temperatures are associated with higher yields.
            </p>
            </div>
            """, unsafe_allow_html=True)

            # Detailed insights for high-risk crops
            high_risk_crops = sensitivity_df[sensitivity_df["Risk Level"] == "🔴 High Risk"]

    # Top Performing States
    st.markdown("---")
    st.markdown('<div class="sub-header">Top Performing States</div>', unsafe_allow_html=True)

    if not filtered.empty:
        state_performance = filtered.groupby("state").agg({
            "production": "sum",
            "yield_efficiency": "mean",
            "temperature": "mean"
        }).reset_index()

        col1, col2 = st.columns(2)

        with col1:
            # Top 5 by production
            top_production = state_performance.nlargest(5, "production")
            st.markdown("**Top 5 States by Production:**")
            for idx, row in top_production.iterrows():
                st.write(f"{idx + 1}. **{row['state']}**: {row['production']:,.0f} MT")

        with col2:
            # Top 5 by yield
            top_yield = state_performance.nlargest(5, "yield_efficiency")
            st.markdown("**Top 5 States by Yield Efficiency:**")
            for idx, row in top_yield.iterrows():
                st.write(f"{idx + 1}. **{row['state']}**: {row['yield_efficiency']:.2f} MT/Ha")

    # -------------------------
    # Production Map
    # -------------------------
    st.markdown('<div class="sub-header">Production Distribution Map</div>', unsafe_allow_html=True)

    if filtered.empty:
        st.info("No data available for the selected filters.")
    else:
        # 1. Define Coordinates
        STATE_COORDS = {
            "Johor": (1.4927, 103.7414),
            "Kedah": (6.1254, 100.3678),
            "Kelantan": (6.1252, 102.2382),
            "Melaka": (2.1896, 102.2501),
            "Negeri Sembilan": (2.7254, 101.9420),
            "Pahang": (3.8079, 102.5485),
            "Penang": (5.4164, 100.3327),
            "Pulau Pinang": (5.4164, 100.3327),
            "Perak": (4.5975, 101.0901),
            "Perlis": (6.4425, 100.2083),
            "Sabah": (5.9804, 116.0735),
            "Sarawak": (1.5533, 110.3593),
            "Selangor": (3.0738, 101.5183),
            "Terengganu": (5.3333, 103.1333),
            "Kuala Lumpur": (3.1390, 101.6869),
            "W.P. Kuala Lumpur": (3.1390, 101.6869),
            "Labuan": (5.2830, 115.2340),
            "W.P. Labuan": (5.2830, 115.2340),
            "Putrajaya": (2.9264, 101.6963),
            "W.P. Putrajaya": (2.9264, 101.6963),
            "Malaysia": (4.2105, 101.9758)
        }


        def clean_state_name(state):
            if isinstance(state, str):
                state = state.strip()
                lower = state.lower()
                if "pulau pinang" in lower or "penang" in lower: return "Penang"
                if "kuala lumpur" in lower: return "Kuala Lumpur"
                if "labuan" in lower: return "Labuan"
                if "putrajaya" in lower: return "Putrajaya"
                return state
            return state


        filtered_clean = filtered.copy()
        filtered_clean["state_clean"] = filtered_clean["state"].apply(clean_state_name)

        # 2. Aggregation: Calculate Production, Area, and Yield
        state_stats = filtered_clean.groupby("state_clean", as_index=False).agg(
            total_production=("production", "sum"),
            total_planted_area=("planted_area", "sum")
        ).rename(columns={"state_clean": "state"})

        # Calculate Yield (Avoid division by zero)
        state_stats["avg_yield"] = state_stats["total_production"] / state_stats["total_planted_area"]
        state_stats = state_stats.replace([np.inf, -np.inf], 0).fillna(0)

        # 3. Identify Top and Bottom Crops
        try:
            # Top Crop
            highest_crop_idx = filtered_clean.groupby("state_clean")["production"].idxmax()
            highest_crop = filtered_clean.loc[highest_crop_idx].set_index("state_clean")[
                ["crop_type"]
            ].rename(columns={"crop_type": "highest_crop"})

            # Bottom Crop
            lowest_crop_idx = filtered_clean.groupby("state_clean")["production"].idxmin()
            lowest_crop = filtered_clean.loc[lowest_crop_idx].set_index("state_clean")[
                ["crop_type"]
            ].rename(columns={"crop_type": "lowest_crop"})
        except:
            highest_crop = pd.DataFrame()
            lowest_crop = pd.DataFrame()

        # 4. Merge Data
        map_df = state_stats.set_index("state").join(highest_crop, how="left").join(lowest_crop,
                                                                                    how="left").reset_index()

        # Add coordinates
        map_df["lat"] = map_df["state"].map(lambda x: STATE_COORDS.get(x, (np.nan, np.nan))[0])
        map_df["lon"] = map_df["state"].map(lambda x: STATE_COORDS.get(x, (np.nan, np.nan))[1])
        map_df = map_df.dropna(subset=["lat", "lon"])

        if map_df.empty:
            st.error("No valid coordinates found for the selected data.")
        else:
            # 5. Normalization for Bubble Size
            min_prod = map_df["total_production"].min()
            max_prod = map_df["total_production"].max()

            # If all values are the same, avoid division by zero
            if max_prod > min_prod:
                map_df["production_norm"] = (map_df["total_production"] - min_prod) / (max_prod - min_prod)
                # Dynamic radius: Base 10km + up to 50km based on production
                map_df["radius"] = map_df["production_norm"] * 50000 + 10000
            else:
                map_df["radius"] = 20000  # Default size


            # Color Logic (Green Intensity)
            def get_green_color(norm_value):
                # Returns a color from dark green to bright green
                # [R, G, B, Alpha]
                return [0, int(100 + norm_value * 155), 0, 180]


            if 'production_norm' in map_df.columns:
                map_df["color"] = map_df["production_norm"].apply(get_green_color)
            else:
                map_df["color"] = [[0, 150, 0, 180]] * len(map_df)

            map_df["highest_crop"] = map_df["highest_crop"].fillna("No data")
            map_df["lowest_crop"] = map_df["lowest_crop"].fillna("No data")

            # 6. PyDeck Layers

            # Bubble Layer
            bubble_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_radius='radius',
                radius_scale=1,
                radius_min_pixels=10,  # Ensures small bubbles are still visible
                radius_max_pixels=60,
                get_fill_color='color',
                pickable=True,
                stroked=True,
                filled=True,
                line_width_min_pixels=1,
                get_line_color=[0, 80, 0, 200],
                auto_highlight=True
            )

            # Text Layer (State Names)
            text_layer = pdk.Layer(
                "TextLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_text='state',
                get_color=[0, 0, 0, 255],
                get_size=60,
                get_alignment_baseline="'center'",  # Text sits slightly above/center
                get_pixel_offset=[0, -15],  # Shift text up slightly
                font_weight="bold"
            )

            # 7. Enhanced Tooltip (HTML)
            tooltip = {
                "html": """
                <div style="
                    font-family: Arial;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    padding: 12px;
                    border: 2px solid #4CAF50;
                    color: #333;
                    min-width: 220px;
                ">
                    <div style="
                        border-bottom: 2px solid #eee;
                        padding-bottom: 8px;
                        margin-bottom: 8px;
                        font-size: 16px;
                        font-weight: 800;
                        text-align: center;
                        color: #1a472a;
                    ">
                        {state}
                    </div>

                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Production:</span>
                        <span style="font-weight: bold;">{total_production} MT</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Planted Area:</span>
                        <span style="font-weight: bold;">{total_planted_area} Ha</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span>Avg Yield:</span>
                        <span style="font-weight: bold; color: #2E7D32;">{avg_yield} T/Ha</span>
                    </div>

                    <div style="border-top: 1px dashed #ccc; margin: 5px 0 10px 0;"></div>

                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                        <div style="background: #e8f5e9; color: #2e7d32; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold;">HIGHEST CROP</div>
                        <span style="font-weight: 600;">{highest_crop}</span>
                    </div>
                     <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="background: #ffebee; color: #c62828; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold;">LOWEST CROP</div>
                        <span style="font-weight: 600;">{lowest_crop}</span>
                    </div>
                </div>
                """,
                "style": {
                    "backgroundColor": "transparent",
                    "color": "black"
                }
            }

            # View State
            view_state = pdk.ViewState(
                latitude=map_df["lat"].mean(),
                longitude=map_df["lon"].mean(),
                zoom=5.5,
                pitch=0
            )

            deck = pdk.Deck(
                layers=[bubble_layer, text_layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="light",
                height=550
            )

            st.pydeck_chart(deck)

        # High-Level Production Forecast
        st.markdown("---")
        st.markdown('<div class="sub-header">High-Level Production Forecast</div>', unsafe_allow_html=True)
        st.caption(
            "This forecast provides a high-level outlook based on historical trends (5-year projection). "
            "Detailed scenario testing is in the Simulation section."
        )

        # Simple Linear Trend for Summary
        summary_forecast_df = filtered.groupby('year')['production'].sum().reset_index()

        if len(summary_forecast_df) > 2:
            # Prepare Data
            X_sum = summary_forecast_df[['year']]
            y_sum = summary_forecast_df['production']

            # Fit Regression
            model_sum = LinearRegression().fit(X_sum, y_sum)

            # Generate 5 Future Years
            # We use np.arange to cleanly generate [max+1, max+2, max+3, max+4, max+5]
            future_years_list = np.arange(max_year + 1, max_year + 6).reshape(-1, 1)
            future_preds = model_sum.predict(future_years_list)

            # Create Forecast DataFrame
            future_df = pd.DataFrame({
                'year': future_years_list.flatten().astype(int),
                'production': future_preds,
                'Status': 'Predicted'
            })

            summary_forecast_df['Status'] = 'Historical'

            # Combine Data
            full_forecast = pd.concat([summary_forecast_df, future_df], ignore_index=True)

            # Visualization
            forecast_chart = alt.Chart(full_forecast).mark_line(point=True).encode(
                x=alt.X('year:O', title="Year"),
                y=alt.Y('production:Q', title="Total Production (MT)", scale=alt.Scale(zero=False)),
                color=alt.Color('Status:N',
                                scale=alt.Scale(domain=['Historical', 'Predicted'], range=['#2E7D32', '#FFA000'])),
                strokeDash=alt.condition(
                    alt.datum.Status == 'Predicted',
                    alt.value([5, 5]),  # Dashed line for prediction
                    alt.value([0])  # Solid line for historical
                ),
                tooltip=['year', 'production', 'Status']
            ).properties(height=350)

            st.altair_chart(forecast_chart, use_container_width=True)
        else:
            st.warning("Not enough historical data points to generate a reliable 5-year trend.")
# -------------------------
# TAB 2: DATA EXPLORATION PAGE
# -------------------------
elif tab_selection == "🔍 Data Exploration":

    # CROP DISTRIBUTION (% BY STATE)
    # -------------------------------------------------
    st.markdown('<div class="sub-header">Crop Distribution Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        sel_state = st.selectbox(
            "Select State for crop composition:",
            filtered["state"].unique(),
            key="dist_state"
        )

        # Additional insights about the selected state
        state_df = filtered[filtered["state"] == sel_state]

        if not state_df.empty:
            # Calculate state metrics
            total_state_production = state_df["production"].sum()
            num_crops_state = state_df["crop_type"].nunique()
            avg_temp_state = state_df["temperature"].mean()
            avg_humidity_state = state_df["humidity"].mean()

            st.markdown(f"""
            <div style="
                background-color: #1a472a;
                padding: 15px;
                border-radius: 10px;
                color: white;
                margin-bottom: 15px;
            ">
                <h4 style="color: white; margin-top: 0;">State Overview: {sel_state}</h4>
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Crop Types:</span>
                        <span style="font-weight: bold;">{num_crops_state}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Avg Temperature:</span>
                        <span style="font-weight: bold;">{avg_temp_state:.1f}°C</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Avg Humidity:</span>
                        <span style="font-weight: bold;">{avg_humidity_state:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        if not state_df.empty:
            # Prepare data for pie chart
            dist = state_df.groupby("crop_type", as_index=False).agg({
                "production": "sum",
                "yield_efficiency": "mean"
            })
            total_prod = dist["production"].sum()
            dist["percentage"] = (dist["production"] / total_prod) * 100

            # Sort by percentage for better visualization
            dist = dist.sort_values("percentage", ascending=False)

            # Assign different colors automatically
            color_scale = alt.Scale(scheme='category20')  # 20 distinct colors

            # Normal pie chart (no inner radius, no stroke)
            base = alt.Chart(dist).encode(
                theta=alt.Theta("percentage:Q", stack=True),
                color=alt.Color(
                    "crop_type:N",
                    legend=alt.Legend(
                        title="Crop Types",
                        labelColor="white",  # Legend labels in white
                        symbolStrokeWidth=0  # Remove outer line on legend symbols
                    ),
                    scale=color_scale
                ),
                tooltip=[
                    alt.Tooltip("crop_type:N", title="Crop"),
                    alt.Tooltip("percentage:Q", title="Percentage", format=".1f"),
                    alt.Tooltip("production:Q", title="Production", format=",.0f"),
                    alt.Tooltip("yield_efficiency:Q", title="Avg Yield Efficiency", format=".2f")
                ]
            )

            pie = base.mark_arc()  # no stroke / border

            # Text labels on slices in WHITE
            text = base.mark_text(
                radiusOffset=20,
                size=12,
                fontWeight="bold",
                color="white"  # <-- WHITE labels
            ).encode(
                text=alt.condition(
                    alt.datum.percentage > 10,  # only for slices >10%
                    alt.Text("crop_type:N"),
                    alt.value("")
                )
            )

            chart = (pie).properties(
                title=f"Crop Distribution in {sel_state}",
                height=400
            )

            st.altair_chart(chart, use_container_width=True)
            st.caption(f"Total Production: {total_prod:,.0f} MT | Shows percentage share of each crop")

    st.markdown('<div class="sub-header">Deep Dive: Data Relationships & Patterns</div>', unsafe_allow_html=True)


    # Interactive exploration controls
    col1, col2 = st.columns(2)

    with col1:
        explore_state = st.selectbox(
            "Select State:",
            options=["All States"] + sorted(filtered["state"].unique()),
            index=0
        )

    with col2:
        explore_crop = st.selectbox(
            "Select Crop:",
            options=["All Crops"] + sorted(filtered["crop_type"].unique()),
            index=0
        )

    # Apply exploration filters
    explore_filtered = filtered.copy()
    if explore_state != "All States":
        explore_filtered = explore_filtered[explore_filtered["state"] == explore_state]
    if explore_crop != "All Crops":
        explore_filtered = explore_filtered[explore_filtered["crop_type"] == explore_crop]

    if explore_filtered.empty:
        st.warning("No data for selected filters. Please adjust selections.")
        st.stop()

    # Aggregate data by year
    time_data = explore_filtered.groupby("year").agg({
        "production": "sum",
        "temperature": "mean",
        "humidity": "mean",
        "yield_efficiency": "mean"
    }).reset_index()

    if len(time_data) > 1:
        # Production trend
        production_chart = alt.Chart(time_data).mark_line(point=True, color="#2E7D32").encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("production:Q", title="Production (MT)", scale=alt.Scale(zero=False)),
            tooltip=["year", "production"]
        ).properties(
            title="Production Over Time",
            height=300
        )

        st.altair_chart(production_chart, use_container_width=True)

        # Climate variables
        col1, col2 = st.columns(2)

        with col1:
            temp_chart = alt.Chart(time_data).mark_line(point=True, color="#FF6B6B").encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("temperature:Q", title="Temperature (°C)", scale=alt.Scale(zero=False)),
                tooltip=["year", "temperature"]
            ).properties(
                title="Temperature Over Time",
                height=250
            )
            st.altair_chart(temp_chart, use_container_width=True)

        with col2:
            humidity_chart = alt.Chart(time_data).mark_line(point=True, color="#42A5F5").encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("humidity:Q", title="Humidity (%)", scale=alt.Scale(zero=False)),
                tooltip=["year", "humidity"]
            ).properties(
                title="Humidity Over Time",
                height=250
            )
            st.altair_chart(humidity_chart, use_container_width=True)
        #----------------------------
        #Climate Anomalies
        # ----------------------------
        # -------------------------
        # CLIMATE ANOMALIES & HEATMAP
        # -------------------------
        st.markdown("---")
        st.markdown('<div class="sub-header">Climate Anomalies & Heatmap</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Temperature Anomaly (vs. Average)**")

            # Calculate deviation from mean
            baseline_temp = filtered['temperature'].mean()
            anomaly_df = filtered.groupby('year')['temperature'].mean().reset_index()
            anomaly_df['anomaly'] = anomaly_df['temperature'] - baseline_temp

            anomaly_chart = alt.Chart(anomaly_df).mark_bar().encode(
                x=alt.X('year:O', title="Year"),
                y=alt.Y('anomaly:Q', title="Deviation from Baseline (°C)"),
                color=alt.condition(
                    alt.datum.anomaly > 0,
                    alt.value("#e45756"),  # Red for hot
                    alt.value("#1c91d4")  # Blue for cool
                ),
                tooltip=['year', alt.Tooltip('anomaly', format='.2f')]
            ).properties(height=350)

            st.altair_chart(anomaly_chart, use_container_width=True)

            # Explanation
            st.info("""
            **How to interpret:**
            * 🔴 **Red Bars:** Years that were **hotter** than the long-term average.
            * 🔵 **Blue Bars:** Years that were **cooler** than the average.
            * *Look for clusters of red bars to identify warming trends.*
            """)

        with col_b:
            st.markdown("**Correlation Matrix**")

            # Calculate Correlation
            corr_matrix = filtered[
                ['temperature', 'humidity', 'production', 'yield_efficiency']
            ].corr().reset_index().melt(id_vars='index')

            heatmap = alt.Chart(corr_matrix).mark_rect().encode(
                x=alt.X('index:N', title=None),
                y=alt.Y('variable:N', title=None),
                color=alt.Color(
                    'value:Q',
                    scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                    legend=alt.Legend(title="Correlation")
                ),
                tooltip=[
                    alt.Tooltip('index', title='Variable 1'),
                    alt.Tooltip('variable', title='Variable 2'),
                    alt.Tooltip('value', title='Correlation', format='.2f')
                ]
            ).properties(height=350)

            # Add text labels on the heatmap for clarity
            text = heatmap.mark_text(baseline='middle').encode(
                text=alt.Text('value:Q', format='.2f'),
                color=alt.condition(
                    alt.datum.value > 0.5,
                    alt.value('white'),
                    alt.value('black')
                )
            )

            st.altair_chart(heatmap + text, use_container_width=True)

            # Explanation
            st.info("""
            **How to interpret:**
            * 🟥 **Dark Red (+1.0):** Strong Positive Correlation (Variables move **together**).
            * 🟦 **Dark Blue (-1.0):** Strong Negative Correlation (Variables move in **opposite** directions).
            * ⬜ **White (0.0):** No relationship.
            """)

        # -------------------------
        # SCATTER PLOTS
        # -------------------------
        st.markdown("---")
        st.markdown('<div class="sub-header">Climate Impact on Yield</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Temperature vs YIELD
            scatter_data = explore_filtered.dropna(subset=["temperature", "yield_efficiency"])

            if len(scatter_data) > 2:
                scatter_chart = alt.Chart(scatter_data).mark_circle(size=60).encode(
                    x=alt.X("temperature:Q", title="Temperature (°C)", scale=alt.Scale(zero=False)),
                    y=alt.Y("yield_efficiency:Q", title="Yield Efficiency (MT/Ha)"),
                    color=alt.Color("crop_type:N", legend=alt.Legend(title="Crop")),
                    tooltip=["state", "crop_type", "year", "temperature", "yield_efficiency"]
                ).properties(title="Temperature vs Yield", height=400)

                reg_chart = scatter_chart.transform_regression(
                    "temperature", "yield_efficiency"
                ).mark_line(color="red", strokeWidth=3)

                st.altair_chart(scatter_chart + reg_chart, use_container_width=True)

                # Correlation Metric
                corr = scatter_data["temperature"].corr(scatter_data["yield_efficiency"])
                st.metric("Temp-Yield Correlation", f"{corr:.3f}", delta_color="off")

                # Simple Explanation in Info Box
                st.info("""
                **How to interpret:**
                * 📉 **Negative Correlation:** As it gets hotter, yield **decreases** (Red line goes down).
                * 📈 **Positive Correlation:** As it gets hotter, yield **increases**.
                """)

        with col2:
            # Humidity vs YIELD
            scatter_data_h = explore_filtered.dropna(subset=["humidity", "yield_efficiency"])

            if len(scatter_data_h) > 2:
                scatter_chart_h = alt.Chart(scatter_data_h).mark_circle(size=60).encode(
                    x=alt.X("humidity:Q", title="Humidity (%)", scale=alt.Scale(zero=False)),
                    y=alt.Y("yield_efficiency:Q", title="Yield Efficiency (MT/Ha)"),
                    color=alt.Color("crop_type:N"),
                    tooltip=["state", "year", "humidity", "yield_efficiency"]
                ).properties(title="Humidity vs Yield", height=400)

                reg_chart_h = scatter_chart_h.transform_regression(
                    "humidity", "yield_efficiency"
                ).mark_line(color="blue", strokeWidth=3)

                st.altair_chart(scatter_chart_h + reg_chart_h, use_container_width=True)

                # Correlation Metric
                corr_h = scatter_data_h["humidity"].corr(scatter_data_h["yield_efficiency"])
                st.metric("Humidity-Yield Correlation", f"{corr_h:.3f}", delta_color="off")

                # Simple Explanation in Info Box
                st.info("""
                **How to interpret:**
                * 📈 **Positive Correlation:** Higher humidity **boosts** yield (Blue line goes up).
                * 📉 **Negative Correlation:** Higher humidity **reduces** yield.
                """)

        # -------------------------
        #HEAT SENSITIVITY ANALYSIS
        # -------------------------
        st.markdown("---")
        st.markdown('<div class="sub-header">Heat Sensitivity Analysis</div>', unsafe_allow_html=True)
        st.caption(
            "This chart shows how much **Yield (MT/Ha)** changes for every **+1°C rise** in temperature. Negative bars indicate crops at risk.")

        # Calculate Sensitivity (Slope of Regression)
        sensitivity_data = []

        # 1. Create a safe copy for analysis
        # We drop rows where Yield or Temp is missing
        safe_data = explore_filtered.dropna(subset=['temperature', 'yield_efficiency']).copy()

        # 2. Handle Division by Zero (Infinity)
        # If planted_area was 0, yield became 'inf'. We replace 'inf' with NaN and drop them.
        safe_data = safe_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['yield_efficiency'])

        for crop in safe_data['crop_type'].unique():
            sub = safe_data[safe_data['crop_type'] == crop]

            # Need enough CLEAN data points to fit a line
            if len(sub) > 5:
                try:
                    # Fit Linear Regression: Yield = a * Temp + b
                    X = sub[['temperature']]
                    y = sub['yield_efficiency']

                    reg = LinearRegression().fit(X, y)
                    slope = reg.coef_[0]
                    sensitivity_data.append({'Crop': crop, 'Sensitivity': slope})
                except Exception as e:
                    # Skip if any specific mathematical error occurs for this crop
                    continue

        sens_df = pd.DataFrame(sensitivity_data)

        if not sens_df.empty:
            sens_chart = alt.Chart(sens_df).mark_bar().encode(
                x=alt.X('Crop:N', sort='y'),
                y=alt.Y('Sensitivity:Q', title="Yield Change per +1°C (MT/Ha)"),
                color=alt.condition(
                    alt.datum.Sensitivity < 0,
                    alt.value("#d32f2f"),  # Red for negative impact
                    alt.value("#388e3c")  # Green for positive/neutral
                ),
                tooltip=['Crop', alt.Tooltip('Sensitivity', format='.4f')]
            ).properties(height=350)

            st.altair_chart(sens_chart, use_container_width=True)

            # Add Interpretation
            st.info(
                "💡 **Interpretation:** Bars pointing **DOWN (Red)** mean the crop produces *less* when it gets hotter. Bars pointing **UP (Green)** mean it tolerates heat well.")
        else:
            st.warning(
                "Insufficient clean data to calculate heat sensitivity. Some crops may have 0 planted area or missing values.")

    # Data Summary
    st.markdown("---")
    st.markdown('<div class="sub-header">Data Summary</div>', unsafe_allow_html=True)

    summary_cols = ["production", "temperature", "humidity", "yield_efficiency"]
    available_cols = [col for col in summary_cols if col in explore_filtered.columns]

    if available_cols:
        summary_stats = explore_filtered[available_cols].describe().round(2)
        st.dataframe(summary_stats, use_container_width=True)

# -------------------------
# TAB 3: CLIMATE SIMULATION PAGE
# -------------------------
else:
    st.markdown('<div class="sub-header">Climate Impact Simulation</div>', unsafe_allow_html=True)

    # 1. Objective Header
    st.markdown("""
    <div style="background-color: #1a472a; padding: 15px; border-radius: 10px; border-left: 5px solid #2e8b57; color: white; margin-bottom: 20px;">
    <b style="color: white;">Simulation Objective:</b> Explore how temperature and humidity changes might affect crop yields.
    Adjust the parameters below to see predicted impacts on production.
    </div>
    """, unsafe_allow_html=True)

    # 2. Selection Controls (Dropdowns)
    sel_c1, sel_c2 = st.columns(2)

    with sel_c1:
        sim_state = st.selectbox(
            "Select State:",
            options=sorted(filtered["state"].unique()),
            key="sim_state_select"
        )

    with sel_c2:
        sim_crop = st.selectbox(
            "Select Crop:",
            options=sorted(filtered["crop_type"].unique()),
            key="sim_crop_select"
        )

    # Data Check
    sim_data = filtered[
        (filtered["state"] == sim_state) &
        (filtered["crop_type"] == sim_crop)
        ].sort_values("year")

    if len(sim_data) < 2:
        st.warning(f"Not enough historical data for {sim_crop} in {sim_state}. Please select another combination.")
        st.stop()

    # 3. Simulation Parameters (Sliders) - NEW COLUMNS
    st.markdown("---")
    st.markdown("### Simulation Parameters")

    param_c1, param_c2 = st.columns(2)

    with param_c1:
        temp_increase = st.slider(
            "Temperature Increase (°C):",
            min_value=0.0, max_value=3.0, value=1.5, step=0.5,
            key="temp_slider"
        )

    with param_c2:
        hum_change = st.slider(
            "Humidity Change (%):",
            min_value=-10.0, max_value=5.0, value=0.0, step=1.0,
            key="hum_slider"
        )

    # -------------------------------------------------------
    # MODEL 1: SCENARIO COMPARISON (Simple Snapshot)
    # -------------------------------------------------------
    st.markdown("---")

    # Check for humidity, default if missing
    if 'humidity' not in sim_data.columns:
        sim_data['humidity'] = 80.0

    # Prepare Model Data
    model_data = sim_data[["year", "temperature", "humidity", "production"]].dropna()

    if len(model_data) > 2:
        X = model_data[["temperature", "humidity"]]
        y = model_data["production"]

        # Simple Regression for the Snapshot Bar Chart
        model_simple = LinearRegression().fit(X, y)

        curr_vals = [[model_data['temperature'].mean(), model_data['humidity'].mean()]]
        future_vals = [[curr_vals[0][0] + temp_increase, curr_vals[0][1] + hum_change]]

        base_pred = model_simple.predict(curr_vals)[0]
        sim_pred = model_simple.predict(future_vals)[0]

        # Calculate % change for later use
        simple_change_pct = ((sim_pred - base_pred) / base_pred) * 100

        # SCENARIO COMPARISON CHART
        comparison_df = pd.DataFrame({
            'Scenario': ['Baseline (Current)', 'Simulated (Future)'],
            'Production': [base_pred, sim_pred]
        })

        col_chart1, col_kpi = st.columns([2, 1])

        with col_chart1:
            comparison_bar = alt.Chart(comparison_df).mark_bar().encode(
                x=alt.X('Scenario:N', axis=alt.Axis(labelAngle=0)),
                y='Production:Q',
                color=alt.Color('Scenario:N', scale=alt.Scale(range=['#2E7D32', '#FF9800']))
            ).properties(height=250, title="Projected Production Impact")
            st.altair_chart(comparison_bar, use_container_width=True)

        with col_kpi:
            st.markdown("#### Impact Summary")
            st.metric("Baseline", f"{base_pred:,.0f} MT")
            st.metric("Simulated", f"{sim_pred:,.0f} MT")
            st.metric("Net Change", f"{simple_change_pct:+.2f}%",
                      delta_color="inverse" if simple_change_pct < 0 else "normal")

    # -------------------------------------------------------
    # MODEL 2: TIME-SERIES FORECASTING (LINEAR REGRESSION)
    # -------------------------------------------------------
    from sklearn.linear_model import LinearRegression
    import numpy as np

    st.markdown("---")
    st.markdown('<div class="sub-header">Time-Series Simulation (Linear Regression)</div>', unsafe_allow_html=True)
    st.caption(
        "This model projects the historical trend forward using Linear Regression, adjusting for your simulated climate scenarios.")

    # 1. Prepare Data
    # We need numerical year for regression
    reg_df = sim_data[['year', 'production', 'temperature', 'humidity']].dropna()

    if len(reg_df) > 3:

        # Define Features (X) and Target (y)
        # We use Year (for trend), Temperature, and Humidity
        X = reg_df[['year', 'temperature', 'humidity']]
        y = reg_df['production']

        # 2. Train Model
        model = LinearRegression()
        model.fit(X, y)

        # Calculate Accuracy (R-Squared)
        r2_score = model.score(X, y)

        # 3. Create Future Scenarios (Next 5 Years)
        last_year = int(reg_df['year'].max())
        future_years = [last_year + i for i in range(1, 6)]

        # Get Baseline Climate (avg of last 3 years to represent 'current' state)
        base_temp = reg_df.tail(3)['temperature'].mean()
        base_hum = reg_df.tail(3)['humidity'].mean()

        # Apply Simulation Sliders (temp_increase, hum_change from your sliders)
        sim_temp = base_temp + temp_increase
        sim_hum = base_hum + hum_change

        # Build Future Data for Prediction
        future_X = pd.DataFrame({
            'year': future_years,
            'temperature': [sim_temp] * 5,
            'humidity': [sim_hum] * 5
        })

        # 4. Predict Future
        future_pred = model.predict(future_X)
        future_X['production'] = future_pred
        future_X['Type'] = 'Forecast'

        # 5. Prepare Historical Data for Chart
        hist_data = reg_df[['year', 'production']].copy()
        hist_data['Type'] = 'Historical'

        # Combine Data
        full_data = pd.concat([hist_data, future_X[['year', 'production', 'Type']]])

        # 6. Visualization

        # Main Line Chart
        base = alt.Chart(full_data).encode(x=alt.X('year:O', title="Year"))

        line = base.mark_line(point=True).encode(
            y=alt.Y('production:Q', title="Production (MT)"),
            color=alt.Color('Type',
                            scale=alt.Scale(domain=['Historical', 'Forecast'], range=['#2E7D32', '#FF9800']),
                            legend=alt.Legend(title="Data Type")
                            ),
            strokeDash=alt.condition(
                alt.datum.Type == 'Forecast',
                alt.value([5, 5]),  # Dashed line for forecast
                alt.value([0])  # Solid line for history
            ),
            tooltip=['year', 'production', 'Type']
        )

        chart = (line).properties(
            title=f"Projected Production Impact for {sim_crop}",
            height=350
        )

        st.altair_chart(chart, use_container_width=True)

        # 7. Impact Explanation Metrics
        col_m1, col_m2, col_m3 = st.columns(3)

        # Calculate Coefficients (Impact per unit)
        temp_impact = model.coef_[1]  # Coefficient for Temperature
        hum_impact = model.coef_[2]  # Coefficient for Humidity

        with col_m1:
            st.metric(
                "Temp Impact (per °C)",
                f"{temp_impact:,.0f} MT",
                delta_color="inverse" if temp_impact < 0 else "normal",
                help="How much production changes for every 1°C increase."
            )
        with col_m2:
            st.metric(
                "Humidity Impact (per %)",
                f"{hum_impact:,.0f} MT",
                help="How much production changes for every 1% humidity increase."
            )

        st.info(f"""
        **Simulation Summary:**
        Based on historical trends, if the temperature stays at **{sim_temp:.1f}°C** and humidity at **{sim_hum:.1f}%**, the model predicts production will 
        **{'increase' if future_pred[-1] > hist_data.iloc[-1]['production'] else 'decrease'}** over the next 5 years.
        """)

    else:
        st.warning("Not enough data to train the simulation model (need at least 3 years).")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p style="font-size: 1.2rem; font-weight: bold;">🌾 Climate Impact on Food Availability Dashboard</p>
    <p>Developed for Final Year Project • Universiti Putra Malaysia</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">

</div>
""", unsafe_allow_html=True)
