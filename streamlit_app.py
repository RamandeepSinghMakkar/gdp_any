import streamlit as st
import pandas as pd
import math
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import numpy as np
import mysql.connector
from datetime import datetime

# New import for XGBoost
import xgboost as xgb

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP Prediction Dashboard (XGBoost)',
    page_icon=':earth_americas:',
    layout="wide"
)

MYSQL_DB_CONFIG = {
    'host': st.secrets.get("mysql", {}).get("host", "localhost"),
    'user': st.secrets.get("mysql", {}).get("user", "your_user"),
    'password': st.secrets.get("mysql", {}).get("password", "your_password"),
    'database': st.secrets.get("mysql", {}).get("database", "gdp_db") # Changed default DB name
}

# Default values for inputs
DEFAULT_STOCK_TICKER = "AAPL" # Not used in GDP app, but kept from previous context
DEFAULT_START_DATE = datetime(2010, 1, 1) # Not directly used for GDP data fetching
DEFAULT_END_DATE = datetime(2023, 1, 1)   # Not directly used for GDP data fetching
DEFAULT_LOOK_BACK = 60 # Not directly used for GDP prediction (RF/XGBoost don't use sequences like LSTM)

# Define min/max years for historical data from the CSV
MIN_CSV_YEAR = 1960
MAX_CSV_YEAR = 2022

# --- MySQL Database Operations ---
@st.cache_resource(ttl="1h") # Cache connection for 1 hour
def get_mysql_connection():
    """Establishes and returns a MySQL database connection."""
    try:
        # Check if credentials are placeholders
        if MYSQL_DB_CONFIG['user'] == "your_user" or MYSQL_DB_CONFIG['password'] == "your_password":
            st.warning("MySQL credentials are not configured. Using placeholder values. Please set up Streamlit Secrets for deployment.")
            return None # Prevent connection with placeholder credentials

        cnx = mysql.connector.connect(**MYSQL_DB_CONFIG)
        return cnx
    except mysql.connector.Error as err:
        st.error(f"Error connecting to MySQL database: {err}. Please check your database configuration and ensure it's accessible.")
        return None

def create_gdp_table(cnx, table_name="gdp_data"):
    """Creates the GDP data table in MySQL if it doesn't exist."""
    try:
        cursor = cnx.cursor()
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                Country_Code VARCHAR(10) NOT NULL,
                Country_Name VARCHAR(255),
                Indicator_Name VARCHAR(255),
                Indicator_Code VARCHAR(255),
                Year INT NOT NULL,
                GDP DOUBLE,
                PRIMARY KEY (Country_Code, Year)
            )
        """
        cursor.execute(create_table_query)
        cnx.commit()
        cursor.close()
        return True
    except mysql.connector.Error as err:
        st.error(f"Error creating MySQL table '{table_name}': {err}")
        return False

def save_gdp_data_to_mysql(df, cnx, table_name="gdp_data"):
    """Saves DataFrame to MySQL."""
    if not cnx or not create_gdp_table(cnx, table_name):
        return False

    st.info(f"Saving GDP data to MySQL table: {table_name}...")
    try:
        cursor = cnx.cursor()
        insert_sql = f"""
            INSERT INTO {table_name} (Country_Code, Country_Name, Indicator_Name, Indicator_Code, Year, GDP)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                Country_Name=VALUES(Country_Name), Indicator_Name=VALUES(Indicator_Name),
                Indicator_Code=VALUES(Indicator_Code), GDP=VALUES(GDP)
        """
        rows_to_insert = []
        for _, row in df.iterrows():
            rows_to_insert.append((
                row['Country Code'], row['Country Name'], row['Indicator Name'],
                row['Indicator Code'], row['Year'], row['GDP']
            ))
        cursor.executemany(insert_sql, rows_to_insert)
        cnx.commit()
        st.success(f"Successfully saved {len(rows_to_insert)} records to MySQL.")
        cursor.close()
        return True
    except mysql.connector.Error as err:
        st.error(f"Error saving data to MySQL: {err}")
        return False

def load_gdp_data_from_mysql(cnx, table_name="gdp_data"):
    """Loads GDP data from MySQL into a DataFrame."""
    if not cnx:
        return None

    st.info(f"Attempting to load GDP data from MySQL table: {table_name}...")
    try:
        cursor = cnx.cursor(dictionary=True)
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY Year ASC, Country_Code ASC")
        data = pd.DataFrame(cursor.fetchall())
        cursor.close()
        if not data.empty:
            # Rename columns to match the CSV loaded DataFrame's column names
            data.rename(columns={
                'Country_Code': 'Country Code',
                'Country_Name': 'Country Name',
                'Indicator_Name': 'Indicator Name',
                'Indicator_Code': 'Indicator Code'
            }, inplace=True)
            st.success(f"Successfully loaded {len(data)} records from MySQL.")
            return data
        st.warning(f"No data found in MySQL table '{table_name}'.")
        return None
    except mysql.connector.Error as err:
        st.warning(f"MySQL table '{table_name}' not found or error loading data: {err}. Will try to load from CSV.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading from MySQL: {e}")
        return None

# --- Data Acquisition (from CSV or MySQL) ---
@st.cache_data(ttl='1d') # Cache the loaded GDP data
def get_gdp_data():
    """
    Attempts to load GDP data from MySQL first. If that fails or no data,
    it falls back to loading from a CSV file and then saves to MySQL.
    """
    cnx = get_mysql_connection()
    df = None

    if cnx:
        df = load_gdp_data_from_mysql(cnx)

    if df is None or df.empty:
        st.info("Loading GDP data from CSV file...")
        DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'

        if not DATA_FILENAME.exists():
            st.error(f"Error: Data file not found at {DATA_FILENAME}. Please ensure 'gdp_data.csv' is in a 'data' subfolder.")
            st.stop() # Stop the app if data file is missing

        raw_gdp_df = pd.read_csv(DATA_FILENAME)

        gdp_df_melted = raw_gdp_df.melt(
            ['Country Code', 'Country Name', 'Indicator Name', 'Indicator Code'],
            [str(x) for x in range(MIN_CSV_YEAR, MAX_CSV_YEAR + 1)],
            'Year',
            'GDP',
        )
        gdp_df_melted['Year'] = pd.to_numeric(gdp_df_melted['Year'])
        gdp_df_melted.dropna(subset=['GDP'], inplace=True)

        if cnx:
            save_gdp_data_to_mysql(gdp_df_melted, cnx)
        df = gdp_df_melted
    
    # Close connection if it was opened
    if cnx:
        cnx.close()

    return df

@st.cache_resource # Use st.cache_resource for models/heavy objects
def train_gdp_prediction_model(df):
    """
    Trains an XGBoostRegressor model for GDP prediction.
    Features: Year, Encoded Country Code
    Target: GDP
    """
    st.info("Training GDP prediction model (XGBoost)...")

    model_df = df.copy()

    # Encode Country Code
    le = LabelEncoder()
    # Fit transform only on unique country codes present in the training data
    le.fit(model_df['Country Code'].unique())
    model_df['Country_Code_Encoded'] = le.transform(model_df['Country Code'])

    # Define features (X) and target (y)
    X = model_df[['Year', 'Country_Code_Encoded']]
    y = model_df['GDP']

    # Initialize and train the XGBoost Regressor
    # n_estimators: number of boosting rounds (trees)
    # random_state: for reproducibility
    # n_jobs=-1: use all available CPU cores
    model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1,
                             objective='reg:squarederror') # Specify objective for regression
    model.fit(X, y)

    st.success("GDP prediction model (XGBoost) trained!")
    return model, le

# Load data (this will try MySQL first, then CSV)
gdp_df = get_gdp_data()

# Train model (only runs once due to st.cache_resource)
gdp_model, country_encoder = train_gdp_prediction_model(gdp_df)

# Get max year from historical data for prediction range
MAX_HISTORICAL_YEAR = gdp_df['Year'].max()

# -----------------------------------------------------------------------------
# Draw the actual page

'''
# :earth_americas: GDP Prediction Dashboard (XGBoost)

Explore historical GDP data and predict future GDP using an XGBoost Machine Learning model.
Data from the [World Bank Open Data](https://data.worldbank.org/).
'''

# Add some spacing
''
''

min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Select Historical Years to Display:',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value]
)

countries = gdp_df['Country Code'].unique()

if not len(countries):
    st.warning("No country data available. Please check your data source (MySQL or CSV).")
    st.stop()

selected_countries = st.multiselect(
    'Which countries would you like to view and predict for?',
    sorted(countries), # Sort countries for consistent display
    ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN']
)

''
''

# Prediction section
st.sidebar.header("Predict Future GDP")
predict_year = st.sidebar.number_input(
    'Enter a year for GDP prediction:',
    min_value=MAX_HISTORICAL_YEAR + 1, # Start prediction from the year after last historical data
    max_value=MAX_HISTORICAL_YEAR + 50, # Arbitrary future limit
    value=MAX_HISTORICAL_YEAR + 5, # Default to 5 years after historical data
    step=1
)

# Filter the historical data for display
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries))
    & (gdp_df['Year'] <= to_year)
    & (from_year <= gdp_df['Year'])
].copy() # Use .copy() to avoid SettingWithCopyWarning

st.header('GDP Over Time (Historical & Predicted)', divider='gray')

''

if selected_countries:
    # Prepare data for prediction
    prediction_data = []
    countries_for_prediction = []
    for country_code in selected_countries:
        try:
            # Only predict for countries that the encoder has seen during training
            if country_code in country_encoder.classes_:
                encoded_country = country_encoder.transform([country_code])[0]
                prediction_data.append([predict_year, encoded_country])
                countries_for_prediction.append(country_code)
            else:
                st.warning(f"Country '{country_code}' not found in historical data for encoding. Skipping prediction for this country.")
        except ValueError:
            # This block should ideally not be reached if `country_code in country_encoder.classes_` check passes
            st.warning(f"Unexpected error encoding country '{country_code}'. Skipping prediction.")
            continue

    if prediction_data:
        predicted_gdp_values = gdp_model.predict(np.array(prediction_data))

        # Create a DataFrame for predictions
        predicted_df = pd.DataFrame({
            'Country Code': countries_for_prediction,
            'Year': predict_year,
            'GDP': predicted_gdp_values,
            'Type': 'Predicted' # Add a type column for differentiation
        })

        # Add 'Type' column to historical data as well
        filtered_gdp_df['Type'] = 'Historical'

        # Combine historical and predicted data for plotting
        # Ensure 'Country Name' is consistent for predicted data for plotting
        # Get Country Name mapping from filtered_gdp_df
        country_name_map = filtered_gdp_df.set_index('Country Code')['Country Name'].drop_duplicates().to_dict()
        predicted_df['Country Name'] = predicted_df['Country Code'].map(country_name_map)

        # Append predicted data, ensuring all columns match
        combined_df = pd.concat([filtered_gdp_df, predicted_df], ignore_index=True)

        # Sort for correct line plotting
        combined_df = combined_df.sort_values(by=['Country Code', 'Year'])

        st.line_chart(
            combined_df,
            x='Year',
            y='GDP',
            color='Country Code',
            # You could add a tooltip to show predicted vs historical
            # tooltip=['Country Code', 'Year', 'GDP', 'Type']
        )
    else:
        st.warning("No countries selected or valid for prediction after filtering.")
else:
    st.info("Select countries from the sidebar to see their GDP data and predictions.")


''
''


# Display GDP in the latest selected historical year (original dashboard feature)
st.header(f'GDP in {to_year} (Historical)', divider='gray')

''

cols = st.columns(4)

# Filter for the last historical year selected by the slider
last_year_data = filtered_gdp_df[filtered_gdp_df['Year'] == to_year]

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]

    with col:
        # Get GDP for the 'from_year' and 'to_year' from the original (unfiltered) gdp_df
        # to ensure we have the full range for growth calculation, but check for NaNs
        first_gdp_row = gdp_df[(gdp_df['Country Code'] == country) & (gdp_df['Year'] == from_year)]
        last_gdp_row = gdp_df[(gdp_df['Country Code'] == country) & (gdp_df['Year'] == to_year)]

        first_gdp = first_gdp_row['GDP'].iat[0] / 1_000_000_000 if not first_gdp_row.empty and not pd.isna(first_gdp_row['GDP'].iat[0]) else float('nan')
        last_gdp = last_gdp_row['GDP'].iat[0] / 1_000_000_000 if not last_gdp_row.empty and not pd.isna(last_gdp_row['GDP'].iat[0]) else float('nan')

        if math.isnan(first_gdp) or math.isnan(last_gdp) or first_gdp == 0:
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        # Display the metric for the last historical year
        display_value = f'{last_gdp:,.0f}B' if not math.isnan(last_gdp) else 'N/A'
        st.metric(
            label=f'{country} GDP',
            value=display_value,
            delta=growth,
            delta_color=delta_color
        )
