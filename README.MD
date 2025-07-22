# :earth_americas: GDP dashboard template


A simple Streamlit app showing the GDP of different countries in the world.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gpd-ramandeep.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```


# :earth_americas: GDP Prediction Dashboard

# A Streamlit app that predicts GDP trends using XGBoost regression and economic indicators.
# Features an interactive dashboard to visualize and simulate country-level GDP forecasts.

# ▶ Try it Live:
# https://gpd-ramandeep.streamlit.app/

# 🛠️ Run Locally

# 1. Clone the repo
git clone https://github.com/RamandeepSinghMakkar/gdp_any.git
cd gdp_any

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run streamlit_app.py

# 📊 Tech Stack:
# - Streamlit
# - XGBoost
# - MySQL (for data warehousing)
# - Pandas, NumPy, Seaborn, Matplotlib

# 📈 Model Performance:
# - R² Score: 0.9522 on unseen economic data

# 📦 Dataset:
# World Bank GDP dataset and enriched economic indicators
