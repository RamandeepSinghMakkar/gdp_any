# :earth_americas: GDP dashboard template
> **Note:** The app may sleep due to inactivity (because I am using a free tier). Please activate it and then refresh the page after 2 minutes.

A simple Streamlit app showing the GDP of different countries in the world.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gdp-dashboard-template.streamlit.app/)

## 🔍 Overview

- Forecasts GDP using historical economic indicators.
- Achieves **R² score of 0.9522** on unseen economic data.
- Integrates **MySQL** for data warehousing and **XGBoost** for regression.
- Streamlit dashboard for intuitive exploration and simulation.

## 📊 Features

- Real-time GDP prediction for selected countries.
- Data visualization with interactive plots and trend charts.
- Country-wise comparison across economic indicators.
- User-friendly Streamlit interface.

## 🚀 How to Run Locally


### 1. Clone the Repository
```bash
git clone https://github.com/RamandeepSinghMakkar/gdp_any.git
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
 ```bash
pip install -r requirements.txt
```

### 4 Run the app

   ```
   $ streamlit run streamlit_app.py
   ```



**Output:**

