import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Page title
st.set_page_config(page_title="House Price Prediction", page_icon="🏡")
st.title("🏡 House Price Prediction")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('train.csv')
    return data

df = load_data()

# --- Preprocessing and Model Training ---
# For simplicity, we'll use a subset of numerical features and handle missing values
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
target = 'SalePrice'

df_model = df[features + [target]].copy()
df_model = df_model.dropna()

X = df_model[features]
y = np.log(df_model[target]) # Use log transformation for the target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Sidebar for User Input ---
st.sidebar.header("Model Parameters")

def user_input_features():
    overall_qual = st.sidebar.slider('Overall Quality', int(X.OverallQual.min()), int(X.OverallQual.max()), int(X.OverallQual.mean()))
    gr_liv_area = st.sidebar.slider('Above Ground Living Area (sq ft)', int(X.GrLivArea.min()), int(X.GrLivArea.max()), int(X.GrLivArea.mean()))
    garage_cars = st.sidebar.slider('Garage Cars', int(X.GarageCars.min()), int(X.GarageCars.max()), int(X.GarageCars.mean()))
    garage_area = st.sidebar.slider('Garage Area (sq ft)', int(X.GarageArea.min()), int(X.GarageArea.max()), int(X.GarageArea.mean()))
    total_bsmt_sf = st.sidebar.slider('Total Basement Area (sq ft)', int(X.TotalBsmtSF.min()), int(X.TotalBsmtSF.max()), int(X.TotalBsmtSF.mean()))
    first_flr_sf = st.sidebar.slider('First Floor Area (sq ft)', int(X['1stFlrSF'].min()), int(X['1stFlrSF'].max()), int(X['1stFlrSF'].mean()))
    full_bath = st.sidebar.slider('Full Bathrooms', int(X.FullBath.min()), int(X.FullBath.max()), int(X.FullBath.mean()))
    tot_rms_abv_grd = st.sidebar.slider('Total Rooms Above Ground', int(X.TotRmsAbvGrd.min()), int(X.TotRmsAbvGrd.max()), int(X.TotRmsAbvGrd.mean()))
    year_built = st.sidebar.slider('Year Built', int(X.YearBuilt.min()), int(X.YearBuilt.max()), int(X.YearBuilt.mean()))
    year_remod_add = st.sidebar.slider('Year Remodeled', int(X.YearRemodAdd.min()), int(X.YearRemodAdd.max()), int(X.YearRemodAdd.mean()))

    data = {'OverallQual': overall_qual,
            'GrLivArea': gr_liv_area,
            'GarageCars': garage_cars,
            'GarageArea': garage_area,
            'TotalBsmtSF': total_bsmt_sf,
            '1stFlrSF': first_flr_sf,
            'FullBath': full_bath,
            'TotRmsAbvGrd': tot_rms_abv_grd,
            'YearBuilt': year_built,
            'YearRemodAdd': year_remod_add}
    features_df = pd.DataFrame(data, index=[0])
    return features_df

input_df = user_input_features()

# --- Model Training and Prediction ---
n_estimators = st.sidebar.slider('Number of Estimators', 10, 500, 100)
max_depth = st.sidebar.slider('Max Depth', 2, 20, 10)

model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

# --- Display Prediction ---
st.subheader("Prediction")

prediction = model.predict(input_df)
predicted_price = np.exp(prediction)[0] # Inverse transform the prediction

st.write(f"**Predicted House Price:** `${predicted_price:,.2f}`")

# --- Model Evaluation ---
st.subheader("Model Performance")

y_pred_test = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
st.write(f"**R-squared (R²):** {r2:.4f}")

# --- Display Data ---
st.subheader("Raw Data")
st.write(df.head())