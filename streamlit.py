import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Chervon: Vehicle Population Prediction")

st.image("cover.png", use_container_width=True)

data = {
    "Column": [
        "Date",
        "Vehicle Category",
        "GVWR Class",
        "Fuel Type",
        "Model Year",
        "Fuel Technology",
        "Electric Mile Range",
        "Number of Vehicles Registered at the Same Address",
        "Region",
        "Vehicle Population",
    ],
}

# Create a DataFrame
df = pd.DataFrame(data)
df.index = range(1, len(df) + 1)


# Display the table in Streamlit
st.markdown("---")
st.title("Column Information")
st.table(df)  # Use st.dataframe(df) if you want interactivity
data = {
    "Variable": [
        "Electric Mile Range",
        "GVWR Class",
        "Model Year",
        "Fuel Type",
        "Number of Vehicles Registered at the Same Address",
        "Date",
        "Vehicle Category",
        "Fuel Technology",
        "Region",
        "Vehicle Population",
    ],
    "Missing Percentage (%)": [
        97.413100,
        55.937447,
        1.468833,
        0.209485,
        0.080384,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
    ],
}

# Create a DataFrame
df = pd.DataFrame(data)
df.index = range(1, len(df) + 1)

st.markdown("---")
# Display the table in Streamlit
st.title("Missing Data Percentage Overview")
st.table(df)  


st.markdown("---")
st.subheader("Feature Engineering")
st.markdown("""
### Vehicle Age in the Model

- **Initially, we included Date and Model Year in our model as numeric columns, but the model's performance did not improve significantly.**
- **Defined Vehicle Age as Date minus Model Year.**
- **For rows where Model Year is missing, we fill the missing value with the median in train data.**
- **Greatly enhanced the model's performance.**
""")
st.image("Vehicle_Age.png", width = 700)

# Data for the table
data = {
    "Variables": [
        "Vehicle Age",
        "Vehicle Category",
        "GVWR Class",
        "Fuel Type",
        "Fuel Technology",
        "Electric Mile Range",
        "Number of Vehicles Registered at the Same Address",
        "Vehicle Population",
    ],
    "Description": [
        "Date - Model Year",
        "Nominal Variable",
        "Nominal Variable",
        "Nominal Variable",
        "Nominal Variable",
        "Nominal Variables, Non-applicable = 1, Miles Related Rows = 2, Unknown = 3",
        "Order Variable",
        "Response Variable",
    ],
}

# Create a DataFrame
df = pd.DataFrame(data)
df.index = range(1, len(df) + 1)

# Display the table in Streamlit
st.subheader("Model Variable Explanation")
st.table(df)  #

st.markdown("---")
st.subheader("Model Performance & Interpretation")
st.subheader("XGBoost Model : RMSE = 6043.28")
st.image("Feature_imp.png", width = 700)

st.image("feature_imp.png", width = 700)