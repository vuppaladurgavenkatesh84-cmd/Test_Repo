import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Beijing Air Quality Analysis", layout="wide")

# Load cleaned dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_combined_air_quality_4stations.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

air_data = load_data()

# Title
st.title("Beijing Air Quality Analysis Application")

# Sidebar navigation
section = st.sidebar.radio(
    "Go to Section",
    ["Dataset Section", "Visualization Section", "Model Outputs Section"]
)

# -------------------------------
# Dataset Section
# -------------------------------
if section == "Dataset Section":
    st.header("Dataset Section")

    st.subheader("Dataset Overview")
    st.write("Shape of dataset:", air_data.shape)

    st.subheader("Preview of Dataset")
    st.dataframe(air_data.head(20))

    st.subheader("Column Names")
    st.write(list(air_data.columns))

    st.subheader("Summary Statistics")
    st.dataframe(air_data.describe())

# -------------------------------
# Visualization Section
# -------------------------------
elif section == "Visualization Section":
    st.header("Visualization Section")

    chart_option = st.selectbox(
        "Choose a chart",
        [
            "PM2.5 by Station",
            "PM2.5 by Season",
            "Average PM2.5 by Hour",
            "Correlation Heatmap"
        ]
    )

    if chart_option == "PM2.5 by Station":
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=air_data, x="station", y="PM2.5", ax=ax)
        ax.set_title("PM2.5 Distribution by Station")
        ax.set_xlabel("Station")
        ax.set_ylabel("PM2.5")
        st.pyplot(fig)

    elif chart_option == "PM2.5 by Season":
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=air_data, x="season", y="PM2.5", ax=ax)
        ax.set_title("PM2.5 Distribution by Season")
        ax.set_xlabel("Season")
        ax.set_ylabel("PM2.5")
        st.pyplot(fig)

    elif chart_option == "Average PM2.5 by Hour":
        hourly_pm25 = air_data.groupby("hour")["PM2.5"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=hourly_pm25, x="hour", y="PM2.5", marker="o", ax=ax)
        ax.set_title("Average PM2.5 by Hour of Day")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Average PM2.5")
        st.pyplot(fig)

    elif chart_option == "Correlation Heatmap":
        corr_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
                     'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        corr_matrix = air_data[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap of Pollutants and Meteorological Variables")
        st.pyplot(fig)

# -------------------------------
# Model Outputs Section
# -------------------------------
elif section == "Model Outputs Section":
    st.header("Model Outputs Section")

    st.subheader("Model Performance Summary")

    results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "Gradient Boosting", "Tuned Random Forest"],
        "MAE": [19.568, 10.419, 15.012, 13.263],
        "RMSE": [30.312, 18.216, 24.131, 22.827],
        "R2": [0.846, 0.944, 0.902, 0.913]
    })

    st.dataframe(results)

    st.subheader("Selected Best Model")
    st.write(
        "The Random Forest Regressor was selected as the final model because it achieved "
        "the lowest MAE and RMSE and the highest R² on the test set."
    )

    st.subheader("Top Feature Importances")

    feature_importance_df = pd.DataFrame({
        "Feature": ["PM10", "CO", "DEWP", "SO2", "TEMP", "NO2", "PRES", "day", "season_Spring", "O3"],
        "Importance": [0.796510, 0.085248, 0.016822, 0.012713, 0.012007, 0.011216, 0.007413, 0.007382, 0.006795, 0.006639]
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=feature_importance_df, x="Importance", y="Feature", ax=ax)
    ax.set_title("Top 10 Feature Importances")
    st.pyplot(fig)

    st.subheader("Model Interpretation")
    st.write(
        "PM10 was the most important predictor of PM2.5, followed by CO. "
        "This suggests that particulate matter and combustion-related emissions played the strongest role in prediction."
    )
