import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

st.title("Stock Price Prediction ")
st.sidebar.title("Stock Prediction App")

dataset_option = st.sidebar.selectbox("Choose a dataset", ["TSLA", "AAPL", "ADANIGREEN", "RELIANCE", "SBIN", "GUJGASLTD"])
st.sidebar.write(f"Selected dataset: {dataset_option}")
st.subheader(f"Selected dataset: {dataset_option}")

def load_data(dataset_option):
    if dataset_option == "TSLA":
        return pd.read_csv("TSLA.csv")
    elif dataset_option == "AAPL":
        return pd.read_csv("AAPL.csv")
    elif dataset_option == "ADANIGREEN":
        return pd.read_csv("ADANIGREEN.csv")
    elif dataset_option == "RELIANCE":
        return pd.read_csv("RELIANCE.csv")
    elif dataset_option == "SBIN":
        return pd.read_csv("SBIN.csv")
    elif dataset_option == "GUJGASLTD":
        return pd.read_csv("GUJGASLTD.csv")

df = load_data(dataset_option)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
if df is None:
    st.error("Dataset failed to load. Please check the file path.")
else:
    X = df[['Open', 'High', 'Low', 'Volume']] 
    y = df['Close'] 
if df is not None:
    st.sidebar.write("### Dataset Shape")
    st.sidebar.write(f"{df.shape[0]} Rows, {df.shape[1]} Columns") 
    st.sidebar.write("### Dataset Preview")
    st.sidebar.write(df.head(5))
    st.sidebar.write("### Dataset Information")
    buffer = pd.DataFrame(df.dtypes).rename(columns={0: 'Data Type'})
    buffer["Non-Null Count"] = df.count()
    st.sidebar.table(buffer)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)

    st.subheader("Predict the Stock Price for New Input")
    open_val = st.number_input("Open Price", value=1.0)
    high_val = st.number_input("High Price", value=1.0)
    low_val = st.number_input("Low Price", value=1.0)
    volume_val = st.number_input("Volume", value=1000.0)

    if st.button("Predict"):
        user_input = [[open_val, high_val, low_val, volume_val]]

        linear_pred = models["Linear Regression"].predict(user_input)[0]
        lasso_pred = models["Lasso"].predict(user_input)[0]
        ridge_pred = models["Ridge"].predict(user_input)[0]

        st.subheader("Predicted Stock Price:")
        st.write(f"Linear Regression Prediction: {linear_pred:.2f}")
        st.write(f"Lasso Prediction: {lasso_pred:.2f}")
        st.write(f"Ridge Prediction: {ridge_pred:.2f}")

    for name, model in models.items():
        model.fit(X_train, y_train)

    st.subheader("Model Evaluation Metrics")
    def calculate_accuracy(y_test, y_pred, threshold=0.05):
        diff = abs(y_test - y_pred)  
        tolerance = threshold * y_test 
        within_tolerance = diff <= tolerance  
        accuracy = np.mean(within_tolerance) * 100  
        return accuracy
    metrics_df = pd.DataFrame({
        "Model": ["Linear Regression", "Lasso", "Ridge"],
        "Mean Squared Error": [mean_squared_error(y_test, models["Linear Regression"].predict(X_test)),
                               mean_squared_error(y_test, models["Lasso"].predict(X_test)),
                               mean_squared_error(y_test, models["Ridge"].predict(X_test))],
        "Mean Absolute Error": [mean_absolute_error(y_test, models["Linear Regression"].predict(X_test)),
                                mean_absolute_error(y_test, models["Lasso"].predict(X_test)),
                                mean_absolute_error(y_test, models["Ridge"].predict(X_test))],
        "R2 Score": [r2_score(y_test, models["Linear Regression"].predict(X_test)),
                     r2_score(y_test, models["Lasso"].predict(X_test)),
                     r2_score(y_test, models["Ridge"].predict(X_test))],
        "Accuracy (%)": [calculate_accuracy(y_test, models["Linear Regression"].predict(X_test)),
                             calculate_accuracy(y_test, models["Lasso"].predict(X_test)),
                             calculate_accuracy(y_test, models["Ridge"].predict(X_test))]
    })
    st.table(metrics_df)

    st.subheader("First 10 Predictions on Test Set")
    predictions_df = pd.DataFrame({
        "Actual": y_test[:10].values,
        "Linear Predicted": models["Linear Regression"].predict(X_test[:10]),
        "Lasso Predicted": models["Lasso"].predict(X_test[:10]),
        "Ridge Predicted": models["Ridge"].predict(X_test[:10])
    })
    st.table(predictions_df)
   
predictions_df1 = pd.DataFrame({
        "Actual": y_test[:50].values,
        "Linear Predicted": models["Linear Regression"].predict(X_test[:50]),
        "Lasso Predicted": models["Lasso"].predict(X_test[:50]),
        "Ridge Predicted": models["Ridge"].predict(X_test[:50])
    })

st.subheader("Bar Plot for Actual vs Predicted ")
plt.figure(figsize=(10, 6))
bar_width = 0.2 
r1 = np.arange(len(predictions_df1["Actual"]))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
plt.bar(r1, predictions_df1["Actual"], color='blue', width=bar_width, label="Actual")
plt.bar(r2, predictions_df1["Linear Predicted"], color='red', width=bar_width, label="Linear Predicted")
plt.bar(r3, predictions_df1["Lasso Predicted"], color='green', width=bar_width, label="Lasso Predicted")
plt.bar(r4, predictions_df1["Ridge Predicted"], color='purple', width=bar_width, label="Ridge Predicted")
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(plt)

st.subheader("Line Plot for Actual vs Predicted ")
plt.figure(figsize=(10, 6))
plt.plot(predictions_df1["Actual"], label="Actual", color='blue', marker='o')
plt.plot(predictions_df1["Linear Predicted"], label="Linear Predicted", color='red', marker='o')
plt.plot(predictions_df1["Lasso Predicted"], label="Lasso Predicted", color='green', marker='o')
plt.plot(predictions_df1["Ridge Predicted"], label="Ridge Predicted", color='purple', marker='o')
plt.legend()
st.pyplot(plt)

for name, model in models.items():
    model.fit(X_train, y_train)

predictions = {name: model.predict(X_test) for name, model in models.items()}
threshold = 0.05 
y_test_class = (y_test > y_test.mean()).astype(int) 
binary_predictions = {
            name: (pred > y_test.mean()).astype(int) for name, pred in predictions.items()
        }
for name, binary_pred in binary_predictions.items():
            st.sidebar.subheader(f"Confusion Matrix for {name}")
            cm = confusion_matrix(y_test_class, binary_pred)
            st.sidebar.write(cm)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Decrease', 'Increase'], yticklabels=['Decrease', 'Increase'])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.sidebar.pyplot(plt)




