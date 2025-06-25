import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import dagshub
import shap

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

# Initialize DagsHub with MLflow integration
dagshub.init(
    repo_owner="deema-hazim",
    repo_name="insurance-final-project",
    mlflow=True,
    quiet=True  # ⛔ 禁用 rich 的动态输出
)

st.set_page_config(
    page_title=" Bank Crunch: Predicting Customer Churn in the Banking Sector",
    layout="centered",
    page_icon="🏦",
)

st.sidebar.title("Customer Churn in the Bank 🏦")
page = st.sidebar.selectbox(
    "Select Page",
    [
        "Presentation 📘",
        "Visualization 📊",
        "Prediction 🤖",
        "Explainability 🔍",
        "MLflow Runs 📈",
    ],
)
# Display header image

## still need the image
st.image("Churn.png")
df = pd.read_csv("cleaned_churn.csv")

# Introduction Page
if page == "Presentation 📘":
    st.subheader("01 Presentation 📘")

    st.markdown("### 🔍  Introduction:")
    st.markdown("### 🎯 Project Goals:")




    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display", 5, 20, 5)
    st.dataframe(df.head(rows))

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)
    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ You have missing values")

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())



# Visulazation Page



# Prediction Page
elif page == "Prediction 🤖":
    st.subheader(" Prediction with different models 🤖")
    
# Model choice
   
    model_option = st.radio("🔘 Select Model", ("Logistic Regression", "Decision Tree","Random Forest"))

    #data processing
    ## One-Hot Encoding for 'Geography'
    df_encoded = pd.get_dummies(df, columns=['Geography'])

    ## Binary Encoding for 'Gender'
    df_encoded['Gender'] = df_encoded['Gender'].map({'Male': 0, 'Female': 1})
    ## Separate the majority and minority classes
    majority_class = df_encoded[df_encoded['Exited'] == 0]
    minority_class = df_encoded[df_encoded['Exited'] == 1]
    ## Oversample the minority class
    minority_oversampled = resample(minority_class,replace=True,n_samples=len(majority_class),random_state=42)
    # Combine the majority class with the oversampled minority class
    data_balanced = pd.concat([majority_class, minority_oversampled])
    X = data_balanced.drop('Exited', axis=1)
    y = data_balanced['Exited']
    feature_names = X.columns.tolist()

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    


    # Scale the training and testing data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

        
    
    #Logistic Regression
    if model_option == "Logistic Regression":
        st.markdown("### 📊 Logistic Regression Evaluation")
        # train the model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

         # predict
        y_pred = model.predict(X_test_scaled)

        # evalution
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # result
        st.dataframe(report_df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1-score": "{:.4f}",
            "support": "{:.0f}"
        }))
        # confusion matrix
        st.markdown("### 🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
            xticklabels=["No Exit", "Exited"], 
            yticklabels=["No Exit", "Exited"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # predict 
        # 🔮 Users input
        st.subheader("🔮 Make a Prediction")

        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure = st.slider("Tenure (Years at Bank)", 0, 10, 5)
        balance = st.number_input("Account Balance", value=50000.0)
        num_of_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", value=50000.0)

        # Input
        user_input = {
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': 1 if has_cr_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active == "Yes" else 0,
            'EstimatedSalary': estimated_salary,
            'Gender': 0 if gender == "Male" else 1,  # 注意：原来 Gender 是 0/1
            'Geography_France': 1 if geography == "France" else 0,
            'Geography_Germany': 1 if geography == "Germany" else 0,
            'Geography_Spain': 1 if geography == "Spain" else 0
        }

        # Input
        input_df = pd.DataFrame([user_input])
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        # Stand
        input_scaled = scaler.transform(input_df)

        # ✅ Pre
        if st.button("Predict Churn Status"):
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0]

            if pred == 1:
                st.error("⚠️ This customer is likely to **EXIT**.")
            else:
                st.success("✅ This customer is likely to **STAY**.")

            st.write("🧪 Probability：[STAY, EXIT] =", prob)

    elif model_option == "Decision Tree":
        st.markdown("### 🌳 Decision Tree Evaluation")
        # training
        max_depth = st.slider("Select Max Depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)  # No scalr
    

        # pre
        y_pred = model.predict(X_test)

        # Viz of decision tree
        with st.expander("🌳 Show Decision Tree Visualization"):
            dot_data = export_graphviz(
                model,
                out_file=None,
                feature_names=feature_names,
                class_names=["Stay", "Exit"],
                filled=True,
                rounded=True,
                special_characters=True
            )
        st.graphviz_chart(dot_data)

        # Evalution report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1-score": "{:.4f}",
            "support": "{:.0f}"
        }))
        # 🔮 Pre
        st.subheader("🔮 Make a Prediction")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure = st.slider("Tenure (Years at Bank)", 0, 10, 5)
        balance = st.number_input("Account Balance", value=50000.0)
        num_of_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", value=50000.0)

        # DataFrame
        user_input = {
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': 1 if has_cr_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active == "Yes" else 0,
            'EstimatedSalary': estimated_salary,
            'Gender': 0 if gender == "Male" else 1,
            'Geography_France': 1 if geography == "France" else 0,
            'Geography_Germany': 1 if geography == "Germany" else 0,
            'Geography_Spain': 1 if geography == "Spain" else 0
        }

        input_df = pd.DataFrame([user_input])
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]



        if st.button("Predict Churn Status"):
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]
            if pred == 1:
                st.error("⚠️ This customer is likely to **EXIT**.")
            else:
                st.success("✅ This customer is likely to **STAY**.")

            st.write("🧪 Probability：[STAY, EXIT] =", prob)

    elif model_option == "Random Forest":
        st.markdown("### 🌲🌲 Random Forest Evaluation")
        # Parameter Choice
        n_estimators = st.slider("Number of Trees", min_value=10, max_value=200, value=100, step=10)
        max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5)
        # Model training
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # pre
        y_pred = model.predict(X_test)

        # evaluation
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1-score": "{:.4f}",
            "support": "{:.0f}"
        }))
        # confusion matrix
        st.markdown("### 🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
            xticklabels=["No Exit", "Exited"], 
            yticklabels=["No Exit", "Exited"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

            # 🔮 Pre
        st.subheader("🔮 Make a Prediction")

        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure = st.slider("Tenure (Years at Bank)", 0, 10, 5)
        balance = st.number_input("Account Balance", value=50000.0)
        num_of_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", value=50000.0)

        user_input = {
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': 1 if has_cr_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active == "Yes" else 0,
            'EstimatedSalary': estimated_salary,
            'Gender': 0 if gender == "Male" else 1,
            'Geography_France': 1 if geography == "France" else 0,
            'Geography_Germany': 1 if geography == "Germany" else 0,
            'Geography_Spain': 1 if geography == "Spain" else 0
        }

        input_df = pd.DataFrame([user_input])
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

     
        if st.button("Predict Churn Status"):
            
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]

            if pred == 1:
                st.error("⚠️ This customer is likely to **EXIT**.")
            else:
                st.success("✅ This customer is likely to **STAY**.")

            st.write(" Probability：[STAY, EXIT] =", prob)

    











        





