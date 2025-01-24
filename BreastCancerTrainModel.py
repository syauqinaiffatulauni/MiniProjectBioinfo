import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle


# Function to preprocess data
def preprocess_data(df, target_col=None):
    st.write("Handling missing values...")
    df.fillna(df.median(numeric_only=True), inplace=True)  # Impute missing numerical values
    df.fillna("Unknown", inplace=True)  # Handle missing categorical values

    for col in df.select_dtypes(include=["object"]).columns:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    if target_col and df[target_col].dtype == "object":
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])

    return df


# Function to train model and save feature names
def train_model(X, y, model_type, n_estimators, feature_names_file):
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    else:
        raise ValueError("Invalid model type")

    model.fit(X, y)

    # Save feature names
    with open(feature_names_file, "wb") as f:
        pickle.dump(X.columns.tolist(), f)
    st.write(f"Feature names saved as: {feature_names_file}")
    return model


# Function to align features during prediction
def align_features(df, feature_names_file):
    try:
        with open(feature_names_file, "rb") as f:
            feature_names = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Feature names file {feature_names_file} not found. Please upload the correct file.")

    # Add missing columns with default value 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Drop extra columns not in feature names
    df = df[feature_names]
    return df


# Function to load model
def load_model(model_file):
    return pickle.load(model_file)


# Streamlit app
def main():
    st.title("Breast Cancer Prediction Platform")

    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Choose Action", ["Train Model", "Predict"])

    if menu == "Train Model":
        st.header("Train a Classification Model")

        data_file = st.file_uploader("Upload Gene Expression Dataset (CSV)", type=["csv"], key="train_file")

        if data_file is not None:
            df = pd.read_csv(data_file)
            st.write("Uploaded Dataset:", df.head())
            st.write("Dataset Summary:")
            st.write(df.describe(include="all"))

            target_col = st.selectbox("Select Target Column", options=df.columns, key="target_column")

            if target_col:
                df = preprocess_data(df, target_col)

                X = df.drop(columns=[target_col])
                y = df[target_col]

                # Display class distribution
                st.write("Class Distribution Before Balancing:", y.value_counts())

                # Apply SMOTE to balance the classes
                smote = SMOTE(random_state=42)
                X, y = smote.fit_resample(X, y)
                st.write("Class Distribution After Balancing:", pd.Series(y).value_counts())

                model_options = ["Random Forest", "Gradient Boosting"]
                model_type = st.selectbox("Choose Model Type", model_options, key="model_type")
                n_estimators = st.slider("Number of Estimators", min_value=10, max_value=200, step=10, value=100, key="n_estimators")

                if st.button("Train Model", key="train_button"):
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y
                        )
                    except ValueError:
                        st.warning("Stratified splitting failed due to rare classes. Using random splitting.")
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    feature_names_file = "random_forest_features.pkl"
                    model = train_model(X_train, y_train, model_type, n_estimators, feature_names_file)

                    st.success(f"{model_type} model trained successfully!")

                    y_pred = model.predict(X_test)

                    st.subheader("Model Performance")
                    st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                    st.text(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
                    st.text("Classification Report:")
                    st.text(classification_report(y_test, y_pred))

                    model_filename = f"{model_type.lower().replace(' ', '_')}_model.pkl"
                    with open(model_filename, "wb") as f:
                        pickle.dump(model, f)
                    st.download_button(
                        label="Download Model",
                        data=pickle.dumps(model),
                        file_name=model_filename,
                        mime="application/octet-stream"
                    )

                    # Save preprocessed dataset
                    st.download_button(
                        label="Download Preprocessed Dataset",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="preprocessed_dataset.csv",
                        mime="text/csv"
                    )

    elif menu == "Predict":
        st.header("Predict Using a Pretrained Model")

        data_file = st.file_uploader("Upload Dataset for Prediction (CSV)", type=["csv"], key="predict_file")
        model_file = st.file_uploader("Upload Pretrained Model File (PKL)", type=["pkl"], key="model_file")
        feature_file = st.file_uploader("Upload Feature Names File (PKL)", type=["pkl"], key="feature_file")

        if data_file and model_file and feature_file:
            df = pd.read_csv(data_file)
            st.write("Uploaded Dataset for Prediction:", df.head())

            try:
                model = load_model(model_file)

                with open(feature_file, "rb") as f:
                    feature_names = pickle.load(f)

                df = preprocess_data(df)
                df = align_features(df, feature_file)

                predictions = model.predict(df)
                df["Prediction"] = predictions
                st.write("Predictions:", df)

                st.download_button(
                    label="Download Predictions",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")


if __name__ == "__main__":
    main()
