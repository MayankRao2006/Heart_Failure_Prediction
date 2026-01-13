import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

MODEL_FILE = "model.pkl"
PIPELINE = "pipeline.pkl"

def build_pipeline(num_attrs, cat_attrs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attrs),
        ("cats", cat_pipeline, cat_attrs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv("heart.csv")
    df["Cholesterol"] = pd.to_numeric(df["Cholesterol"], errors="coerce")
    df.loc[df["Cholesterol"] == 0, "Cholesterol"] = np.nan
    df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
    df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})

    # Splitting the data
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["ST_Slope"]):
        train_set = df.iloc[train_index]
        test_set = df.iloc[test_index]
    
    test_set.to_csv("Testing_data.csv", index=False)

    features = train_set.drop("HeartDisease", axis=1)
    labels = train_set["HeartDisease"].copy()

    num_attrs = features.drop(["ChestPainType", "RestingECG", "ST_Slope"], axis=1).columns
    cat_attrs = ["ChestPainType", "RestingECG", "ST_Slope"]

    pipeline = build_pipeline(num_attrs, cat_attrs)
    prepared_data = pipeline.fit_transform(features)
    
    model = LogisticRegression()
    model.fit(prepared_data, labels)

    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE)

    print("Model and pipeline have been saved.")
    
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE)

    test_df = pd.read_csv("Testing_data.csv")
    test_features = test_df.drop("HeartDisease", axis=1)
    test_labels = test_df["HeartDisease"].copy()

    prepared_testing_data = pipeline.transform(test_features)
    predictions = model.predict(prepared_testing_data)

    test_df["Predicted_HeartDisease"] = predictions
    test_df.to_csv("Predicted_heart_disease.csv", index=False)

    print("Inference completed and results saved.")