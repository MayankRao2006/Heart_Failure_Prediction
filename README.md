# ❤️ Heart Failure Predictor

Heart Failure Predictor is a **machine learning project** that predicts the likelihood of heart disease in patients using medical data. This project demonstrates a complete ML pipeline: from preprocessing raw data to training a model, and making predictions on new data.

---

## 🧰 Features

- Handles missing or inconsistent values in the dataset.
- Encodes categorical variables and scales numerical features.
- Implements a **Logistic Regression** model for heart disease prediction.
- Saves the trained model and preprocessing pipeline for future inference.
- Generates predictions on test data and saves results in CSV format.

---

## 📊 Dataset

The project uses a dataset named `heart.csv` with patient health metrics including:

- Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
- Target: `HeartDisease` (1 = disease, 0 = no disease)

> **Note:** Ensure `heart.csv` is in the project directory before running the code.
Dataset source: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
---

## ⚙️ How to use
1. Open Main.ipynb to explore the full workflow, including model comparisons and preprocessing.
2. Alternatively, run model.py on your local machine to train the model and generate predictions.

---

### 🛠️ How it works

1. Data Preprocessing:

   - Converts numeric columns and handles missing values.

   - Maps categorical variables (Sex, ExerciseAngina) to numerical values.

   - One-hot encodes other categorical features (ChestPainType, RestingECG, ST_Slope).

   - Scales numerical features using StandardScaler.

2. Model Training:

   - Uses Logistic Regression on the preprocessed training data.

   - Saves the trained model (model.pkl) and preprocessing pipeline (pipeline.pkl) using joblib.
     

3. Inference:

   - Loads the model and pipeline.

   - Transforms test data and predicts heart disease.

   - Saves the predictions to a CSV file.

---

## 📂 Project Structure

HeartPredictor/
│
├─ model.py                                         # Main Python script for training and inference               
├─ Predicted_heart_disease.csv                      # Output predictions
├─ heart.csv                                        # Dataset (to be provided by user)
├─ Main.ipynb                                       # jupyter notebook which explains the whole process as well as model comparision
└─ README.md

---

## 🙏 Credits
Dataset by Fedesoriano on Kaggle - https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
