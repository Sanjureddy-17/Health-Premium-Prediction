# Health-Premium-Prediction

---

## 📌 Project Overview

This project predicts insurance-related values (such as premium or risk category) based on user details including:

* Age
* Income
* Medical History
* BMI
* Smoking Status
* Employment Status
* Region
* Insurance Plan

The system uses **two trained machine learning models**:

* `model_young.joblib` → For users aged ≤ 25
* `model_rest.joblib` → For users aged > 25

It dynamically selects the correct model based on age and applies appropriate preprocessing and scaling.

---

## 🧠 How It Works

The prediction process follows these steps:

### 1️⃣ Risk Score Calculation

Medical history is converted into a normalized risk score using predefined disease weights.

Example risk weights:

| Disease             | Risk Score |
| ------------------- | ---------- |
| Heart Disease       | 8          |
| Diabetes            | 6          |
| High Blood Pressure | 6          |
| Thyroid             | 5          |
| None                | 0          |

The total score is normalized between **0 and 1**.

---

### 2️⃣ Data Preprocessing

The system:

* Encodes categorical features (Gender, Region, BMI, etc.)
* Converts insurance plans into numeric encoding
* Adds normalized medical risk
* Applies scaling using:

  * `scaler_young.joblib`
  * `scaler_rest.joblib`

---

### 3️⃣ Model Selection

```python
if Age <= 25:
    use model_young
else:
    use model_rest
```

Each model was trained separately for better accuracy across age groups.

---

## 📂 Project Structure

```
project/
│
├── artifacts/
│   ├── model_young.joblib
│   ├── model_rest.joblib
│   ├── scaler_young.joblib
│   └── scaler_rest.joblib
│
├── app.py
├── prediction_helper.py
└── README.md
```

---

## ⚙️ Installation

Install required dependencies:

```bash
pip install pandas joblib scikit-learn
```

---

## 🚀 Usage

### Example Input

```python
input_data = {
    "Age": 24,
    "Number of Dependants": 2,
    "Income in Lakhs": 6,
    "Insurance Plan": "Silver",
    "Genetical Risk": 1,
    "Gender": "Male",
    "Region": "Southeast",
    "Marital Status": "Unmarried",
    "BMI Category": "Overweight",
    "Smoking Status": "Occasional",
    "Employment Status": "Salaried",
    "Medical History": "diabetes & high blood pressure"
}
```

### Prediction

```python
result = predict(input_data)
print("Predicted Value:", result)
```

---

## 📊 Key Features

* Age-based model segmentation
* Custom medical risk normalization
* Manual categorical encoding
* Conditional feature scaling
* Clean modular function structure

---

## 🛠 Core Functions

### `calculate_normalized_risk(medical_history)`

Computes a normalized risk score based on diseases.

### `preprocess_input(input_dict)`

Encodes categorical values and prepares the dataframe.

### `handle_scaling(age, df)`

Applies appropriate scaler depending on age group.

### `predict(input_dict)`

Returns final predicted value as an integer.

---

## 🧩 Design Decisions

* Separate models improve performance across different age distributions.
* Manual encoding ensures model compatibility with training pipeline.
* Custom risk scoring allows domain-based weighting.

---

## ⚠️ Notes

* Ensure the `artifacts/` folder contains all required `.joblib` files.
* Input dictionary keys must match exactly (case-sensitive).
* Model performance depends on training dataset quality.

---

## 📌 Future Improvements

* Replace manual encoding with pipeline-based preprocessing
* Add API layer using Flask/FastAPI
* Add input validation
* Add logging & error handling
* Add model performance metrics to documentation

---
