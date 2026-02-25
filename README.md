# 🌱 Soil Moisture Prediction using ML & Deep Learning

## 📌 Project Overview

This project focuses on building a **regression model to predict soil moisture content** using radar backscatter signals and satellite-derived features.

The objective is to:

* Perform detailed **Exploratory Data Analysis (EDA)**
* Apply **Feature Engineering**
* Train multiple **Machine Learning models**
* Compare with a **Deep Learning model**
* Evaluate performance using RMSE, MAE, and R²

---

## 📂 Dataset Information

* **Total Samples:** 30,747
* **Features:** 3 Independent + 1 Target
* **Target Variable:** `soil_moisture`
* **Input Features:**

  * `VV`
  * `VH`
  * `smap_am`

### Data Quality Checks

| Check                                         | Result                |
| --------------------------------------------- | --------------------- |
| Missing Values                                | None                |
| Duplicate Rows                                | 13                    |
| Physically Invalid Values (soil_moisture > 1) | 22                    |
| Final Clean Dataset                           | Filtered to 0–1 range |

---

# 📊 Exploratory Data Analysis (EDA)

## 1️⃣ Distribution Analysis

* `VV` and `VH` show negatively skewed dB distributions.
* `smap_am` is right-skewed.
* `soil_moisture` mostly lies within **0–1 range**.

Outliers (>1) were removed as physically unrealistic.

---

## 2️⃣ Correlation Analysis

Correlation with `soil_moisture`:

| Feature | Correlation |
| ------- | ----------- |
| VH      | -0.070      |
| VV      | -0.044      |
| smap_am | 0.041       |

<img width="515" height="418" alt="image" src="https://github.com/user-attachments/assets/2365b4ab-0e41-4fc8-9413-8e9177e6926a" />

📌 **Observation:**
Very weak linear correlation → Linear models are not sufficient.

This justified using:

* Tree-based models
* Non-linear models
* Feature engineering

---

# ⚙️ Feature Engineering

To capture non-linear relationships:

### 1️⃣ dB → Linear Conversion

```
VV_lin = 10^(VV/10)
VH_lin = 10^(VH/10)
```

### 2️⃣ Interaction Features

* VV × VH
* VV × smap
* VH × smap

### 3️⃣ Ratio Features

* VV / VH
* VH / VV
* VV − VH

### 4️⃣ Polynomial Features

* VV²
* VH²
* smap²

### 5️⃣ Log Transform

* log(1 + smap)

---

## Selected Final Features

```
VV_lin
VH_lin
smap_am
VV_minus_VH
VV_VH_ratio
VV_smap_interaction
VH_smap_interaction
smap_sq
```

---

# 🤖 Machine Learning Models

### Train-Test Split

* 80% Training
* 20% Testing
* random_state = 42

---

## Models Implemented

* Random Forest
* Gradient Boosting
* XGBoost
* Support Vector Regression (RBF)
* KNN
* Deep Neural Network (ANN)

---

# 📈 Model Performance (Before Tuning)

| Model             | RMSE    | MAE     | R²     |
| ----------------- | ------- | ------- | ------ |
| Gradient Boosting | 0.01477 | 0.10186 | 0.072  |
| SVR               | 0.01503 | 0.10271 | 0.056  |
| Random Forest     | 0.01546 | 0.10263 | 0.029  |
| XGBoost           | 0.01678 | 0.10601 | -0.054 |
| KNN               | 0.01681 | 0.10612 | -0.056 |

📌 Gradient Boosting performed best initially.

---

# 🔍 Hyperparameter Tuning

## Best Random Forest Parameters

```
n_estimators = 600
max_depth = 15
min_samples_split = 5
min_samples_leaf = 2
```

### Tuned RF Results

* RMSE: 0.01458
* MAE: 0.10076
* R²: 0.08445

---

## Best Gradient Boosting Parameters

```
n_estimators = 200
max_depth = 4
learning_rate = 0.1
subsample = 0.8
```

### Tuned GB Results

* RMSE: 0.01467
* MAE: 0.10130
* R²: 0.07893

---

## Best XGBoost Parameters

```
n_estimators = 300
max_depth = 3
learning_rate = 0.1
subsample = 0.8
```

### Tuned XGB Results

* RMSE: 0.01474
* MAE: 0.10170
* R²: 0.07415

---

# 🏆 Best Model

✅ **Random Forest (Tuned)**

* Lowest error
* Highest R² (0.084)

---

# 📉 Residual Analysis

Residual plot observations:

* Residuals centered around zero → Low bias
* Funnel-shaped pattern → Heteroscedasticity
* Model performance varies across moisture ranges
  
<img width="578" height="432" alt="image" src="https://github.com/user-attachments/assets/de5c706c-e2be-463a-8697-726ce54319b4" />

📌 Conclusion:
Model is acceptable but variance is not constant.

---

# 🧠 Deep Learning Model

### Architecture

Input → 128 → 64 → 32 → 16 → Output
Activation: ReLU
Optimizer: Adam (lr = 0.001)
Loss: MSE
Batch Size: 64
Early Stopping: patience = 10

---

## DL Performance

* RMSE: 0.01496
* MAE: 0.10266
* R²: 0.06055

📌 Observation:
Deep Learning did not outperform tuned Random Forest.

---

# 📊 Training Curve Analysis

From the training plot:

- Training and validation loss converge.
- No severe overfitting due to Early Stopping.
- Training loss drops quickly in early epochs.
- Validation loss stabilizes early.
- No major divergence between training and validation curves.

## Interpretation

- The model is not overfitting.
- However, the model is unable to extract stronger predictive signal from the available features.

<img width="556" height="413" alt="image" src="https://github.com/user-attachments/assets/f365ba86-d321-4d69-9c73-bd8e435ed26d" />

---

# 🔎 Final Insights

1. Soil moisture has weak linear correlation with raw radar features.
2. Feature engineering significantly improved performance.
3. Tree-based models outperform linear and deep learning models.
4. Random Forest achieved best overall performance.
5. Residual analysis shows heteroscedasticity remains.

---

# 🚀 Possible Improvements

* Add temporal features
* Use spatial context (geolocation)
* Apply ensemble stacking
* Use advanced DL models (TabNet / Attention-based models)
* Use larger dataset

---

# 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* TensorFlow / Keras
* Matplotlib
* Seaborn

---

# 📁 Project Structure

```
Soil-Moisture-Prediction/
│
├── data/t_s1_am_6am.csv
├── notebook/soil_moisture_model.ipynb
├── report/soil_moisture_Project_Documentation.pdf
└── README.md
```

---

# 👨‍💻 Author

**N. Arun Nenavath**
B.Tech – Mathematics and Computing

---

# 📌 Resume-Ready Project Description

Developed a regression system to predict soil moisture using radar backscatter features. Performed advanced feature engineering and compared multiple ML/DL models. Achieved best R² of 0.084 using tuned Random Forest, outperforming Gradient Boosting, XGBoost, SVR, and ANN models.
