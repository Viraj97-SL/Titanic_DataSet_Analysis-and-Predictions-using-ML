
-----

# 🚢 Titanic Survival Prediction Analysis

This repository contains an end-to-end data science project focused on predicting survival on the Titanic, using the famous Kaggle dataset. It encompasses a full Exploratory Data Analysis (EDA) notebook and a Machine Learning (ML) notebook comparing different classification models.

## 📁 Repository Contents

| File | Description | Key Findings/Goal |
| :--- | :--- | :--- |
| `Titanic_DataSet_EDA.ipynb` | Detailed Exploratory Data Analysis (EDA) notebook. | Cleaning missing data, visualizing feature distributions, and calculating feature correlations. |
| `ML_predictions.ipynb` | Machine Learning notebook for model training and comparison. | Feature engineering, one-hot encoding, and evaluation of **Logistic Regression**, **Decision Tree**, and **Random Forest** classifiers. |
| `README.md` | This file. | Project overview and summary of results. |

-----

## 🔎 Exploratory Data Analysis (`Titanic_DataSet_EDA.ipynb`)

The EDA focused on understanding the raw data, handling missing values, and visualizing potential relationships between features and the target variable (`Survived`).

### 📊 Key Data Insights

  * **Missing Data:** The columns `Age`, `Cabin`, and `Embarked` initially contained missing values.
      * `Age` was imputed using the **median age grouped by `Pclass` and `Sex`** for a more accurate estimate.
      * `Embarked` missing values (only 2) were filled with the **mode** (most frequent value).
      * `Cabin` had too many missing values (687/891), so a new binary feature, `HasCabin`, was created, and the original `Cabin` column was dropped.
  * **Survival Rate by Class:** There is a clear correlation between `Pclass` and `Survived`, demonstrating the "women and children first" or "first-class privilege" effect.
  * **Feature Correlation:** A heatmap of numerical features showed that `Fare` has a **slight positive correlation** with `Survived` (around **0.26**), and `FamilySize` (derived from `SibSp` + `Parch` + 1) has a **slight negative correlation** (around **-0.07**), suggesting smaller families or single travellers may have had slightly lower survival rates.

### 🐍 Key EDA Code Snippets

```python
# Check for missing values
print(df.isnull().sum())

# Impute Age using grouped medians
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

# Plot Survival Rate by Passenger Class
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Correlation Heatmap
correlation_matrix = df[['Survived', 'Age', 'Fare', 'FamilySize']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
```

-----

## 🤖 Machine Learning Model Comparison (`ML_predictions.ipynb`)

This notebook prepares the cleaned data from the EDA phase for modeling and compares the performance of three common classification algorithms.

### ✨ Feature Engineering & Preprocessing

1.  **Title Extraction:** A `Title` feature (e.g., 'Mr', 'Miss', 'Master', 'Mrs', 'Rare') was extracted from the `Name` column, as title is often a powerful predictor of survival.
2.  **Fare Binning:** The continuous `Fare` feature was converted into 4 categorical bins (`Low`, `Med`, `High`, `VeryHigh`).
3.  **Encoding:** All remaining categorical features (`Sex`, `Embarked`, `Title`, `FareBin`) were converted into numerical formats using **One-Hot Encoding** (`pd.get_dummies(..., drop_first=True)`).
4.  **Data Split:** The final feature set (`X`) was split into 80% training and 20% testing sets (`random_state=42`).

### ⚙️ Model Performance Summary

The performance of the models was evaluated using **Accuracy** (overall correct predictions) and **F1-Score** (harmonic mean of precision and recall).

| Model | Accuracy | F1-Score | Comments |
| :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | 0.8156 | 0.7755 | A solid linear baseline. |
| **Decision Tree (max\_depth=5)** | **0.8324** | **0.7917** | **Best performance.** Shows the power of non-linear decision boundaries for this dataset. |
| **Random Forest (n\_estimators=100, max\_depth=8)** | 0.8212 | 0.7681 | Robust ensemble method, slightly underperformed the tuned Decision Tree on this split. |

The **Decision Tree Classifier** yielded the highest accuracy and F1-Score on the test set.

### ❓ Confusion Matrix (Logistic Regression Example)

The Logistic Regression model's confusion matrix provides a detailed breakdown of its predictions on the 179-sample test set:

| | Predicted Died (0) | Predicted Survived (1) |
| :--- | :--- | :--- |
| **Actual Died (0)** | 89 (TN) | 16 (FP) |
| **Actual Survived (1)** | 17 (FN) | 57 (TP) |

  * **True Positives (TP):** 57 survivors were correctly predicted.
  * **False Negatives (FN):** 17 actual survivors were incorrectly predicted to have died (*Type II Error*).

### 🥇 Random Forest Feature Importance

The Random Forest model identified the most influential features for predicting survival:

1.  **Title\_Mr** (`0.206`): The passenger's title (proxy for age, gender, and social status).
2.  **Age** (`0.164`): Age remains a strong numerical factor.
3.  **Sex\_male** (`0.135`): Gender is a primary determinant due to the "women and children first" rule.
4.  **Pclass** (`0.109`): Passenger class (socio-economic status).
5.  **Title\_Mrs** (`0.084`): The title for married women.

-----

Would you like me to generate a new Python snippet, perhaps to perform a cross-validation on the Decision Tree model to confirm its stability?
