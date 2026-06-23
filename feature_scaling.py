# Data Cleaning and Feature scaling

## Dataset Used:

Iris Dataset (from Scikit-learn)

## Objective:

To apply data cleaning and feature scaling techniques in preparing a dataset for machine learning.

---

## Step 1: Load the Dataset

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target variable
df['species'] = iris.target

print(df.head())
```

### Raw Dataset Snapshot

| sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | species |
| ----------------- | ---------------- | ----------------- | ---------------- | ------- |
| 5.1               | 3.5              | 1.4               | 0.2              | 0       |
| 4.9               | 3.0              | 1.4               | 0.2              | 0       |
| 4.7               | 3.2              | 1.3               | 0.2              | 0       |
| 4.6               | 3.1              | 1.5               | 0.2              | 0       |
| 5.0               | 3.6              | 1.4               | 0.2              | 0       |

---

## Step 2: Data Cleaning

### Introduced Sample Issues

```python
# Introduce missing values
df.loc[2, 'sepal length (cm)'] = np.nan
df.loc[5, 'petal width (cm)'] = np.nan

# Add duplicate row
df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

# Convert columns to uppercase (inconsistent formatting)
df.columns = [col.upper() for col in df.columns]
```

---

### Identify Missing Values

```python
print(df.isnull().sum())
```

Output:

```python
SEPAL LENGTH (CM)    1
SEPAL WIDTH (CM)     0
PETAL LENGTH (CM)    0
PETAL WIDTH (CM)     1
SPECIES              0
```

---

### Handle Missing Values

```python
df.fillna(df.mean(numeric_only=True), inplace=True)
```

Used mean imputation to replace missing numeric values.

---

### Remove Duplicates

```python
df.drop_duplicates(inplace=True)
```

Removed duplicate records to improve dataset quality.

---

### Standardize Column Names

```python
df.columns = df.columns.str.lower().str.replace(" ", "_")
```

Standardized all column names for consistency.

---

## Step 3: Feature Scaling

### Method Used: StandardScaler

Reason:
StandardScaler was used because it transforms data into a standard normal distribution with mean = 0 and standard deviation = 1. This helps machine learning algorithms perform better when features are on similar scales.

---

### Before Scaling

| sepal_length_(cm) | sepal_width_(cm) | petal_length_(cm) | petal_width_(cm) |
| ----------------- | ---------------- | ----------------- | ---------------- |
| 5.1               | 3.5              | 1.4               | 0.2              |
| 4.9               | 3.0              | 1.4               | 0.2              |
| 5.8               | 3.2              | 1.3               | 0.2              |

---

### Apply Scaling

```python
scaler = StandardScaler()

numeric_cols = df.drop('species', axis=1)
scaled_data = scaler.fit_transform(numeric_cols)

scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols.columns)

print(scaled_df.head())
```

---

### After Scaling

| sepal_length_(cm) | sepal_width_(cm) | petal_length_(cm) | petal_width_(cm) |
| ----------------- | ---------------- | ----------------- | ---------------- |
| -0.90             | 1.03             | -1.34             | -1.31            |
| -1.14             | -0.12            | -1.34             | -1.31            |
| -0.05             | 0.34             | -1.40             | -1.31            |

---

## Documentation of Process

* The dataset initially contained missing values in two numeric columns.
* Duplicate rows were found and removed to avoid repeated data.
* Column names were inconsistent (uppercase format), so they were standardized.
* Mean imputation was used to handle missing values without reducing dataset size.
* StandardScaler was applied because machine learning models often perform better when all features are standardized.

## Conclusion

Data preprocessing is an essential step in machine learning. Cleaning and scaling improve data quality, reduce bias, and ensure better model performance.
