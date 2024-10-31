# EDA of House Prices Prediction

This project involves exploratory data analysis (EDA) and preprocessing of the House Prices dataset. The goal is to clean and prepare the data for predictive modeling.

## Data Processing

### Loading Libraries and Dataset

First, import the required libraries and load the dataset. We use `pandas` for data manipulation, `matplotlib` and `seaborn` for visualizations, and `numpy` for numeric operations.

### Initial Inspection of Data

Check the structure and basic statistics of the data:

- Data shape and types
- Summary statistics

### Missing Data

Analyze and handle missing values:

- **High Missing Values**: Columns like `PoolQC`, `MiscFeature`, and `Alley` have more than 90% missing values and are dropped.
- **Moderate Missing Values**: Columns such as `Fence`, `MasVnrType`, and `FireplaceQu` have significant missing data and are imputed with the mode.
- **Low Missing Values**: Columns like `LotFrontage`, `GarageYrBlt`, and `BsmtFinType2` have less than 20% missing values and are imputed using appropriate techniques.

### Outlier Removal

Remove outliers that are erroneous or clearly unrepresentative of the data (e.g., a house listed with 1000 rooms).

### Checking Data Types and Converting Columns

Convert date columns to `datetime` and ensure numeric columns are correctly typed.

### Final Consistency and Missing Checking

Ensure no missing values remain after preprocessing.

### Visualizing Target Variable

Explore the distribution of the target variable `SalePrice` to understand its range and trends.

### Feature Relationships

Check the relationship between features and the target variable (`SalePrice`).

- **Correlation Matrix**: Visualize correlations between numeric features.
- **Pairplot and Scatterplots**: Analyze relationships between features and `SalePrice`.

### Categorical Encoding

#### Ordinal Encoding

Map quality-related categorical features to numeric values.

#### Target Encoding

Replace categories with mean target values for high cardinality features and features with strong price correlation.

#### One-Hot Encoding

Apply one-hot encoding for features with few unique values.

### Outliers and Range Consistency Check

Ensure numeric values are within a reasonable range and correct any anomalies.

### Saving Processed Data

Save the cleaned and processed data to CSV files for further analysis and modeling.

## Files

- `data/`: Contains the raw data files.
- `src/`: Contains the preprocessing scripts.
- `HousePrices.ipynb`: Jupyter notebook with the EDA and preprocessing steps.
- `README.md`: This file.

## Usage

To run the preprocessing steps, open and execute the cells in the [HousePrices.ipynb](HousePrices.ipynb) notebook.
