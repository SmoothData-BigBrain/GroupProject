# GroupProject
## Data Preprocessing

Before conducting any in-depth analysis or modeling, it's essential to preprocess the data to ensure consistency, reliability, and usability. Our preprocessing workflow involves the following steps:

### 1. Data Cleaning

* **Null Values**: Columns with more than 10% missing values will be dropped. For others, we will apply imputation using either the mean or median, depending on which results in the lowest RMSE during testing.
* **Outliers**: We will identify and handle outliers using statistical methods such as the IQR rule or z-scores, depending on the distribution.

### 2. Data Transformation

* **Standardization/Normalization**: Numerical features will be standardized or normalized to ensure uniform scale, especially for models sensitive to feature magnitudes. After an initial analysis, nothing stands out that would require this, but we will be sure to mention here if that changes.
* **Unit Conversion**: For example, to calculate average speed (in miles/hour), we will convert elapsed time to hours and use distance traveled.

### 3. Feature Engineering

* **New Features**: We will create new variables that may help with analysis or modeling. Example: `speed = distance / elapsed_time`.
* **Categorical Encoding**: If any categorical features are included, we will encode them using one-hot encoding or label encoding. This ensures usability for models the machine learning models we plan to implement (random forest etc.).

### 4. Data Integration & Reduction

* **External Data**: If beneficial, we may integrate external datasets (e.g., city coordinates for geospatial plotting).
* **Dimensionality Reduction**: We will use PCA or combine low-variance features to reduce complexity.

### Output

The preprocessed dataset will be saved as `cleaned_data.csv` for further analysis.

### Jupyter Notebook

See the full preprocessing pipeline in [Preprocessing\_Notebook.ipynb](./Preprocessing_Notebook.ipynb).

