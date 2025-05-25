# Data Preprocessing - Milestone 2

Before conducting any in-depth analysis or modeling, it's essential to preprocess the data to ensure consistency, reliability, and usability. Our preprocessing workflow involves the following steps:

## Preprocessing Steps Taken in This Notebook

For the purposes of this milestone, we have conducted in-depth analyses of our data and performed several preprocessing steps to prepare the dataset for subsequent milestones.

### 1. Data Exploration

- **Feature Exploration:** We began the exploration process by presenting the schema of our DataFrame, displaying the column names, data types, and nullability.
- **Dataset Shape:** After loading the data and reviewing the feature schemas, we identified the dataset dimensions: 29,193,782 rows by 120 columns.
- **Null Values:** Columns with more than 10% missing values were dropped. For others, we plan to apply imputation using either the mean or median, depending on which results in the lowest RMSE during testing.
- **Outliers:** Outliers will be handled using statistical methods appropriate to each feature's distribution. Initial distribution analysis is covered in the next section.

### 2. Dataset Statistics and Distributions

- **Statistical Summary:** For each numerical feature, we examined the following statistics: count, mean, standard deviation, minimum, maximum, and the 25th, 50th (median), and 75th percentiles.
- **Data Distributions:** To visualize the data and identify potential outliers, we calculated skewness and plotted histograms for the 20 most skewed features.

### 3. Data Visualization

- **Questions and Corresponding Plots:** We posed several analytical questions related to flight delays and visualized relevant features to explore potential insights. Each question is accompanied by a corresponding plot and a brief discussion. This exploratory step is crucial for identifying useful features and will aid in our final feature selection process.

## Preprocessing Steps for the Future

As our objectives become more refined and our understanding of the data improves, the preprocessing steps outlined above may need to be expanded. Below are potential future enhancements:

### 1. Data Transformation

- **Standardization/Normalization:** We plan to standardize or normalize numerical features to ensure consistent scaling, particularly for models sensitive to feature magnitudes. At this stage, no features appear to require it, but we will revisit this as needed.
- **Unit Conversion:** For example, to compute average speed (in miles/hour), we will convert elapsed time to hours and divide the distance by this value.

### 2. Feature Engineering

- **New Features:** We plan to create additional variables that could enhance analysis or modeling. For instance, `speed = distance / elapsed_time`.
- **Categorical Encoding:** Any categorical features will be encoded using one-hot or label encoding to ensure compatibility with machine learning models such as random forests.

### 3. Data Integration & Reduction

- **External Data:** If useful, we may incorporate external datasets (e.g., city coordinates for geospatial analysis).
- **Dimensionality Reduction:** Techniques such as PCA or feature aggregation will be explored to reduce dimensionality and improve model performance.

## Jupyter Notebook

The full preprocessing pipeline can be found in [GroupProject.ipynb](./GroupProject.ipynb).

# Model Generation - Milestone 3

## 1. PreProcessing Finalization

- **Removal of Redundant Features:** columns deemed to have unnecessary information or information covered by other features were removed leaving 31 remaining features.
- **Handling Missing Data:** to account for the vast majority of missing data, entries with cancelled flights were removed resulting in only 0.27% of observations having null values. These rows were subsequently removed.

## 2. Feature Expansions

- **Route Column:** this new feature combines information from 'Origin' and 'Dest' to make a route identifier.
- **Speed Column:** this feature takes the ratio of 'AirTime' and 'Distance' to get the average speed during the flight.
- **Route Visualization:** for visualization purposes, we created an interactive map of the U.S. that includes the top 1000 most popular routes.

## 3. Model Creation

- **Model Selection:** a RandomForest classifier was selected to be our first model
- **String Indexing:** after preprocessing and feature expansion, there were 12 columns that needed to be converted from strings to doubles. Two of these new indices resulted in too many distinct categories and were therefore removed.
- **Label Column Reduction:** initially, there were 16 label categories. We reduced this to 4 and balanced the category counts as best as possible,
- **Dataset Split:** this dataset was split into smaller training, validation, and testing datasets (70/15/15 respectively).

## 4. Model Evaluation

- **Accuracy Scores:** the trained model was used to predict scores resulting in training and testing accuracies of around 42%
- **Importance Scores:** we used the gini importance metric to see what impact our features had on the model

## 5. Conclusion
- Our initial model provides a solid baseline, but its moderate accuracy suggests potential underfitting or label noise.

- Future work will include:

  - Exploring other model types (e.g., XGBoost, Gradient Boosted Trees)

  - Hyperparameter tuning

  - Further refining the label categories and applying SMOTE or other balancing techniques
 
**For our full code and detailed outputs, see our main notebook [GroupProject.ipynb](./GroupProject.ipynb).**
