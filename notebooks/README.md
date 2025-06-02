# Introduction

...

# Dataset

**Our data set can be found here: [Flight Status Prediction](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022)**

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

# Model Optimization and Final Submission - Milestone 4

## Introduction:
Why chosen? why is it cool? General/Broader impact of having a good predictive mode. i.e. why is this important?

... INTRO ...
**Why chosen?** We chose this model and dataset because  

## Figures:
Figures (of your choosing to help with the narration of your story) with legends (similar to a scientific paper) For reference you search machine learning and your model in google scholar for reference examples.

... GENERATE IMAGES OF RANDOM FOREST MODEL...
**Image Example:**
![RandomForest](https://media.datacamp.com/legacy/image/upload/v1677239992/image1_73caef2811.png)

## Methods:
Methods section (this section will include the exploration results, preprocessing steps, models chosen in the order they were executed. Parameters chosen. Please make sub-sections for every step. i.e Data Exploration, Preprocessing, Model 1, Model 2, additional models are optional , (note models can be the same i.e. DNN but different versions of it if they are distinct enough. Changes can not be incremental). You can put links here to notebooks and/or code blocks using three ` in markup for displaying code. so it would look like this: ``` MY CODE BLOCK ```
- Note: A methods section does not include any why. the reason why will be in the discussion section. This is just a summary of your methods

... METHODS  LIST AND DESCRIPTION...

## Results:
This will include the results from the methods listed above (C). You will have figures here about your results as well.
- No exploration of results is done here. This is mainly just a summary of your results. The sub-sections will be the same as the sections in your methods section.
- Your final model and final results summary will go in the last paragraph.

... RESULTS FROM METHODS ...

## Discussion:
This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!

... 

## Conclusion:
This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts

...

## Collaboration:
This is a statement of contribution by each member. This will be taken into consideration when making the final grade for each member in the group. Did you work as a team? was there a team leader? project manager? coding? writer? etc. Please be truthful about this as this will determine individual grades in participation. There is no job that is better than the other. If you did no code but did the entire write up and gave feedback during the steps and collaborated then you would still get full credit. If you only coded but gave feedback on the write up and other things, then you still get full credit. If you managed everyone and the deadlines and setup meetings and communicated with teaching staff only then you get full credit. Every role is important as long as you collaborated and were integral to the completion of the project. If the person did nothing. they risk getting a big fat 0. Just like in any job, if you did nothing, you have the risk of getting fired. Teamwork is one of the most important qualities in industry and academia!!!
- Start with Name: Title: Contribution. If the person contributed nothing then just put in writing: Did not participate in the project.

- **Ahyo:**
- **Hailey:**
- **Mihir:**
- **Nam:**
- **Rita:**




