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

The full preprocessing pipeline can be found in [GroupProject.ipynb](./notebooks/GroupProject.ipynb).

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
 
**For our full code and detailed outputs, see our main notebook [GroupProject.ipynb](./notebooks/GroupProject.ipynb).**


---
# Model Optimization and Final Submission - Milestone 4

## Introduction:
Why chosen? why is it cool? General/Broader impact of having a good predictive mode. i.e. why is this important?
**** Mihir Start


**** Mihir End

### ... INTRO ...
**Why chosen?** We chose this model and dataset because  
**** Mihir start

**** Mihir End


## Figures:
Figures (of your choosing to help with the narration of your story) with legends (similar to a scientific paper) For reference you search machine learning and your model in google scholar for reference examples.
**** Nam Start
### ... GENERATE IMAGES OF RANDOM FOREST MODEL...
**Image Example:**

![RandomForest](https://media.datacamp.com/legacy/image/upload/v1677239992/image1_73caef2811.png)


**** Nam End

## Methods:
Methods section (this section will include the exploration results, preprocessing steps, models chosen in the order they were executed. Parameters chosen. Please make sub-sections for every step. i.e Data Exploration, Preprocessing, Model 1, Model 2, additional models are optional , (note models can be the same i.e. DNN but different versions of it if they are distinct enough. Changes can not be incremental). You can put links here to notebooks and/or code blocks using three ` in markup for displaying code. so it would look like this: ``` MY CODE BLOCK ```
- Note: A methods section does not include any why. the reason why will be in the discussion section. This is just a summary of your methods

### ... METHODS  LIST AND DESCRIPTION...
**** Mihir Start
Enter text here...

**** Mihir End


**** Hailey Start
1. Data Exploration
- Evaluated null value abundance across all features.
- Features with >10% null values were excluded.
- Assessed distribution and summary statistics for each feature.
- Identified skewed distributions to inform preprocessing steps.

2. Preprocessing
- Removed redundant features.
- Excluded cancelled flights to focus on factors predictive of flight delay duration.
Feature Engineering:
- Created new feature Route by combining Origin and Dest.
- Generated ratio feature: AirTime / Distance.
- Applied log transformation to skewed numeric features (e.g., Distance with >3× skew).
- Indexed categorical variables for ML compatibility.
- Created multiclass label by binning delay durations into 4 categories.

3. Model 1: Random Forest Classifier
- Trained a Random Forest model to predict delay duration category.
- Used feature importance scores to identify the most influential predictors.
- Planned hyperparameter tuning via a validation dataset if SDSC resources were available.

4. Model 2: K-Means Clustering (Planned if SDSC resources were available) 
- Fit K-Means on the Feature Matrix
- Apply K-Means to the preprocessed data (excluding the target/label column).
- Choose k based on the elbow method or aligned with the number of delay categories (e.g., 4 clusters for 4 delay duration classes).
- analyze cluster composition by assigning the original delay labels back to each sample after clustering.
- calculate label proportions in each cluster to identify which clusters are enriched for high-delay flights.
- Compare feature distributions by cluster. For each feature, plot a heatmap to compare how feature values differ between clusters.
- Rank features by cluster separation (ANOVA F-value?) to help quantify how strongly a feature is driving cluster formation - how it relates to delay outcomes. 

5. Model 1 & 2 comparisons (Planned if SDSC resources were available)
- Features with highest Gini scores were to be compared to high ranked features by clustering.
- Potential interpretations: features that are top-ranked in both models are likely truly important for explaining flight delay. If a feature is ranked highly in clustering but not random forest, it might be associated with delay patterns that the model didn't learn well — useful for model refinement or domain insights.

**** Hailey End

**** Ahyo Start (if needed)
Enter text here...

**** Ahyo End

**** Nam start
Enter text here...


**** Name end


**** Rita Start
Enter text here...


**** Rita End


## Results:
This will include the results from the methods listed above (C). You will have figures here about your results as well.
- No exploration of results is done here. This is mainly just a summary of your results. The sub-sections will be the same as the sections in your methods section.
- Your final model and final results summary will go in the last paragraph.

### ... RESULTS FROM METHODS ...
**** Rita Start
Enter text here...


**** Start

**** Hailey start
Potential interpretations if both supervised and unsupervised models were able to be evaluated. Missing ranked features from unsupervised clustering, unable to complete this model without SDSC resources: 
- Supervised learning (Random Forest): - Used feature importance (Gini) scores to identify the most influential features for accurately predicting flight delay duration.
-  Unsupervised learning (K-Means): Rank features by cluster separation (ANOVA F-value?) to help quantify how strongly a feature is driving cluster formation - how it relates to delay outcomes.
- Comparison of top ranked features from both approaches: features that are top-ranked in both models are likely truly important for explaining flight delay.
- If a feature is ranked highly in clustering but not random forest, it might be associated with delay patterns that the model didn't learn well.

**** Hailey end

## Discussion:
This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!

**** As needed write discussion under your block
**** Mihir Start
Enter text here...

**** Mihir End


**** Hailey Start

This project aimed to utilize two distinct methods—Random Forest classification and K-Means clustering—to identify factors associated with flight delays. While both methods aimed to uncover relationships between features and flight delay durations, they operate on fundamentally different principles, making direct comparison both informative and limited.

Random Forest, a supervised learning algorithm, ranks features by their predictive power—specifically, how much each feature contributes to reducing impurity in classifying delay categories. This results in a quantitative feature importance score that highlights variables most useful for prediction, such as origin, distance, or taxi-out time.

In contrast, K-Means clustering is an unsupervised method that groups flights based on feature similarity, without reference to the actual delay categories. After clustering, we can examine the distribution of delay categories within each cluster and analyzed the underlying feature distributions. This approach identifies which feature values (not just features) are common in clusters enriched for each delay category. For example, a cluster characterized by longer delays might also have high values for taxi-out time and low values for flight distance.

Although both methods can highlight relevant features, they do so through different approaches:
- Random Forest emphasizes which features help a model distinguish between delay categories.
- K-Means emphasizes which feature values tend to group together in clusters associated with specific delay outcomes.

Despite these differences, the two methods offer complementary insights, features that are ranked highly by both methods (e.g., high Random Forest importance and distinct cluster patterns) provide strong evidence of their relevance to flight delays. Discrepancies between the two can guide further analysis—e.g., a feature important in clustering but not in RF may suggest a variable underrepresented in the model.

**** Hailey End

**** Ahyo Start (if needed)
Enter text here...

**** Ahyo End

**** Nam start
Enter text here...


**** Name end


**** Rita Start
Enter text here...


**** Rita End

## Conclusion:
This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts

**** Group Start
Enter text here...

**** Group end

## Collaboration:
This is a statement of contribution by each member. This will be taken into consideration when making the final grade for each member in the group. Did you work as a team? was there a team leader? project manager? coding? writer? etc. Please be truthful about this as this will determine individual grades in participation. There is no job that is better than the other. If you did no code but did the entire write up and gave feedback during the steps and collaborated then you would still get full credit. If you only coded but gave feedback on the write up and other things, then you still get full credit. If you managed everyone and the deadlines and setup meetings and communicated with teaching staff only then you get full credit. Every role is important as long as you collaborated and were integral to the completion of the project. If the person did nothing. they risk getting a big fat 0. Just like in any job, if you did nothing, you have the risk of getting fired. Teamwork is one of the most important qualities in industry and academia!!!
- Start with Name: Title: Contribution. If the person contributed nothing then just put in writing: Did not participate in the project.

1. **Ahyo:** Coding/Management. I set up initial communication through discord and email and helped organize meetings throughout the project. After selecting our dataset, I created the foundational notebook for us to use for the remainder. I aided in exploring the dataset through pyspark. Specifically, I developed several stacked barcharts showing flight departure delays grouped by origin airport, year, and month categorized as early, on time, or late. To preprocess the data, I, along with the rest of the team, wrote code to clean, feature engineer, and transform the dataset, preparing it for modeling. Finally, I was responsible for writing and organizing much of the README.md file that outlines this project.
2. **Hailey:** Coder/Writer/Machine Learning lead. I led the development of the machine learning strategy to assess factors contributing to flight delays, including proposing the use of Random Forest for feature selection and unsupervised clustering for distribution analysis, as I was most familiar with machine learning applications in the group. I designed the data preprocessing workflow, ensuring the dataset was cleaned, structured, and optimized for machine learning implementation using PySpark in which everyone contributed to coding a separate piece of the preprocessing steps. I also contributed to understanding data distributions, skews, and general statistics of the dataset features to set up the preprocessing workflow. 
3. **Mihir:** Project Manager/Lead/Dev. I contributed by creating document for setting up the local env for everyone to use. Set up github repo and managed the processes to do with SDLC. I contributed to helping set up the data and extracting it to being able to use it by everyone, and contributed to data engineering process for coding. My main focus was to make sure everyone was on same page about the project, and had effective communication and strategy for the project. This work easy to manage since everyone in the project was equally helpful and communicated with each other as needed. I maintained the Git repo and and Pull request members did to make sure no merge conflcits deleted work by others since I was the only member in group experienced in Git. 
4. **Nam:**
5. **Rita:**

