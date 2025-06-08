# Model Optimization and Final Submission - Milestone 4
## Abstract 

We aim to analyze a dataset of approximately 30 million U.S. domestic flights from 2018 to 2022 - around 11GB in size - to understand the key factors contributing to flight delays. The dataset includes features such as scheduled and actual departure/arrival times, delay durations, and reasons for delays or cancellations. Our approach will involve comparing two methods - feature selection using random forest and feature distribution through unsupervised clustering - to identify the most significant factors influencing flight delays. Due to the limitation of the SDSC resources, we weren't able to get to the latter but we hope as a group to continue researching this if possible post class. The ultimate goal is to extract insights that can be used to predict flight delay statuses. Given the dataset’s large size (29 million rows and 120 columns), we will leverage PySpark for efficient data processing.

## Introduction

Modern transportation has significantly evolved, ranging from walking and horse-drawn carriages to advanced systems such as cars, buses, subways, and airplanes. Among these, air travel stands out due to its speed, convenience, and the ability to bridge large distances quickly. Accurate travel planning, including precise departure and arrival times, is crucial in our fast-paced world, as delays can result in lost productivity, increased stress, and disrupted schedules. In 2023 alone, over 2.6 billion passengers used domestic air travel in the U.S., emphasizing the scale and impact of delays.

We selected the U.S. Domestic Flights Delay dataset (2018-2022) from Kaggle due to its extensive coverage, containing roughly 30 million records detailing various factors associated with flight scheduling and delays. This comprehensive dataset provides rich insights into scheduled versus actual departure and arrival times, delay durations, and specific reasons for cancellations or delays. Our goal is to identify and analyze the most influential factors leading to delays by applying advanced data analysis techniques, specifically feature selection through random forest and feature distribution via unsupervised clustering.

By effectively predicting flight delays, travelers, airlines, and airport authorities can proactively manage disruptions, significantly improving passenger satisfaction, reducing operational costs, and enhancing overall efficiency. Ultimately, our analysis aims to deliver valuable insights that can facilitate better planning, optimize time usage, and positively influence millions of travelers each year.

## Methods

### 1. Data Exploration
- Evaluated null value abundance across all features.
- Features with >10% null values were excluded.
- Assessed distribution and summary statistics for each feature.
- Identified skewed distributions to inform preprocessing steps.
```py
# Finding stats
for col_name in cont_cols:
    q1, median, q3 = filtered_df.approxQuantile(col_name, [0.25, 0.5, 0.75], 0.01)
    stats["25%"][col_name] = str(q1)
    stats["50%"][col_name] = str(median)
    stats["75%"][col_name] = str(q3)

# Removing Null Values
columns_above_90 = [col_name for col_name, pct in non_null_percentages.items() if pct >= 90]
filtered_df = df.select(columns_above_90)

# Managing skewed data
# build rows of (column, absolute_diff, skew direction)
result_rows = []
for c in cols: # for each col
    mean_val = float(mean_row[c]) # get mean
    median_val = float(median_row[c]) # get median
    diff = __builtins__.abs(mean_val - median_val) # get abs difference
    skew = "right" if mean_val > median_val else "left" if mean_val < median_val else "none" # get skew direction
    result_rows.append(Row(column=c, absolute_diff=diff, skew=skew)) # aggregate
```

### 2. Preprocessing
- Removed redundant features.
- Excluded cancelled flights to focus on factors predictive of flight delay duration.
Feature Engineering:
- Created new feature Route by combining Origin and Dest.
- Generated ratio feature: AirTime / Distance.
- Applied log transformation to skewed numeric features (e.g., Distance with >3× skew).
- Indexed categorical variables for ML compatibility.
- Created multiclass label by binning delay durations into 4 categories.
```py
# Creating new features
filtered_df = filtered_df.withColumn(
    "route",
    F.concat_ws(
        " - ",
        F.coalesce(F.col("Origin"), F.lit("Unknown")),
        F.coalesce(F.col("Dest"), F.lit("Unknown"))
    )
)

filtered_df = filtered_df.withColumn(
    "avg_speed_mph",
    F.when(
        F.col("AirTime").isNotNull() & (F.col("AirTime") != 0),
        F.round((F.col("Distance") / F.col("AirTime")) * 60)
    ).otherwise(None)
)

```

### 3. Model 1: Random Forest Classifier
- Trained a Random Forest model to predict delay duration category.
- Used feature importance scores to identify the most influential predictors.
- Preliminary hyperparemeter tuning done by measuring accuracy when varying `numTrees` and `maxDepth`.
- Further hyperparameter tuning via a validation dataset was planned if SDSC resources were available.
```py

# assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
# Generate random forest model
rf = RandomForestClassifier(labelCol="ArrivalDelayGroups", featuresCol="features", seed=42, maxBins=800, numTrees = 5, maxDepth = 3)

...
# Feature importance
rf_model = model.stages[-1]  # assuming RF is the last stage
importances = rf_model.featureImportances
feature_names = assembler.getInputCols()

importances_list = list(zip(feature_names, importances.toArray()))
importances_sorted = sorted(importances_list, key=lambda x: x[1], reverse=True)

print("Feature Importances from RF Gini Importance:\n")
for feat, score in importances_sorted:
    print(f"{feat}: {score:.4f}")
```

### 4. Model 2: K-Means Clustering (Planned if SDSC resources were available) 
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

## Results:

### 1. Data Exploration
Our unfiltered dataset had a shape of 29193782 rows and 120 columns. In the initial data exploration, we chose to remove the columns that more than 10% of null values, leaving us with 62 columns out of the original 120 columns to work with. After removing these columns, we decided to explore the top 20 most skewed columns as seen below. Most of which are skewed right, aside from WheelsOn, ArrTime, and CRSArrTime, which are skewed left. 

![hist_plots](./images/histogram_plots.png)

### 2. Preprocessing 
We decided to remove columns we thought to be redundant for analysis or were related to other columns. For instance, we removed FlightDate because of the Year, Month, and DayofWeek columns. For missing data, columns with more than 10% of null values were excluded from our analysis. Columns that had 1-3% of msising values were Tail_Number, DepTime, DepDelay, DepDelayMinutes, DepDel15, DepartureDelayGroups, WheelsOff, TaxiOut, ArrTime, WheelsOn, TaxiIn, ActualElapsedTime, ArrDelay, ArrDelayMinutes, ArrDel15,  ArrivalDelayGroups, and AirTime. Their missing values were handled by just removing the missing rows. 

For feature expansion, Route was created by combining the Origin and Dest columns. Avg_speed_mph (Distance/AirTime * 60) and num_flights (the count of each route) were created to aid in model performance. Categorical variables were indexed using StringIndexer to convert them to numeric values to be used in the Random Forest model.

### 3. Model 1: Random Forest Classifier
Our first model yielded an accuracy score of ~42% for both the training and test data. Overfitting was not strongly observed, as both training and test accuracies were similar. The most influential features were TaxiOut and ArrTime, both of which are likely indicators of delay and are available before the flight departs. The model's accuracy was below our expectations, and further optimizations and alternative models such as XGBoost were planned to improve performance.

#### Preliminary Hyperparameter tuning
Despite not being able to fully optimize our model, we did take some preliminary steps by tuning parameters. The results of this process are given by the images below:

**Decreasing Sample Size**

![SampleSize](./images/train_and_test_accuracies_as_SampleSize_decreases.jpg)

To check the stability of our model, we measured the effect on training and testing accuracy as the sample size was significantly decreased. As can be seen from the plot above, decreasing the sample size did not result in any meaningful changes to accuracy, suggesting that our model is relatively stable. This further allows us to use a much smaller subset of the data for hyper parameter tuning, before eventually scaling back to size. For tuning, we use 0.1% of the data. At this split, accuracy remains the same, but memory usage is greatly improved.

**Increasing `numTrees`**

![NumTrees](./images/train_and_test_accuracies_as_NumTrees_increases.jpg)

The first parameter tuned is NumTrees and the accuracy of the model is measured with 10 trees, 30 trees, and 40 trees, while maintaining MaxDepth at 10. From the plot above, there is no discernable change in accuracy as this parameter is increased.

**Increasing `maxDepth`**

![MaxDepth](./images/train_and_test_accuracies_as_MaxDepth_increases.jpg)

Next, we tune MaxDepth using 10, 20, and 30 for this parameter while keeping NumTrees at a constant 10. Here, we observe overfitting as the training accuracy greatly improves while validation accuracy slightly diminishes.

**Best Model Performance**

![BestModel](./images/best_model_performance.jpg)

After tuning the parameters, the best model was found to have numTrees equal to 40 and a max depth of 20. Compared with our originally trained model, these parameters increased our test set accuracy by around 10%, resulting in a new accuracy of 52%. While this is a significant increase, and remains well above a model randomly guessing (25%), it still leaves a lot of room for improvement. Given more time and available resources, we would have continued to tune these parameters, feature engineer, and compare different machine learning methods.

## Discussion
To enhance our predictive capability, we began with extensive preprocessing and feature engineering. In the data-exploration phase, we removed redundant and duplicate fields to streamline the dataset and reduce noise. Because real-time factors like weather or mechanical issues weren’t directly captured, we crafted new, more informative predictors—combining Origin and Destination into a single Route feature, and computing an AirTime-to-Distance ratio to better reflect flight characteristics. We hope these derived variables sharpened the Random Forest’s ability to spot delay patterns by supplying clearer signals.

We acknowledge that some dynamic variables were unavailable or may have been inadvertently dropped during cleaning. Our focus here is strictly on departure delays—knowing pilots often recover lost time in flight when conditions permit, so departure and arrival delays don’t always correlate. Predicting arrival delays would demand a separate study and additional data outside this project’s scope.

This project aimed to utilize two distinct methods—Random Forest classification and K-Means clustering—to identify factors associated with flight delays. While both methods aimed to uncover relationships between features and flight delay durations, they operate on fundamentally different principles, making direct comparison both informative and limited.

Random Forest, a supervised learning algorithm, ranks features by their predictive power—specifically, how much each feature contributes to reducing impurity in classifying delay categories. This results in a quantitative feature importance score that highlights variables most useful for prediction, such as origin, distance, or taxi-out time.

In contrast, K-Means clustering is an unsupervised method that groups flights based on feature similarity, without reference to the actual delay categories. After clustering, we can examine the distribution of delay categories within each cluster and analyzed the underlying feature distributions. This approach identifies which feature values (not just features) are common in clusters enriched for each delay category. For example, a cluster characterized by longer delays might also have high values for taxi-out time and low values for flight distance.

Although both methods can highlight relevant features, they do so through different approaches:
- Random Forest emphasizes which features help a model distinguish between delay categories.
- K-Means emphasizes which feature values tend to group together in clusters associated with specific delay outcomes.

Despite these differences, the two methods offer complementary insights, features that are ranked highly by both methods (e.g., high Random Forest importance and distinct cluster patterns) provide strong evidence of their relevance to flight delays. Discrepancies between the two can guide further analysis—e.g., a feature important in clustering but not in RF may suggest a variable underrepresented in the model.

## Conclusion

This project offered valuable insight into the complex and multifaceted nature of flight delays across the U.S. domestic air travel system. By leveraging a large-scale dataset and applying both supervised and unsupervised learning techniques—Random Forest classification and K-Means clustering (planned)—we aimed to identify which factors most influence flight delay durations.

Through extensive preprocessing and feature engineering, we distilled a highly dimensional dataset into a manageable, informative form. Our Random Forest model, after preliminary hyperparameter tuning, achieved an improved accuracy of approximately 52%, up from an initial 42%. While modest, this result suggests that meaningful predictive signals exist in the data and that our feature selection strategies enhanced model interpretability and utility.

However, the project faced limitations due to resource constraints, particularly in implementing the unsupervised clustering pipeline. Completing this aspect would have allowed a deeper understanding of how delay patterns naturally group and how certain feature values correlate with those clusters—potentially uncovering delay causes that supervised models overlook.

If given more time and computing power, future iterations of this project could include:

- **Expanded hyperparameter tuning** and model selection using larger subsets of the data and alternate algorithms like XGBoost or Gradient Boosted Trees for improved performance.

- **Full execution of the clustering workflow**, including feature distribution analysis and ANOVA-based feature ranking, to compare with supervised insights.

- **Integration of external data sources**, such as real-time or historical weather data, airport congestion levels, or FAA airspace restrictions, to enhance predictive power and explainability.

- **Exploration of arrival delay prediction**, recognizing that on-time departures do not always equate to on-time arrivals, and understanding the interplay between the two delay types could yield more comprehensive models.

Ultimately, while our current findings already offer actionable insights—such as the critical importance of features like TaxiOut and ArrTime—this work serves as a foundational step. With continued development, the framework and learnings from this project can contribute meaningfully to efforts aimed at minimizing delays, optimizing flight operations, and improving the overall travel experience for millions of passengers.

## Collaboration:

1. **Ahyo:** *Coding/Management:* I set up initial communication through discord and email and helped organize meetings throughout the project. After selecting our dataset, I created the foundational notebook for us to use for the remainder. I aided in exploring the dataset through pyspark. Specifically, I developed several stacked barcharts showing flight departure delays grouped by origin airport, year, and month categorized as early, on time, or late. To preprocess the data, I, along with the rest of the team, wrote code to clean, feature engineer, and transform the dataset, preparing it for modeling. Finally, I was responsible for writing and organizing much of the README.md file that outlines this project.
2. **Hailey:** *Coder/Writer/Machine Learning lead:* I led the development of the machine learning strategy to assess factors contributing to flight delays, including proposing the use of Random Forest for feature selection and unsupervised clustering for distribution analysis, as I was most familiar with machine learning applications in the group. I designed the data preprocessing workflow, ensuring the dataset was cleaned, structured, and optimized for machine learning implementation using PySpark in which everyone contributed to coding a separate piece of the preprocessing steps. I also contributed to understanding data distributions, skews, and general statistics of the dataset features to set up the preprocessing workflow.
3. **Mihir:** *Project Manager/Lead/Dev:* I contributed by creating document for setting up the local env for everyone to use. Set up github repo and managed the processes to do with SDLC. I contributed to helping set up the data and extracting it to being able to use it by everyone, and contributed to data engineering process for coding. My main focus was to make sure everyone was on same page about the project, and had effective communication and strategy for the project. This work easy to manage since everyone in the project was equally helpful and communicated with each other as needed. I maintained the Git repo and and Pull request members did to make sure no merge conflcits deleted work by others since I was the only member in group experienced in Git.
4. **Nam:** *Coder/Machine Learning Specialist/Feature Engineer*: I explored, cleaned, and pioneered a pathforward for our dataset through milestone 3 and 4. I trained and validated different Random Forest models for milestone 4. Before the ML portion of milestone-4 being cancelled, I started validating smaller subsets of the dataset for easy handling and training. I further worked on hyper-parameter tuning with numTrees and maxDepths improving model's performance by 10%. I intended to explore a new direction for our project by training a Boosted Gradient Tree since the Random Forest is much better than random guessing (0.25 accuracy). I wrote up report and conclusion for my parts in each milestone. For milestone 3, I contributed in the group writeup and README.md.
5. **Rita:** *Coder:* I aided in finding and selecting the dataset and exploring some of the dataset through PySpark. In milestone 2, I developed bar charts exploring the top 10 most delayed routes and the top 10 most delayed airlines. I also developed a scatterplot to explore if delayed flights and distance had any sort of correlation. In milestone 3, I assisted in selecting redundant columns in our dataset and removing them to make preprocessing run more smoothly and ensure that the model was not affected on irrelevant or highly correlated data.

