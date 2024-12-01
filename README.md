# california-housing
An attempt to analyze the california housing price dataset

# Dataset
The dataset was obtained from https://www.kaggle.com/datasets/camnugent/california-housing-prices/data as per assignment's instruction

# Exploratory data analysis
Exploratory data analysis results are available in [EDA.ipynb](https://github.com/m10w/california-housing/blob/main/notebooks/EDA.ipynb)

Note: In the exploratory phase, no change should was committed to the dataset (e.g., handling missing values and outliers, feature engineering, etc.) to avoid data leakage.
After getting general insights about the data, data should be split into training and testing sets. Only the training set should be used for handling missing values, feature engineering, etc.

## Key findings are:
- There are 20640 rows in the dataset
- "total_bedrooms" contain 207 missing values which should be handled accordingly(to be replaced with the median value?)
- All columns are numerical (float64) except the "ocean_proximity" (object)
- "ocean_proximity" is a categorical feature (with 5 distinct categories) and should be taken care of (e.g., one-hot-encoded) and probably some categories could be merged
- Numerical features have different ranges and should be scaled
- median_house_value and housing_median_age column values have high peaks in their respective histograms' last bins which might suggest that these values are capped.
- Latitude and Longitude values have 2 peaks which might suggests that there are certain hubs with dense population (or entries).
- There is a strong correlation between the following columns (>0.85):
        - longitude & latitude
        - total_room & total_bedrooms & population & household
        - It's important to handle these. Probably we need to only keep the features with highest correlation with the target.
- median_income has the strongest correlation with median_house_value (target), without any preprocessing
- latitude has higher correlation to the median_house_value (negative), compared to the longitude. This means that as you go north, the prices drop slightly
- In the histogram plot, in the cell of latitude vs longitude, the map of california is visible!
- Houses in 'INLAND' have lower median age
- Houses in 'ISLAND' have higher median price (over blocks)
- Houses in 'INLAND' have lower median, but there are a lot of points beyond the IQR which suggests skewness




*For details please check the notebook*

# Modeling
Feature Engineering, Preprocessing, and modeling is done in [Modeling.ipynb](https://github.com/m10w/california-housing/blob/main/notebooks/modeling.ipynb)

- The data was split into train/test using stratified splitting based on the target variable (median_house_value) to make sure both training and test sets contain similar distributions of the target variable.

- The following features were created as part of the feature engineering based on the insights from EDA:
  - train_set['rooms_per_household'] = train_set['total_rooms'] / train_set['households'] # A measure of average house size (assumption: more rooms per household = larger house)
  - train_set['bedrooms_per_rooms'] = train_set['total_bedrooms'] / train_set['total_rooms'] # How many bedrooms relative to the total rooms
  - train_set['income_per_population'] = train_set['median_income'] / train_set['population'] # Average median income per population
  - train_set['rooms_per_population'] = train_set['total_rooms'] / train_set['population']
  - train_set['households_per_population'] = train_set['households'] / train_set['population'] # Average household size

- Preprocessing decisions:
  - Imputed the missing values (numerical features) by replacing every missing value (of the numerical features) with its respective column's median from the training set. In the current dataset only the total_bedrooms has missing values, but we can keep the median values of all numerical features, in case a missing value is encountered for any of the features after the deployment
  - Scaled numerical features in order to prevent the bias towards features with relatively larger scales, all numerical features will be scaled to mean=0 and std=1, by subtracting the mean and dividing by variance. This is crucial to be able to compare different ML algorithms as some are sensitive to the scale of features.
  - As most of the ML algorithms expect numerical features, it's always a good idea to create dummy variables for the categorical features. This was done using One-hot encoding method
  - All these preprocessing steps were done as part of a preprocessing pipeline which makes it more readable and easier to reproduce on the test set.

- Model exploration and hyper-parameter tuning:
  - The following models are selected to be explored, with the rationale mentioned below:
    - (Regularized) Linear regression: A simple model to be used as the baseline
    - Random Forests: handles non-linear relationships with low risk of overfitting
    - lightGBM (gradient boosting): performs very well on tabular data, handles non-linear relationships and is optimized for memory use (compared to XGBoost)
    - KNN: A simple non-parametric algorithm (no assumption is made about the data). However, it does not scale well on large datasets. As long as the end goal for the deployed model is to do single instance inference and not batch, it should be fine.

  - RandomizedSearchCV was used to explore the hyper-parameters instead of e.g., GridSearch. Because it lets us use explore a wider range of hyper parameters for each algorithm by trying random combniations of hyperparameters.
  - RMSE was used as the metric to track the performance as a easy to compute metric

  **Note**: In general, hyper-parameters grids for each of the algorithms are not exhaustive and are chosen to have a trade-off between computation time, model complexity and to prevent overfitting

  - Best models from each algorithm with their respective RMSE:
    - Lasso: 65925.41
    - Random Forest: 50236.95
    - **LightGBM: 44926.28**
    - k-NN: 57265.09

- Model Training Conclusions:
  - LightGBM has the lowest RMSE of 44926.28. Which means that on average, the model makes an error of 45k when predicting the median price for a block.
  - On its own and without knowing about the business requirements and acceptance criteria and the model's intended use, it's difficult to judge if this is an acceptable error or not.
  - Given the time and resource constraints in the context of this assignment, the error seems reasonable, but requires further improvements in future.
  - On the positive side, by looking at the CV folds errors of the best LightGBM model, the model seem stable (with std of test scores ~ 978)
  - Given that the range of the target variable (median_house_value) is 500000-15000 = 485000 (see EDA), then the 45000 error is roughly 10% of the range, which might be acceptable.
  - However looking at the lasso regression as a baseline, it improves the CV error on the training set by ~30% which is significant.

 
- The best model (lightGBM) was retrained on the full training set to use all the information in the training set. Note that we only update the model parameters and hyper-parameters are not touched (they are the same as found by the randomizedSearchCV)

- Then the performance was evaluated on the unseen test set. 
**Note**: According to best practices and to prevent data leakage, we should not update the model once it's evaluated on the test set.
Test set RMSE: 42636.85870692916

Conclusion: The error on the test set (~ 42000) seems to be aligned with the training CV error (~45000), which suggests that the model is robust and can generalize well on unseen data

Finally the best model and the preprocessing pipeline were saved as joblib files to be used in deployment.

*For details please check the notebook*

# Deployment
The API is built using FastAPI and predictions can be made using /predict/ endpoint by sending the required features in JSON format.
It is available as a native python service and is deployed via Render.

# Improvements to be done in future:
## DevOps/MLOps & coding best practices
- Dockerize the application
- Monitoring the performance using service like NannyML
- Apply best practices for the code versioning (creating Dev/Release/Main branches and merging individual feature branches via PR)
- Setting up continuous deployment (every merge into master, trigger a new build/deployment)
- Using ML lifecycle management tools such as MLflow to manage the ML artifacts and metrics and also deployments
- Automatic retraining pipeline
- Using Poetry for dependency management
- Structure the modeling part in python files (instead of jupyter notebooks)
- Store data in a database (and not in Git)
- Adding unit tests

## Preprocessing & Modeling 
- Handling missing values using more sophisticated methods (e.g., MICE)
- Feature importance analysis using SHAP values
- Generate confidence intervals for each prediction
- Perform feature engineering using custom sklearn estimators
- Augment the dataset with more data points

## Front-end
- Authentication and monitoring for the API
- Building a front-end 

--END--