#!/usr/bin/env python
# coding: utf-8

# # Classification

# In[1]:


#Import data
import pandas as pd
kickstarter=pd.read_excel('/Users/yvette/Desktop/INSY 662/Kickstarter.xlsx')
kickstarter.head()


# In[2]:


##check the distribution counts of "state" variable
state_counts = kickstarter['state'].value_counts()
numerical_summary = kickstarter.describe()
state_counts,numerical_summary


# In[3]:


#check the state variable categories and only reserve rows that is "successful"and "failed"
filtered_data = kickstarter[kickstarter['state'].isin(['successful', 'failed'])]
filtered_data.head()


# In[4]:


# #check whether the value in pledged column under 'USD' in currency is equal to the values in use_pledged
# usd_currency = filtered_data[filtered_data['currency'] == 'USD']
# comparison_usd = usd_currency['pledged'] == usd_currency['usd_pledged']
# comparison_result = comparison_usd.all()
# comparison_result

# #validate the consistency between the division of "usd_pledged" amounts and "pleged" values and the conversion rates("static_usd_rate")
# # non_usd_projects = filtered_data[(filtered_data['currency'] != 'USD') & (filtered_data['pledged'] != 0)]non_usd_projects['calculated_usd_rate'] = non_usd_projects['usd_pledged'] / non_usd_projects['pledged']

# # Comparing the calculated_usd_rate with the static_usd_rate
# #comparison_non_usd = non_usd_projects[['static_usd_rate', 'calculated_usd_rate']]

# #tolerance=0.1
# #consistent_rates_non_usd = (comparison_non_usd['static_usd_rate'] - comparison_non_usd['calculated_usd_rate']).abs() < tolerance
# #consistency_result_non_usd = consistent_rates_non_usd.all()
# #comparison_result,consistency_result_non_usd


# In[5]:


#drop irrelevant columns
columns_to_drop = ['id', 'name','pledged','state_changed_at', 'backers_count', 'usd_pledged', 
                   'spotlight', 'state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr','launch_to_state_change_days','state_changed_at_weekday','state_changed_at','deadline']

filtered_data.drop(columns=columns_to_drop, axis=1, inplace=True)


# In[6]:


#check correlation
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
corr_matrix = filtered_data.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()


# In[7]:


#drop high correlated columns
columns_to_drop2 = ['deadline_yr', 'deadline_month', 'deadline_day',
                   'launched_at_yr', 'launched_at_month', 'launched_at_day',
                   'created_at_yr', 'created_at_month', 'created_at_day','blurb_len','name_len',
                   'deadline_hr','created_at_hr', 'launched_at_hr','currency']

filtered_data.drop(columns=columns_to_drop2, inplace=True)


# In[8]:


# Counting missing values for each column
missing_values = filtered_data.isnull().sum()
print(missing_values)


# In[9]:


# Drop rows with any missing values in the dataset
filtered_data.dropna(inplace=True)


# In[10]:


#dummify categorical columns

categorical_cols = ['country','category', 
                    'deadline_weekday', 'created_at_weekday', 
                    'launched_at_weekday']

filtered_data = pd.get_dummies(filtered_data, columns=categorical_cols)

filtered_data.drop(filtered_data.columns[filtered_data.columns.str.startswith('country_') & ~filtered_data.columns.isin(['country_US', 'country_CA'])], axis=1, inplace=True)




# In[11]:


# Convert 'goal' to USD
filtered_data['goal_usd'] = filtered_data['goal'] * filtered_data['static_usd_rate']
filtered_data.drop(columns=['goal'], inplace=True)


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, accuracy_score


# Separate the target variable and encode it
encoder = LabelEncoder()
y = encoder.fit_transform(filtered_data['state'])

# Drop the target variable from the features dataset
X = filtered_data.drop('state', axis=1)

# Separate numeric and categorical features
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = X.select_dtypes(include=['uint8', 'bool', 'object']).columns 
X_numeric = X[numeric_columns]
X_categorical = pd.get_dummies(X[categorical_columns]) 

# Scale numeric features
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Convert scaled numeric features back to DataFrame
X_numeric_scaled_df = pd.DataFrame(X_numeric_scaled, columns=numeric_columns, index=X.index)

# Concatenate scaled numeric features with categorical features
X_final = pd.concat([X_numeric_scaled_df, X_categorical], axis=1)

# Split the dataset 
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=50)


# In[13]:


#Try random forest
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest Classifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("Random Forest Classifier Metrics:")
print("  Accuracy:", accuracy_rf)
print("  Precision:", precision_rf)
print("  Recall:", recall_rf)
print("  F1 Score:", f1_rf)


# In[14]:


# Try Gradient Boosting model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

gradient_boosting_model = GradientBoostingClassifier()

# Train the model
gradient_boosting_model.fit(X_train, y_train)

# Predict on the test data
y_pred_gb = gradient_boosting_model.predict(X_test)

# Calculate metrics
accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)

# Print the results
print("Gradient Boosting Classifier Metrics:")
print("  Accuracy:", accuracy_gb)
print("  Precision:", precision_gb)
print("  Recall:", recall_gb)
print("  F1 Score:", f1_gb)


# In[15]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
##hyperparameter tuning
# Define the parameter grid
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2]
}

gb = GradientBoostingClassifier()

# Instantiate the grid search model
grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb, 
                              cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the grid search to the data
grid_search_gb.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters for Gradient Boosting:", grid_search_gb.best_params_)



# In[16]:


##retrain the model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


# Train the optimized Gradient Boosting Classifier
gb_optimal = GradientBoostingClassifier(n_estimators=200, learning_rate=0.2, max_depth=3, random_state=50)
gb_optimal.fit(X_train, y_train)


y_pred_optimal = gb_optimal.predict(X_test)

report_dict = classification_report(y_test, y_pred_optimal, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
required_rows = ['0', '1', 'accuracy', 'macro avg', 'weighted avg']
print(report_df.loc[required_rows])


# # cluster

# In[17]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
# Recreating the preprocessing steps 
# 1. Filter the data for "successful" or "failed" projects
filtered_data2 = kickstarter[kickstarter['state'].isin(['successful', 'failed'])]

# 2. Feature Engineering: Create 'goal_usd' column and drop the original 'goal' column
filtered_data2['goal_usd'] = filtered_data2['goal'] * filtered_data2['static_usd_rate']
filtered_data2.drop(columns=['goal'], inplace=True)

# 3. Handling Missing Values: Drop rows where 'category' is missing
filtered_data2.dropna(subset=['category'], inplace=True)

# 4. Feature Selection: Drop unnecessary and categorical columns
selected_features = filtered_data2.drop(columns=['state','static_usd_rate',
    'id', 'name', 'deadline', 'state_changed_at', 'created_at', 'launched_at','category',
    'currency', 'pledged','country','spotlight','disable_communication', 'staff_pick',
    'deadline_weekday', 'state_changed_at_weekday', 'created_at_weekday', 'launched_at_weekday',
    'deadline_month', 'deadline_day', 'deadline_yr', 'deadline_hr',
    'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr',
    'created_at_month', 'created_at_day', 'created_at_yr', 'created_at_hr',
    'launched_at_month', 'launched_at_day', 'launched_at_yr', 'launched_at_hr','name_len','blurb_len'])
import numpy as np

# 5. Normalizing the selected features
scaler = StandardScaler()
numeric_features = selected_features.select_dtypes(include=['float64', 'int64'])
normalized_features = scaler.fit_transform(numeric_features)

# Proceed with PCA analysis
pca = PCA()
pca.fit(normalized_features)
explained_variance = pca.explained_variance_ratio_.cumsum()

# Plotting the explained variance to decide on the number of components for PCA
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()


# In[18]:


variance_threshold = 0.95
n_components = next(x for x, cumulative_variance in enumerate(explained_variance) if cumulative_variance > variance_threshold) + 1
n_components


# In[19]:


# Applying PCA with 6 components
pca_7 = PCA(n_components=6)
pca_features = pca_7.fit_transform(normalized_features)

# Determining the optimal number of clusters using the Elbow Method
sum_of_squared_distances = []
K = range(3, 15)  # Range for potential number of clusters

for k in K:
    km = KMeans(n_clusters=k, random_state=50)
    km = km.fit(pca_features)
    sum_of_squared_distances.append(km.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[20]:


# Perform K-Means Clustering with 6 clusters
kmeans = KMeans(n_clusters=6, random_state=50)
kmeans.fit(pca_features)
clusters = kmeans.labels_
selected_features['cluster'] = clusters

cluster_analysis = selected_features.groupby('cluster').mean()

cluster_analysis


# In[21]:


from matplotlib import pyplot
selected_features['cluster'] = clusters
############# Examining Clusters#############
# Distribution of the clusters
pl = sns.countplot(x=selected_features['cluster'])
pl.set_title("Distribution Of The Clusters")
pyplot.show()

for i in selected_features.columns[:-1]:
  pl = sns.boxplot(x="cluster", y=i, data=selected_features)
  pl.set_title(i+" by clusters")
  pyplot.show()


# In[22]:


# Count the number of observations in each cluster
cluster_counts = selected_features['cluster'].value_counts()
print(cluster_counts)


# # Grading

# In[23]:


# # Grading

# In[17]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

#import 
grading_data = pd.read_excel('Kickstarter-Grading.xlsx')
grading_data = grading_data[grading_data['state'].isin(['successful', 'failed'])]

# Drop irrelevant columns
columns_to_drop = ['id', 'name', 'pledged', 'state_changed_at', 'backers_count', 'usd_pledged', 
                   'spotlight', 'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr', 
                   'state_changed_at_hr', 'launch_to_state_change_days', 'state_changed_at_weekday', 'state_changed_at','deadline']
grading_data.drop(columns=columns_to_drop, inplace=True)

# Drop highly correlated columns
columns_to_drop2 = ['deadline_yr', 'deadline_month', 'deadline_day',
                    'launched_at_yr', 'launched_at_month', 'launched_at_day',
                    'created_at_yr', 'created_at_month', 'created_at_day', 'blurb_len', 'name_len',
                    'deadline_hr', 'created_at_hr', 'launched_at_hr','currency']
grading_data.drop(columns=columns_to_drop2, inplace=True)

# Drop rows with missing values
grading_data.dropna(inplace=True)

# Dummify categorical columns
categorical_cols = ['country','category', 'deadline_weekday', 'created_at_weekday', 'launched_at_weekday']
grading_data = pd.get_dummies(grading_data, columns=categorical_cols)
#only focus on country and canada
grading_data.drop(grading_data.columns[grading_data.columns.str.startswith('country_') & ~grading_data.columns.isin(['country_US', 'country_CA'])], axis=1, inplace=True)
# Convert 'goal' to USD and remove the original 'goal' column

grading_data['goal_usd'] = grading_data['goal'] * grading_data['static_usd_rate']
grading_data.drop(columns=['goal'], inplace=True)

# Encode the target variable
encoder = LabelEncoder()
y_grading = encoder.fit_transform(grading_data['state'])
X_grading = grading_data.drop('state', axis=1)

# Scale numeric features
numeric_columns = X_grading.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = X_grading.select_dtypes(include=['uint8', 'bool', 'object']).columns 
X_categorical = pd.get_dummies(X_grading[categorical_columns])
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_grading[numeric_columns])
X_numeric_scaled_df = pd.DataFrame(X_numeric_scaled, columns=numeric_columns, index=X_grading.index)

# Concatenate scaled numeric features with categorical features
X_final_grading = pd.concat([X_numeric_scaled_df, X_categorical], axis=1)


y_pred_grading = gb_optimal.predict(X_final_grading)

# Calculate and print accuracy, precision, and recall
accuracy = accuracy_score(y_grading, y_pred_grading)
precision = precision_score(y_grading, y_pred_grading)
recall = recall_score(y_grading, y_pred_grading)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


# In[ ]:




