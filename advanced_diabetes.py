
# Data wrangling
import pandas as pd
import numpy as np

# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from lazypredict.Supervised import LazyClassifier

data = pd.read_csv("diabetes.csv", delimiter=",")
# Sample data
print(data.head())
# List of all columns
print(data.info())
# Correlation between attributes and target
plt.figure(figsize=(8, 8))
sns.histplot(data['Outcome'])
plt.title('Distribution Outcome')
plt.savefig("diabetes_outcome_dist.png")
plt.clf()
correlation = data.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm')
plt.savefig("diabetes_correlation.png")
plt.clf()

# Set features and target
x = data.drop("Outcome", axis=1)
y = data["Outcome"]
# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Normalize data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# Train 1 model
clf = SVC()
clf.fit(x_train, y_train)
# Run prediction on test set
y_predict = clf.predict(x_test)
# Metrics report
print(classification_report(y_test, y_predict))

cm = np.array(confusion_matrix(y_test, y_predict, labels=[0, 1]))
confusion = pd.DataFrame(cm, index=["Diabetic", "Not Diabetic"], columns=["Predicted Diabetes", "Predicted Healthy"])
sns.heatmap(confusion, annot=True, fmt="g")
plt.savefig("diabetes_conf_matrix.png")


# USE GridSearchCV (FOR SMALL NUMBER OF COMBINATIONS)

param_grid = {'n_estimators': [100, 200],
              'max_features': ['auto', 'sqrt'],
              'max_depth': [5, 10],
              'min_samples_split': [10, 50],
              'min_samples_leaf': [2, 5]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=10)
grid_search.fit(x_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
y_predict = grid_search.predict(x_test)
print("GridSearchCV's report:")
print(classification_report(y_test, y_predict))



# USE RandomizedSearchCV (FOR LARGE NUBMER OF COMBINATIONS)
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [5, 10, 20, 30]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 50, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Random search of parameters, using 5-fold cross validation, search across 100 different combinations

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=random_grid,
                        scoring='neg_mean_squared_error', n_iter=10, cv=5,
                        verbose=1, random_state=42, n_jobs=1)
random_search.fit(x_train, y_train)
print("Best parameters: {}".format(random_search.best_params_))
y_predict = random_search.predict(x_test)
print("RandomizedSearchCV's report:")
print(classification_report(y_test, y_predict))


# SEARCH FOR ALL REGRESSORS
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
