import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Example dataset
data = pd.read_csv("../data/data.csv", delimiter=';')
X = data.drop("Target", axis=1)  # Assuming "target" is the column with labels
y = data["Target"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize SelectKBest with the ANOVA F-value scoring function and k=10
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)  # Use the same features selected from training data

# Fit a model using the selected features
model = RandomForestClassifier()
model.fit(X_train_selected, y_train)
score = model.score(X_test_selected, y_test)

print("Model accuracy with selected features:", score)
print("Selected feature indices:", selector.get_support(indices=True))  # Print selected feature indices
print("Selected feature names:", X.columns[selector.get_support()])  # Print selected feature names

# plot the results of the feature selection in a bar chart with importance scores
import matplotlib.pyplot as plt

# Get the scores and feature names
feature_scores = selector.scores_
feature_names = X.columns

# Combine scores and feature names, then sort them
feature_scores_names = list(zip(feature_names, feature_scores))
feature_scores_names.sort(key=lambda x: x[1], reverse=True)

# Plot the results with the 10 most important features marked in a green color, and the rest in a blue color
plt.figure(figsize=(12, 10))
plt.barh(*zip(*feature_scores_names))

plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('SelectKBest ranking', fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.savefig('../plots/selectkbest-ranking.png', dpi=300, bbox_inches='tight')
plt.show()