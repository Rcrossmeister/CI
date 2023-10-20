from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

url = 'https://www.gairuo.com/file/data/dataset/iris.data'
iris_data = pd.read_csv(url)

# Drop the unnecessary column and split data into features and target
X = iris_data.drop(columns=['Unnamed: 0', 'species'])
y = iris_data['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy

# Initialize and train the random forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Predict on the test set
rf_y_pred = rf_clf.predict(X_test)

# Calculate the accuracy
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_accuracy
