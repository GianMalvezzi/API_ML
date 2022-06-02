import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv("iris.csv")

#Feature and target selection

X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y = df["Class"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling using StandartScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Creating the model
classifier = GradientBoostingClassifier()

# Fit the model 
classifier.fit(X_train, y_train)

# Saving the model
pickle.dump(classifier, open("model.pkl", "wb"))