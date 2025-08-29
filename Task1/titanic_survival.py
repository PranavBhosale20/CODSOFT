import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(filename):
    return pd.read_csv(filename)

def clean_data(data):
    data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    
    data["Age"].fillna(data["Age"].median(), inplace=True)
    data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)
    
    return data

def encode_data(data):
    le = LabelEncoder()
    data["Sex"] = le.fit_transform(data["Sex"])
    data["Embarked"] = le.fit_transform(data["Embarked"])
    return data

def split_data(data):
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"The trained model can predict if a passenger on the Titanic survived or not with approximately {accuracy * 100:.2f}% accuracy.")


data = load_data("Titanic-Dataset.csv")
data = clean_data(data)
data = encode_data(data)
X_train, X_test, y_train, y_test = split_data(data)
model = train_model(X_train, y_train)

evaluate_model(model, X_test, y_test)
