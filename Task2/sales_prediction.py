import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def load_data(filename):
    return pd.read_csv(filename)

def split_features_target(data):
    X = data.drop("Sales", axis=1)  
    y = data["Sales"]            
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"The trained model can predict future sales with an accuracy of {score * 100:.2f}%.")
    return y_pred

data = load_data("Sales Prediction.csv")
X_train, X_test, y_train, y_test = split_features_target(data)
model = train_model(X_train, y_train)
predictions = evaluate_model(model, X_test, y_test)
