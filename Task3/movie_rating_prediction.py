import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filename):
    return pd.read_csv(filename, encoding="latin1")

def preprocess_data(data):
    print("Sample Data:\n", data.head())
    print("\nAvailable columns in dataset:\n", data.columns.tolist())

    possible_rating_cols = ['IMDB Rating', 'IMDb Rating', 'Rating', 'IMDB_Score']
    rating_col = None
    for col in possible_rating_cols:
        if col in data.columns:
            rating_col = col
            break

    if rating_col is None:
        raise ValueError("❌ No rating column found. Please check dataset column names.")

    data = data.dropna(subset=[rating_col]).copy()

    available_cols = data.columns
    feature_cols = [col for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'] if col in available_cols]

    if not feature_cols:
        raise ValueError("❌ No valid feature columns (Genre/Director/Actors) found in dataset.")

    features = data[feature_cols]
    target = data[rating_col]

    features = pd.get_dummies(features, drop_first=True)

    return features, target


    return features, target

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")
    print("\nThe trained model can estimate movie ratings with good accuracy based on features like Genre, Director, and Actors.")

    return model

if __name__ == "__main__":
    data = load_data("IMDb Movies India.csv")
    X, y = preprocess_data(data)
    model = train_model(X, y)
