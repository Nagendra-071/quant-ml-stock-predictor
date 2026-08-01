from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Import data processor
from preprocessing import scaled_bse_data


def train_pipeline(ticker="RELIANCE.NS"):
    # Load dataset
    df_features = scaled_bse_data(ticker)

    # Separate Features (X) and Target (y)
    X = df_features.drop(columns=["Target"])
    y = df_features["Target"]

    # Chronological Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Fit Scaler ONLY on X_train to prevent leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Print scaled training data cleanly
    print(f"\nScaled Data Preview (Shape: {X_train_scaled.shape}):")
    print(f"Latest Available Date in Dataset: {df_features.index[-1].strftime('%Y-%m-%d')}")
    
    print(pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index).tail(3))
    

    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Predictions & Results
    y_pred = rf_model.predict(X_test_scaled)
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return rf_model, scaler


if __name__ == "__main__":
    model, scaler = train_pipeline("RELIANCE.NS")