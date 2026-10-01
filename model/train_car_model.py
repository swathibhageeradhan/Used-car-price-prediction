import pandas as pd
# Load dataset
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv(r"C:\Users\vivek\Downloads\CAR DETAILS FROM CAR DEKHO.csv")  

# Drop 'name' (not useful for regression directly)
df.drop('name', axis=1, inplace=True)

# Create new feature: car_age
current_year = 2025
df['car_age'] = current_year - df['year']
df.drop('year', axis=1, inplace=True)

# Define features and target
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Identify categorical columns
#Machine learning models (like RandomForestRegressor) can only work with numerical data. So:
#Categorical columns like "fuel", "seller_type", or "transmission" (which contain text) need to be converted into numbers.

categorical_cols = X.select_dtypes(include='object').columns.tolist()

# Preprocessing: OneHotEncoding for categorical features
#'cat'This is just a name you give to the transformation (not functionally important but useful for debugging or inspecting pipelines).
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    #Without handle_unknown='ignore', the encoder will crash when it encounters unknown categories.
    ],
    remainder='passthrough'  # Keep numeric columns(not removed or droped without encoded columns)
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline: preprocessing + model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])




pipeline.fit(X_train, y_train)

# Predict on test set
# y_pred = pipeline.predict(X_test)

# # Evaluate
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Model Evaluation:")
# print(f"MSE: {mse:.2f}")
# print(f"R²: {r2:.2f}")



# Save model

joblib.dump(pipeline,'car_price_model.pkl')

print("✅ Model and preprocessing pipeline saved successfully.")