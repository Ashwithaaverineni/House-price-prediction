import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data/house_data.csv")

# Features and target
X = data[["Area", "Bedrooms", "Bathrooms"]]
y = data["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# User input
area = int(input("Enter area: "))
bedrooms = int(input("Enter bedrooms: "))
bathrooms = int(input("Enter bathrooms: "))

# Prediction
prediction = model.predict([[area, bedrooms, bathrooms]])

print("Predicted House Price:", int(prediction[0]))