# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Step 2: Create dataset
data = {
    'experience': [1, 3, 5, 0, 4, 2, 6, 3],
    'test_score': [80, 70, 90, 75, 85, 72, 92, 65],
    'interview_score': [85, 65, 88, 60, 80, 70, 90, 75],
    'salary': [40000, 45000, 60000, 35000, 58000, 42000, 68000, 47000]
}

df = pd.DataFrame(data)

# Step 3: Prepare features and label
X = df[['experience', 'test_score', 'interview_score']]
y = df['salary']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict on test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 8: Predict on new input
new_sample = [[5, 85, 82]]  # Example input
predicted_salary = model.predict(new_sample)

# Step 9: Save model
joblib.dump(model, 'salary_model.pkl')

# Step 10: Print results
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
print("Predicted Salary (for [5, 85, 82]):", round(predicted_salary[0], 2))
