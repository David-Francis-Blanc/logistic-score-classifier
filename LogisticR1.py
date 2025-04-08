# Predict pass/fail based on study hours using logistic regression

from sklearn.linear_model import LogisticRegression
import numpy as np

# X = hours studied, y = 1 if passed, 0 if failed
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])  # Passed if studied 4+ hours

# Train model
model = LogisticRegression()
model.fit(X, y)

# Predict for 3 and 5 hours
test_hours = np.array([[3], [5]])
predictions = model.predict(test_hours)
probs = model.predict_proba(test_hours)

# Output
for i, hours in enumerate(test_hours):
    print(f"Study hours: {hours[0]}")
    print(f"Predicted: {'Pass' if predictions[i]==1 else 'Fail'}")
    print(f"Confidence: {probs[i][1]*100:.2f}%\n")
