import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'study_hours': [2, 4, 6, 8, 10],
    'sleep_hours': [5, 6, 7, 8, 9],
    'attendance': [60, 70, 80, 90, 100],
    'marks': [50, 60, 70, 85, 95]
}

df = pd.DataFrame(data)

# Features & Target
X = df[['study_hours', 'sleep_hours', 'attendance']]
y = df['marks']

# Model
model = LinearRegression()
model.fit(X, y)

# User Input
study = float(input("Enter study hours: "))
sleep = float(input("Enter sleep hours: "))
att = float(input("Enter attendance (%): "))

# Prediction
pred = model.predict([[study, sleep, att]])
print("📊 Predicted Marks:", round(pred[0], 2))

# Graph (Study Hours vs Marks)
plt.scatter(df['study_hours'], df['marks'])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()