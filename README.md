# Stock-Price-Prediction-using-Linear-Regression
This project uses Machine Learning to predict the next-day closing stock price of Apple Inc. using historical stock market data from Yahoo Finance.
The model is built using Linear Regression from Scikit-learn and trained on Apple stock data downloaded using the yfinance library.

Features Used

The model uses the following stock market features:

High Price
Low Price
Open Price
Trading Volume

Target Variable:

Next Day Closing Price
Technologies Used
Python
Pandas
NumPy
yFinance
Scikit-learn
Matplotlib
Dataset Source

Stock data is collected from Yahoo Finance using:

yf.download("AAPL")

Library:

yfinance Documentation

Project Workflow
1. Import Libraries
import yfinance as yf
import numpy as np
import pandas as pd
2. Download Stock Data
df = yf.download("AAPL", start="2025-01-01", end="2026-04-28")
3. Data Preprocessing
Check Missing Values
df.isnull().sum()
Create Target Variable
df['Target'] = df[('Close','AAPL')].shift(-1)

This shifts the closing price by one day to predict the next day's price.
Remove Null Values
df.dropna(inplace=True)
4. Feature Selection
features = [
    ('High', 'AAPL'),
    ('Low', 'AAPL'),
    ('Open', 'AAPL'),
    ('Volume', 'AAPL')
]
5. Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    shuffle=False
)

shuffle=False is used because stock market data is time-series data.

6. Model Training
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)
7. Prediction
y_pred = regression.predict(X_test)
8. Visualization
import matplotlib.pyplot as plt

plt.plot(y_test.values, label='Actual Close Price')
plt.plot(y_pred, label='Predicted Close Price')

plt.title("Actual vs Predicted Closing Prices")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
9. Model Evaluation
Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)
R² Score
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
Output Example
MAE :  2.31
R2_score :  0.96
Train MSE: 5.12
Test MSE: 7.45
Installation
Clone the repository:
git clone <your-github-repo-link>
Install dependencies:
pip install yfinance numpy pandas matplotlib scikit-learn

Run the project:
python app.py
Project Structure
stock-price-prediction/
│
├── app.py
├── README.md
└── requirements.txt
Future Improvements
Add LSTM Deep Learning model
Predict multiple future days
Add technical indicators
Deploy using Flask or Streamlit
Real-time stock prediction dashboard
