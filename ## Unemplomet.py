##  Unemplomet Python 


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

##load data
df = pd.read_csv("C:/Users/Shabnam Kabir/Downloads/Unemployment_Rate_upto_11_2020.csv")
print(df.head())
print(df.info())


# Renaming columns for ease of use
df.rename(columns={
    'Region': 'State',
    ' Date': 'Date',
    'Frequency ': 'Frequency',
    'Estimated Unemployment Rate (%)': 'Unemployment Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour Participation Rate'
}, inplace=True)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

##unemploment state analysis
# plt.figure(figsize=(16,6))
# sns.lineplot(data=df, x='Date', y='Unemployment Rate', hue='State', legend=False)
# plt.title("Unemployment Rate Over Time (All States)")
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()
# plt.show()



##state wise average unemployment
# state_avg = df.groupby('State')['Unemployment Rate'].mean().sort_values()

# plt.figure(figsize=(12,6))
# sns.barplot(x=state_avg.values, y=state_avg.index, palette="viridis")
# plt.title("Average Unemployment Rate by State")
# plt.xlabel("Unemployment Rate (%)")
# plt.tight_layout()
# plt.show()

# monthly unemployment rate
# monthly_avg = df.groupby(df['Date'].dt.to_period("M"))['Unemployment Rate'].mean()

# plt.figure(figsize=(12,5))
# monthly_avg.plot(marker='o')
# plt.title("Monthly Average Unemployment Rate in India")
# plt.xlabel("Month")
# plt.ylabel("Unemployment Rate (%)")
# plt.grid(True)
# plt.show()

##correlation analsis
# plt.figure(figsize=(6,4))
# sns.heatmap(df[['Unemployment Rate', 'Employed', 'Labour Participation Rate']].corr(), annot=True, cmap='coolwarm')
# plt.title("Correlation Heatmap")
# plt.show()

##predictive modeling
# X = df[['Employed', 'Labour Participation Rate']]
# y = df['Unemployment Rate']

# Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
# model = LinearRegression()
# model.fit(X_train, y_train)

# Prediction
# y_pred = model.predict(X_test)

# Evaluation
# print("R² Score:", r2_score(y_test, y_pred))
# print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

##Bivaraite analysis
## scatter plot
# sns.scatterplot(x='Employed', y='Unemployment Rate', data=df)
# plt.title("Unemployment Rate vs Employed")
# plt.show()

##regression plot
# sns.lmplot(x='Labour Participation Rate', y='Unemployment Rate', data=df)
# plt.title("Unemployment Rate vs Labour Participation Rate")
# plt.show()

# unemploment by state
# plt.figure(figsize=(12,6))
# sns.boxplot(x='State', y='Unemployment Rate', data=df)
# plt.xticks(rotation=90)
# plt.title("Unemployment Rate by State")
# plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['Unemployment Rate'], kde=True, bins=20)
plt.title('Distribution of Unemployment Rate')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.show()

