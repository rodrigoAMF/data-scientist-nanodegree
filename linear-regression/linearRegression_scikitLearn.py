# Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')

x = bmi_life_data[['BMI']]
y = bmi_life_data[['Life expectancy']]

# Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(x, y)

# Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict([[21.07931]])
