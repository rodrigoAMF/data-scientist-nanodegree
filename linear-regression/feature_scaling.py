# Add import statements
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Load the data
train_data = pd.read_csv('feature_scaling_data.csv', header=None)
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

# Create the standardization scaling object.
scaler = StandardScaler()

# Fit the standardization parameters and scale the data.
X_scaled = scaler.fit_transform(X)

# Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# Fit the model.
lasso_reg(X_scaled, y)

# Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)