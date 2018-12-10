# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('All violent crime set 2.csv')
#df = pd.DataFrame(df)
#del df['Unique County']
#del df['Population']
#del df['Pacific']
#del df['Murder per 100k']
#del df['Rape per 100k']
#del df['Robbery per 100k']
#del df['Aggravated assault per 100k']
#del df['Burglary per 100k']
#del df['All violent crime']
X = df.iloc[:, :-1].values
y = df.iloc[:, 5].values

# Encoding categorical data (Independent variable(s))
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()"""

# Avoiding the Dummy Variable Trap (Done automatically in Python - but do it anyway)
    # Try with and without it
"""X = X[:, 1:]"""

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature scaling - scales variables so they make sense in euclidean terms
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X = sc_X.transform(X)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Predicting the Test set results
"""y_pred = regressor.predict(X_test)"""

# Building the optimal model using Backward Elimination
    # Step 1: Significance level = 0.05
import statsmodels.formula.api as sm
        # Add a column of 1's as constants to the regression model (the intercept)
            # This is necessary for OLS
X = np.append(arr = np.ones((2999, 1)).astype(int), values = X, axis = 1)

    #Step 2 - Fit the full model with all possible predictors
        # First iteration
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
        # Second iteration
X_opt = X[:, [0, 1, 2, 4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
        # Third iteration

        # Fourth iteration

        # Fifth iteration

        # Sixth iteration

        # Seventh iteration

        # Eighth iteration

        # Nineth iteration
