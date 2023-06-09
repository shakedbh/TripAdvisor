import EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


df = pd.read_csv("Dataset_after_all.csv", encoding='cp1252')


def remove_commas_spaces():
    """
    This function removes any commas and spaces from each column in the dataset, using string manipulation methods.
    :return: None.
    """
    params = ['Rating', 'Review Count', 'Great for walkers', 'Restaurants', 'Attractions', 'Excellent', 'Very good',
              'Average', 'Poor', 'Terrible', 'Picture']
    for p in params:
        try:
            df[p] = df[p].str.replace(',', '').str.replace(' ', '')
        except Exception:
            pass


def get_features_importances():
    """
    This function provides you with the feature importances ranked in descending order.
    Higher importance values indicate more significant contributions to the prediction.
    :return: Print a table with the importance of each feature.
    """
    # Split the dataset into features (X) and target variable (y)
    X = df[['Review Count', 'Great for walkers', 'Restaurants', 'Attractions', 'Excellent', 'Very good', 'Average', 'Poor', 'Terrible', 'Picture']]
    y = df['Rating']

    # Initialize and fit a Random Forest model
    rf_model = RandomForestRegressor()
    rf_model.fit(X, y)

    # Get the feature importances
    feature_importances = rf_model.feature_importances_

    # Create a DataFrame of feature importances
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    # Sort the DataFrame by importance values
    sorted_feature_importances = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(sorted_feature_importances)


# Linear Regression

def rating_ReviewPicture_LinearRegression():  # rating against review count and picture - Linear Regression
    """
    The code aims to perform a linear regression analysis to predict hotel ratings based on the features 'Review Count' and 'Picture'.
    It trains a linear regression model, evaluates its performance using MSE and R-squared scores, and visualizes the predicted ratings against the actual ratings.
    :return: Print Mean Squared Error, R-squared, Coefficients, Intercept. Show the scatter plot of Actual vs. Predicted Ratings.
    """
    X = df[['Review Count', 'Picture']]
    y = df['Rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    coefficients = model.coef_
    intercept = model.intercept_

    print("Coefficients:", coefficients)
    print("Intercept:", intercept)

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Linear Regression - Actual vs. Predicted Ratings")
    plt.show()


def rating_LinearRegression():  # rating against all params - Linear Regression
    """
    The code aims to perform a linear regression analysis to predict hotel ratings based on the all numeric features.
    It trains a linear regression model, evaluates its performance using MSE and R-squared scores, and visualizes the predicted ratings against the actual ratings.
    :return: Print Mean Squared Error, R-squared, Coefficients, Intercept. Show the scatter plot of Actual vs. Predicted Ratings.
    """
    X = df.drop(['Rating', 'Name', 'Link'], axis=1)
    y = df['Rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Linear Regression:")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    coefficients = model.coef_
    intercept = model.intercept_

    print("Coefficients:", coefficients)
    print("Intercept:", intercept)

    sse = np.sum((y_test - y_pred) ** 2)
    print("Sum of Squared Errors (SSE):", sse)

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Linear Regression - Actual vs. Predicted Ratings")
    plt.show()


def rating_location_LinearRegression():  # rating against the params of location: Great for walkers, Restaurants, Attractions - Linear Regression
    """
    The code aims to perform a linear regression analysis to predict hotel ratings based on the location's features.
    It trains a linear regression model, evaluates its performance using MSE and R-squared scores, and visualizes the predicted ratings against the actual ratings.
    :return: Print Mean Squared Error, R-squared, Coefficients, Intercept. Show the scatter plot of Actual vs. Predicted Ratings.
    """
    X = df[['Great for walkers', 'Restaurants', 'Attractions']]
    y = df['Rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    coefficients = model.coef_
    intercept = model.intercept_

    print("Coefficients:", coefficients)
    print("Intercept:", intercept)

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Linear Regression - Actual vs. Predicted Ratings")
    plt.show()


def rating_excellent_LinearRegression():  # rating against excellent - Linear Regression
    """
    The code aims to perform a linear regression analysis to predict hotel ratings based on the feature 'Excellent'.
    It trains a linear regression model, evaluates its performance using MSE and R-squared scores, and visualizes the predicted ratings against the actual ratings.
    :return: Print Mean Squared Error, R-squared, Coefficients, Intercept. Show the scatter plot of Actual vs. Predicted Ratings.
    """
    X = df['Excellent'].values.reshape(-1, 1)
    y = df['Rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    coefficients = model.coef_
    intercept = model.intercept_

    print("Coefficients:", coefficients)
    print("Intercept:", intercept)

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Linear Regression - Actual vs. Predicted Ratings")
    plt.show()


# Polynomial Regression

def rating_PolynomialRegression():  # rating against all params - Polynomial Regression
    """
    This code will plot the predicted ratings on the x-axis and the residuals (difference between actual and predicted ratings) on the y-axis.
    The red dashed line represents the zero residual line.
    :return: Print Mean Squared Error, R-squared. Show the scatter plot of Polynomial Regression - Residual Plot.
    """
    X = df.drop(['Rating', 'Name', 'Link'], axis=1)
    y = df['Rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    degree = 2  # Set the degree of the polynomial
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Polynomial Regression:")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    sse = np.sum((y_test - y_pred) ** 2)
    print("Sum of Squared Errors (SSE):", sse)

    residuals = y_test - y_pred

    plt.scatter(y_pred, residuals, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Residuals')
    plt.title('Polynomial Regression - Residual Plot')
    plt.show()


# Decision Tree Regression

def rating_Decision_Tree_Regression():  # rating against all params - Decision Tree Regression
    """
    This function performs a Decision Tree Regression analysis to predict hotel ratings based on the all numeric features.
    It trains a linear regression model, evaluates its performance using MSE and R-squared scores, and visualizes the predicted ratings against the actual ratings.
    :return: Print Mean Squared Error, R-squared, Coefficients, Intercept. Show the scatter plot of Actual vs. Predicted Ratings.
    """
    X = df.drop(['Rating', 'Name', 'Link'], axis=1)
    y = df['Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Decision Tree Regression:")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    sse = np.sum((y_test - y_pred) ** 2)
    print("Sum of Squared Errors (SSE):", sse)
    # Plotting the predicted values against the actual values
    plt.scatter(y_test, y_pred, color='blue')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Decision Tree Regression - Actual vs. Predicted Values')
    plt.show()


# Random Forest Regression

def rating_Random_Forest_Regression():  # rating against all params - Random Forest Regression
    """
    This function performs a Random Forest Regression analysis to predict hotel ratings based on the all numeric features.
    It trains a linear regression model, evaluates its performance using MSE and R-squared scores, and visualizes the predicted ratings against the actual ratings.
    :return: Print Mean Squared Error, R-squared, Coefficients, Intercept. Show the scatter plot of Actual vs. Predicted Ratings.
    """
    X = df.drop(['Rating', 'Name', 'Link'], axis=1)
    y = df['Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Random Forest Regression:")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    sse = np.sum((y_test - y_pred) ** 2)
    print("Sum of Squared Errors (SSE):", sse)
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Random Forest Regression - Actual vs. Predicted")
    plt.show()


remove_commas_spaces()
EDA.change_to_numeric()

# linear Regression
rating_LinearRegression()
rating_ReviewPicture_LinearRegression()
rating_location_LinearRegression()
rating_excellent_LinearRegression()

# Polynomial Regression
rating_PolynomialRegression()

# Decision Tree Regression
rating_Decision_Tree_Regression()

# Random Forest Regression
rating_Random_Forest_Regression()
