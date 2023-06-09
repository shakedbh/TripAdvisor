# TripAdvisor Data Science Project

This project is a data science analysis of TripAdvisor data to predict hotel ratings. It utilizes various machine learning models and techniques to analyze the data and make predictions.
 
# Project Overview
The TripAdvisor Data Science Project is aimed at analyzing and predicting hotel ratings using machine learning techniques. The objective of this project is to develop a model that can accurately predict hotel ratings based on various features and attributes.

TripAdvisor is a popular travel website that provides information and reviews of hotels, restaurants, and attractions. With a vast amount of user-generated data available, it presents an opportunity to extract insights and create predictive models to understand what factors contribute to higher or lower hotel ratings.

The main problem addressed in this project is to build a model that can accurately predict hotel ratings based on a set of input features such as location, amenities, pricing, and user reviews. By analyzing these features and their impact on ratings, the model can help hotel owners and travelers make informed decisions.

The project utilizes a combination of data exploration, preprocessing, feature engineering, and machine learning algorithms. It involves cleaning the dataset, handling missing values, encoding categorical variables, and scaling numerical features. Various regression models such as Linear Regression, Polynomial Regression, Decision Tree Regression, and Random Forest Regression are applied to train and evaluate the predictive performance.

The project aims to provide insights into the factors influencing hotel ratings and deliver a reliable rating prediction model. By accurately predicting ratings, hotel owners can identify areas of improvement and prioritize efforts to enhance customer satisfaction. Travelers can also benefit from the model by using it as a guide to select hotels that align with their preferences.

Overall, this project combines data science techniques with TripAdvisor data to address the challenge of predicting hotel ratings, ultimately benefiting both hotel owners and travelers in making informed decisions.

# Data Description
The TripAdvisor Data Science Project utilizes a dataset containing information about hotels and their corresponding ratings. The dataset is obtained by scraping TripAdvisor website using web scraping techniques.

The dataset consists of the following columns:

Name: The name of the hotel.
Link: The link to the hotel's TripAdvisor page.
Rating: The rating of the hotel (target variable).
Review count: The number of reviews for the hotel.
Great for walkers: The number of places that close to hotel on foot.
Restaurants: The number of count of restaurants that close to hotel.
Attractions: The number of count of attractions that close to hotel.
Excellent: The number of excellent ratings for the hotel.
Very Good: The number of very good ratings for the hotel.
Average: The number of average ratings for the hotel.
Poor: The number of poor ratings for the hotel.
Terrible: The number of terrible ratings for the hotel.
Picture: The count of picture.

The dataset contains numerical features, allowing for various analyses and modeling approaches to predict hotel ratings based on different parameters.

# Results
After performing the data analysis and applying various regression models, the following are the key findings and results of the project:

Linear Regression:

Mean Squared Error (MSE): 1.18533
R-squared Score: 0.13175

Polynomial Regression:

Mean Squared Error (MSE): 2.03027
R-squared Score: -0.48716

Decision Tree Regression:

Mean Squared Error (MSE): 0.06678
R-squared Score: 0.95108

Random Forest Regression:

Mean Squared Error (MSE): 0.03313
R-squared Score: 0.97572

Based on these results, we can draw the following conclusions:
Among the models we evaluated, the Random Forest Regression model performed the best in terms of prediction accuracy and capturing the underlying patterns in the data.
The Random Forest Regression model can explain approximately 97.57% of the variance in the Rating variable.

It's important to note that the performance and results may vary depending on the specific dataset, preprocessing techniques, and model configurations. These findings provide an initial assessment of the models' performance and serve as a basis for further analysis and improvement.

