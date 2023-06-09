import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


df = pd.read_csv("Dataset_after_all.csv", encoding='cp1252')


def change_to_numeric():
    """
    This function changes all the params in the dataset to numeric values.
    :return: None
    """
    params = ['Rating', 'Review Count', 'Great for walkers', 'Restaurants', 'Attractions', 'Excellent', 'Very good',
              'Average', 'Poor', 'Terrible', 'Picture']
    try:
        for p in params:
            df[p] = pd.to_numeric(df[p], errors='coerce')
    except Exception:
        pass


def rating_reviews_PlotBar():
    """
    This function creates a bar plot showing the average value of 'Review Count' parameter for each category of 'Rating' parameter.
    :return: Picture of the bar plot.
    """
    # Select the parameters you want to analyze
    x_parameter = 'Rating'
    y_parameter = 'Review Count'

    # Group the data by the x_parameter and calculate the mean of the y_parameter
    grouped_data = df.groupby(x_parameter)[y_parameter].mean()

    # Plot the bar plot
    plt.bar(grouped_data.index, grouped_data.values)
    plt.xlabel(x_parameter)
    plt.ylabel(y_parameter)
    plt.title(f"Average {y_parameter} by {x_parameter}")
    plt.show()


def rating_restaurants_PlotBar():
    """
    This function creates a bar plot showing the average value of 'Restaurants' parameter for each category of 'Rating' parameter.
    :return: Picture of the bar plot.
    """
    # Select the parameters you want to analyze
    x_parameter = 'Rating'
    y_parameter = 'Restaurants'

    # Group the data by the x_parameter and calculate the mean of the y_parameter
    grouped_data = df.groupby(x_parameter)[y_parameter].mean()

    # Plot the bar plot
    plt.bar(grouped_data.index, grouped_data.values)
    plt.xlabel(x_parameter)
    plt.ylabel(y_parameter)
    plt.title(f"Average {y_parameter} by {x_parameter}")
    plt.show()


def attractions_reviews_PlotBar():
    """
    This function creates a bar plot showing the average value of 'Attractions' parameter for each category of 'Review Count' parameter.
    :return: Picture of the bar plot.
    """
    # Select the parameters you want to analyze
    x_parameter = 'Review Count'
    y_parameter = 'Attractions'

    # Group the data by the x_parameter and calculate the mean of the y_parameter
    grouped_data = df.groupby(x_parameter)[y_parameter].mean()

    # Plot the bar plot
    plt.bar(grouped_data.index, grouped_data.values)
    plt.xlabel(x_parameter)
    plt.ylabel(y_parameter)
    plt.title(f"Average {y_parameter} by {x_parameter}")
    plt.show()


def attractions_reviews_ScatterPlot():
    """
    This function creates a bar plot showing the average value of 'Attractions' parameter for each category of 'Review Count' parameter.
    :return: Picture of the scatter plot
    """
    # Define the parameters
    x_parameter = 'Attractions'
    y_parameter = 'Review Count'
    color_parameter = 'Rating'  # Categorical variable to represent color

    # Scatter plot with colored points
    plt.scatter(df[x_parameter], df[y_parameter], c=df[color_parameter], cmap='Reds')
    plt.xlabel(x_parameter)
    plt.ylabel(y_parameter)
    plt.title(f'Correlation between {x_parameter} and {y_parameter}')
    plt.colorbar(label=color_parameter)
    plt.show()


def rating_level_PlotBar():
    """
    This function creates a stacked bar plot where each bar represents a 'Rating' value, and the different colors within each bar represent the average values of the parameters.
    :return: Picture of the bar plot.
    """
    # Group the data by the rating and calculate the average values for the parameters
    grouped_data = df.groupby('Rating')[['Excellent', 'Very good', 'Average', 'Poor', 'Terrible']].mean()

    # Plot the stacked bar chart
    grouped_data.plot(kind='bar', stacked=True)

    # Set the labels and title
    plt.xlabel('Rating')
    plt.ylabel('Average Parameter Value')
    plt.title('Impact of Parameters on Rating')

    # Show the plot
    plt.show()


def rating_location_hotel_PlotBar():
    """
    This function creates a bar plot showing the average value of location parameters for each category of 'Rating' parameter.
    :return: Picture of the bar plot.
    """
    # Calculate the average rating for each parameter
    grouped_data = df.groupby('Rating')[['Great for walkers', 'Restaurants', 'Attractions']].mean()

    # Plot the stacked bar chart
    grouped_data.plot(kind='bar', stacked=True)

    # Set the labels and title
    plt.xlabel('Rating')
    plt.ylabel('Parameters of location the hotel')
    plt.title('Average Rating by Parameters')

    # Show the plot
    plt.show()


def picture_reviews_ScatterPlot():
    """
    This function creates a scatter plot showing the average value of 'Review Count' parameters for each category of 'Picture' parameter.
    :return: Picture of the scatter plot.
    """
    # Extract the required columns from the DataFrame
    x_parameter = 'Picture'
    y_parameter = 'Review Count'

    # Plot the scatter plot
    plt.scatter(df[x_parameter], df[y_parameter])

    # Set the labels and title
    plt.xlabel(x_parameter)
    plt.ylabel(y_parameter)
    plt.title('Review Count vs Picture')

    # Show the plot
    plt.show()


def picture_reviews_ScatterPlot_with_log():
    """
    This function is like the function: picture_reviews_ScatterPlot() that mentions above, but it has the plt.yscale('log'), plt.xscale('log') lines,
    the y-axis and x-axis will be displayed on a logarithmic scale, which can help in visualizing the data better when dealing with large numbers.
    :return: Picture of the scatter plot.
    """
    # Extract the required columns from the DataFrame
    x_parameter = 'Picture'
    y_parameter = 'Review Count'

    # Plot the scatter plot
    plt.scatter(df[x_parameter], df[y_parameter])

    # Set the labels and title
    plt.xlabel(x_parameter)
    plt.ylabel(y_parameter)
    plt.title('Review Count vs Picture')

    # Set log scale for the y-axis
    plt.yscale('log')
    plt.xscale('log')

    # Show the plot
    plt.show()


def rating_Excellent_Terrible_PlotBar():
    """
    This code will create a bar plot where the x-axis represents the Rating, and the y-axis represents the average of the parameters 'Excellent' and 'Terrible'.
    Each parameter will have a corresponding bar, and you can observe the difference in ratings between the two parameters.
    :return: Picture of the bar plot
    """
    # Extract the required columns from the DataFrame
    x_parameter = 'Rating'
    y_parameter = ['Excellent', 'Terrible']

    # Group the data by the x_parameter and calculate the mean rating
    grouped_data = df.groupby(x_parameter)[y_parameter].mean()

    # Plot the bar plot
    grouped_data.plot(kind='bar', stacked=True)

    # Set the labels and title
    plt.xlabel('Rating')
    plt.ylabel('Parameters')
    plt.title('Rating vs Parameters')

    # Show the plot
    plt.show()


def picture_level_ScatterBar():
    """
    In this example, the count_of_pictures list represents the 'Picture' parameter for each data point.
    The lists excellent, very_good, average, poor, and terrible represent the corresponding parameter values for each data point.
    The colors list assigns different colors to each parameter level.
    The scatter plot is created using the plt.scatter() function, where the c parameter is used to specify the color for each parameter level.
    :return: Picture of the scatter plot.
    """
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    # Create a scatter plot
    plt.scatter(df['Picture'], df['Excellent'], c=colors[0], label='Excellent')
    plt.scatter(df['Picture'], df['Very good'], c=colors[1], label='Very Good')
    plt.scatter(df['Picture'], df['Average'], c=colors[2], label='Average')
    plt.scatter(df['Picture'], df['Poor'], c=colors[3], label='Poor')
    plt.scatter(df['Picture'], df['Terrible'], c=colors[4], label='Terrible')

    # Set labels and title
    plt.xlabel('Count of Pictures')
    plt.ylabel('Parameter Value')
    plt.title('Effect of Picture Count on Parameters')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


def rating_picture_PlotBar():
    """
    This function creates a bar plot showing the average value of the y_parameter for each category of the x_parameter.
    :return: Picture of the bar plot.
    """
    # Select the parameters you want to analyze
    x_parameter = 'Rating'
    y_parameter = 'Picture'

    # Group the data by the x_parameter and calculate the mean of the y_parameter
    grouped_data = df.groupby(x_parameter)[y_parameter].mean()

    # Plot the bar plot
    plt.bar(grouped_data.index, grouped_data.values)
    plt.xlabel(x_parameter)
    plt.ylabel(y_parameter)
    plt.title(f"Average {y_parameter} by {x_parameter}")
    plt.show()


def rating_picture_ScatterPlot():
    """
    In this code, df[x_parameter] represents the values of the 'Picture' column, and df[y_parameter] represents the values of the 'Review Count' column.
    The scatter plot will show the relationship between these two variables. Each point on the plot represents a hotel.
    The scatter plot allows you to visually assess any potential relationship between the two variables.
    :return: Picture of the scatter plot.
    """
    # Extract the required columns from the DataFrame
    x_parameter = 'Rating'
    y_parameter = 'Picture'

    # Plot the scatter plot
    plt.scatter(df[x_parameter], df[y_parameter])

    # Set the labels and title
    plt.xlabel(x_parameter)
    plt.ylabel(y_parameter)
    plt.title('Review Count vs Picture')

    # Show the plot
    plt.show()


def heatmap():
    """
    This function creates a heatmap. This table shows the p-value between every two variables.
    :return: Picture of the p-value table.
    """
    # Heatmap: Correlation between price and numerical features
    numeric_vars = ['Rating', 'Review Count', 'Great for walkers', 'Restaurants', 'Attractions', 'Excellent', 'Very good',
                    'Average', 'Poor', 'Terrible', 'Picture']

    plt.figure(figsize=(11, 8))
    corr_matrix = df[numeric_vars].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap: P-VALUE between every two variables")
    plt.show()


change_to_numeric()

# bar plot
rating_reviews_PlotBar()
rating_restaurants_PlotBar()
rating_level_PlotBar()
rating_location_hotel_PlotBar()
rating_Excellent_Terrible_PlotBar()
rating_picture_PlotBar()
attractions_reviews_PlotBar()

# scatter plot
attractions_reviews_ScatterPlot()
picture_reviews_ScatterPlot()
picture_reviews_ScatterPlot_with_log()
picture_level_ScatterBar()
rating_picture_ScatterPlot()

# heatmap
heatmap()
