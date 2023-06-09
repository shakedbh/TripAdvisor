import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep


links = []  # Array of the links to hotel pages.
data = []  # Array of the data of each hotel.

url_london = 'https://www.tripadvisor.in/Hotels-g186338-London_England-Hotels.html'  # The results of all hotels in london.

user_agent = ({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
    AppleWebKit/537.36 (KHTML, like Gecko) \
    Chrome/90.0.4430.212 Safari/537.36',
               'Accept-Language': 'en-US, en;q=0.5'})


def get_page_contents(url_web_page):
    """
    This function does a request get to the url and does a parsing to the HTML.
    :param url_web_page: The URL of the website.
    :return: The HTML of a website after a parsing.
    """
    page = requests.get(url_web_page, headers=user_agent)
    return BeautifulSoup(page.text, 'html.parser')


def get_num_pages(url_web_page) -> int:
    """
    This function checks how many pages there is in the url_web_page and returns it.
    :param url_web_page: The URL of the website.
    :return: The number of pages in the url_web_page
    """
    soup = get_page_contents(url_web_page)
    num_pages = soup.find(name="div", class_="unified ui_pagination standard_pagination ui_section listFooter").get("data-numpages")
    return (int)(num_pages) + 1


def scraping_url_of_all_hotels():
    """
    This function scrapes the url_london and takes out the parameters: names, links, ratings, reviews count,
    Great for walkers, Restaurants count, Attractions count, Excellent grade, Very good grade, Average grade,
    Poor grade, Terrible grade
    :return: None
    """
    num_pages = get_num_pages(url_london)  # Get the number of pages in url_london

    for page_num in range(1, num_pages):  # loop of all pages in url_london

        # the url of each page
        url = f"https://www.tripadvisor.in/Hotels-g186338-oa{30 * (page_num - 1)}-London_England-Hotels.html"
        sleep(15)  # sleep 15 seconds until the page is loaded

        # parsing HTML of each page with a get_page_contents(url) function.
        soup = get_page_contents(url)

        # get all the hotels in each page
        container = soup.findAll(name="div",
                                 class_="prw_rup prw_meta_hsx_responsive_listing ui_section listItem reducedWidth "
                                        "rounded")

        for j in range(len(container)):  # loop of all hotels in each page

            # names of hotels
            try:
                name = container[j].find('div', {'class': 'prw_rup prw_meta_hsx_listing_name listing-title'}).find(
                    name="a").getText().strip()
            except Exception:
                name = None

            # links of hotels
            try:
                link_element = container[j].find('div', {'class': 'prw_rup prw_meta_hsx_listing_name listing-title'}).find(
                    name="a").get("href")
                link = "https://tripadvisor.in" + link_element
            except Exception:
                link = None

            # each hotel:
            s = get_page_contents(link)  # paring each hotel page

            # count of locations for walkers
            try:
                great_for_walkers = s.find(name="span", class_="iVKnd fSVJN").getText()
            except Exception:
                great_for_walkers = None

            # count of restaurants
            try:
                restaurants = s.find(name="span", class_="iVKnd Bznmz").getText()
            except Exception:
                restaurants = None

            # count of attractions
            try:
                attractions = s.find(name="span", class_="iVKnd rYxbA").getText()
            except Exception:
                attractions = None

            # count of reviews:

            # excellent grade
            try:
                excellent = s.find(id="ReviewRatingFilter_5").find(name="span", class_="NLuQa").getText()
            except Exception:
                excellent = None

            # very good grade
            try:
                very_Good = s.find(id="ReviewRatingFilter_4").find(name="span", class_="NLuQa").getText()
            except Exception:
                very_Good = None

            # average grade
            try:
                average = s.find(id="ReviewRatingFilter_3").find(name="span", class_="NLuQa").getText()
            except Exception:
                average = None

            # poor grade
            try:
                poor = s.find(id="ReviewRatingFilter_2").find(name="span", class_="NLuQa").getText()
            except Exception:
                poor = None

            # terrible grade
            try:
                terrible = s.find(id="ReviewRatingFilter_1").find(name="span", class_="NLuQa").getText()
            except Exception:
                terrible = None

            # pictures
            try:
                picture = s.find(name="span", class_="is-hidden-mobile krMEp").getText()
            except Exception:
                picture = None

            # ratings of hotels
            try:
                rating_element = container[j].find('a', {'class': 'ui_bubble_rating'})
                rating = rating_element['alt']
            except Exception:
                rating = None

            # count of reviews of hotels
            try:
                review = container[j].find('a', {'class': 'review_count'}).text.strip()
            except Exception:
                review = None

            # save the data of each hotel in hotel_data value, and save it in data array
            hotel_data = {"Name": name, "Link": link, "Rating": rating, "Review Count": review, "Great for walkers": great_for_walkers, "Restaurants": restaurants, "Attractions": attractions,
                          "Excellent": excellent, "Very good": very_Good, "Average": average, "Poor": poor, "Terrible": terrible, "Picture": picture}
            data.append(hotel_data)


def create_full_dataset():
    """
    This function creates a data frame by panda. The columns are the parameters that we collate in a scraping_url_of_all_hotels function.
    :return: The data frame
    """
    df = pd.DataFrame(data, columns=["Name", "Link", "Rating", "Review Count", "Great for walkers", "Restaurants", "Attractions", "Excellent", "Very good", "Average", "Poor", "Terrible", "Picture"])
    df.to_csv("TripAdvisor_Dataset.csv", index=False)
    return df


def remove_missing_data():
    """
    This function copies the original dataset to a copy and removes missing values in the data frame that is created in a create_dataset_with_panda function.
    :return: The copy of dataset without missing values.
    """
    without_missing_dataset = create_full_dataset().copy()
    without_missing_dataset = without_missing_dataset.dropna()
    without_missing_dataset.to_csv("Without_Missing_Dataset.csv", index=False)
    return without_missing_dataset


def remove_duplicates():
    """
    This function copies the original dataset to a copy and removes duplicates rows.
    :return: The copy of dataset without duplicates rows.
    """
    without_duplicates_dataset = remove_missing_data().copy()
    without_duplicates_dataset = without_duplicates_dataset.drop_duplicates()
    without_duplicates_dataset.to_csv("Without_Duplicates_Dataset.csv", index=False)
    return without_duplicates_dataset


def check_count_data():
    """
    This function checks the size of dataset.
    :return: Print the size of the full dataset, dataset without missing data and dataset without duplicates rows.
    """
    # count_of_dataset = create_full_dataset().size
    # count_of_copied_dataset = remove_missing_data().size
    # count_of_without_duplicates = remove_duplicates().size()
    # print(f"Count of Dataset: {count_of_dataset}\nCount of Copied Dataset: {count_of_copied_dataset}\nCount of dataset without duplicates: {count_of_without_duplicates}")

    full_dataset = pd.read_csv("TripAdvisor_Dataset.csv", encoding='cp1252')
    missing_dataset = pd.read_csv("Without_Missing_Dataset.csv", encoding='cp1252')
    duplicates_dataset = pd.read_csv("Without_Duplicates_Dataset.csv", encoding='cp1252')
    print(f"Full dataset: {full_dataset.size}\nMissing_dataset: {missing_dataset.size}\nDuplicates dataset: {duplicates_dataset.size}")


def normalize_rating_column():
    """
    This function changes the rating column that it is written only the grade from 5.
    from example: instead of: 3.5 of the bubbles, its written: 3.5
    This change is done on the dataset that after removing the duplicates rows, and it is saved to new dataset: Dataset_after_all.csv
    :return: None
    """
    df = pd.read_csv("Without_Duplicates_Dataset.csv", encoding='cp1252')
    df['Rating'] = df['Rating'].apply(lambda x: x.split()[0])
    df.to_csv("Dataset_after_all.csv", index=False)


def normalize_review_count_column():
    """
    This function changes the review count column that it is written only the count without the word: reviews.
    from example: instead of: 675 reviews, its written: 675
    This change is done on the: Dataset_after_all.csv
    :return: None
    """
    df = pd.read_csv("Dataset_after_all.csv", encoding='cp1252')
    df['Review Count'] = df['Review Count'].apply(lambda x: x.split()[0])
    df.to_csv("Dataset_after_all.csv", index=False)


def normalize_picture_column():
    """
    This function changes the picture column that it is written only the count of pictures without: ()
    from example: instead of: (401), its written: 401
    This change is done on the: Dataset_after_all.csv
    :return: None
    """
    df = pd.read_csv("Dataset_after_all.csv", encoding='cp1252')
    df['Picture'] = df['Picture'].str.replace('(', '').str.replace(')', '')
    df.to_csv("Dataset_after_all.csv", index=False)


# scraping
scraping_url_of_all_hotels()

# save the data in csv file
create_full_dataset()

# remove missing rows
remove_missing_data()

# remove duplicates rows
remove_duplicates()

# data's amount
check_count_data()

# normalizing columns
normalize_rating_column()
normalize_review_count_column()
normalize_picture_column()
