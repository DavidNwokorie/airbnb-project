import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer


airbnb = pd.read_csv("AirbnbSample6.csv")

print(airbnb.info())

print("Number of missing values: ")
print("\n")
for col in airbnb.columns:
    variable = sum(airbnb[col].isna())
    print('{}:{}'.format(col, variable))

# listed all the neighborhoods provided in the dataset
print("List of neighborhoods: ")
print(airbnb['neighborhood'].unique())

# current analysis goal: predict which airbnb listings are likley to have high booking demand


# basic cleaning
# dropped irrelevant columns
# kept same column names
airbnb = airbnb.drop(['id','name','country',], axis=1)
print(airbnb.info())
print(airbnb.head())


# handling missing values

# filling host verification missing values with all caps "UNCONFIRMED"
mv_hiv = "host_identity_verified"
airbnb[mv_hiv] = airbnb[mv_hiv].fillna('UNCONFIRMED')

# filling neighborhood misinng values with "N/A"
mv_neighborhood = "neighborhood"
airbnb[mv_neighborhood] = airbnb[mv_neighborhood].fillna('N/A')
airbnb['neighborhood'] = airbnb['neighborhood'].replace('manhatan', 'Manhattan')


# filling instant bookings misinng values with "N/A"
mv_ib = "instant_bookable"
airbnb[mv_ib] = airbnb[mv_ib].fillna('N/A')

# filling cancellation policy misinng values with "N/A"
mv_cp = "cancellation_policy"
airbnb[mv_cp] = airbnb[mv_cp].fillna('N/A')

# filling room types misinng values with "N/A"
mv_rt = "room_type"
airbnb [mv_rt] = airbnb[mv_rt].fillna('N/A')


# quick insight i ran into: creating dummy variables is more for numerical variables/columns and not really categorical columns.
# dummy variable for missing construction year numerical values. indicating missing value as 1 and non missing values as 0. will be useful for modeling
airbnb["year_missing"] = airbnb["construction_year"].isna().astype(int)
airbnb["construction_year"] = airbnb["construction_year"].fillna(airbnb["construction_year"].median())


# filling the missing values for price
# the missing values for price were filled based off the median prices per neighborhood
print("Missing prices before: ", airbnb['price'].isna().sum())
airbnb['price'] = airbnb['price'].fillna(
    airbnb.groupby('neighborhood')['price'].transform('median')
)
print("Missing prices after: ", airbnb['price'].isna().sum())


# filling the missing values for service fee
# insight: i came to the question of whether the missing values in the service fee column were random or deliberate
# insight: i calculated the mean for service fee, price, and star ratings; in doing so to see the difference in price for properties that had service fees and properties that didn't
# insight: with the results it made sense that properites without service fees had a lower mean average, but what caught me by suprise was star rating
# insight: the mean star ratings for properties that did not have service fees were higher than properties which did
# insight: it dawned on me that the service fees might only be implied in some properties, particularily smaller private properties, so i calculated the proportion of listings missing a service fee
# insight: the results showed that YES given the group that has no service fees, private rooms are more notiable meaning they are correlated to the higher star rating
missing_count = airbnb['service_fee'].isna().sum()
total_count = len(airbnb)
print(f"Missing service_fee values: {missing_count} out of {total_count}")

comparison = airbnb.groupby(airbnb['service_fee'].isna()).agg({
    'service_fee': 'mean',
    'price': 'mean',
    'star_rating': 'mean'
}).rename(index={True: 'Missing service_fee', False: 'Has service_fee'})

print(comparison)

print(airbnb.groupby(airbnb['service_fee'].isna())['room_type'].value_counts(normalize=True))
airbnb['service_fee_missing'] = airbnb['service_fee'].isna().astype(int)
airbnb['service_fee'] = airbnb['service_fee'].fillna(0)


# filling missing data for minimum nights
# made a seperate dummy column so the model can recognize the original, i went ahead a filled the missing data to 1 so those lisings will have a minimum of 1 night
airbnb['minimum_nights_missing'] = airbnb['minimum_nights'].isna().astype(int)
airbnb['minimum_nights'] = airbnb['minimum_nights'].fillna(1)


# filling missing data for number of reviews
# i just went ahead and filled it in with 0, assuming these listings are brand new and the empty column are not random
airbnb['number_of_reviews_missing'] = airbnb['number_of_reviews'].isna().astype(int)
airbnb['number_of_reviews'] = airbnb['number_of_reviews'].fillna(0)


# filling missing data for last review time
# exact dates not needed so turned into a binary dummy, meaning if it has a date it was reviewed, if not it wasn't reviewed
# likewise dropped the original column after because it is not needed
airbnb['last_review_missing'] = airbnb['last_review_time'].isna().astype(int)
airbnb = airbnb.drop(columns=['last_review_time'])


# filling missing data for reviews per month 
# created a binary dummy holding if the property has been reviewed or not
# if it has not been reviewed it'll be 0, if it has it'll be 1
airbnb['reviews_per_month_missing'] = airbnb['reviews_per_month'].isna().astype(int)
airbnb['reviews_per_month'] = airbnb['reviews_per_month'].fillna(0)


# filling missing data for star rating
# created a binary dummy as well to just inticate no guest rating yet
# including the 0 doesn't mean bad quality just "no rating yet"
airbnb['star_rating_missing'] = airbnb['star_rating'].isna().astype(int)
airbnb['star_rating'] = airbnb['star_rating'].fillna(0)


# filling missing data for host listing count
# made binary to keep it notiable where it was missing
# filled missing value with 1 because host needs at least 1 property to use airbnb
airbnb['host_listings_count_missing'] = airbnb['host_listings_count'].isna().astype(int)
airbnb['host_listings_count'] = airbnb['host_listings_count'].fillna(1)


# filling missing data for available days in the future
# saw some negative numbers which doesn't make sense, so i changed them to zero
# also changed all missing values to equal 0
airbnb.loc[airbnb['available_days_in_future'] < 0, 'available_days_in_future'] = 0
airbnb['available_days_missing'] = airbnb['available_days_in_future'].isna().astype(int)
airbnb['available_days_in_future'] = airbnb['available_days_in_future'].fillna(0)
print(airbnb['available_days_in_future'].describe())



# EDA and VISUALIZATIONS

# airbnb.info()
# airbnb.describe(include='all')

# corr = airbnb.corr(numeric_only=True)
# plt.figure(figsize=(10, 6))
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap of Numeric Variables')
# plt.show()


# Average reviews per month (demand indicator) by neighborhood

### IMPORTANT: look at the differnce between the graphs that have both "Manhattan" and "manhatan"
demand_by_neighborhood = (
    airbnb.groupby('neighborhood')['reviews_per_month']
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10,5))
demand_by_neighborhood.plot(kind='bar', color='mediumseagreen')
plt.title('Average Monthly Reviews by Neighborhood')
plt.ylabel('Average Reviews per Month')
plt.xlabel('Neighborhood')
plt.xticks(rotation=45, ha='right')
plt.show()




# Scatter Plot
# There isn’t a clear linear relationship between price and booking demand.
# Some higher-priced neighborhoods (like Staten Island) show relatively high review frequency,
# While others (like Manhattan and Brooklyn) have lower demand despite higher prices.
# This suggests that factors beyond price, such as location and listing type, drive booking activity
nb_stats = airbnb.groupby('neighborhood').agg({
    'price': 'mean',
    'reviews_per_month': 'mean'
}).reset_index()

plt.figure(figsize=(7,5))
sns.scatterplot(x='price', y='reviews_per_month', hue='neighborhood', data=nb_stats, s=80)
plt.title('Neighborhood Price vs. Booking Demand')
plt.xlabel('Average Price ($)')
plt.ylabel('Average Reviews per Month')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# Scatter Plot 
# Listings with high booking demand (many reviews per month) tend to have lower availability,
# While listings with many available days are less frequently booked
plt.figure(figsize=(7,5))
sns.scatterplot(x='available_days_in_future', y='reviews_per_month', data=airbnb, alpha=0.6)
plt.title('Availability vs. Booking Demand')
plt.xlabel('Available Days in Future')
plt.ylabel('Reviews per Month')
plt.show()



# Text Mining
nltk.download('vader_lexicon')
text_data = airbnb['house_rules'].dropna().astype(str)
text = " ".join(text_data)

airbnb["house_rules"] = airbnb["house_rules"].str.strip()
airbnb["house_rules"] = airbnb["house_rules"].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)


# Word Cloud Visualization
wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", max_words = 20, width=800, height=400).generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Terms in Airbnb House Rules')
plt.show()


# Sentimnet Analysis Visualization
SentimentAnalysis = SentimentIntensityAnalyzer()

airbnb['sentiment_score'] = airbnb['house_rules'].fillna('').apply(lambda x: SentimentAnalysis.polarity_scores(str(x))['compound'])
print(airbnb['sentiment_score'].describe())
plt.hist(airbnb['sentiment_score'], bins=20, color='lightblue', edgecolor='black')
plt.title('Sentiment Distribution of Airbnb House Rules')
plt.xlabel('Sentiment Score (-1 = Negative, 1 = Positive)')
plt.ylabel('Count')
plt.show()


