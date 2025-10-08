import pandas as pd
import matplotlib as plt
import seaborn as sns

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
airbnb = airbnb.drop(['id','name','country','house_rules'], axis=1)
print(airbnb.info())
print(airbnb.head())


# handling missing values

# filling host verification missing values with all caps "UNCONFIRMED"
mv_hiv = "host_identity_verified"
airbnb[mv_hiv] = airbnb[mv_hiv].fillna('UNCONFIRMED')

# filling neighborhood misinng values with "N/A"
mv_neighborhood = "neighborhood"
airbnb[mv_neighborhood] = airbnb[mv_neighborhood].fillna('N/A')

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
airbnb['last_review_misisng'] = airbnb['last_review_time'].isna().astype(int)
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
