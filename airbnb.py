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


airbnb['minimum_nights_missing'] = airbnb['minimum_nights'].isna().astype(int)
airbnb['minimum_nights'] = airbnb['minimum_nights'].fillna(1)


# should be all good now i hope lol


