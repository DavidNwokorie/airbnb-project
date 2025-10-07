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
# dummy variable for missing construction year numerical values. indicating missing value as 1 and non missing values as 0
airbnb["year_missing"] = airbnb["construction_year"].isna().astype(int)
airbnb["construction_year"] = airbnb["construction_year"].fillna(airbnb["construction_year"].median())




