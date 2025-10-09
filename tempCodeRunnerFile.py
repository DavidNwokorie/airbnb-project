import pandas as pd
import matplotlib.pyplot as plt
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
airbnb['neighborhood'] = airbnb['neighborhood'].replace('manhatan', 'Manhattan')
mv_neighborhood = "neighborhood"
airbnb[mv_neighborhood] = airbnb[mv_neighborhood].fillna('N/A')
print("List of neighborhoods: ")
print(airbnb['neighborhood'].unique())