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


# dropped irrelevant columns
airbnb = airbnb.drop(['id','name','country','house_rules'], axis=1)
print(airbnb.info())


# handling missing values

