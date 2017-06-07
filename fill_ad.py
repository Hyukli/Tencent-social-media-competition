import pandas as pd
ad=pd.read_csv('database/ad.csv')
print ad['adID'].value_counts()