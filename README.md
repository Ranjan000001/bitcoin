# 🪙 Crypto Market Analysis Datasets

## 📘 Project Overview

The project involves analyzing crypto market data from including midterm scores, Participation Score, Projects_Score and avcerage score of  Assignments, Quizzes ,  Gender , Department , Extracurricular Activities, Internet Access at Home, Parent Education Level, Family_Income_Level, Grade. The analysis encompasses data cleaning, featur engineering and visualization to understand crypto market patterns and trends.

## Data Cleaning and Preprocessing

Import Libraries 
```jupyter
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

```
Read files
```jupyter# read files
# file name 
file1 = "1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf"                     
file2 = "1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs"
# url link download
url1 = f"https://drive.google.com/uc?export=download&id={file1}"
url2 = f"https://drive.google.com/uc?export=download&id={file2}"

df = pd.read_csv(url1) # behavier of market
df1 = pd.read_csv(url2) # trade inferomation

```
Information of data
```jupyter
# info and head
df.info()
df.head()
```
```jupyter
# info and head
df1.info()
df1.head()
```
Cherck null cells
```jupyter
# checking null cells
student_df.isnull().sum()
```
Columns name
```jupyter
# column 
student_df.columns
```
## EDA

### Convert into Datetime Data Type
```jupyter
# convert column datatype 
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# But to change the print data look, we must apply dt.strftime (convert formatted string)
# Change or reform the date into dd-mm-yyyy
df['date'] = df['date'].dt.strftime('%d-%m-%Y')  # again it convert into object 

# check the changes
df.info()
df.head()
```

```jupyter
# Convert the data type of timestamp isd
df1['Timestamp IST'] = pd.to_datetime(df1['Timestamp IST'], errors='coerce')
# creat new column and change or reform timestamp into dd-mm-YYYY
df1['date'] = df1['Timestamp IST'].dt.strftime('%d-%m-%Y')

# Check the changes
df1.info()
df1.head()
```
### New DataSet by Fillter it with Another DataSet
```jupyter
# creating new df1 by fillter by df
new_df1 = df1[df1['date'].isin(df['date'])]
# check new df1 
new_df1.info()
new_df1.head()
```

### Remove Null Cells if Any
```jupyter
# remove the null cells if any 
new_df1.dropna(inplace=True)
df.dropna()
# Check the changes
new_df1.info()
df.info()
```


```jupyter
# despation statics of new_df1
new_df1.describe()
# despation statics of df
df.describe()
```

### Creating new columns
```jupyter
# New column netpnl(netprofit)
new_df1['net pnl'] = new_df1['Closed PnL'] - new_df1['Fee']
```
```jupyter
# new column of hours and session by day
new_df1['hours'] = new_df1['Timestamp IST'].dt.hour

def day(time):
    if 4<= time <12:
        return 'Morning'
    elif 12<=time<16:
        return 'Afternoon'
    elif 16 <= time < 20:
        return 'Evning'
    else:
        return 'Night'

new_df1['session'] = new_df1['hours'].apply(day)
```

```jupyter
# new column by usd per token
new_df1['usd per token'] = new_df1['Size USD']/new_df1['Size Tokens']
```

```jupyter
# new column position type
new_df1['position type'] = new_df1.apply(lambda x: 'long' if x['Start Position'] > 0  else( 'no position' if x['Start Position']==0 else 'short'), axis=1)  # axis = 1 for only column because it show error without axis.
# basic problem: elif can't be used in a lambda or comprehension because they are single-line expressions

#  Check the changes
new_df1 .info()
new_df1.head()
```
### Merging DataSet by Another DataSet

```jupyter
# merging two datasets  
bitcoin = new_df1.merge(df, on='date', how='left')  # merging in this way because they are of diff sizes and to get columns of df into new_df1 and the merger name is bitcoin.

# check changes
bitcoin.info()
bitcoin.head()
```
### Numerical-Categorial
```jupyter
# Violint graph of Numerical-Categorial 
for col in bitcoin.select_dtypes(include=np.number):
    plt.figure(figsize=(12, 5))
    sns.violinplot(data=bitcoin, x='classification', y=col )
    plt.title(f'graph of {col} vs classification')
    plt.show()
```

### Categorial-Categorial
```jupyter
# Stacked bar graphs for category to category
cols = [ 'Coin', 'Side',  'Direction',   'Crossed', 'session','position type']
for col in cols:
    plt.figure(figsize=(12, 7))
    cross_tab = pd.crosstab(bitcoin[col], bitcoin['classification'])
    cross_tab.plot(kind='bar', stacked=True)
    plt.title(f'graphs of {col} per classification')
    plt.show()
```

### Convert Categorial to numerical
```jupyter
# Convert category into numerical
[ 'Coin', 'Side',  'Direction',   'Crossed', 'session','position type']
df = bitcoin.copy()
# comprehenction on dicitionary for get str(key): int(value).
dis1 = {value:index for index, value in enumerate(df['Coin'])}
df['Coin'] = df['Coin'].map(dis1)
dis2 = {'BUY':0, 'SELL':1}
df['Side'] = df['Side'].map(dis2)
dis3 = {'Buy':0, 'Sell':1, 'Open Long':2, 'Close Long':3, 'Spot Dust Conversion':4, 'Open Short':5,'Close Short':6, 'Long > Short':7, 'Short > Long':8}
df['Direction'] = df['Direction'].map(dis3)
dis4 = {True:0, False:1}
df['Crossed'] = df['Crossed'].map(dis4)
dis5 = {'Night':0, 'Afternoon':1, 'Evning':2, 'Morning':3}
df['session'] = df['session'].map(dis5)
dis6 = {'no position':0, 'long':1, 'short':2}
df['position type'] = df['position type'].map(dis6)
dis7 = {'Greed':0, 'Extreme Greed':1, 'Fear':2, 'Extreme Fear':3, 'Neutral':4}
df['classification'] = df['classification'].map(dis7)
```


### Heatmap of Correlation Matrix
```jupyter

# correlation matrix with heatmap
corr_matrix = df.select_dtypes(include=np.number).corr()
# heatmap for correlation matrix
plt.figure(figsize=(12, 5))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Heatmap of Correlation Matrix')
plt.show()
```

### Trying Tree Model
```jupyter
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X, y)
print("Random Forest R²:", model_rf.score(X, y))
```

### Top 10 Coins
```jupyter
# top 10 coins
df = bitcoin.groupby('Coin').size().reset_index(name='count')
top10 = df.nlargest(10, 'count')
top10
```
### Cross validation
```jupyter
# check every coin by clasification
ct = pd.crosstab(bitcoin[bitcoin['Coin']=='@4']['Coin'], bitcoin['classification'], normalize='index')*100
ct
```


```jupyter
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

### Average of Numerical columns by Classification
```jupyter
# avgerag data of numerical vs classification
for col in bitcoin.select_dtypes(include=np.number):
    print(bitcoin.groupby('classification')[col].median().reset_index())
```

### Checking Target distribution
```jupyter
sns.histplot(y, kde=True)
```

### Try Similar Model
```jupyter
from sklearn.linear_model import RidgeCV, LassoCV
ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10])
lasso = LassoCV(cv=5)

ridge.fit(X_imputed, y)
lasso.fit(X_imputed, y)

print("Ridge R²:", ridge.score(X_imputed, y))
print("Lasso R²:", lasso.score(X_imputed, y))
```


## 📈 Conclusions

- Extreme Greed is the best time for profit-booking.
- Extreme Fear is the best time to buy any coin while having the 2nd highest fee.
- Best Session for selling is night because from morning and night, greed is increasing and fear is the same.
- Most liquid/popular coins is overall fear-driven HYPE and SOL, and overall greed-driven are @107, BTC, FTT, WLD, FARTCOIN, @4 and PURR/USD.
- Features like fee, order ID, timestamp, and execution price can be used for model calibration or abnormal detection trend.
- Greed is very dangerous for buying due to  higher token size, larger position, and negative Net pnl because buying at a higher price or overvaluation.   

## 📉 Risks

- Extreme greed selling risky due to overvaluation (suddenly convert extreme fear quickly, or coin fraud comes out).
- Extreme fear of buying risky due to the market going further downwards.
- Night is more greedy risky due to market uncertainty, it may change or not follow the trend this time.
- Features give an abnormal trend risky due to market uncertainty,  may give a wrong trend or change the trend when we enter.

## Power BI Dashboad
<img src="crypto market .png" alt="Click to visit Example.com">
