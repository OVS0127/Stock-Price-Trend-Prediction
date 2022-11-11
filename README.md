# **Stock-Price-Trend-Prediction**

## ***I.Proposal***

### *Introduction*

Stock is one of the most significant forms of investments sparked off in current centuries. Stock markets are hard to predict, triggering intense study and research topics related to machine learning in order to spark off new methods to predict the tendency. Specifically, only a few of the parameters of a stock can be quantified precisely, such as prices, volume of trade, open interest, etc. However, the value of a stock is also highly related to a company’s investment strategies, intrinsic value, or even a piece of news, all of which are difficult to express mathematically. The unstable nature of the stock market makes it a challenging task to predict the exact price of a stock. As a result, we will focus on predicting the short-term trend of a stock using historical data[1].

### *Plans and Methods*

Our dataset will be the open, close, low, and high prices, as well as volume and open interest of a unique stock over a period of time[2]. We will then use preprocessing to explicitly clean the data and compare horizontally to obtain several technical indicators such as relative return, Momentum, price rate of change, etc. These can be obtained by database managements methods including pandas, numpy and SQL as well as technical realization of entity matching. After useful data and indicators are obtained, our machine learning model then takes these technical indicators as inputs and predicts a short-term future trend[4].

Over recent years, many methods like LSTM, random forest, and multi layer perceptron have been proved to be great methods for stock prediction[3]. The idea right now is to choose two or more methods to implement several models (the decision now is random forest and forward neural network) and find out which model gives us a higher accuracy of prediction after fine tuning. We are also considering using the two models together. First we will use RF to fit the training data and use PCA to extract the most important feature. Then we will reconstruct the input based on the new features and feed into the FNN[4].

For now, we assume the input will be a window of 30 days. In other words, we are using data from the past 30 days to predict the trend of the price on the 31st day[4]. The problem will then become a binary classification problem. The accuracy is calculated by classification metrics: accuracy scores (F-measure), Hamming loss, and ROCAUC to compare the predicted market price trend and the ground truth market price trend of the period depending on which matrix represents the accuracy better and what our output will be (market price of each time interval in the day or a matrix represents the trend) , since what we are looking for is more of an accurate trend of market price over the period than market price itself.

### *Gantt Chart for Group Work*
***Group Member/Task***|Data Sourcing 10/7-10/14 | Model Selection 10/7-10/14 | Data pre-processing 10/7-10/20| Model Coding 10/14-11/19| Result Eval & tuning 11/19-11/24| Report & Recording 11/24-12/6
--------- | -------------| -------------| -------------| -------------| -------------| -------------
Lifu Wang | R| R| | R for Model 1| |
Xiaofeng Wu| |R| | R for DL Method| R for DL Method| R
Yelu Wang ||R|| R for Model 1| R for Model 1| R
Hanran Wu | | R || R|R|R
Zhonghui Shen |R for both data|R|R for both Data | | | R for Report

***Group Member/Task***|Data Sourcing | Model Selection | Data pre-processing| Model Coding| Result Eval & tuning | Report & Recording
--------- | -------------| -------------| -------------| -------------| -------------| -------------
Lifu Wang |C| C| | C|C |C
Xiaofeng Wu| |C| | C| C| C
Yelu Wang ||C|| C| C| C
Hanran Wu | | C||C|C|C
Zhonghui Shen |C|C|C| | |C

### *References:*

[1]ProjectPro. (2022, June 16). Stock price prediction using machine learning with source code. ProjectPro. Retrieved October 7, 2022, from https://www.projectpro.io/article/stock-price-prediction-using-machine-learning-project/571

[2]Marjanovic, B. (2017, November 16). Huge stock market dataset. Kaggle. Retrieved October 7, 2022, from https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs

[3]Huang, Y., Capretz, L. F., & Ho, D. (2022, January 26). Machine learning for stock prediction based on fundamental analysis. arXiv.org. Retrieved October 7, 2022, from https://arxiv.org/abs/2202.05702

[4]Ma, Yilin & Han, Ruizhu & Fu, Xiaoling. (2019). Stock prediction based on random forest and LSTM neural network. 126-130. 10.23919/ICCAS47443.2019.8971687. Dataset: https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs

### *Dataset*

[1]https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
<br>
<br>
<br>
## ***Midterm Report***
The datasets and method we should select have triggered intense discussion in our group. At this stage, we scraped the data from Yahoo Finance and collected the trend within 1000 days of stock from each company. Moreover, we utilized feature engineering and random forest to predict the trend of any stock in the market within the future of 5 days, 1 month and 1 year in the future. We also figured out possible improvements at current status.
### *Part I*
#### ***I. Importing Data and Required Packages***<br />
```
import yfinance as yf
import talib
import datetime as dt
import pandas as pd
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
```
**We imported yfinance as our data source; also, we utilized several data packages that would manifest the result effectively. For instance, we empleyed tools to plot the diagrams of stock trend we predict. Also, for our future analysis and prediction, we would continue with random forest method.**

#### ***II. Feature Engineering***<br />
```
stock_list = ['0005.HK', '0006.HK', '0066.HK', '0700.HK', '2800.HK']
sp500data = yf.download('0066.HK', start="2020-01-01", end="2022-11-16")
feature_names = []
for n in [14, 30, 50, 200]:
    sp500data['ma' + str(n)] = talib.SMA(sp500data['Adj Close'].values, timeperiod=n)
    sp500data['rsi' + str(n)] = talib.RSI(sp500data['Adj Close'].values, timeperiod=n)
    feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]
sp500data['Volume_1d_change'] = sp500data['Volume'].pct_change()
volume_features = ['Volume_1d_change']
feature_names.extend(volume_features)
```
**In order to make our prediction with a more solid ground with less interference, we select an essential approach to reduce complexity of our dataset. Since our dataset contains a lot of columns, including the open and close value with and without adjustment, the highest and lowest point, etc. it might be useful for us to examine the correlation and effective weight of every feature to drop the features that do not need to be stressed while selecting and incorporating more significant ones into our further analysis.**

#### ***III. Forming the datasets into CSV file & Cleaning***<br />
```
sp500_df = pd.DataFrame(sp500data)
sp500_df.to_csv("sp500_data.csv")
print(sp500_df)
read_df = pd.read_csv("sp500_data.csv")
read_df.set_index("Date", inplace=True)
pd.set_option('mode.use_inf_as_na', True)
read_df.dropna(
    axis=0,
    how='any',
    subset=None,
    inplace=True
)

read_df['Adj Close'].plot()
plt.ylabel("Adjusted Close Prices")
plt.show()
df = pd.read_csv("sp500_data.csv")
df.set_index("Date", inplace=True)
df.dropna(inplace=True)
```
**We extracted the data from source website to a CSV file that contain the information we would like to predict out of our method.**

#### ***IV. Splitting Training and Testing Datasets***<br />
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/blob/main/images%20in%20report/split.png)<br />

#### ***V. Model***<br />
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/blob/main/images%20in%20report/model.png)<br />

#### ***VI. Result and Output***<br />
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/blob/main/images%20in%20report/result.png)
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/blob/main/images%20in%20report/output.png)<br />









