# **Stock-Price-Trend-Prediction**

## **Proposal**

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
## **Midterm Report**
The datasets and method we should select have triggered intense discussion in our group. At this stage, we scraped the data from Yahoo Finance and collected the trend within 1000 days of stock from each company. Moreover, we utilized feature engineering and random forest to predict the trend of any stock in the market within the future of 5 days, 1 month and 1 year in the future. We also figured out possible improvements at current status. **Several comparatively indicative parts of the code and the result we obtained are included in our report below to make the process clear.**

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
We imported yfinance as our data source; also, we utilized several data packages that would manifest the result effectively. For instance, we empleyed tools to plot the diagrams of stock trend we predict. Also, for our future analysis and prediction, we would continue with random forest method.

#### ***II. Forming the datasets into CSV file & Cleaning***<br />
```
stock_list = ['0005.HK', '0006.HK', '0066.HK', '0700.HK', '2800.HK']
sp500data = yf.download('0066.HK', start="2020-01-01", end="2022-11-16")
sp500_df = pd.DataFrame(sp500data)
sp500_df.to_csv("sp500_data.csv")
```
```
pd.set_option('mode.use_inf_as_na', True)
read_df.dropna(
    axis=0,
    how='any',
    subset=None,
    inplace=True
)
```
We extracted the data from source website to a CSV file that contain the information we would like to predict out of our method. We also conducted data cleaning to drop empty values that might affect the validity of our training dataset(sometimes feature not applicable or sometimes there contains NaN as value missing. We selected to plot "Ädjusted Close" to indicate the trend.

#### ***III. Feature Engineering***<br />
```
feature_names = []
for n in [14, 30, 50, 200]:
    sp500data['ma' + str(n)] = talib.SMA(sp500data['Adj Close'].values, timeperiod=n)
    sp500data['rsi' + str(n)] = talib.RSI(sp500data['Adj Close'].values, timeperiod=n)
    feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]
sp500data['Volume_1d_change'] = sp500data['Volume'].pct_change()
volume_features = ['Volume_1d_change']
feature_names.extend(volume_features)
```
In order to make our prediction with a more solid ground with less interference, we select an essential approach to reduce complexity of our dataset. Since our dataset contains a lot of columns, including the open and close value with and without adjustment, the highest and lowest point, etc. it might be useful for us to examine the correlation and effective weight of every feature to drop the features that do not need to be stressed while selecting and incorporating more significant ones into our further analysis.

#### ***IV. Splitting Training and Testing Datasets***<br />
```
x = df.iloc[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].values
y = df.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.26,  random_state=0)
```
We selected two group of data, x and y, each into their training and testing datasets and normalize to fit them into appropriate scale.

#### ***V. Model***<br />
```
model = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_split=2, min_samples_leaf=1, max_depth=10, bootstrap=True)
model.fit(x_train, y_train)
predict = model.predict(x_test)
```
```
rscv = RandomizedSearchCV(estimator=model, param_distributions=grid_rf, cv=3, n_jobs=-1, verbose=2, n_iter=200)
rscv_fit = rscv.fit(x_train, y_train)
best_parameters = rscv_fit.best_params_
```
Using random forest regression method, we want to see the current result we could obtain. Also, we used cross validation as a resampling method that uses different portions of the data to test and train this model on different iterations. In each iteration, it would select the features randomly with fairness.

#### ***VI. Result and Output***<br />
```
print(f'Train Score : {model.score(x_train, y_train) * 100:.2f}% and Test Score : {model.score(x_test, y_test) * 100:.2f}% using Random Tree Regressor.')
errors = abs(predict - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
```
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/blob/main/images%20in%20report/output.png)<br />

Before we conducted feature engineering as described above, we tried to predict the adjusted close value with simply the raw data, considering all features available. As the first general result, it is a result that generally met our expectation with an R^2 score around 0.87. We used this result to consolidate the process and effect of our effort in feature engineering.<br />

![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/blob/main/images%20in%20report/After_feature_engineering.jpg)<br />
After our implementation of Random Forest with the cleaned dataset after feature engineering and it turns out that the accuracy with random forest on first-cleaned dataset is around 98%.<br />

#### ***VII. Improvement and Visualization***<br />
```
def data_manage(data, date):
    y = data[date + 1:]
    x = []
    for i in range(date, data.shape[0] - 1):
        x.append(data[i - date: i])
    return np.array(x), y
```
In order to test the output of our prediction more effectively, we changed the input a bit with a defined function to include x as the first 30 days and y as the next day right after in a way which we intend to predict the adjusted price of the day right after 30 days that we inputed data. Afterwards, we ran the random forest method again as below.
```
model = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_split=2, min_samples_leaf=1, max_depth=10, bootstrap=True)
model.fit(x_train, y_train)
predict = model.predict(x_test)
```
```
one_estimator = rscv_fit.estimator[0]
f_name, c_name = feature_target_name(date)
plt.figure()  
_ = tree.plot_tree(one_estimator, feature_names = f_name, filled=True, rounded = True, proportion = False, precision = 2, fontsize=6)
plt.show()
plt.savefig('tree.png')
```
We begin to visualize the random forest as combination of decision trees. We will begin by determining the difference between the true value and the prediction values. This difference will be used for calculating the accuracy as well as indicating how well we perform overall. Therefore, we obtained the graph of decision trees as below.

![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/blob/main/images%20in%20report/Trees.png) <br />

We tried two different means to see if our method works.<br />

![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/blob/main/images%20in%20report/Predicted_1.png)<br />

We tried to use our model to train with two companies as inputs, and give out the plot for prediction of one among them. The overall trend meets our expectation.<br />

![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/blob/main/images%20in%20report/Predicted_2.png)<br />

Then, we tried to use our model to train one company to see if the predicted value are closely associated with the the correct ground truth from 2016 till now. And we obtained the graphs below. We can see that the accuracy is generally appropriate and constantly high.<br />

![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/blob/main/images%20in%20report/result_2.png)<br />

#### ***VIII. Direction for our next Phase***<br />
Currently, our implementation of feature engineering and random forest method met our expectation and proved that we are on the right track. However, we have several more aspects to consider before we reach a final conclusion. The high accuracy was probably due to a correct feature chose instead of applicable to every case. Therefore, here are several points we would like to list out.<br />
1. We would like to further test and configure to determine the correlation thresholds between different column features, and simultaneously, considering the possibility of adding dimensions including variance or moving average, etc. Also, we are considering using tools to visualize the correlation, including SNS's heat map under Exploratory Data Analysis. Hopefully it would boost accuracy.<br />
2. Overfitting to one or specific sets of data can be detrimental when generalizing the result to other companies. Continuous cross-validation(using different methods, such as kfold), and training with more data with augmentation is required in our next phase.<br />
3. If possible, we could try out on different methods and compare the effectiveness, including multiple linear regression and forecasting, or deep learning method, to see if any stage could overperform ultimately. There are two major deep learning methods we would like to use for our next phase: Sequence to Sequence Model and Transformer Model. Compare to random forest model, they capture the concept of time better and, for the Transformer Model, could process accompanying time information and a longer time of memory. We would like to try out to see if accuracy hit our target.<br />















