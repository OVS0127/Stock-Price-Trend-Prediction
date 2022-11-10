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
### *Part I*
>**Step I. Importing Data and Required Packages**\
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/blob/main/import_midterm.png)\
We imported yfinance as our data source; also, we utilized several data packages that would manifest the result effectively.

>**Step II. Forming the datasets into CSV file**\
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/blob/main/convert_csv_midterm.png)\
We extracted the data from source website to a CSV file that contain the information we would like to predict out of our method.






