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

For our Dataset selection, we choose to import from Yahoo finance, which is a widely known website that provide most of the data necessary for the stock market. If we download and import yfinance will help us downloading the data we need using code. The dataset from yfinance look like this. It has seven columns, date, open, high, low, close, adjusted close and volume for a company's stock on continuitive dates.
```
            Open       High       Low        Close      Adj Close  Volume
Date
2020-01-02  60.849998  60.950001  60.599998  60.900002  55.998753  14629077
2020-01-03  60.900002  61.200001  60.250000  60.400002  55.538994  14419537
2020-01-06  60.099998  60.400002  59.799999  60.000000  55.171185  13809308
2020-01-07  60.200001  60.299999  59.799999  59.900002  55.079235   8818594
2020-01-08  59.299999  59.400002  58.849998  59.299999  54.527519  16826669
...               ...        ...        ...        ...        ...       ...
2022-11-09  43.049999  43.599998  42.500000  43.000000  43.000000   9781099
2022-11-10  42.349998  42.500000  41.799999  42.500000  42.500000   8914894
2022-11-11  43.599998  43.700001  42.750000  43.400002  43.400002  23447735
2022-11-14  43.400002  44.700001  43.349998  43.650002  43.650002  16908635
2022-11-15  43.900002  44.700001  43.799999  44.700001  44.700001  20487782
```
Our further implementation in our midterm and final phase would begin with this dataset for learning.

<br>
<br>
<br>
## **Phase I**
We divided our project in two phases. As we described in our proposal, the datasets and method we should select have triggered intense discussion in our group. At this stage, we scraped the data from Yahoo Finance and collected the trend within 1000 days of stock from each company. Moreover, we utilized feature engineering and random forest to predict the trend of any stock in the market in the future of 1 day, 5 days, 1 month and 1 year. We also figured out possible improvements at current status. **Several comparatively indicative parts of the code and the result we obtained are included in our report below to make the process clear.**

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
We imported yfinance as our data source; also, we utilized several data packages that would manifest the result effectively. For instance, we employed tools to plot the diagrams of stock trend we predict. Also, for our future analysis and prediction, we would continue with random forest method.

#### ***II. Forming the datasets into CSV file & Cleaning***<br />
```
sp500data = yf.download('0066.HK', start="2020-01-01", end="2022-11-16")
sp500_df = pd.DataFrame(sp500data)
sp500_df.to_csv("sp500_data.csv")
```
Our dataset from the sourse website looks like this.
```
            Open       High       Low        Close      Adj Close  Volume
Date
2020-01-02  60.849998  60.950001  60.599998  60.900002  55.998753  14629077
...
```
We made some cleaning to it for further use.
```
pd.set_option('mode.use_inf_as_na', True)
read_df.dropna(
    axis=0,
    how='any',
    subset=None,
    inplace=True
)
```
We extracted the data from source website to a CSV file that contain the information we would like to predict out of our method. We also conducted data cleaning to drop empty values or infinity values that might affect the validity of our training dataset(sometimes feature not applicable or sometimes there contains NaN as value missing. We selected to plot "Ädjusted Close" to indicate the trend.

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
```
def data_manage(data, date):
    y = data[date + 1:]
    x = []
    for i in range(date, data.shape[0] - 1):
        x.append(data[i - date: i])
    return np.array(x), y 
```
We selected the fifth column 'Adjusted Close' as the ground truth(y). The rest of the columns becomes the input(x). we also split x and y into their training and testing datasets and normalized them to fit them into appropriate scale. Afterwards, we split the adjusted close into two sets, one set in the next day and one set in next thirty days.

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
Using random forest regression method, we want to see the current result we could obtain. Also, we used cross validation to search for the best hyperparameters. The final result is based on the best hyperparameters.
#### ***VI. Result and Output***<br />
```
print(f'Train Score : {model.score(x_train, y_train) * 100:.2f}% and Test Score : {model.score(x_test, y_test) * 100:.2f}% using Random Tree Regressor.')
errors = abs(predict - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
```
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/raw/main/images%20in%20report/output.png)<br />

Before we conducted feature engineering as described above, we tried to predict the adjusted close value with simply the raw dataset, considering all features available. We used many metrics here, including Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, etc. As the first general result, it is a result that generally met our expectation with an R^2 score around 0.87. We used this result to consolidate the process and effect of our effort in feature engineering.<br />

![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/raw/main/images%20in%20report/After_feature_engineering.jpg)<br />

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

![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/raw/main/images%20in%20report/Trees.png) <br />

Then, we tried to use our model to train one company to see if the predicted value are closely associated with the the correct ground truth from 2016 till now. And we obtained the graphs below. We can see that the accuracy is generally appropriate and constantly high.<br />

![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/raw/main/images%20in%20report/result_2.png)
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/raw/main/images%20in%20report/rf.png)


We tried two different means to see if our method works.<br />

![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/raw/main/images%20in%20report/Predicted_1.png)

We tried to use our model to train with two companies as inputs, and give out the plot for prediction of one among them. The overall trend meets our expectation.<br />
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/raw/main/images%20in%20report/Predicted_2.png)



#### ***VIII. Direction for our next Phase***<br />
Currently, our implementation of feature engineering and random forest method met our expectation and proved that we are on the right track. However, we have several more aspects to consider before we reach a final conclusion. The high accuracy was probably due to a correct feature chose instead of applicable to every case. Therefore, here are several points we would like to list out.<br />
1. We would like to further test and configure to determine the correlation thresholds between different column features, and simultaneously, considering the possibility of adding dimensions including variance or moving average, etc. Also, we are considering using tools to visualize the correlation, including SNS's heat map under Exploratory Data Analysis. Hopefully it would boost accuracy.<br />
2. Overfitting to one or specific sets of data can be detrimental when generalizing the result to other companies. Continuous cross-validation(using different methods, such as kfold), and training with more data with augmentation is required in our next phase.<br />
3. If possible, we could try out on different methods and compare the effectiveness, including multiple linear regression and forecasting, or deep learning method, to see if any stage could overperform ultimately. There are two major deep learning methods we would like to use for our next phase: Sequence to Sequence Model and Transformer Model. Compare to random forest model, they capture the concept of time better and, for the Transformer Model, could process accompanying time information and a longer time of memory. We would like to try out to see if accuracy hit our target.<br />


## **Phase II**
We aim to use different methods try to predict the stock price with higher accuracy compared to previous endeavor. The background information, problem definition, data collection are the same as stated in our proposal and midterm report.

After the first checkpoint, we have implemented two new models to train data: Feature Engineering and Random Forest based on decision trees in order to compare the accuracy between different models and try to find the most accurate way to predict the stock price. The key difference between the previous and current method we used is that the ones we employed in our midterm did not capture the concept of time; however, our new model employed deep learning method and achieved better concept of time. We select LSTM method and transformer.

### *Part I LSTM Model*
#### ***I. Model Overview***<br />
LSTM model, with full name of long short-term memory, is also a method in deep learning that incurrs a variety of recurrent neural networks with the capability of learning long-term dependencies between two variables in sequencial prediction related problems. Since it has the ability to optimize memorizing the past data and capture the concept of time, it is useful to evaluate time-related sequence. We employed the model to help us doing the selection and prediction. 
Below, as usual, we listed the parts we considered significant. <br />
Here is the class for us to utilize the function of LSTM model. The layer is consequential that learns long-term dependencies between time steps in time series and sequence data.<br />
```
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes 
        self.num_layers = num_layers
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.seq_length = seq_length 

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
```
The model, the loss function and the optimizer are nevertheless important.<br />
```
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]) #our lstm class 
criterion = torch.nn.MSELoss()   
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 
```
We divided the datasets as usual and try to plot out the result.<br />
```
train_predict = lstm1(X_test_tensors_final)
data_predict = train_predict.data.numpy() 
dataY_plot = y_test_tensors.data.numpy()
data_predict = mm.inverse_transform(data_predict) 
dataY_plot = mm.inverse_transform(dataY_plot)
plt.figure(figsize=(10,6)) =
plt.axvline(x=200, c='r', linestyle='--')
```

#### ***II. Result and Output***<br />
We obtained the output of accuracy as below using transformer. We are using $R^2$ to evaluate and found out the accuracy is approaching 1. <br />
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/raw/main/images%20in%20report/LSTM_r_squared.png)<br />
After visualization, we obtained the prediction of a stock for 200 days. The results are also obvious differentiating different parts in different color. The trends of the blue and the orange curve are approaching each other.<br />
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/raw/main/images%20in%20report/LSTM_result.png)<br />
However, we tried to train the same LSTM model with another company in order to see if the good overall result is a little bit overfitted and cannot be extented to the others. <br />
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/raw/main/images%20in%20report/LSTM_2.png)<br />
We can see the overall trend is correct but there are large differences between ground truth and the predicted value, which motivate us with a new method.

### *Part II Transformer*
#### ***I. Model Overview***<br />
Transformer is a deep learning method that adopts the mechanism of self-attention, which helps differentially weighting the significance of each part of the input data. We import Transformer and induced parameters as defined below. We added them to our previous model.

```
class transf_params:
    n_layers = 11
    num_heads = 12
    model_dim = 16  # nr of features
    forward_dim = 2048
    output_dim = 1
    dropout = 0
    n_epochs = 20
    lr = 0.01
```
```
class TransformerModel(nn.Module):
    def __init__(self, params):
        super(TransformerModel, self).__init__()
        self.transf = transformer.TransformerModel(n_layers=params.n_layers,
                                                   num_heads=params.num_heads,
                                                   model_dim=params.model_dim,
                                                   forward_dim=params.forward_dim,
                                                   output_dim=16,
                                                   dropout=params.dropout)
        self.linear = nn.Linear(16, params.output_dim)
    def forward(self, x):
        transf_out = self.transf(x)
        out = self.linear(transf_out)
        return out
```
We did the preprocessing steps again in order to make the data appropraite to fit into the model. Here are several parts as illustration. Also, we divided the traning and testing dataset and plotted the graph in order to visualize our result.
```
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f'Daily Return of stock "{df["Company stock name"][0]}"',
                                        f'Daily change in % of stock "{df["Company stock name"][0]}"'))
    fig.add_trace(go.Histogram(x=df['Daily Return'].dropna(), marker_color='#330C73', opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=xDR, y=df['Daily Return'].dropna(), mode='lines', line_color='#330C73'), row=1, col=1)
```
```
train_data, test_data, train_data_len = dataset.split(train_split_ratio=0.8, time_period=30)
    train_data, test_data = dataset.get_torchdata()
```
```
if stationary:
        predictions = Analysis.inverse_stationary_data(old_df=df, new_df=predictions,
                                                       orig_feature='Actual', new_feature='Predictions',
                                                       diff=12, do_orig=False)
    plot_predictions(df, train_data_len, predictions["Predictions"].values, model_type)

```
#### ***II. Result and Output***<br />
We obtained the output of accuracy as below using transformer. The accuracy rate is indicated by $R^2$ score 0.8333. <br />
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/raw/main/images%20in%20report/transformer_accuracy.png)<br />
After visualization, we obtained the prediction of a stock for two years, from September 2020 to September 2022. The results are straightforward, differentiating different parts in different color. The prediction data is in pink. <br />
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/raw/main/images%20in%20report/transformer_output.png)<br />
We tried to train the same LSTM model with another company in order to see if the good overall result is currently overfitted and cannot be extented to the others.<br />
![Alt](https://github.com/OVS0127/Stock-Price-Trend-Prediction/raw/main/images%20in%20report/transformer_final.png)<br />
The overall result is performing well and higher than our expectation.

#### ***Review***<br />
Throughout our project, we manipulated the dataset with initial feature engineered prediction and used three different models to predict the price of the stock market. First, we concreted our previous result by selecting and adding features relevant put under the random forest regressor. In our second phase, we used two deep learning methods: Transformer and LSTM(without position encoding layer). All of these three method are making suitable predictions and justifiable outputs. However, both of them have advantages and disadvantages, making every step and tryouts significant for the progress though we can see fluctuations. After training with one set of data, we tried another set of another company and strengthen our confident in our prediction. It is nevertheless significant for us to continue improving while learning.
We have made a video to present our process. Here is the link to our video:













