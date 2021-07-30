import sys
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler

from keras.utils import to_categorical
from keras.layers import Input
from math import ceil
import json
import os
import sklearn.externals
import joblib

from technical_analysis.generate_labels import Genlabels
from technical_analysis.macd import Macd
from technical_analysis.rsi import StochRsi
from technical_analysis.poly_interpolation import PolyInter
from technical_analysis.dpo import Dpo
from technical_analysis.coppock import Coppock
from technical_analysis.date import Date
from technical_analysis.sp500 import Sp500
from technical_analysis.y import Y
from out.pred import Pred

def info(min_, i, coin, ep, X):
    print(min_)
    print(i)
    print(coin)
    print(ep)
    print(X.shape)

def insert_data(path, choice=True, end=None, add=None):
    df = pd.read_csv(path)
    df = df.set_index('dates')
    df = df.dropna()
    
    exclud = 110000

    return df.iloc[113420:, :]

    # if (end != None):
    #     return df.iloc[exclud: (exclud + end), :] # predict
    # if (choice):
    #     return df.iloc[0:exclud, :] # test/train
    # return df.iloc[exclud:, :] # predict

def extract_data(close_data, all_data, no_tend=True):
    date = Date(all_data)

    # obtain labels
    trend = Genlabels(close_data, window=25, polyorder=3).labels
    l = Y(close_data)
    size = 30 

    # obtain features
    macd = Macd(close_data, 6, 12, 3).values
    stoch_rsi = StochRsi(close_data, period=14).hist_values
    dpo = Dpo(close_data, period=4).values
    cop = Coppock(close_data, wma_pd=10, roc_long=6, roc_short=3).values
    inter_slope = PolyInter(close_data, progress_bar=True).values
    label = l.get_y()[(size - 1): -1]
    days = date.days()
    hours = date.hours()


    X = np.array([
                macd[size:-1]
                ,stoch_rsi[size:-1]
                ,dpo[size:-1]
                ,cop[size:-1]
                ,inter_slope[size:-1]
                ,hours[size:-1]
                ,days[size:-1]   
                ,trend[size:-1]
                ,np.array(l.get_y()[(size - 1): -1])
                ,close_data[size:-1]
                # ,np.array(all_data['Volume'])[size:-1]
                ])

    X = np.transpose(X)
    y = np.array(l.get_y()[(size):])
    y_2 = pd.DataFrame(to_categorical(l.get_y()[(size):], 2)).iloc[:, :1]

    data = pd.DataFrame(close_data[(size):-1])
    data['y'] = y
    data['y_2'] = y_2
    
    return X, y, data.iloc[7:].reset_index()

def adjust_y(y, size=2):

    label = []
    for i in y:
        aux = np.zeros(size)
        if (i == 0):
            aux[0] = 1
        else:
            aux[1] = 1
        label.append(aux)

    return np.array(label)

def adjust_data(X, y, split=0.8):
    # count the number of each label
    count_1 = np.count_nonzero(y)
    count_0 = y.shape[0] - count_1
    cut = min(count_0, count_1)

    # save some data for testing
    train_idx = int(cut * split)
    print(train_idx)

    # shuffle data
    np.random.seed(42)

    shuffle_index = np.random.permutation(X.shape[0])

    print(X.shape)
    print(y.shape)

    X, y = X[shuffle_index], y[shuffle_index]

    # find indexes of each label
    idx_1 = np.argwhere(y == 1).flatten()
    idx_0 = np.argwhere(y == 0).flatten()

    print(idx_1)
    print(idx_0)

    # grab specified cut of each label put them together 
    X_train = np.concatenate((X[idx_1[:train_idx]], X[idx_0[:train_idx]]), axis=0)
    X_test = np.concatenate((X[idx_1[train_idx:cut]], X[idx_0[train_idx:cut]]), axis=0)
    y_train = np.concatenate((y[idx_1[:train_idx]], y[idx_0[:train_idx]]), axis=0)
    y_test = np.concatenate((y[idx_1[train_idx:cut]], y[idx_0[train_idx:cut]]), axis=0)

    # shuffle again to mix labels
    np.random.seed(7)
    shuffle_train = np.random.permutation(X_train.shape[0])
    shuffle_test = np.random.permutation(X_test.shape[0])

    X_train, y_train = X_train[shuffle_train], y_train[shuffle_train]
    X_test, y_test = X_test[shuffle_test], y_test[shuffle_test]

    return X_train, X_test, y_train, y_test


def shape_data(X, y, timesteps=10):
    # scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if not os.path.exists('models'):
        os.mkdir('models')

    joblib.dump(scaler, 'models/scaler.dump')

    # reshape data with timesteps
    reshaped = []
    for i in range(timesteps, X.shape[0] + 1):
        reshaped.append(X[i - timesteps:i])
    
    # account for data lost in reshaping
    X = np.array(reshaped)
    y = y[timesteps - 1:]

    print(y.shape)

    return X, y


def build_model():

    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))

    # fourth layer and output
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # compile layers
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def calculate_trade(trend, trade, percent, resp):
    percent_high = float(trend[0][1].split(' ')[1])
    percent_close = float(trend[1][1].split(' ')[1])
    percent_low = float(trend[2][1].split(' ')[1])
    price_high = float(trend[0][0])
    price_close = float(trend[1][0])
    price_low = float(trend[2][0])

    s = 0.15

    if ((percent_high > percent + s) & (percent_close > percent) & (percent_low > percent + s)):
        trade['p_high'] = ceil((100 - ((price_close * 100) / price_high)) * 1000)
        trade['p_low'] = ceil((100 - ((price_low * 100) / price_close)) * 1000)
        trade['status'] = True
        if ((resp[0][0] > 0.5) & (resp[1][0] > 0.5) & (resp[2][0] > 0.5)):
            trade['resp'] = True
        else:
            trade['resp'] = False

    if ((percent_high < (1 - percent + s)) & (percent_close < (1 - percent)) & (percent_low < (1 - percent + s))):
        trade['p_high'] = ceil((100 - ((price_close * 100) / price_high)) * 1000)
        trade['p_low'] = ceil((100 - ((price_low * 100) / price_close)) * 1000)
        trade['status'] = False
        if ((resp[0][1] < 0.5) & (resp[1][1] < 0.5) & (resp[2][1] < 0.5)):
            trade['resp'] = True
        else:
            trade['resp'] = False

        
def predict(min_, ep, coin, end):
    trend = []
    resp = []
    trade = {
        'status': None,
        'p_high': None,
        'p_low': None,
        'resp': None
    }
    percent = 0.70

    for i in ['high', 'close', 'low']:
            model_path = './models/' + coin + '/' + min_ + '/' + ep + '/' + i + '/lstm_model.h5'
            df_name = './historical_data/' + coin + '/' + coin + min_ + i + '.csv'
            data = None
            data = insert_data(df_name, False, int(end[0]))#,  # Predict
            X, y, data = extract_data(np.array(data['close']), data)
            X, y = shape_data(X, y, timesteps=8)
     
            pred = Pred(data, y, X, model_path)
            status = pred.status(min_, percent, False)

            info(min_, i, coin, ep, X)
            
            resp.append(pred.get_resp())
            trend.append([status[0], status[1]])
            print('============================================================')
    
    calculate_trade(trend, trade, percent, resp)
    print(trade)

    if (trade['status'] != None):
        f1 = open("./out/out" + min_ + '_in' + ".txt", "a")
        f1.write(str(trade['status']) + ',' + str(trade['p_high']) + ',' + str(trade['p_low']) + '\n')
        f1.close()
        f2 = open("./out/out" + min_ + '_in_resp' + ".txt", "a")
        f2.write(str(trade) + '\n')
        f2.close()

    f3 = open("./out/out" + min_ + '_all' + ".txt", "a")
    f3.write(str(trade) + '\n')
    f3.close()

    # values = (trend[1][2]).split(' ')
    # if((float(values[0]) > percent) | (float(values[1]) >  percent)):
    #     f.write(str(trend) + ' call' + '\n')
    # else:
    #     f.write(str(trend) + '\n')



if __name__ == '__main__':
    args = sys.argv[1:]
    flag = 1
    min_ = '60' # 60 240
    prep = 'close' # close high loew # treining
    coin = 'EURUSD' # EURUSD BTCUSD
    ep = '120EP' # sys.argv[1:][0] # 10EP 30EP 60EP 120EP 240EP

    if (flag == 1):
        data = None
        df_name = './historical_data/' + coin + '/' + coin + min_ + prep + '.csv'
        data = insert_data(df_name) # Test/Train
        X, y, data = extract_data(np.array(data['close']), data)
        print(y.shape)
        X, y = shape_data(X, y, timesteps=8)
        X_train, X_test, y_train, y_test = adjust_data(X, y)
        # y_train, y_test = to_categorical(y_train, 2), to_categorical(y_test, 2)
        # info(min_, prep, coin, ep, X)
        # model = build_model()
        # model.fit(X_train, y_train, epochs=120, batch_size=8, shuffle=True, validation_data=(X_test, y_test), verbose=1)
        # model.save('models/'  + coin + '/' + min_ + '/lstm_model.h5')
    else:
        predict(min_, ep, coin, args)
        