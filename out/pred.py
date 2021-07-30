import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from keras.utils import to_categorical

class Pred():
    def __init__(self, data, y, X, path='../models/lstm_model.h5'):
        self.data = data
        self.X = X
        self.y = y

        model = tf.keras.models.load_model(path)
        self.pred = model.predict(X)

    def get_pred(self):
        return self.pred

    def status(self, out='', perc=0.5, desc=True):

        pred_df = pd.DataFrame(self.pred, columns=['BAIXA', 'ALTA'])
        self.data = self.data.join(pred_df)
        self.data = self.data.set_axis(['LINHA', 'PRECO', 'ALTA_REAL', 'BAIXA_REAL', 'BAIXA', 'ALTA'], axis='columns')
        self.resp = self.data.iloc[-1:, 2:4].values[0]
        print('============================================================')
        print(self.data.iloc[-1:, :])
        aux = self.data.iloc[-1:, :]
        d = (aux['BAIXA'].values[0] > perc) & (aux['BAIXA_REAL'] == 1)
        u = (aux['ALTA'].values[0] > perc) & (aux['ALTA_REAL'] == 1)
        
        
        w = None
        if (u.values | d.values):
            w = True
        elif ((aux['BAIXA'].values[0] > perc) | (aux['ALTA'].values[0] > perc)):
            w = False
        else:
            w = None

        f = open("./out/out" + out + '_test' + ".txt", "a")
        f.write('-------------------------------------------------------------------------\n')
        f.write(str(self.data.iloc[-1:, 1:]) + ' - ' +  str(w)  +  '\n')
        f.close()
        
        [trash, price] = self.data.iloc[-1:, 1:2].to_string(index=False).split('\n')
        [trash, percent] = self.data.iloc[-1:, 4:].to_string(index=False).split('\n')

        return [price, percent]

    def get_resp(self):
        return self.resp

    def graf(self,inter_slope, stoch_rsi, dpo, cop, macd): 
        pred_n = []
        for i in self.pred[:]:
            v = str(i)[1:-1].replace(' 0.', ',0.').split(',')
            pred_n.append(v)

        df = pd.DataFrame(pred_n, columns=['UP', 'DOWN'])
        df['UP'] = df['UP'].astype(float)
        df['DOWN'] = df['DOWN'].astype(float)

        trace7 = go.Scatter(y=np.array(self.data['close'])[40:], name='data')
        trace0 = go.Scatter(y=np.array(df['UP'] + 0.90), name='UP')
        trace1 = go.Scatter(y=np.array(df['DOWN'] + 0.90), name='DOWN')
        trace2 = go.Scatter(y=inter_slope[40:], name='inter_slope')
        trace3 = go.Scatter(y=stoch_rsi[40:], name='stoch_rsi')
        trace4 = go.Scatter(y=dpo[40:], name='dpo')
        trace5 = go.Scatter(y=cop[40:], name='cop')
        trace6 = go.Scatter(y=macd[40:], name='macd')
        data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7]

        layout = go.Layout(
            title='Labels',
            yaxis=dict(
                title='USDT value'
            ),
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig)
