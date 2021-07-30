import pandas as pd
import numpy as np

class Sp500():
    def __init__(self, sp500):
        # Compute the logarithmic returns using the Closing price 
        sp500['Log_Ret_1d']=np.log(sp500['close'] / sp500['close'].shift(1))
        # Compute logarithmic returns using the pandas rolling mean function
        sp500['Log_Ret_1w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=5).sum()
        sp500['Log_Ret_2w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=10).sum()
        sp500['Log_Ret_3w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=15).sum()
        sp500['Log_Ret_4w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=20).sum()
        sp500['Log_Ret_8w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=40).sum()
        sp500['Log_Ret_12w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=60).sum()
        sp500['Log_Ret_16w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=80).sum()
        sp500['Log_Ret_20w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=100).sum()
        sp500['Log_Ret_24w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=120).sum()
        sp500['Log_Ret_28w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=140).sum()
        sp500['Log_Ret_32w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=160).sum()
        sp500['Log_Ret_36w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=180).sum()
        sp500['Log_Ret_40w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=200).sum()
        sp500['Log_Ret_44w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=220).sum()
        sp500['Log_Ret_48w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=240).sum()
        sp500['Log_Ret_52w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=260).sum()
        sp500['Log_Ret_56w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=280).sum()
        sp500['Log_Ret_60w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=300).sum()
        sp500['Log_Ret_64w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=320).sum()
        sp500['Log_Ret_68w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=340).sum()
        sp500['Log_Ret_72w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=360).sum()
        sp500['Log_Ret_76w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=380).sum()
        sp500['Log_Ret_80w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=400).sum()

        # Compute Volatility using the pandas rolling standard deviation function
        sp500['Vol_1w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=5).std()*np.sqrt(5)
        sp500['Vol_2w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=10).std()*np.sqrt(10)
        sp500['Vol_3w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=15).std()*np.sqrt(15)
        sp500['Vol_4w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=20).std()*np.sqrt(20)
        sp500['Vol_8w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=40).std()*np.sqrt(40)
        sp500['Vol_12w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=60).std()*np.sqrt(60)
        sp500['Vol_16w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=80).std()*np.sqrt(80)
        sp500['Vol_20w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=100).std()*np.sqrt(100)
        sp500['Vol_24w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=120).std()*np.sqrt(120)
        sp500['Vol_28w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=140).std()*np.sqrt(140)
        sp500['Vol_32w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=160).std()*np.sqrt(160)
        sp500['Vol_36w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=180).std()*np.sqrt(180)
        sp500['Vol_40w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=200).std()*np.sqrt(200)
        sp500['Vol_44w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=220).std()*np.sqrt(220)
        sp500['Vol_48w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=240).std()*np.sqrt(240)
        sp500['Vol_52w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=260).std()*np.sqrt(260)
        sp500['Vol_56w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=280).std()*np.sqrt(280)
        sp500['Vol_60w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=300).std()*np.sqrt(300)
        sp500['Vol_64w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=320).std()*np.sqrt(320)
        sp500['Vol_68w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=340).std()*np.sqrt(340)
        sp500['Vol_72w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=360).std()*np.sqrt(360)
        sp500['Vol_76w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=380).std()*np.sqrt(380)
        sp500['Vol_80w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=400).std()*np.sqrt(400)

        # Compute Volumes using the pandas rolling mean function
        sp500['Volume_1w']=pd.Series(sp500['Volume']).rolling(window=5).mean()
        sp500['Volume_2w']=pd.Series(sp500['Volume']).rolling(window=10).mean()
        sp500['Volume_3w']=pd.Series(sp500['Volume']).rolling(window=15).mean()
        sp500['Volume_4w']=pd.Series(sp500['Volume']).rolling(window=20).mean()
        sp500['Volume_8w']=pd.Series(sp500['Volume']).rolling(window=40).mean()
        sp500['Volume_12w']=pd.Series(sp500['Volume']).rolling(window=60).mean()
        sp500['Volume_16w']=pd.Series(sp500['Volume']).rolling(window=80).mean()
        sp500['Volume_20w']=pd.Series(sp500['Volume']).rolling(window=100).mean()
        sp500['Volume_24w']=pd.Series(sp500['Volume']).rolling(window=120).mean()
        sp500['Volume_28w']=pd.Series(sp500['Volume']).rolling(window=140).mean()
        sp500['Volume_32w']=pd.Series(sp500['Volume']).rolling(window=160).mean()
        sp500['Volume_36w']=pd.Series(sp500['Volume']).rolling(window=180).mean()
        sp500['Volume_40w']=pd.Series(sp500['Volume']).rolling(window=200).mean()
        sp500['Volume_44w']=pd.Series(sp500['Volume']).rolling(window=220).mean()
        sp500['Volume_48w']=pd.Series(sp500['Volume']).rolling(window=240).mean()
        sp500['Volume_52w']=pd.Series(sp500['Volume']).rolling(window=260).mean()
        sp500['Volume_56w']=pd.Series(sp500['Volume']).rolling(window=280).mean()
        sp500['Volume_60w']=pd.Series(sp500['Volume']).rolling(window=300).mean()
        sp500['Volume_64w']=pd.Series(sp500['Volume']).rolling(window=320).mean()
        sp500['Volume_68w']=pd.Series(sp500['Volume']).rolling(window=340).mean()
        sp500['Volume_72w']=pd.Series(sp500['Volume']).rolling(window=360).mean()
        sp500['Volume_76w']=pd.Series(sp500['Volume']).rolling(window=380).mean()
        sp500['Volume_80w']=pd.Series(sp500['Volume']).rolling(window=400).mean()
        
        # # Label data: Up (Down) if the the 1 month (≈ 21 trading days) logarithmic return increased (decreased)
        # sp500['Return_Label']=pd.Series(sp500['Log_Ret_1d']).shift(-21).rolling(window=21).sum()
        # # sp500['Label']=np.where(sp500['Return_Label'] > 0, 1, 0)

        # # Label data: Up (Down) if the the 1 month (≈ 21 trading days) logarithmic return increased (decreased)
        # sp500['Return_Label']=pd.Series(sp500['Log_Ret_1d']).shift(-21).rolling(window=21).sum()
        # sp500['Label']=np.where(sp500['Return_Label'] > 0, 1, 0)

        # Drop NA´s
        sp500=sp500.dropna("index")
        sp500=sp500.drop(['Open', 'High', 'Low', 'close', 'Volume'], axis=1)

        self.data = sp500

    def get_sp500(self):
        return np.array(self.data)