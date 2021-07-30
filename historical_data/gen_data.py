import pandas as pd
import numpy as np

# for j in ["03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13"]:
#     data = pd.read_csv('../data/EURUSD/DAT_MT_EURUSD_M1_20' + j + '.csv', names=['time', 'data_vector', 'b', 'c', 'd', 'e'])

#     data['dates'] = data.index +  data['time']
#     data['dates'] = [i.replace('.', '').replace(':', '') for i in data['dates']]
#     data = data.drop(columns=['time', 'e', 'data_vector', 'c', 'd'])
#     data = data.set_index(pd.to_datetime(data['dates'], format='%Y%m%d%H%M')).drop(columns=['dates'])

#     df = np.array([])
#     flag = True
#     time = np.array([0])
#     end_avg = None
#     start_avg = None
#     data_vector = []

#     for i in range(len(data)):
#         if (data.iloc[i:i+1, :].index.minute.isin(time)):
#             t = str(data.iloc[i:i+1, :].index.values)
#             t = (t[2:21]).replace('T', ' ')
#             aux = np.append(data.iloc[i:i+1, :]['b'], t)
#             data_vector.append(aux)
#             flag = False
#         elif (data.iloc[i:i+1, :].index.minute.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) & (flag)):
#             t = str(data.iloc[i:i+1, :].index.values)
#             t = (t[:16] + '00' + t[18:21]).replace('T', ' ').replace("['", '')
#             start_avg = data.iloc[i:i+1, :]['b']
#             aux = np.append(((start_avg.values + end_avg.values) / 2), t)
#             data_vector.append(aux)
#             flag = False
#         if (data.iloc[i:i+1, :].index.minute.isin([40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])):
#             end_avg = data.iloc[i:i+1, :]['b']
#             flag = True

#     print(i)
#     data = pd.DataFrame(data_vector, columns=['max', 'dates'])
#     data = data.set_index('dates')
#     data.to_csv('EURUSD20' + str(j) + '.csv')


data = pd.read_csv('EURUSD2003.csv')
for i in ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13']:
    data_ = pd.read_csv('EURUSD20' + i + '.csv')
    data = data.append(data_)

data['dates'] = [ str(i)[:-3] for i in data['dates']]


d_aux = pd.read_csv('EURUSD2014.csv', sep='\t', names=['dates', 'dx','max','min','close','ax'])
df = d_aux.iloc[:, :1]
df = df.join(d_aux['max'])
data = data.append(df)
 


data.to_csv('EURUSD60.csv', index=False)
