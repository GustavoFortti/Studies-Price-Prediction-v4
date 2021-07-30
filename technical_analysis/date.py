import numpy as np

class Date():
    def __init__(self, data):
        self.data = data
    
    def days(self):
        days = []
        for i in np.array(self.data.index):
            days.append(int(str(i)[8:10]))
        
        return days

    def hours(self):
        hours = []
        for i in np.array(self.data.index):
            hours.append(int(str(i)[11:-1].replace(':3', '5').replace(':', '')))
        
        return hours