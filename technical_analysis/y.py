
class Y():
    def __init__(self, close_data):
        self.y = []
        for i, index in zip(close_data, range(0, len(close_data))):
            if ((index + 1) < len(close_data)):
                if (i > close_data[index + 1]):
                    self.y.append(0)
                else:
                    self.y.append(1)

    def get_y(self):
        return self.y