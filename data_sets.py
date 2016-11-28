class Data_set:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y


class Data_sets:
    def __init__(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test
