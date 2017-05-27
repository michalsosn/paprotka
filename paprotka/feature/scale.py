class StandardScaler:
    def __init__(self):
        self.mean = []
        self.scale = []
        self.scale_nonzero = []

    def fit(self, features):
        self.mean = features.mean(0)
        self.scale = features.std(0)
        self.scale_nonzero = self.scale != 0

    def transform(self, features):
        scaled = (features - self.mean)
        scaled[:, self.scale_nonzero] /= self.scale[self.scale_nonzero]
        return scaled

    def fit_transform(self, features):
        self.fit(features)
        return self.transform(features)
