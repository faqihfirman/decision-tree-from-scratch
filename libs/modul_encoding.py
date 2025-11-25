import numpy as np


class LabelEncoder:

    def __init__(self):
        self.classes_ = None
        self.class_to_index = {}
        self.index_to_class = {}
    
    def fit(self, y):
        self.classes_ = np.unique(y)
        self.class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        self.index_to_class = {idx: label for idx, label in enumerate(self.classes_)}
        return self
    
    def transform(self, y):
        if self.classes_ is None:
            raise ValueError("LabelEncoder belum di-fit. Panggil fit() terlebih dahulu.")
        return np.array([self.class_to_index[label] for label in y])
    
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y):
        if self.classes_ is None:
            raise ValueError("LabelEncoder belum di-fit.")
        return np.array([self.index_to_class[idx] for idx in y])

