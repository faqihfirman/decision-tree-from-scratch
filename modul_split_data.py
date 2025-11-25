import numpy as np

def manual_train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):

    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    
    if stratify is not None:
       
        unique_classes = np.unique(stratify)
        train_indices = []
        test_indices = []
        
        for cls in unique_classes:
            
            cls_indices = np.where(stratify == cls)[0]
            n_cls_test = int(len(cls_indices) * test_size)
            
            np.random.shuffle(cls_indices)
            
            test_indices.extend(cls_indices[:n_cls_test])
            train_indices.extend(cls_indices[n_cls_test:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        
    else:
       
        n_test = int(n_samples * test_size)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


class CrossValidation:

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
    
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            
            test_indices = indices[start:stop]
            
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            
            yield train_indices, test_indices
            current = stop
            
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits