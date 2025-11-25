import numpy as np
from copy import deepcopy
from modul_split_data import KFold

def accuracy_score(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, n_classes=None):
 
    if n_classes is None:
        n_classes = len(np.unique(y_true))
        
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if 0 <= true < n_classes and 0 <= pred < n_classes:
            cm[true, pred] += 1
    return cm

def _precision_recall_f1_support(cm):
  
    n_classes = cm.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    support = np.zeros(n_classes)
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        support[i] = cm[i, :].sum()
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if (precision[i] + recall[i]) > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1[i] = 0
            
    return precision, recall, f1, support

def classification_report(y_true, y_pred, target_names, digits=2):
    
    n_classes = len(target_names)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    precision, recall, f1, support = _precision_recall_f1_support(cm)
    
    width = max([len(name) for name in target_names] + [10])
    width = max(width, 12)
    
    head_fmt = '{:>{width}s}  {:>9}  {:>9}  {:>9}  {:>9}\n'
    report = head_fmt.format('', 'precision', 'recall', 'f1-score', 'support', width=width)
    report += "\n"
    
    row_fmt = '{:>{width}s}  {:>9.{digits}f}  {:>9.{digits}f}  {:>9.{digits}f}  {:>9}\n'
    
    for i, name in enumerate(target_names):
        report += row_fmt.format(name, precision[i], recall[i], f1[i], int(support[i]), width=width, digits=digits)
    
    report += "\n"
    
    accuracy = accuracy_score(y_true, y_pred)
    total_support = int(np.sum(support))
    
    macro_prec = np.mean(precision)
    macro_rec = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    if total_support > 0:
        weighted_prec = np.average(precision, weights=support)
        weighted_rec = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
    else:
        weighted_prec = weighted_rec = weighted_f1 = 0

    report += '{:>{width}s}  {:>9}  {:>9}  {:>9.{digits}f}  {:>9}\n'.format('accuracy', '', '', accuracy, total_support, width=width, digits=digits)
    
    report += row_fmt.format('macro avg', macro_prec, macro_rec, macro_f1, total_support, width=width, digits=digits)
    
    report += row_fmt.format('weighted avg', weighted_prec, weighted_rec, weighted_f1, total_support, width=width, digits=digits)
    
    return report

def cross_val_score(estimator, X, y, cv=5):

    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kfold.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model_copy = deepcopy(estimator)
        
        model_copy.fit(X_train_fold, y_train_fold)
        
        y_pred = model_copy.predict(X_val_fold)
        score = accuracy_score(y_val_fold, y_pred)
        scores.append(score)
        
    return np.array(scores)