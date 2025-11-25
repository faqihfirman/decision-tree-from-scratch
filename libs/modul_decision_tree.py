import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, criterion='entropy', max_depth=100, min_samples_split=2, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        
    def _calculate_entropy(self, y):
        class_counts = np.bincount(y)
        probabilities = class_counts[class_counts > 0] / len(y)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _calculate_information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        
        return self._calculate_entropy(parent) - (
            weight_left * self._calculate_entropy(left_child) +
            weight_right * self._calculate_entropy(right_child)
        )
    
    def _split_data(self, X, y, feature_idx, threshold):
        left_indices = np.where(X[:, feature_idx] <= threshold)[0]
        right_indices = np.where(X[:, feature_idx] > threshold)[0]
        return left_indices, right_indices
    
    def _find_best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            possible_thresholds = np.unique(feature_values)
            
            for threshold in possible_thresholds:
                left_indices, right_indices = self._split_data(X, y, feature_idx, threshold)
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                gain = self._calculate_information_gain(y, y[left_indices], y[right_indices])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            return Node(value=self._most_common_label(y))
        
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            return Node(value=self._most_common_label(y))
        
        left_indices, right_indices = self._split_data(X, y, best_feature, best_threshold)
        
        if (len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf):
            return Node(value=self._most_common_label(y))
        
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(best_feature, best_threshold, left_child, right_child)
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        self.tree_ = self._build_tree(X, y)
        self._calculate_feature_importances(X, y)
        return self
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.tree_) for x in X])
    
    def _calculate_feature_importances(self, X, y):
        importances = np.zeros(self.n_features_in_)
        
        def calculate_importance(node, X, y):
            if node.is_leaf_node():
                return
            
            left_indices, right_indices = self._split_data(X, y, node.feature, node.threshold)
            
            if len(left_indices) > 0 and len(right_indices) > 0:
                gain = self._calculate_information_gain(y, y[left_indices], y[right_indices])
                importances[node.feature] += gain * len(y)
            
            if not node.left.is_leaf_node():
                calculate_importance(node.left, X[left_indices], y[left_indices])
            if not node.right.is_leaf_node():
                calculate_importance(node.right, X[right_indices], y[right_indices])
        
        calculate_importance(self.tree_, X, y)
        
        total = np.sum(importances)
        if total > 0:
            importances = importances / total
        
        self.feature_importances_ = importances
    
    def get_depth(self):
        def _get_depth(node):
            if node.is_leaf_node():
                return 0
            return 1 + max(_get_depth(node.left), _get_depth(node.right))
        return _get_depth(self.tree_)
    
    def get_n_leaves(self):
        def _count_leaves(node):
            if node.is_leaf_node():
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)
        return _count_leaves(self.tree_)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)