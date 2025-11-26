import uuid
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.id = str(uuid.uuid4())

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, random_state=None):

        self.criterion = criterion
        self.max_depth = max_depth if max_depth is not None else 1000
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        
    def _entropy(self, y):
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        probabilities = probabilities[probabilities > 0]
        
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _information_gain(self, parent, left_child, right_child):
        if len(left_child) == 0 or len(right_child) == 0:
            return 0.0
        
        n = len(parent)
        n_left = len(left_child)
        n_right = len(right_child)
        
        parent_entropy = self._entropy(parent)
        weighted_child_entropy = (n_left / n) * self._entropy(left_child) + \
                                (n_right / n) * self._entropy(right_child)
        
        return parent_entropy - weighted_child_entropy
    
    def _split(self, X, y, feature_idx, threshold):
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        return left_mask, right_mask
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i + 1]) / 2.0

                left_mask, right_mask = self._split(X, y, feature_idx, threshold)

                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                gain = self._information_gain(y, left_y, right_y)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _most_common_label(self, y):
        if len(y) == 0:
            return 0
        
        counter = Counter(y)
        most_common = counter.most_common()
        
        if len(most_common) == 0:
            return 0
        
        max_count = most_common[0][1]
        tied_labels = [label for label, count in most_common if count == max_count]
        
        return min(tied_labels)
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_mask, right_mask = self._split(X, y, best_feature, best_threshold)
        
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(best_feature, best_threshold, left_child, right_child)
    
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.int64)

        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        
        self.tree_ = self._build_tree(X, y, depth=0)
        
        self._calculate_feature_importances(X, y)
        
        return self
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        predictions = [self._traverse_tree(x, self.tree_) for x in X]
        return np.array(predictions, dtype=np.int64)
    
    def _calculate_feature_importances(self, X, y):
        importances = np.zeros(self.n_features_in_)
        total_samples = len(y)
        
        def traverse_and_calculate(node, X, y):
            if node.is_leaf_node():
                return
            
            left_mask, right_mask = self._split(X, y, node.feature, node.threshold)
            
            if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                gain = self._information_gain(y, left_y, right_y)
                
                weighted_gain = (len(y) / total_samples) * gain
                importances[node.feature] += weighted_gain

            if not node.left.is_leaf_node():
                traverse_and_calculate(node.left, X[left_mask], y[left_mask])
            if not node.right.is_leaf_node():
                traverse_and_calculate(node.right, X[right_mask], y[right_mask])
        
        traverse_and_calculate(self.tree_, X, y)
        
        total = np.sum(importances)
        if total > 0:
            importances = importances / total
        
        self.feature_importances_ = importances
    
    def get_depth(self):
        def _depth(node):
            if node.is_leaf_node():
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        
        return _depth(self.tree_)
    
    def get_n_leaves(self):
        def _count_leaves(node):
            if node.is_leaf_node():
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)
        
        return _count_leaves(self.tree_)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)