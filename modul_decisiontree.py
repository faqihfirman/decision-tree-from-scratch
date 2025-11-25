import numpy as np
from collections import Counter

class Node:
    """
    Representasi node dalam Decision Tree
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # Index fitur untuk split
        self.threshold = threshold      # Threshold value untuk split
        self.left = left               # Child node kiri
        self.right = right             # Child node kanan
        self.value = value             # Nilai prediksi jika leaf node

    def is_leaf_node(self):
        """Cek apakah node adalah leaf"""
        return self.value is not None


class DecisionTreeClassifier:
    """
    Decision Tree Classifier dengan Entropy Criterion
    
    Implementasi manual decision tree yang menggunakan entropy untuk 
    mengukur impurity dan information gain untuk memilih split terbaik.
    
    Parameters
    ----------
    criterion : str, default='entropy'
        Fungsi untuk mengukur kualitas split. Hanya mendukung 'entropy'.
    
    max_depth : int, default=100
        Kedalaman maksimum tree. None berarti unlimited.
    
    min_samples_split : int, default=2
        Jumlah minimum sampel yang diperlukan untuk melakukan split.
    
    min_samples_leaf : int, default=1
        Jumlah minimum sampel yang diperlukan di leaf node.
    
    Attributes
    ----------
    n_classes_ : int
        Jumlah kelas dalam target
    
    n_features_in_ : int
        Jumlah fitur saat training
    
    tree_ : Node
        Root node dari tree yang sudah dibangun
    
    feature_importances_ : array
        Importance dari setiap fitur
    """
    
    def __init__(self, criterion='entropy', max_depth=100, 
                 min_samples_split=2, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        
    def _calculate_entropy(self, y):
        """
        Menghitung entropy dari label y
        
        Formula: H(S) = -∑(p_i * log2(p_i))
        dimana p_i adalah proporsi sampel kelas i
        
        Parameters
        ----------
        y : array-like
            Label target
            
        Returns
        -------
        entropy : float
            Nilai entropy
        """
        # Hitung proporsi setiap kelas
        class_counts = np.bincount(y)
        probabilities = class_counts[class_counts > 0] / len(y)
        
        # Hitung entropy: -∑(p * log2(p))
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _calculate_information_gain(self, parent, left_child, right_child):
        """
        Menghitung Information Gain dari split
        
        Formula: IG = H(parent) - [weighted_avg(H(left), H(right))]
        
        Parameters
        ----------
        parent : array-like
            Label parent node
        left_child : array-like
            Label left child
        right_child : array-like
            Label right child
            
        Returns
        -------
        information_gain : float
            Nilai information gain
        """
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        
        # IG = H(parent) - weighted_avg(H(children))
        gain = self._calculate_entropy(parent) - (
            weight_left * self._calculate_entropy(left_child) +
            weight_right * self._calculate_entropy(right_child)
        )
        return gain
    
    def _split_data(self, X, y, feature_idx, threshold):
        """
        Split data berdasarkan fitur dan threshold
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data fitur
        y : array-like of shape (n_samples,)
            Label target
        feature_idx : int
            Index fitur untuk split
        threshold : float
            Nilai threshold untuk split
            
        Returns
        -------
        left_indices : array
            Indices untuk left child
        right_indices : array
            Indices untuk right child
        """
        left_indices = np.where(X[:, feature_idx] <= threshold)[0]
        right_indices = np.where(X[:, feature_idx] > threshold)[0]
        return left_indices, right_indices
    
    def _find_best_split(self, X, y):
        """
        Mencari split terbaik untuk node saat ini
        
        Iterasi semua fitur dan semua possible threshold,
        hitung information gain, pilih yang terbaik.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data fitur
        y : array-like of shape (n_samples,)
            Label target
            
        Returns
        -------
        best_feature : int
            Index fitur terbaik
        best_threshold : float
            Threshold terbaik
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        # Iterasi setiap fitur
        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            possible_thresholds = np.unique(feature_values)
            
            # Iterasi setiap possible threshold
            for threshold in possible_thresholds:
                left_indices, right_indices = self._split_data(
                    X, y, feature_idx, threshold
                )
                
                # Skip jika salah satu child kosong
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                # Hitung information gain
                gain = self._calculate_information_gain(
                    y, y[left_indices], y[right_indices]
                )
                
                # Update best split jika gain lebih besar
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """
        Membangun tree secara rekursif
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data fitur
        y : array-like of shape (n_samples,)
            Label target
        depth : int
            Kedalaman saat ini
            
        Returns
        -------
        node : Node
            Node yang sudah dibangun
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Kondisi stopping criteria
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            # Buat leaf node dengan majority class
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Cari best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        # Jika tidak ada split yang baik, buat leaf node
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Split data
        left_indices, right_indices = self._split_data(
            X, y, best_feature, best_threshold
        )
        
        # Cek min_samples_leaf
        if (len(left_indices) < self.min_samples_leaf or 
            len(right_indices) < self.min_samples_leaf):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Rekursif build left dan right subtree
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(best_feature, best_threshold, left_child, right_child)
    
    def _most_common_label(self, y):
        """
        Mengembalikan label yang paling sering muncul
        
        Parameters
        ----------
        y : array-like
            Label target
            
        Returns
        -------
        most_common : int
            Label yang paling sering muncul
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def fit(self, X, y):
        """
        Membangun decision tree classifier dari training set (X, y)
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator
        """
        # Convert to numpy array
        X = np.array(X)
        y = np.array(y)
        
        # Simpan informasi
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        
        # Build tree
        self.tree_ = self._build_tree(X, y)
        
        # Hitung feature importances
        self._calculate_feature_importances(X, y)
        
        return self
    
    def _traverse_tree(self, x, node):
        """
        Traverse tree untuk satu sampel
        
        Parameters
        ----------
        x : array-like
            Satu sampel data
        node : Node
            Node saat ini
            
        Returns
        -------
        prediction : int
            Predicted class
        """
        # Jika leaf node, return value
        if node.is_leaf_node():
            return node.value
        
        # Traverse ke left atau right child
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        """
        Predict class untuk X
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples untuk diprediksi
            
        Returns
        -------
        y : array of shape (n_samples,)
            Predicted classes
        """
        X = np.array(X)
        predictions = [self._traverse_tree(x, self.tree_) for x in X]
        return np.array(predictions)
    
    def _calculate_feature_importances(self, X, y):
        """
        Menghitung feature importances berdasarkan information gain
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        importances = np.zeros(self.n_features_in_)
        
        def calculate_importance(node, X, y):
            if node.is_leaf_node():
                return
            
            # Hitung information gain untuk split ini
            left_indices, right_indices = self._split_data(
                X, y, node.feature, node.threshold
            )
            
            if len(left_indices) > 0 and len(right_indices) > 0:
                gain = self._calculate_information_gain(
                    y, y[left_indices], y[right_indices]
                )
                importances[node.feature] += gain * len(y)
            
            # Rekursif untuk children
            if not node.left.is_leaf_node():
                calculate_importance(node.left, X[left_indices], y[left_indices])
            if not node.right.is_leaf_node():
                calculate_importance(node.right, X[right_indices], y[right_indices])
        
        calculate_importance(self.tree_, X, y)
        
        # Normalisasi
        total = np.sum(importances)
        if total > 0:
            importances = importances / total
        
        self.feature_importances_ = importances
    
    def get_depth(self):
        """
        Mengembalikan kedalaman maksimum tree
        
        Returns
        -------
        depth : int
            Kedalaman maksimum
        """
        def _get_depth(node):
            if node.is_leaf_node():
                return 0
            return 1 + max(_get_depth(node.left), _get_depth(node.right))
        
        return _get_depth(self.tree_)
    
    def get_n_leaves(self):
        """
        Mengembalikan jumlah leaf nodes
        
        Returns
        -------
        n_leaves : int
            Jumlah leaf nodes
        """
        def _count_leaves(node):
            if node.is_leaf_node():
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)
        
        return _count_leaves(self.tree_)
    
    def score(self, X, y):
        """
        Menghitung accuracy score
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels
            
        Returns
        -------
        score : float
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
