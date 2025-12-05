import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


class TreeVisualizer:
    
    def __init__(self, tree_model, feature_names=None, class_names=None):
        self.tree = tree_model
        self.feature_names = feature_names
        self.class_names = class_names
        self.node_positions = {}
        self.leaf_counter = 0
        
    def _calculate_entropy(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def _get_node_color(self, y, is_leaf=False):
        if len(y) == 0:
            return '#ffffff'
        
        n_classes = len(self.class_names) if self.class_names is not None else len(np.unique(y))
        
        if n_classes > 0:
            class_counts = np.bincount(y, minlength=n_classes)
            dominant_class = np.argmax(class_counts)
        else:
            dominant_class = 0
        
        colors = [
            '#ff9999', '#66b3ff', '#99ff99', '#ffcc99', 
            '#c2c2f0', '#ffb3e6', '#c4e17f', '#76d7c4',
            '#f7c6c7', '#f9e79f', '#d2b4de', '#aed6f1',
            '#e59866', '#a9dfbf', '#fad7a0', '#f5b7b1',
            '#abebc6', '#d7bde2', '#a3e4d7', '#f9e79f'
        ]
        
        if is_leaf:
            color_idx = dominant_class % len(colors)
            return colors[color_idx]
        else:
            return '#e6f2ff'
    
    def _count_leaves(self, node):
        if node is None:
            return 0
        if node.is_leaf_node():
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)
    
    def _assign_positions(self, node, depth=0, pos_x=0, spacing=1.0):
        if node is None:
            return pos_x
        
        if node.is_leaf_node():
            x = pos_x
            y = -depth * 1.2
            self.node_positions[node.id] = (x, y)
            return pos_x + spacing
        else:
            left_x = self._assign_positions(node.left, depth + 1, pos_x, spacing)
            right_x = self._assign_positions(node.right, depth + 1, left_x, spacing)
            
            x = (pos_x + right_x - spacing) / 2.0
            y = -depth * 1.2
            self.node_positions[node.id] = (x, y)
            return right_x
    
    def _draw_node(self, ax, node, X, y, fontsize=7):
        if node is None:
            return
        
        x, y_pos = self.node_positions[node.id]
        
        entropy = self._calculate_entropy(y)
        samples = len(y)
        
        n_classes = len(self.class_names) if self.class_names is not None else 2
        value = [0] * n_classes
        if samples > 0:
            value = list(np.bincount(y, minlength=n_classes))
        
        if node.is_leaf_node():
            majority_class = np.argmax(value) if samples > 0 else 0
            class_name = self.class_names[majority_class] if self.class_names is not None else str(majority_class)
            
            text = f"entropy={entropy:.2f}\n"
            text += f"samples={samples}\n"
            text += f"class={class_name}"
            is_leaf = True
        else:
            feature_name = self.feature_names[node.feature] if self.feature_names else f"X[{node.feature}]"
            majority_class = np.argmax(value) if samples > 0 else 0
            class_name = self.class_names[majority_class] if self.class_names is not None else str(majority_class)
            
            text = f"{feature_name} <= {node.threshold:.2f}\n"
            text += f"entropy={entropy:.2f}\n"
            text += f"samples={samples}\n"
            text += f"class={class_name}"
            is_leaf = False
        
        box_color = self._get_node_color(y, is_leaf)
        
        bbox_props = dict(boxstyle="round,pad=0.5", fc=box_color, ec="black", alpha=1.0)
        
        ax.text(x, y_pos, text,
                ha='center', va='center',
                fontsize=fontsize,
                fontfamily='monospace',
                bbox=bbox_props,  
                zorder=3)

    
 
    def _draw_edges(self, ax, node, X, y, depth=0, max_depth=None):
        if node is None or node.is_leaf_node():
            return
        if max_depth is not None and depth >= max_depth:
            return
        
        x_parent, y_parent = self.node_positions[node.id]
        x_left, y_left = self.node_positions[node.left.id]
        x_right, y_right = self.node_positions[node.right.id]
        
        y_start = y_parent - 0.18
        y_end_left = y_left + 0.18
        y_end_right = y_right + 0.18
        
        mid_y = (y_start + y_end_left) / 2
        
        ax.plot([x_parent, x_parent], [y_start, mid_y],
                color='#666666', linewidth=1.2, zorder=1)
        ax.plot([x_parent, x_left], [mid_y, y_end_left],
                color='#666666', linewidth=1.2, zorder=1)
        
        ax.plot([x_parent, x_parent], [y_start, mid_y],
                color='#666666', linewidth=1.2, zorder=1)
        ax.plot([x_parent, x_right], [mid_y, y_end_right],
                color='#666666', linewidth=1.2, zorder=1)
        
        left_mask = X[:, node.feature] <= node.threshold
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]
        
        self._draw_edges(ax, node.left, X_left, y_left, depth + 1, max_depth)
        self._draw_edges(ax, node.right, X_right, y_right, depth + 1, max_depth)
    
    def _draw_tree_recursive(self, ax, node, X, y, depth, fontsize, max_depth=None):
        if node is None:
            return
     
        if max_depth is not None and depth > max_depth:
            return
        
        self._draw_node(ax, node, X, y, fontsize)
        
        if not node.is_leaf_node():
            left_mask = X[:, node.feature] <= node.threshold
            X_left, y_left = X[left_mask], y[left_mask]
            X_right, y_right = X[~left_mask], y[~left_mask]
            
            self._draw_tree_recursive(ax, node.left, X_left, y_left, depth + 1, fontsize, max_depth)
            self._draw_tree_recursive(ax, node.right, X_right, y_right, depth + 1, fontsize, max_depth)
    
    def plot_tree_graph(self, X, y, figsize=(24, 12), fontsize=7, max_depth=None):
        self.leaf_counter = 0
        self.node_positions = {}
        
        n_leaves = self._count_leaves(self.tree.tree_)
        spacing = 25.0 
        
        self._assign_positions(self.tree.tree_, depth=0, pos_x=0, spacing=spacing)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('auto')
        ax.axis('off')
        
        self._draw_edges(ax, self.tree.tree_, X, y, depth=0, max_depth=max_depth)
        self._draw_tree_recursive(ax, self.tree.tree_, X, y, 0, fontsize, max_depth=max_depth)
        
        if self.node_positions:
            all_x = [pos[0] for pos in self.node_positions.values()]
            all_y = [pos[1] for pos in self.node_positions.values()]
            
            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)
            
            x_margin = max(2.0, x_range * 0.1)
            y_margin = max(0.5, y_range * 0.15)
            
            ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
            ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        
        plt.title('Decision Tree Visualization', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_usage(self, figsize=(10, 6)):
        feature_count = {name: 0 for name in self.feature_names}
        
        def _count_features(node):
            if node is None or node.is_leaf_node():
                return
            feature_count[self.feature_names[node.feature]] += 1
            _count_features(node.left)
            _count_features(node.right)
        
        _count_features(self.tree.tree_)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        features = list(feature_count.keys())
        counts = list(feature_count.values())
        
        bars = ax.bar(features, counts, color='steelblue', edgecolor='black', alpha=0.7)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Usage Count', fontsize=12, fontweight='bold')
        ax.set_title('Feature Usage in Decision Tree', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def plot_decision_paths(self, X_sample, y_sample=None, figsize=(12, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        
        path = []
        node = self.tree.tree_
        
        while not node.is_leaf_node():
            feature_val = X_sample[node.feature]
            
            if feature_val <= node.threshold:
                decision = f"{self.feature_names[node.feature]} = {feature_val:.2f} <= {node.threshold:.2f}"
                path.append((decision, 'left'))
                node = node.left
            else:
                decision = f"{self.feature_names[node.feature]} = {feature_val:.2f} > {node.threshold:.2f}"
                path.append((decision, 'right'))
                node = node.right
        
        prediction = self.class_names[node.value]
        
        y_pos = len(path)
        colors = ['#3498db', '#e74c3c']
        
        for i, (decision, direction) in enumerate(path):
            color = colors[0] if direction == 'left' else colors[1]
            ax.barh(y_pos - i, 1, color=color, alpha=0.6, edgecolor='black')
            ax.text(0.5, y_pos - i, decision, ha='center', va='center',
                   fontsize=10, fontweight='bold')
        
        ax.barh(0, 1, color='#2ecc71', alpha=0.6, edgecolor='black')
        pred_text = f"Prediction: {prediction}"
        if y_sample is not None:
            actual = self.class_names[y_sample]
            pred_text += f"\nActual: {actual}"
            pred_text += f"\n{'✓ Correct' if prediction == actual else '✗ Incorrect'}"
        ax.text(0.5, 0, pred_text, ha='center', va='center',
               fontsize=12, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(path) + 0.5)
        ax.axis('off')
        ax.set_title('Decision Path Visualization', fontsize=14, fontweight='bold', pad=20)
        
        left_patch = mpatches.Patch(color=colors[0], alpha=0.6, label='Left (≤)')
        right_patch = mpatches.Patch(color=colors[1], alpha=0.6, label='Right (>)')
        pred_patch = mpatches.Patch(color='#2ecc71', alpha=0.6, label='Prediction')
        ax.legend(handles=[left_patch, right_patch, pred_patch], 
                 loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def extract_rules(self, X, y, max_depth=None):
        rules = []
        
        def _extract_rules_helper(node, X_node, y_node, rule_conditions, depth):
            if node is None or (max_depth and depth > max_depth):
                return
            
            n_samples = len(y_node)
            
            if node.is_leaf_node():
                rule = {
                    'conditions': rule_conditions.copy(),
                    'prediction': self.class_names[node.value],
                    'samples': n_samples
                }
                rules.append(rule)
            else:
                left_mask = X_node[:, node.feature] <= node.threshold
                X_left, y_left = X_node[left_mask], y_node[left_mask]
                X_right, y_right = X_node[~left_mask], y_node[~left_mask]
                
                left_condition = f"{self.feature_names[node.feature]} <= {node.threshold:.2f}"
                _extract_rules_helper(node.left, X_left, y_left,
                                    rule_conditions + [left_condition], 
                                    depth + 1)
                
                right_condition = f"{self.feature_names[node.feature]} > {node.threshold:.2f}"
                _extract_rules_helper(node.right, X_right, y_right,
                                     rule_conditions + [right_condition], 
                                     depth + 1)
        
        _extract_rules_helper(self.tree.tree_, X, y, [], 0)
        return rules
    
    def print_rules(self, X, y, max_depth=None, max_rules=10):
        rules = self.extract_rules(X, y, max_depth)
        
        print("=" * 80)
        print("DECISION RULES")
        print("=" * 80)
        print(f"Total Rules: {len(rules)}")
        print(f"Showing: {min(max_rules, len(rules))} rules")
        print("=" * 80)
        
        for i, rule in enumerate(rules[:max_rules], 1):
            print(f"\nRule #{i}:")
            print(f"  IF:")
            for condition in rule['conditions']:
                print(f"    - {condition}")
            print(f"  THEN:")
            print(f"    - Prediction: {rule['prediction']}")
            print(f"    - Samples: {rule['samples']}")
            print("-" * 80)