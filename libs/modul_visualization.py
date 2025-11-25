import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx

class ManualTreeVisualizer:
    def __init__(self, tree_model, feature_names, class_names):
        self.tree = tree_model
        self.feature_names = feature_names
        self.class_names = class_names
        self.node_info = []
        self.root = tree_model.tree_

    def _count_samples(self, node, X, y):
        if node.is_leaf_node():
            return len(y)
        left_mask = X[:, node.feature] <= node.threshold
        return len(y)

    def _extract_tree_info(self, node, X, y, depth=0, parent_info="Root"):
        if node is None:
            return
        
        n_samples = len(y)
        info = {
            'depth': depth,
            'parent': parent_info,
            'is_leaf': node.is_leaf_node(),
            'samples': n_samples
        }

        if node.is_leaf_node():
            info['class'] = self.class_names[node.value]
            info['type'] = 'leaf'
        else:
            info['feature'] = self.feature_names[node.feature]
            info['threshold'] = node.threshold
            info['type'] = 'split'
            
            left_mask = X[:, node.feature] <= node.threshold
            X_left, y_left = X[left_mask], y[left_mask]
            X_right, y_right = X[~left_mask], y[~left_mask]
        
        self.node_info.append(info)

        if not node.is_leaf_node():
            self._extract_tree_info(node.left, X_left, y_left, depth + 1, 
                                   f"{info['feature']} <= {info['threshold']:.2f}")
            self._extract_tree_info(node.right, X_right, y_right, depth + 1, 
                                   f"{info['feature']} > {info['threshold']:.2f}")

    def plot_tree_structure(self, X, y, figsize=(20, 12), max_depth=None):
        self.node_info = []
        self._extract_tree_info(self.root, X, y)
        
        if max_depth is not None:
            self.node_info = [n for n in self.node_info if n['depth'] <= max_depth]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, max([n['depth'] for n in self.node_info]) + 2)
        ax.axis('off')
        
        split_color = '#E8F4F8'
        leaf_color = '#D4E6F1'
        x_offset = {}
        
        for i, node in enumerate(self.node_info):
            depth = node['depth']
            if depth not in x_offset:
                x_offset[depth] = 0
            
            x = x_offset[depth]
            y = -depth
            x_offset[depth] += 2
            
            if node['type'] == 'leaf':
                box_color = leaf_color
                text = f"Class: {node['class']}\nSamples: {node['samples']}"
            else:
                box_color = split_color
                text = f"{node['feature']}\n<= {node['threshold']:.2f}\nSamples: {node['samples']}"
            
            bbox = FancyBboxPatch((x - 0.8, y - 0.3), 1.6, 0.6,
                                 boxstyle="round,pad=0.1", 
                                 edgecolor='black', 
                                 facecolor=box_color,
                                 linewidth=2)
            ax.add_patch(bbox)
            ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
        
        plt.title('Decision Tree Structure', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig

    def plot_tree_graph(self, X, y, figsize=(16, 12), max_depth=None):
        G = nx.DiGraph()
        root_label = self._build_graph(self.root, X, y, G, parent=None, depth=0, max_depth=max_depth)
        
        fig, ax = plt.subplots(figsize=figsize)
        pos = self._hierarchy_pos(G, root_label)
        
        node_colors = []
        for node in G.nodes():
            if isinstance(node, str) and 'Class:' in node:
                node_colors.append('#90EE90')
            else:
                node_colors.append('#87CEEB')
        
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000,
               font_size=9, font_weight='bold', arrows=True, arrowsize=20,
               edge_color='gray', linewidths=2, ax=ax)
        
        plt.title('Decision Tree Graph Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def _build_graph(self, node, X, y, G, parent, depth, max_depth):
        if node is None or (max_depth and depth > max_depth):
            return None
        
        n_samples = len(y)
        
        if node.is_leaf_node():
            label = f"Class: {self.class_names[node.value]}\n({n_samples})"
        else:
            label = f"{self.feature_names[node.feature]}\n<= {node.threshold:.2f}\n({n_samples})"
        
        G.add_node(label)
        
        if parent is not None:
            G.add_edge(parent, label)
        
        if not node.is_leaf_node():
            left_mask = X[:, node.feature] <= node.threshold
            X_left, y_left = X[left_mask], y[left_mask]
            X_right, y_right = X[~left_mask], y[~left_mask]
            
            self._build_graph(node.left, X_left, y_left, G, label, depth + 1, max_depth)
            self._build_graph(node.right, X_right, y_right, G, label, depth + 1, max_depth)
        
        return label

    def _hierarchy_pos(self, G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        def _hierarchy_pos_helper(G, node, left, right, pos, parent, parsed, level):
            if node not in parsed:
                parsed.append(node)
                neighbors = list(G.neighbors(node))
                if len(neighbors) == 0:
                    pos[node] = ((left + right) / 2, -level)
                else:
                    dx = (right - left) / len(neighbors)
                    nextx = left
                    for neighbor in neighbors:
                        pos = _hierarchy_pos_helper(G, neighbor, nextx, nextx + dx,
                                                   pos, node, parsed, level + 1)
                        nextx += dx
                    pos[node] = ((left + right) / 2, -level)
            return pos
        
        parsed = []
        pos = {}
        pos = _hierarchy_pos_helper(G, root, 0, width, pos, None, parsed, 0)
        return pos

    def plot_decision_paths(self, X_sample, y_sample=None, figsize=(12, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        path = []
        node = self.root
        
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
            ax.text(0.5, y_pos - i, decision, ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax.barh(0, 1, color='#2ecc71', alpha=0.6, edgecolor='black')
        pred_text = f"Prediction: {prediction}"
        if y_sample is not None:
            actual = self.class_names[y_sample]
            pred_text += f"\nActual: {actual}"
            pred_text += f"\n{'✓ Correct' if prediction == actual else '✗ Incorrect'}"
        ax.text(0.5, 0, pred_text, ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(path) + 0.5)
        ax.axis('off')
        ax.set_title('Decision Path Visualization', fontsize=14, fontweight='bold', pad=20)
        
        left_patch = mpatches.Patch(color=colors[0], alpha=0.6, label='Left (≤)')
        right_patch = mpatches.Patch(color=colors[1], alpha=0.6, label='Right (>)')
        pred_patch = mpatches.Patch(color='#2ecc71', alpha=0.6, label='Prediction')
        ax.legend(handles=[left_patch, right_patch, pred_patch], loc='upper right', fontsize=10)
        
        plt.tight_layout()
        return fig

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
                                    rule_conditions + [left_condition], depth + 1)
                
                right_condition = f"{self.feature_names[node.feature]} > {node.threshold:.2f}"
                _extract_rules_helper(node.right, X_right, y_right,
                                     rule_conditions + [right_condition], depth + 1)
        
        _extract_rules_helper(self.root, X, y, [], 0)
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

    def plot_feature_usage(self, figsize=(10, 6)):
        feature_count = {name: 0 for name in self.feature_names}
        
        def _count_features(node):
            if node is None or node.is_leaf_node():
                return
            feature_count[self.feature_names[node.feature]] += 1
            _count_features(node.left)
            _count_features(node.right)
        
        _count_features(self.root)
        
        fig, ax = plt.subplots(figsize=figsize)
        features = list(feature_count.keys())
        counts = list(feature_count.values())
        
        bars = ax.bar(features, counts, color='steelblue', edgecolor='black', alpha=0.7)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(count)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Usage Count', fontsize=12, fontweight='bold')
        ax.set_title('Feature Usage in Decision Tree', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig