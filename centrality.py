import numpy as np
import networkx as nx
import random
import torch

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class DynamicCPNodeCount(metaclass=SingletonMeta):
    def __init__(self):
        return
        
    def setup(self, num_nodes, preservation_duration):
        self.node_counts = [0 for _ in range(num_nodes)]
        self.duration = preservation_duration + 1
    
    def add_new_preservation(self, processed_nodes):
        for node in processed_nodes:
            if self.node_counts[node] == 0:
                self.node_counts[node] += self.duration
        
    def update_node_count(self):
        for node in range(len(self.node_counts)):
            if self.node_counts[node] > 0:
                self.node_counts[node] -= 1
    
    def get_nodes_count(self):
        return self.node_counts
    
    def get_nodes_num(self):
        return len(self.node_counts)
    
def centrality_preservation(G, centrality, preservation_prcnt, adj, oriadj):
    node_occasions = DynamicCPNodeCount()
    
    if centrality == 'BC':
        centrality_map = nx.betweenness_centrality(G, k=100, normalized=True, weight=None, endpoints=False, seed=42)
    elif centrality == 'DC':
        centrality_map = nx.degree_centrality(G)
    elif centrality == 'PR':
        centrality_map = nx.pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, dangling=None)
    elif centrality == 'EC':
        centrality_map = nx.eigenvector_centrality(G, max_iter=10000, tol=1e-06, nstart=None, weight=None)
    elif centrality == 'CC':
        centrality_map = nx.closeness_centrality(G, u=None, distance=None, wf_improved=True)
    elif centrality == 'RAND':
        centrality_map = {}
        for i in range(G.number_of_nodes()):
            centrality_map[i] = random.random()
    elif centrality == 'STAT':
        top_indices = np.argsort(oriadj, axis=1)[:, -3:]
        
        for node in range(len(adj)):
            for top_index in top_indices[node]:
                adj[node][top_index] = oriadj[node][top_index]
                print(oriadj[node][top_index])
        return adj
    else:
        print("Centrality Type Undefined, No Preservation Done!")
        return adj
    non_zero_centrality = {key: value for key, value in centrality_map.items() if value >= 0}
    min_pcentile_value = np.percentile(list(non_zero_centrality.values()), preservation_prcnt)
    nodes_to_preserve = {key: value for key, value in centrality_map.items() if value >= min_pcentile_value}
    
    node_occasions.add_new_preservation(nodes_to_preserve)
    node_occasions.update_node_count()
    nodes_to_restore = [key for key in range(node_occasions.get_nodes_num()) if node_occasions.get_nodes_count()[key] > 0]

    # preserved = 0
    # preserved_idx = []
    # adj_copy = adj.copy()
    for node in nodes_to_restore:
        adj[node] = oriadj[node]
        # for j in range(len(G.nodes)):
        #     if not (adj[node][j] == oriadj[node][j]): 
        #         preserved += 1 
        #         preserved_idx.append((node, j))
    return adj         

def additive_centrality_preservation(G, centrality, preservation_prcnt, adj, oriadj, ac_selection):
    node_occasions = DynamicCPNodeCount()

    centrality_functions = {
        "EC": compute_eigenvector_centrality,
        "BC": compute_betweenness_centrality,
        "DC": compute_degree_centrality,
        "CC": compute_closeness_centrality
    }
    
    centrality_codes = centrality.split('_')

    # compute additive centrality
    centrality_values = sum(centrality_functions[code](G, ac_selection) for code in centrality_codes if code in centrality_functions)

    # Update the dictionary with scaled values
    centrality_map = {node: val for node, val in enumerate(centrality_values)}

    non_zero_centrality = {key: value for key, value in centrality_map.items() if value >= 0}
    min_pcentile_value = np.percentile(list(non_zero_centrality.values()), preservation_prcnt)
    nodes_to_preserve = {key: value for key, value in centrality_map.items() if value >= min_pcentile_value}
    
    node_occasions.add_new_preservation(nodes_to_preserve)
    node_occasions.update_node_count()
    nodes_to_restore = [key for key in range(node_occasions.get_nodes_num()) if node_occasions.get_nodes_count()[key] > 0]

    for node in nodes_to_restore:
        adj[node] = oriadj[node]
        
    return adj 

def compute_closeness_centrality(G, ac_select):
    closeness_centrality = nx.closeness_centrality(G)
    return scaling_factor_selection(closeness_centrality, ac_select)
    
def compute_degree_centrality(G, ac_select):
    degree_centrality = nx.degree_centrality(G)
    return scaling_factor_selection(degree_centrality, ac_select)
    
def compute_betweenness_centrality(G, ac_select):
    betweenness_centrality = nx.betweenness_centrality(G, k=100, normalized=True, weight=None, endpoints=False, seed=42)
    return scaling_factor_selection(betweenness_centrality, ac_select)

def compute_eigenvector_centrality(G, ac_select):
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=10000, tol=1e-06, nstart=None, weight=None)
    return scaling_factor_selection(eigenvector_centrality, ac_select)

def scaling_factor_selection(centrality_map, ac_select):
    centrality_values = np.array(list(centrality_map.values()))

    # If ac_select is "minmax", apply Min-Max scaling
    if ac_select == "minmax":
        
        # Apply Min-Max scaling
        min_val = centrality_values.min()
        max_val = centrality_values.max()
        scaled_values = (centrality_values - min_val) / (max_val - min_val)
        
    elif ac_select == "zscore":
        # Perform Z-score normalization
        mean_val = centrality_values.mean()
        std_val = centrality_values.std()
        zscore_values = (centrality_values - mean_val) / std_val
        
        # Apply Min-Max scaling to the Z-score normalized values
        min_val = zscore_values.min()
        max_val = zscore_values.max()
        scaled_values = (zscore_values - min_val) / (max_val - min_val)
        
    # scaled_centrality = {node: scaled_values[i] for i, node in enumerate(centrality_map.keys())}
        
    return scaled_values
    

def dynamic_additive_centrality_preservation(G, scale_factors, centrality, preservation_prcnt, adj, oriadj, device):
    node_occasions = DynamicCPNodeCount()
    
    centrality_functions = {
        "BC": plain_betweenness_centrality,
        "CC": plain_closeness_centrality,
        "DC": plain_degree_centrality,
        "EC": plain_eigenvector_centrality
    }

    for i, factor in enumerate(scale_factors):
        print("scale factor:", i, factor)

    centrality_codes = centrality.split('_')

    # compute additive centrality
    centrality_values = sum(centrality_functions[code](G).to(device) * scale_factors[i] for i, code in enumerate(centrality_codes) if code in centrality_functions)

    # Update the dictionary with scaled values
    centrality_map = {node: val for node, val in enumerate(centrality_values)}

    non_zero_centrality = {key: value for key, value in centrality_map.items() if value >= 0}
    min_pcentile_value = np.percentile(list(non_zero_centrality.values()), preservation_prcnt)
    nodes_to_preserve = {key: value for key, value in centrality_map.items() if value >= min_pcentile_value}
    
    node_occasions.add_new_preservation(nodes_to_preserve)
    node_occasions.update_node_count()
    nodes_to_restore = [key for key in range(node_occasions.get_nodes_num()) if node_occasions.get_nodes_count()[key] > 0]

    for node in nodes_to_restore:
        adj[node] = oriadj[node]
        
    return adj 

def plain_closeness_centrality(G):
    return torch.tensor(list(nx.closeness_centrality(G).values()))
    
def plain_degree_centrality(G):
    return torch.tensor(list(nx.degree_centrality(G).values()))
    
def plain_betweenness_centrality(G):
    return torch.tensor(list(nx.betweenness_centrality(G, k=100, normalized=True, weight=None, endpoints=False, seed=42).values()))
    
def plain_eigenvector_centrality(G):
    return torch.tensor(list(nx.eigenvector_centrality(G, max_iter=10000, tol=1e-06, nstart=None, weight=None).values()))
