import networkx as nx

def set_edge_weight(G, wt):
    """Sets the edge weights in G to wt."""
    for _, _, e in G.edges(data=True):
        e["weight"] = wt


def cartesian_product(graphs):
    """Performs a cartesian product between a list of networkx graphs."""
    G = nx.cartesian_product(graphs[0], graphs[1])
    for i in range(2, len(graphs)):
        G = nx.cartesian_product(G, graphs[i])
    mapping = {}
    for node in G.nodes():
        mapping[node] = tuple(flatten(node))
    return nx.relabel_nodes(G, mapping)


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i