import itertools
from collections import deque
import copy
import networkx as nx
from apted import APTED, Config
from apted.helpers import Tree

from utils.kinematic_utils import to_DAG


# ref: https://g github.com/JoaoFelipe/apted


def find_root_node(G):
    root_node = None
    for node in G:
        if len(nx.descendants(G, node)) == 0:
            if root_node is None:
                root_node = node
                break
    return root_node


def bfs_traverse_topo(G, root_node):
    # G: direct graph: edge from child to parent
    num_node = len(G.nodes)
    bfs_dict = dict(nx.bfs_successors(G.reverse(), root_node))
    bfs_lists = []

    def bfs_backtrack(queue, visited, result):
        parent = queue.popleft()
        result.append(parent)
        if len(result) == num_node:
            bfs_lists.append(result)
            return
        if parent in bfs_dict:
            children = bfs_dict[parent]
            permute_children = itertools.permutations(children, len(children))
            for children in permute_children:
                queue_ = copy.deepcopy(queue)
                visited_ = copy.deepcopy(visited)
                result_ = copy.deepcopy(result)
                for child in children:
                    if child not in visited_:
                        visited_.add(child)
                        queue_.append(child)
                bfs_backtrack(queue_, visited_, result_)
        else:
            bfs_backtrack(queue, visited, result)

    visited = {root_node}
    queue = deque([root_node])
    bfs_backtrack(queue, visited, [])
    return bfs_lists


def get_node_attr(G, root_node):
    # G: directed graph
    topo = [root_node]
    topo.extend([t for (s, t) in nx.bfs_edges(G, root_node, reverse=True)])
    # nx.topological_sort not guarantee BFS order
    char_idx = 97
    value_dict = {}
    for idx, node in enumerate(topo):
        value_dict[node] = chr(char_idx + idx)
    return value_dict


def get_node_attr_list(G, root_node):
    # G: directed graph
    topo_lists = bfs_traverse_topo(G, root_node)
    value_dict_lists = []
    char_idx = 97
    for topo in topo_lists:
        value_dict = {}
        for idx, node in enumerate(topo):
            value_dict[node] = chr(char_idx + idx)
        value_dict_lists.append(value_dict)
    return value_dict_lists


def to_nested_tuple(T, root_node, traverse=True):
    """
    T : NetworkX graph, An directed graph object representing a tree.
    """
    def _make_tuple(graph, root, _parent, str_dict):
        """Recursively compute the nested tuple representation of the
        given rooted tree.

        ``_parent`` is the parent node of ``root`` in the supertree in
        which ``T`` is a subtree, or ``None`` if ``root`` is the root of
        the supertree. This argument is used to determine which
        neighbors of ``root`` are children and which is the parent.

        """
        # Get the neighbors of `root` that are not the parent node. We
        # are guaranteed that `root` is always in `T` by construction.
        children = set(graph[root]) - {_parent}
        node_str = str_dict[root]
        nested = f"{{{node_str}"
        for v in sorted(children, key=lambda node: str_dict[node]):
             nested += _make_tuple(graph, v, root, str_dict)
        nested += "}"
        return nested

    # Do some sanity checks on the input.
    if not nx.is_tree(T):
        raise nx.NotATree("provided graph is not a tree")

    T_undirected = T.to_undirected()
    if traverse:
        str_dicts = get_node_attr_list(T, root_node)
        str_list = []
        for str_dict in str_dicts:
            str_list.append(_make_tuple(T_undirected, root_node, None, str_dict))
        return str_list
    else:
        str_dict = get_node_attr(T, root_node)
        return _make_tuple(T_undirected, root_node, None, str_dict)


class CustomConfig(Config):
    def rename(self, node1, node2):
        return 0


def compute_ted(pred_edges_list, pred_root_node, gt_edges_list, gt_root_node, traverse=True, verbose=False):
    pred_graph = nx.from_edgelist(pred_edges_list, create_using=nx.Graph())
    pred_graph = to_DAG(pred_graph, pred_root_node)

    gt_graph = nx.from_edgelist(gt_edges_list, create_using=nx.DiGraph())
    if not traverse:
        gt_tree_str = to_nested_tuple(gt_graph, gt_root_node)
        pred_tree_str = to_nested_tuple(pred_graph, pred_root_node)

        tree1, tree2 = Tree.from_text(pred_tree_str), Tree.from_text(gt_tree_str)
        apted = APTED(tree1, tree2, CustomConfig())
        ted = apted.compute_edit_distance()
        if verbose:
            print("gt tree str", gt_tree_str)
            print("pred tree str", pred_tree_str)
            print("tree edit distance {}".format(ted))
    else:
        gt_tree_str_list = to_nested_tuple(gt_graph, gt_root_node, traverse=traverse)
        pred_tree_str_list = to_nested_tuple(pred_graph, pred_root_node, traverse=traverse)
        ted = 9999
        for gt_str in gt_tree_str_list:
            for pred_str in pred_tree_str_list:
                tree1, tree2 = Tree.from_text(pred_str), Tree.from_text(gt_str)
                apted = APTED(tree1, tree2, CustomConfig())
                dist = apted.compute_edit_distance()
                if dist < ted:
                    ted = dist
        if verbose:
            print("final tree edit distance {}".format(ted))
    return ted