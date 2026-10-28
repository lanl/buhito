from collections import Counter

import networkx as nx

from buhito.census_graphlet import take_census


def test_petersen_graph():
    g = nx.petersen_graph()
    
    # unlabelled, all graphlets
    graphlets = take_census(g, graphlet_sizes = [x for x in range(1,10)] )
    assert set({(len(info), number) for info, number in graphlets.items()}) == set({ (9, 10), (3, 30), (6, 15), (7, 20), (6, 10), (8, 15), (7, 30), (5, 60), (8, 30), (4, 10), (6, 60), (4, 60), (1, 10), (7, 60), (5, 12), (2, 15) })  

    # count only some graphlets
    graphlets = take_census(g, subtree_starts = [0])
    assert Counter({frozenset({(0, frozenset({1})), (1, frozenset({0}))}): 3}) == graphlets
    
        # with a vertex coloring
    graphlets = take_census(g, vertex_coloring = {1 : "one"}, graphlet_sizes = [3])
    #print(graphlets, "\n")
    assert graphlets == Counter({(frozenset({(2, frozenset({0, 1})), (0, frozenset({2})), (1, frozenset({2}))}), ((0, 'unlabelled'), (1, 'unlabelled'), (2, 'unlabelled'))): 21, (frozenset({(0, frozenset({1})), (2, frozenset({1})), (1, frozenset({0, 2}))}), ((0, 'one'), (1, 'unlabelled'), (2, 'unlabelled'))): 6, (frozenset({(2, frozenset({0})), (1, frozenset({0})), (0, frozenset({1, 2}))}), ((0, 'one'), (1, 'unlabelled'), (2, 'unlabelled'))): 3})
    
    # with edge coloring
    graphlets = take_census(g, edge_coloring = {frozenset({1,2}) : "one"}, graphlet_sizes = [1,2,3])
    assert graphlets == Counter({(frozenset({(0, frozenset({3})),
                     (1, frozenset({4})),
                     (2, frozenset({3, 4})),
                     (3, frozenset({0, 2})),
                     (4, frozenset({1, 2}))}),
          ((0, 'unlabelled'),
           (1, 'unlabelled'),
           (2, 'unlabelled'),
           (3, 'unlabelled'),
           (4, 'unlabelled'))): 26,
         (frozenset({(0, frozenset({2})),
                     (1, frozenset({2})),
                     (2, frozenset({0, 1}))}),
          ((0, 'unlabelled'), (1, 'unlabelled'), (2, 'unlabelled'))): 14,
         (frozenset({(0, frozenset())}), ((0, 'unlabelled'),)): 10,
         (frozenset({(0, frozenset({1, 2})),
                     (1, frozenset({0, 4})),
                     (2, frozenset({0})),
                     (3, frozenset({4})),
                     (4, frozenset({1, 3}))}),
          ((0, 'one'),
           (1, 'unlabelled'),
           (2, 'unlabelled'),
           (3, 'unlabelled'),
           (4, 'unlabelled'))): 4,
         (frozenset({(0, frozenset({1, 2})),
                     (1, frozenset({0})),
                     (2, frozenset({0}))}),
          ((0, 'one'), (1, 'unlabelled'), (2, 'unlabelled'))): 1})
    # with multiprocessing 
    graphlets = take_census(g, graphlet_sizes = [x for x in range(1,10)] , use_pool = True)
    assert set({(len(info), number) for info, number in graphlets.items()}) == set({ (9, 10), (3, 30), (6, 15), (7, 20), (6, 10), (8, 15), (7, 30), (5, 60), (8, 30), (4, 10), (6, 60), (4, 60), (1, 10), (7, 60), (5, 12), (2, 15) })  
