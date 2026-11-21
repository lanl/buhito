"""Public graphlet imports share the canonical implementations."""

def test_public_graphlet_implementations():
    import buhito
    from buhito.breadth_graphlet import generate_subgraphs_breadthwise
    from buhito.depth_graphlet import generate_subgraphs_depthwise
    assert buhito.generate_subgraphs_breadthwise is generate_subgraphs_breadthwise
    assert buhito.generate_subgraphs_depthwise is generate_subgraphs_depthwise
