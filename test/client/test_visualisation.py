from matchbox.client.visualisation import draw_model_tree
from matplotlib.figure import Figure


def test_draw_resolution_graph():
    plt = draw_model_tree()
    assert isinstance(plt, Figure)
