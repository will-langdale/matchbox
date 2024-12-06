from matchbox.client.visualisation import draw_resolution_graph
from matplotlib.figure import Figure


def test_draw_resolution_graph():
    plt = draw_resolution_graph()
    assert isinstance(plt, Figure)
