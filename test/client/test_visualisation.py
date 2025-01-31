import respx
from httpx import Response
from matplotlib.figure import Figure

from matchbox.client._handler import url
from matchbox.client.visualisation import draw_resolution_graph


@respx.mock
def test_draw_resolution_graph(resolution_graph):
    respx.get(url("/report/resolutions")).mock(
        return_value=Response(200, content=resolution_graph.model_dump_json()),
    )

    plt = draw_resolution_graph()
    assert isinstance(plt, Figure)
