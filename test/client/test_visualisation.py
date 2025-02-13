import pytest
from httpx import Response
from matplotlib.figure import Figure
from respx import MockRouter

from matchbox.client.visualisation import draw_resolution_graph
from matchbox.common.graph import ResolutionGraph


@pytest.mark.respx(base_url="http://localhost:8000")
def test_draw_resolution_graph(
    respx_mock: MockRouter, resolution_graph: ResolutionGraph
):
    respx_mock.get("/report/resolutions").mock(
        return_value=Response(200, content=resolution_graph.model_dump_json()),
    )

    plt = draw_resolution_graph()
    assert isinstance(plt, Figure)
