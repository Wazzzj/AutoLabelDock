from src.controllers.model import ModelController


class _FakeResult:
    def __init__(self):
        self.calls = []

    def plot(self, **kwargs):
        self.calls.append(kwargs)
        return "image-array"


def test_native_plot_uses_default_ultralytics_visualization():
    result = _FakeResult()

    plotted = ModelController._plot_result_for_display(result)

    assert plotted == "image-array"
    assert result.calls == [{}]
