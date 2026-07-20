from src.engine.trainer import TrainConfig
from src.utils.workers import TrainWorker


class _FakeTrainer:
    cancelled = False

    def train(self, config, on_epoch_end=None):
        pass

    def get_best_metrics(self):
        return {"metric": 1.0}


class _OrderCheckingTrainWorker(TrainWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.released = False

    def _release_trainer(self, trainer) -> None:
        self.released = True


def test_train_worker_emits_finished_after_resource_release():
    config = TrainConfig(data_yaml="data.yaml", model="model.pt", task="detect")
    worker = _OrderCheckingTrainWorker(config, trainer_cls=_FakeTrainer)
    events = []

    worker.finished_ok.connect(
        lambda metrics: events.append(("finished", worker.released, metrics))
    )

    worker.run()

    assert events == [("finished", True, {"metric": 1.0})]
