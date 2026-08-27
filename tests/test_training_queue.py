from src.core.training_queue import TrainingQueue


def test_training_queue_runs_in_fifo_order():
    queue = TrainingQueue[str]()

    assert queue.enqueue("first") == 1
    assert queue.start_next() == "first"
    assert queue.enqueue("second") == 1
    assert queue.enqueue("third") == 2

    assert queue.finish_active() == ("first", True)
    assert queue.start_next() == "second"
    assert queue.finish_active() == ("second", True)
    assert queue.start_next() == "third"
    assert queue.finish_active() == ("third", True)
    assert len(queue) == 0


def test_single_job_batch_allows_auto_load_policy():
    queue = TrainingQueue[str]()
    queue.enqueue("only")
    queue.start_next()

    assert queue.finish_active() == ("only", False)


def test_clearing_waiting_jobs_keeps_multi_job_policy_for_active_job():
    queue = TrainingQueue[str]()
    queue.enqueue("active")
    queue.start_next()
    queue.enqueue("discarded")

    assert queue.clear_waiting() == ["discarded"]
    assert queue.finish_active() == ("active", True)
