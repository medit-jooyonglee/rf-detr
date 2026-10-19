from rfdetr.util.utils import BestMetricHolder


def test_ema_checkpoint_improves_below_regular_best():
    holder = BestMetricHolder(use_ema=True)
    assert holder.update(0.60, 0)
    assert holder.update(0.58, 0, is_ema=True)
    assert holder.update(0.59, 1, is_ema=True)
    assert not holder.update(0.58, 2, is_ema=True)
    assert holder.summary()['ema_best_ep'] == 1
    assert holder.summary()['all_best_ep'] == 0
    assert holder.summary()['all_best_res'] == 0.60


def test_regular_checkpoint_improves_below_ema_best():
    holder = BestMetricHolder(use_ema=True)
    assert holder.update(0.60, 0, is_ema=True)
    assert holder.update(0.58, 0)
    assert holder.update(0.59, 1)
    assert not holder.update(0.59, 2)
    assert holder.summary()['regular_best_ep'] == 1
    assert holder.summary()['all_best_res'] == 0.60


def test_without_ema_tracks_overall_best():
    holder = BestMetricHolder()
    assert holder.update(0.60, 0)
    assert not holder.update(0.59, 1)
    assert holder.update(0.61, 2)
    assert holder.summary() == {'best_res': 0.61, 'best_ep': 2}
