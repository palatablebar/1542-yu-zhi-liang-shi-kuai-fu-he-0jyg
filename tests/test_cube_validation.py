from dataclasses import replace
import pytest
from beamcure.cube_validation.service import check_point, prepare_verification, performance, stable_loss, boundary_diagnostics, sector_replay


def test_loss_is_finite_at_probability_limits():
    # TRAP-31
    assert stable_loss(0,1) == pytest.approx(27.6310211159)
    assert stable_loss(1,0) == pytest.approx(27.6310432379)


def test_no_outcome_gets_undefined_metrics(prediction):
    # TRAP-43
    row = replace(prediction,outcome=None)
    metric = performance([check_point(row,prepare_verification([]))])
    assert (metric.status,metric.count,metric.mae,metric.error) == ('NO_MATURE_RESULT',0,None,None)


def test_empty_calibration_requires_review(prediction):
    # TRAP-44
    checked = check_point(prediction,prepare_verification([]))
    assert checked.uncertain and checked.lower is None


def test_first_class_only_is_sparse(prediction):
    negatives = [replace(prediction,score=.25,outcome=0,observed=True)] * 12
    assert prepare_verification(negatives).method == 'sparse-risk'
    metric = performance([check_point(p,prepare_verification(negatives)) for p in negatives])
    assert metric.auxiliary == 0 and metric.status == 'NO_POSITIVES'


def test_fitted_calibrator_is_order_preserving(prediction):
    observed = [replace(prediction,key=f'k{i}',score=i/20,outcome=int(i>=10),observed=True) for i in range(1,19)]
    trained = prepare_verification(observed)
    estimates = [check_point(row,trained).estimate for row in observed]
    assert all(0<=estimate<=1 for estimate in estimates)
    assert estimates == sorted(estimates) and trained.size == 18
    assert not any(check_point(row,trained).uncertain for row in observed)


def test_censored_target_does_not_fit_error_band(prediction):
    point = replace(prediction,observed=False,outcome=0)
    assert prepare_verification([point]).size == 0


def test_cutoff_confusion_counts_have_exact_boundaries(prediction):
    verified = [check_point(replace(prediction,score=value,outcome=target,observed=True),prepare_verification([]))
                for value,target in [(0.2,1),(0.5,0),(0.8,1)]]
    actual=boundary_diagnostics(verified,.5)
    assert (actual['实绩超线'],actual['估计超线'],actual['额外复测'],actual['漏查']) == (2,2,1,1)
    assert len(sector_replay(verified,.5)) == 5


# --- Per-line calibration version selection ---------------------------------
from beamcure.cube_validation.contracts import Verification
from beamcure.cube_validation.service import (UNCALIBRATED, choose_versions, holdout_scores,
                                              select_version, selection_evidence, validate_by_line,
                                              versions_of)
from beamcure.strength_classifier.contracts import Prediction


def pair(key, score, outcome, *, cohort='线甲', day=90, observed=True):
    return Prediction(key, cohort, day, score, (), (), (), outcome, observed, ())


def test_holdout_score_is_undefined_without_outcomes(prediction):
    censored = replace(prediction, observed=False, outcome=0)
    assert holdout_scores(prepare_verification([prediction] * 12), [censored]) == (None, None)


def test_holdout_score_counts_positives_below_the_dispatch_cutoff(prediction):
    anchor = prepare_verification([replace(prediction, key=f'k{i}', score=0.4 + 0.02*i,
                                           outcome=int(i >= 6)) for i in range(12)])
    # Raw scores 0.95 and 0.30 with an identity calibrator map straight through.
    identity = Verification('press_scale', (1.0, 0.0), 12, 6)
    mae, missed = holdout_scores(identity, [pair('a', 0.95, 1), pair('b', 0.30, 1), pair('c', 0.10, 0)])
    assert missed == 1
    assert mae == pytest.approx((0.05 + 0.70 + 0.10) / 3)


def test_line_needs_both_outcomes_before_switching():
    ready = Verification('press_scale', (1.3, 0.0), 12, 6)
    holdout = [pair(f'h{i}', 0.9, 1) for i in range(8)]  # no negatives on replay
    choice = select_version('线甲', ready, ready, holdout)
    assert choice.version == 'prior'
    assert choice.reason == '回放缺正常或异常实绩'


def test_line_without_holdout_cannot_use_new_version():
    ready = Verification('press_scale', (1.3, 0.0), 12, 6)
    choice = select_version('线甲', ready, ready, [])
    assert choice.version == 'prior' and choice.reason == '回放缺正常或异常实绩'
    assert choice.chosen is ready
    # An unregistered line resolves to the explicit sparse anchor, not a guess.
    anchor = UNCALIBRATED
    assert anchor.method == 'sparse-risk' and not anchor.parameters


def test_unready_version_blocks_switch_even_if_error_drops():
    sparse = Verification('sparse-risk', (), 12, 0)
    ready = Verification('press_scale', (1.3, 0.0), 12, 6)
    holdout = [pair(f'h{i}', 0.2 if i % 2 else 0.9, i % 2) for i in range(10)]
    assert select_version('线甲', sparse, ready, holdout).reason == '版本未就绪'
    assert select_version('线甲', ready, sparse, holdout).reason == '版本未就绪'


def test_new_version_adopted_only_when_error_drops_and_misses_do_not_rise():
    prior = Verification('press_scale', (2.6, -0.3), 12, 6)   # steeper, pulls away from 0.5
    recent = Verification('press_scale', (1.0, 0.0), 12, 6)   # identity, well matched
    holdout = [pair(f'h{i}', 0.2 if i % 2 else 0.8, i % 2) for i in range(10)]
    choice = select_version('线甲', prior, recent, holdout)
    assert choice.version == 'recent'
    assert choice.reason == '误差下降且漏报未增加'
    assert choice.challenger_mae < choice.incumbent_mae
    assert choice.challenger_misses <= choice.incumbent_misses


def test_plant_error_gain_cannot_add_misses_on_a_line():
    # A version that lowers mean absolute error while pushing a true positive
    # below the cutoff must be rejected: the miss count wins over the average.
    prior = Verification('press_scale', (1.0, 0.0), 12, 6)   # keeps the positive above 0.49
    recent = Verification('press_scale', (1.0, -1.2), 12, 6)  # shifts the positive below cutoff
    holdout = [pair('pos', 0.6, 1)] + [pair(f'n{i}', 0.9, 0) for i in range(5)]
    choice = select_version('线甲', prior, recent, holdout)
    assert choice.incumbent_misses == 0 and choice.challenger_misses == 1
    assert choice.challenger_mae < choice.incumbent_mae
    assert choice.version == 'prior' and choice.reason == '漏报增加'


def test_exact_tie_keeps_the_prior_version():
    same = Verification('press_scale', (1.0, 0.0), 12, 6)
    holdout = [pair(f'h{i}', 0.2 if i % 2 else 0.8, i % 2) for i in range(10)]
    choice = select_version('线甲', same, same, holdout)
    assert choice.version == 'prior' and choice.reason == '平均绝对误差未下降'


def test_versions_are_chosen_per_line_and_recorded():
    prior = [pair(f'a{i}', 0.3 + 0.01*i, int(i % 2), cohort='线甲') for i in range(12)]
    prior += [pair(f'b{i}', 0.3 + 0.01*i, int(i % 2), cohort='线乙') for i in range(12)]
    recent = [pair(f'ra{i}', 0.7 - 0.01*i, int(i % 2), cohort='线甲') for i in range(12)]
    recent += [pair(f'rb{i}', 0.7 - 0.01*i, int(i % 2), cohort='线乙') for i in range(12)]
    holdout = [pair(f'ha{i}', 0.2 if i % 2 else 0.9, i % 2, cohort='线甲') for i in range(10)]
    holdout += [pair(f'hb{i}', 0.2 if i % 2 else 0.9, i % 2, cohort='线乙') for i in range(10)]
    choices, _, _ = choose_versions(prior, recent, holdout, scope={'线丙'})
    by_line = {c.cohort: c for c in choices}
    assert set(by_line) == {'线甲', '线乙', '线丙'}
    assert by_line['线丙'].version == 'prior' and by_line['线丙'].holdout == 0
    evidence = selection_evidence(choices)
    assert {row['养护线'] for row in evidence} == {'线甲', '线乙', '线丙'}
    assert {row['采用'] for row in evidence} <= {'新版', '旧版'}


def test_validate_by_line_uses_each_line_and_falls_back_to_review():
    anchor = prepare_verification([replace(Prediction('x', '蒸养线二', 89, 0.5, (), (), (), 1, True, ()),
                                           key=f'k{i}', score=0.4 + 0.02*i, outcome=int(i >= 6))
                                   for i in range(12)])
    table = {'蒸养线二': anchor}
    scored = [Prediction('R', '蒸养线二', 90, 0.6, (), (), (), None, True, ()),
              Prediction('S', '未登记', 90, 0.6, (), (), (), None, True, ())]
    known, stranger = validate_by_line(scored, table)
    assert known.uncertain is False and stranger.uncertain is True
    assert versions_of((select_version('未登记', UNCALIBRATED, UNCALIBRATED, []),))['未登记'].method == 'sparse-risk'


def _versions(prior, recent, holdout):
    choices, _, _ = choose_versions(prior, recent, holdout)
    choice, = choices
    return {'incumbent': choice.incumbent, 'challenger': choice.challenger, 'chosen': choice.chosen}


def test_replay_outcomes_never_reach_the_calibration_fit():
    # Both versions are fitted on their own windows only, so rewriting every
    # replay outcome leaves the fitted parameters untouched.
    prior = [pair(f'p{i}', 0.2 + 0.03*i, int(i % 2)) for i in range(12)]
    recent = [pair(f'r{i}', 0.25 + 0.03*i, int(i % 2)) for i in range(12)]
    holdout = [pair(f'h{i}', 0.2 if i % 2 else 0.85, i % 2) for i in range(10)]
    flipped = [replace(entry, outcome=1 - entry.outcome) for entry in holdout]
    a = _versions(prior, recent, holdout)
    b = _versions(prior, recent, flipped)
    assert a['incumbent'].parameters == b['incumbent'].parameters
    assert a['challenger'].parameters == b['challenger'].parameters


def test_selection_is_identical_on_rerun():
    prior = [pair(f'p{i}', 0.2 + 0.03*i, int(i % 2)) for i in range(12)]
    recent = [pair(f'r{i}', 0.25 + 0.03*i, int(i % 2)) for i in range(12)]
    holdout = [pair(f'h{i}', 0.2 if i % 2 else 0.85, i % 2) for i in range(10)]
    assert _versions(prior, recent, holdout) == _versions(prior, recent, holdout)
