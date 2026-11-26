from bisect import bisect_right
from collections import defaultdict
from hashlib import blake2s
from math import exp, fsum, isfinite, log, log1p
from statistics import mean

from ..retest_press.contracts import RISK_CUTOFF
from .contracts import Verification, Assessment, CalibrationRow, ModelReport, VersionChoice
from .slices import replay_slices
import numpy as np

# 漏报判定与派单共用同一条强度不足阈值，避免两处口径漂移。
RISK_THRESHOLD = RISK_CUTOFF
UNCALIBRATED = Verification('sparse-risk', (), 0, None)


def clip_risk(probability):
    return min(1-1e-12,max(1e-12,probability))


def sigmoid(log_odds):
    if log_odds < 0:
        factor = exp(log_odds)
        return factor / (1+factor)
    return 1/(1+exp(-log_odds))


def logit(probability):
    clipped = clip_risk(probability)
    return log(clipped)-log1p(-clipped)


def stable_loss(probability, outcome):
    estimate = clip_risk(probability)
    return -log(estimate) if outcome else -log1p(-estimate)





def press_scale(checks):
    probability = np.array([clip_risk(weight.score) for weight in checks],dtype=float)
    responses = np.array([weight.outcome for weight in checks],dtype=float)
    features = np.column_stack((np.log(probability) - np.log1p(-probability),np.ones(probability.size)))
    slope, offset = 1., 0.
    for _ in range(36):
        theta = np.array((slope,offset))
        scores = features @ theta
        expected = np.array([sigmoid(float(value)) for value in scores])
        error = features.T @ (expected-responses) / len(checks) + .05*(theta-(1.,0.))
        slope = max(.05, slope-.16*error[0]); offset -= .16*error[1]
    return (float(slope),float(offset))


def calibrated_press_scale(model, p):
    slope, origin = model
    return sigmoid(slope*logit(p)+origin)


def prepare_verification(pairs, *, coverage=.85):
    mature = [entry for entry in pairs if entry.observed and entry.outcome is not None]
    if any(not isfinite(entry.score) or entry.outcome not in (0,1) for entry in mature):
        raise ValueError('风险校准要求有限预测和二元实绩')
    positives = sum(entry.outcome == 1 for entry in mature)
    if len(mature) < 8 or positives in (0,len(mature)):
        return Verification('sparse-risk', (), len(mature), positives)
    return Verification('press_scale', press_scale(mature), len(mature), positives)


def check_point(scored, anchor):
    if not isfinite(scored.score):
        raise ValueError('预测风险必须有限')
    value = clip_risk(scored.score)
    if not anchor.parameters:
        return Assessment(scored,value,None,None,True)
    return Assessment(scored, calibrated_press_scale(anchor.parameters,value),None,None,False)


def validate_predictions(scored, prepared):
    return tuple(check_point(sample, prepared) for sample in scored)


def _mature(pairs):
    return [entry for entry in pairs if entry.observed and entry.outcome is not None]


def holdout_scores(anchor, holdout):
    """Return (mean absolute error, missed high-risk count) for one calibration version.

    A missed high-risk beam is a replayed case whose actual outcome is 1 but whose
    calibrated estimate stays below the shared dispatch cutoff. No recorded outcome
    means no score at all — never a zero-error substitute.
    """
    if not anchor.parameters or not holdout:
        return None, None
    errors = [abs(calibrated_press_scale(anchor.parameters, entry.score) - entry.outcome) for entry in holdout]
    missed = sum(calibrated_press_scale(anchor.parameters, entry.score) < RISK_THRESHOLD
                 for entry in holdout if entry.outcome == 1)
    return mean(errors), missed


def select_version(cohort, incumbent, challenger, holdout_pairs):
    """Decide one curing line's calibration version from the shared replay set.

    Both versions are scored on the same replay records, which neither was
    calibrated on. The line only moves to the newer version when it has both
    normal and high-risk outcomes on replay, its mean absolute error strictly
    drops, and no extra high-risk beam is missed. Any other outcome keeps the
    prior version so a plant-wide gain can never quietly raise one line's misses.
    """
    holdout = sorted(_mature(holdout_pairs), key=lambda entry: (entry.day, entry.key))
    positives = sum(entry.outcome == 1 for entry in holdout)
    prior_mae, prior_missed = holdout_scores(incumbent, holdout)
    recent_mae, recent_missed = holdout_scores(challenger, holdout)
    choice = dict(cohort=cohort, incumbent=incumbent, challenger=challenger, holdout=len(holdout),
                  positives=positives, incumbent_mae=prior_mae, challenger_mae=recent_mae,
                  incumbent_misses=prior_missed, challenger_misses=recent_missed)
    if not incumbent.parameters or not challenger.parameters:
        return VersionChoice(version='prior', reason='版本未就绪', chosen=incumbent, **choice)
    if positives == 0 or positives == len(holdout):
        return VersionChoice(version='prior', reason='回放缺正常或异常实绩', chosen=incumbent, **choice)
    if not recent_mae < prior_mae:
        return VersionChoice(version='prior', reason='平均绝对误差未下降', chosen=incumbent, **choice)
    if recent_missed > prior_missed:
        return VersionChoice(version='prior', reason='漏报增加', chosen=incumbent, **choice)
    return VersionChoice(version='recent', reason='误差下降且漏报未增加', chosen=challenger, **choice)


def choose_versions(prior, recent, holdout, *, scope=()):
    """Pick one calibration version per curing line.

    The prior (old) calibration is fitted on the fit-window actuals and the recent
    (new) calibration on the later calibration window. Both are plant-wide fits;
    neither ever sees the replay window, and both are scored on the same replay
    records. Each line then decides for itself: it moves to the new version only
    when that line has both normal and high-risk outcomes on replay, its own mean
    absolute error strictly drops, and no extra high-risk beam is missed. A
    whole-plant error gain therefore cannot carry a line into more misses, and a
    line without complete replay evidence keeps the old version instead of being
    scored by an unproven one.

    Returns ``(choices, incumbent, challenger)`` so the caller can reuse the two
    fitted versions without re-fitting them.
    ``scope`` names every line that must receive an explicit decision (for
    example live lines, so an unregistered line still resolves to review).
    """
    incumbent = prepare_verification(prior)
    challenger = prepare_verification(recent)
    holdout_by = defaultdict(list)
    for entry in holdout:
        holdout_by[entry.cohort].append(entry)
    names = sorted(set(holdout_by) | {entry.cohort for entry in prior} |
                   {entry.cohort for entry in recent} | set(scope))
    choices = tuple(select_version(name, incumbent, challenger, holdout_by.get(name, ()))
                    for name in names)
    return choices, incumbent, challenger


def plant_holdout(incumbent, challenger, holdout):
    """Plant-wide holdout scores for the old and new versions, on shared replay."""
    replay = sorted(_mature(holdout), key=lambda entry: (entry.day, entry.key))
    return {'回放实绩': len(replay), '阳性例': sum(entry.outcome == 1 for entry in replay),
            '旧版校准': incumbent.method, '新版校准': challenger.method,
            '旧版平均绝对误差': holdout_scores(incumbent, replay)[0],
            '新版平均绝对误差': holdout_scores(challenger, replay)[0]}


def versions_of(choices):
    return {choice.cohort: choice.chosen for choice in choices}


def validate_by_line(scored, versions):
    return tuple(check_point(sample, versions.get(sample.cohort, UNCALIBRATED)) for sample in scored)


def selection_stamp(choices):
    digest = blake2s(digest_size=8)
    for choice in choices:
        payload = (choice.cohort, choice.version, choice.chosen.method, choice.chosen.parameters,
                   choice.reason, choice.holdout, choice.positives)
        digest.update(repr(payload).encode())
    return digest.hexdigest()[:12]


def selection_evidence(choices):
    labels = {'recent': '新版', 'prior': '旧版'}
    rows = []
    for choice in choices:
        rows.append({'养护线': choice.cohort, '采用': labels.get(choice.version, choice.version),
                     '判定': choice.reason, '回放实绩': choice.holdout, '阳性例': choice.positives,
                     '旧版绝对误差': choice.incumbent_mae, '新版绝对误差': choice.challenger_mae,
                     '旧版漏报': choice.incumbent_misses, '新版漏报': choice.challenger_misses,
                     '旧版校准': choice.incumbent.method, '新版校准': choice.challenger.method})
    return rows


def reliability_rows(cases):
    groups = defaultdict(list)
    for raw, label in cases:
        groups[min(3,int(raw*4))].append((raw,label))
    for position in range(4):
        group = groups[position]
        yield CalibrationRow(position, mean(x[0] for x in group) if group else None,
                             mean(x[1] for x in group) if group else None, len(group))


def performance(assessments):
    review = [(point.estimate,point.raw.outcome) for point in assessments if point.raw.observed and point.raw.outcome is not None]
    if not review:
        return ModelReport('NO_MATURE_RESULT',0,None,None,(),None)
    error = mean(abs(estimate-outcome) for estimate,outcome in review)
    positive = sum(outcome==1 for _,outcome in review)
    return ModelReport('READY' if positive else 'NO_POSITIVES',len(review),error,
                       mean(stable_loss(*item) for item in review),tuple(reliability_rows(review)),positive)


def calibration_evidence(report):
    reads = [{'核验':report.status,'样本':report.count,'平均绝对误差':report.mae,'对数损失':report.error,'阳性例':report.auxiliary}]
    reads += [{'分箱':f'{row.index*25}–{(row.index+1)*25}%', '样本':row.count,
               '预测均值':row.predicted,'实绩均值':row.observed} for row in report.buckets]
    return reads


def sector_replay(checked, line):
    return replay_slices(checked,line,risk_model=True)


def boundary_diagnostics(checks, threshold):
    points = [(row.estimate>=threshold,row.raw.outcome==1) for row in checks if row.raw.observed and row.raw.outcome is not None]
    return {'临界判断':'强度不足', '实绩超线':sum(label for _,label in points),
            '估计超线':sum(pred for pred,_ in points), '额外复测':sum(p and not y for p,y in points),
            '漏查':sum(y and not p for p,y in points)}
