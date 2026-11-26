from dataclasses import dataclass
from ..strength_classifier.contracts import Prediction

RISK_MODE = True


@dataclass(frozen=True, slots=True)
class Verification:
    method: str
    parameters: tuple
    size: int
    positives: int | None

    def __post_init__(self):
        if self.size < 0 or self.positives is not None and not 0 <= self.positives <= self.size:
            raise ValueError('校准回查计数失衡')


@dataclass(frozen=True, slots=True)
class VersionChoice:
    cohort: str
    version: str
    reason: str
    incumbent: Verification
    challenger: Verification
    chosen: Verification
    holdout: int
    positives: int
    incumbent_mae: float | None
    challenger_mae: float | None
    incumbent_misses: int | None
    challenger_misses: int | None

    def __post_init__(self):
        if not self.cohort or not self.reason:
            raise ValueError('选版需要养护线和判定原因')
        if self.version not in ('recent', 'prior'):
            raise ValueError('未知校准版本')
        if self.holdout < 0 or not 0 <= self.positives <= self.holdout:
            raise ValueError('选版回放计数失衡')


@dataclass(frozen=True, slots=True)
class Assessment:
    raw: Prediction
    estimate: float
    lower: float | None
    upper: float | None
    uncertain: bool


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    index: int
    predicted: float | None
    observed: float | None
    count: int


@dataclass(frozen=True, slots=True)
class ModelReport:
    status: str
    count: int
    mae: float | None
    error: float | None
    buckets: tuple[CalibrationRow, ...]
    auxiliary: int | float | None
