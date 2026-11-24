"""Market-only baselines and a linear multinomial model with a fixed offset."""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy import sparse
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import log_softmax
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from .market import N_CLASSES, validate_probabilities, validate_targets

L2_GRID = (0.0001, 0.001, 0.01, 0.1)
MAX_ITER = 1000
TOL = 1e-8
# Orthogonal coordinates for the sum-to-zero class logits. L2 on these
# coordinates equals L2 on the corresponding full set of class slopes.
CONTRASTS = helmert(N_CLASSES).T


@dataclass(frozen=True)
class Convergence:
    converged: bool
    iterations: int
    message: str


def _training_targets(y: np.ndarray, n_rows: int) -> np.ndarray:
    target = validate_targets(y, n_rows)
    if np.unique(target).size != N_CLASSES:
        raise ValueError("Training requires examples of all three outcomes")
    return target


def _validate_l2(l2: float) -> float:
    if not np.isfinite(l2) or l2 <= 0:
        raise ValueError("L2 regularization must be finite and strictly positive")
    return l2


class RawMarket:
    """M0 returns the proportional market probabilities without fitting."""

    def predict_proba(self, market: np.ndarray) -> np.ndarray:
        return validate_probabilities(market)


class MarketLogistic(BaseEstimator):
    """M1 uses standardized log market probabilities and L2-penalized slopes."""

    def __init__(self, l2: float):
        self.l2 = _validate_l2(l2)

    def fit(self, market: np.ndarray, y: np.ndarray):
        features = np.log(validate_probabilities(market, positive=True))
        target = _training_targets(y, len(features))
        # sklearn's C weights the penalty relative to summed loss. Express
        # l2 relative to mean loss so its units agree with M2 across folds.
        model = LogisticRegression(solver="lbfgs", C=1 / (len(target) * self.l2), max_iter=MAX_ITER, tol=TOL)
        pipeline = make_pipeline(StandardScaler(), model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(features, target)
        messages = []
        for warning in caught:
            if issubclass(warning.category, ConvergenceWarning):
                messages.append(str(warning.message))
            else:
                warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
        self.pipeline_ = pipeline
        self.convergence_ = Convergence(not messages, int(model.n_iter_[0]), "\n".join(messages))
        return self

    def predict_proba(self, market: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.predict_proba(np.log(validate_probabilities(market, positive=True)))


def residual_probabilities(market: np.ndarray, corrections: np.ndarray) -> np.ndarray:
    """Apply additive corrections to fixed log market probabilities (H/D/A)."""
    probabilities = validate_probabilities(market, positive=True)
    corrections = np.asarray(corrections, dtype=float)
    if corrections.shape != probabilities.shape or not np.isfinite(corrections).all():
        raise ValueError("Corrections must be finite and have the same shape as the market probabilities")
    with np.errstate(over="ignore", under="ignore"):
        centered = corrections - corrections.max(axis=1, keepdims=True)
        return np.exp(log_softmax(np.log(probabilities) + centered, axis=1))


class MarketResidual(BaseEstimator):
    """M2 fits a linear correction to the fixed log market offset.

    Supply a row-aligned dense or sparse feature matrix to define the correction
    inputs. By default the correction uses log market probabilities. Scaling is
    learned in fit. Sparse inputs are scaled without centering to preserve sparsity.
    Set standardize=False to pass preprocessed inputs through unchanged.
    With text_l2, pass numerical_width to fit and put the text columns after
    the numerical columns. l2 then applies only to the numerical slopes.
    """

    def __init__(self, l2: float, *, standardize: bool = True, text_l2: float | None = None):
        self.l2 = _validate_l2(l2)
        self.standardize = standardize
        self.text_l2 = None if text_l2 is None else _validate_l2(text_l2)

    @staticmethod
    def _features(offset: np.ndarray, features):
        matrix = offset if features is None else check_array(features, accept_sparse="csr", dtype=float)
        if matrix.shape[0] != offset.shape[0]:
            raise ValueError("Correction features and market probabilities must have the same row count")
        return matrix

    def fit(self, market: np.ndarray, y: np.ndarray, features=None, *, numerical_width: int | None = None):
        offset = np.log(validate_probabilities(market, positive=True))
        target = _training_targets(y, len(offset))
        matrix = self._features(offset, features)
        slope_penalties = self.l2
        if self.text_l2 is not None:
            if (features is None or not isinstance(numerical_width, (int, np.integer))
                    or isinstance(numerical_width, bool) or not 0 < numerical_width < matrix.shape[1]):
                raise ValueError("Block penalties require a numerical_width separating two nonempty feature blocks")
            slope_penalties = np.full((matrix.shape[1], 1), self.text_l2)
            slope_penalties[:numerical_width] = self.l2
        elif numerical_width is not None:
            raise ValueError("numerical_width requires text_l2")
        scaler = None
        if self.standardize:
            scaler = StandardScaler(with_mean=not sparse.issparse(matrix))
            matrix = scaler.fit_transform(matrix)
        rows = np.arange(len(target))
        parameter_shape = (matrix.shape[1] + 1, N_CLASSES - 1)

        def objective(parameters):
            coefficients = parameters.reshape(parameter_shape)
            intercept, slopes = coefficients[0], coefficients[1:]
            corrections = (matrix @ slopes + intercept) @ CONTRASTS.T
            log_probabilities = log_softmax(offset + corrections, axis=1)
            penalty_gradient = slope_penalties * slopes
            loss = -log_probabilities[rows, target].mean() + (slopes * penalty_gradient).sum() / 2
            errors = np.exp(log_probabilities)
            errors[rows, target] -= 1
            gradient = (errors @ CONTRASTS) / len(target)
            slope_gradient = matrix.T @ gradient + penalty_gradient
            return loss, np.vstack((gradient.sum(axis=0), slope_gradient)).ravel()

        result = minimize(
            objective, np.zeros(np.prod(parameter_shape)), method="L-BFGS-B", jac=True,
            options={"maxiter": MAX_ITER, "ftol": 1e-12, "gtol": TOL},
        )
        coefficients = result.x.reshape(parameter_shape)
        self.intercept_ = coefficients[0] @ CONTRASTS.T
        self.coef_ = coefficients[1:] @ CONTRASTS.T
        self.scaler_ = scaler
        self.custom_features_ = features is not None
        self.convergence_ = Convergence(bool(result.success), int(result.nit), str(result.message))
        return self

    def predict_proba(self, market: np.ndarray, features=None) -> np.ndarray:
        check_is_fitted(self, "scaler_")
        if (features is not None) != self.custom_features_:
            raise ValueError("Use the same correction feature inputs as in fit")
        offset = np.log(validate_probabilities(market, positive=True))
        matrix = self._features(offset, features)
        if matrix.shape[1] != self.coef_.shape[0]:
            raise ValueError("Correction features must have the same column count as in fit")
        if self.scaler_ is not None:
            matrix = self.scaler_.transform(matrix)
        return residual_probabilities(market, matrix @ self.coef_ + self.intercept_)
