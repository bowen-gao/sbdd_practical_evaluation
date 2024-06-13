import logging
import math

import numpy as np
import torch
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from sklearn.metrics import roc_auc_score, top_k_accuracy_score
from torch import Tensor
from torchmetrics import (
    MeanSquaredError,
    Metric,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)
from torchmetrics.metric import jit_distributed_available
from torchmetrics.utilities.data import dim_zero_cat

logger = logging.getLogger(__name__)


def cat_states(func):
    def wrapper(self, *args, **kwargs):
        if not jit_distributed_available() and not self._is_synced:
            output_dict = {
                attr: getattr(self, attr) for attr in self._reductions
            }

            for attr, reduction_fn in self._reductions.items():
                # pre-concatenate metric states that are lists to reduce number of all_gather operations
                if (
                    reduction_fn == dim_zero_cat
                    and isinstance(output_dict[attr], list)
                    and len(output_dict[attr]) >= 1
                ):
                    setattr(self, attr, dim_zero_cat(output_dict[attr]))
        return func(self, *args, **kwargs)

    return wrapper


def fill_prob(prob: torch.Tensor, num_classes: int):
    """Fill the prob tensor with 0.0 for the classes that are not in the
    target."""
    assert prob.dim() == 2
    assert prob.size(1) <= num_classes, f"{prob.size(1)} > {num_classes}"
    if prob.size(1) < num_classes:
        prob = torch.cat(
            [
                prob,
                torch.zeros(
                    prob.size(0), num_classes - prob.size(1), device=prob.device
                ),
            ],
            dim=1,
        )
    return prob


def bedroc_score(y_true, y_pred, alpha=80.5):
    """Boltzmann-Enhanced Discrimination ROC score.

    Implementation of Boltzmann-Enhanced Discrimination ROC metric. This is a
    "weighted" ROC score assigning more weight on early recognition. Refer to
    paper below for details. This is  simply a wrapper function around the
    rdkit implementation.

    See: https://pubs.acs.org/doi/10.1021/ci600426e

    Args:
        y_true: 1d array-like
            Ground truth (correct) labels indicating whether the corresponding
            ligand binds to the target protein.

        y_pred: 1d array-like
            Predicted scores, one for each ligand, as returned by a scoring
            function.

        alpha: float
            Early recognition parameter.

    Returns:
        score: float
            BEDROC score for specified ``alpha``.
    """
    sort_ind = np.argsort(y_pred)[::-1]  # Descending order
    # BedROC only considers the ground truth vector sorted wrt predictions
    # but reshape to match the expected rdkit format (2D array)
    return CalcBEDROC(y_true[sort_ind].reshape(-1, 1), 0, alpha)


def ef_score(y_true, y_pred, alpha):
    """Enhancement factor score.

    EF-score is calculated as: EF_a  = NTB_a / (NTB_total x a), where NTB_a
    is the number of true binders observed among the a% top-ranked candidates
    selected by a given scoring function, NTB_total is the total number of
    true binders for a given target protein.

    See: https://doi.org/10.1021/acs.jcim.8b00545

    Args:
        y_true: 1d array-like
            Ground truth (correct) labels indicating whether the corresponding
            ligand binds to the target protein.

        y_pred: 1d array-like
            Predicted scores, one for each ligand, as returned by a scoring
            function.

        alpha: float, range (0, 1]
            Τop-ranking threshold (%). Set to ``0.01`` for EF1,
            ``0.1`` for EF10 and so on.

    Returns:
        score: float
            EF score for specified ``alpha``.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(
            "``alpha`` argument must be in range (0, 1] but {} "
            "was provided.".format(alpha)
        )

    # Sort in descending order and store indexes for labels sorting
    sort_ind = np.argsort(y_pred)[::-1]
    y_true, y_pred = y_true[sort_ind], y_pred[sort_ind]
    top_pred = y_true[0 : math.ceil((len(y_true) * alpha))]
    return top_pred.sum() / (y_true.sum() * alpha)


# able to not set num_classess
class Accuracy(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(self):
        super().__init__()

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num_classes = None

    def update(self, preds: torch.Tensor, target: torch.Tensor, **kwargs):
        assert preds.size() == target.size()
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class TopkAccuracy(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(self, topk: int = 3):
        super().__init__()

        self.topk = topk
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor, target: torch.Tensor, **kwargs):
        assert probs.size(0) == target.size(0)
        self.correct += top_k_accuracy_score(
            target.cpu(), probs.cpu(), k=self.topk, normalize=False
        )
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class AUCROC(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self):
        super().__init__()

        self.num_classes = None
        self.add_state("probs", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, probs: torch.Tensor, target: torch.Tensor, **kwargs):
        assert probs.size(0) == target.size(0)
        if self.num_classes is None:
            self.num_classes = probs.size(1)
        probs = fill_prob(probs, self.num_classes)
        self.probs.append(probs)
        self.targets.append(target)

    @cat_states
    def compute(self):
        return roc_auc_score(
            self.targets.detach().cpu(),
            self.probs.detach().cpu(),
            average="macro",
            multi_class="ovr",
        )


class BEDROC(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, alpha: float = 80.5):
        super().__init__()

        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.alpha = alpha

    def update(self, probs: torch.Tensor, target: torch.Tensor, **kwargs):
        assert probs.size(0) == target.size(0)
        labels = torch.zeros_like(probs)
        labels[torch.arange(labels.size(0)), target] = 1
        for y_true, y_pred in zip(labels, probs):
            self.score += bedroc_score(
                y_true.cpu().numpy(), y_pred.cpu().numpy(), self.alpha
            )
            self.total += 1

    def compute(self):
        return self.score.float() / self.total


class Spearman(SpearmanCorrCoef):
    def update(self, probs: torch.Tensor, affinities: torch.Tensor, **kwargs):
        preds = probs.diagonal()
        super().update(preds, affinities)


class Pearson(PearsonCorrCoef):
    def update(self, probs: torch.Tensor, affinities: torch.Tensor, **kwargs):
        preds = probs.diagonal()
        super().update(preds, affinities)


class RMSE(MeanSquaredError):
    def compute(self) -> Tensor:
        return math.sqrt(super().compute())
