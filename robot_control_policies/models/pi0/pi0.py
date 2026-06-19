from torch import nn


class Pi0Policy(nn.Module):
    """Placeholder for the eventual reference-aligned pi0 implementation.

    Use `Pi0Tiny` while learning the data contract and flow-matching objective.
    This class is intentionally not implemented yet so the toy model and the
    real model do not get mixed together.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "Pi0Policy is reserved for the full pi0 implementation. "
            "Use robot_control_policies.models.pi0.Pi0Tiny for the tiny scaffold."
        )
