"""CLRBezierLane (sparse Bezier-anchor CLR head) for the official CLRerNet repo."""
from .head import CLRBezierHead, StagePredictions  # noqa: F401  (registers CLRBezierHead)

__all__ = ["CLRBezierHead", "StagePredictions"]
