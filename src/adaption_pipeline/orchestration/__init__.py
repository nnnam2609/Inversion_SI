"""Small dependency runner used locally and from OAR jobs."""

from .dag import Dag, Stage

__all__ = ["Dag", "Stage"]
