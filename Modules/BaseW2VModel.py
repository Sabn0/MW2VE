
# import packages
import numpy as np
from abc import abstractmethod
from typing import *

class BaseW2VModel:
    def __init__(self):
        pass

    def computeSigmoidLikeDerivative(self, v: np.ndarray, u: np.ndarray, sign: int) -> tuple:
        sig_like_score = self.computeSigmoidLike(v, u, sign=sign)
        v_tag = u * np.gradient(v)
        u_tag = v * np.gradient(u)
        dv = -sign * (sig_like_score * (1 - sig_like_score) * v_tag) / sig_like_score
        du = -sign * (sig_like_score * (1 - sig_like_score) * u_tag) / sig_like_score
        return dv, du

    def computeSigmoidLike(self, v: np.ndarray, u: np.ndarray, sign: int) -> float:
        assert sign in [-1, 1]
        return 1/(1+(np.exp(sign * np.dot(v, u))))

    def computeLog(self, x: float) -> float:
        return np.log(x)

    def computeScore(self, v: np.ndarray, u: np.ndarray, sign: int) -> float:
        return self.computeLog(self.computeSigmoidLike(v, u, sign=sign))

    def computeLossAndGrads(self, w: np.ndarray, c: np.ndarray, r: np.ndarray) -> tuple:
        loss = - (self.computeScore(w, c, sign=-1) + self.computeScore(w, r, sign=1))
        dw, dc = self.computeSigmoidLikeDerivative(w, c, sign=-1)
        dw_tag, dr = self.computeSigmoidLikeDerivative(w, r, sign=1)
        return loss, (dw+dw_tag, dc, dr)

