
# import packages
import numpy as np

class w2vModel:
    def __init__(self, w2i: dict):
        self.w2i = w2i

    def computeSigmoidLikeDerivative(self, v: np.ndarray, u: np.ndarray, v_word: str, u_word: str, sign: int) -> tuple:
        sig_like_score = self.computeSigmoidLike(v, u, sign=sign)
        '''
        dv = -1 * -sign * (sig_like_score * (1 - sig_like_score) * u) / sig_like_score
        du = -1 * -sign * (sig_like_score * (1 - sig_like_score) * v) / sig_like_score
        '''
        dv = sign * (1 - sig_like_score) * u
        du = sign * (1 - sig_like_score) * v

        # move to matrix
        v_size = len(self.w2i)
        dv = np.eye(v_size)[self.w2i[v_word]].reshape(v_size, 1) * dv.reshape(1, dv.shape[0])
        du = np.eye(v_size)[self.w2i[u_word]].reshape(v_size, 1) * du.reshape(1, du.shape[0])
        return dv, du

    def computeSigmoidLike(self, v: np.ndarray, u: np.ndarray, sign: int) -> float:
        assert sign in [-1, 1]
        return 1/(1+(np.exp(sign * np.dot(v, u))))

    def computeLog(self, x: float) -> float:
        return np.log(x)

    def computeScore(self, v: np.ndarray, u: np.ndarray, sign: int) -> float:
        return self.computeLog(self.computeSigmoidLike(v, u, sign=sign))

    def computeLossAndGrads(self, params: tuple) -> tuple:

        (E, E_tag, word, context, rnd_word) = params

        w = E[self.w2i[word]]
        c = E_tag[self.w2i[context]]
        r = E_tag[self.w2i[rnd_word]]

        # as minimization problem max(f) = - min (-f)
        good = -1 * self.computeScore(w, c, sign=-1)
        bad = -1 * self.computeScore(w, r, sign=1)
        loss = (good + bad)
        dw, dc = self.computeSigmoidLikeDerivative(w, c, word, context, sign=-1)
        dw_tag, dr = self.computeSigmoidLikeDerivative(w, r, word, rnd_word, sign=1)
        return loss, (dw+dw_tag, dc, dr)

