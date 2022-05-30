
from typing import List
import numpy as np
from cued_sf2_lab.familiarisation import quantise
from cued_sf2_lab.laplacian_pyramid import rowdec,rowint


def lp_enc(X: np.ndarray, n: int, h: np.ndarray=np.array([1, 2, 1])) -> List[np.ndarray]:

    pyramid = [X]

    for _ in range(n):

        Xn = pyramid[-1]

        # Decimate X 2:1
        Xd = rowdec(rowdec(Xn, h).T, h).T

        # Interpolate Xd to upscale and subtract from Xn to obtain Y
        Y = Xn - rowint(rowint(Xd, 2*h).T, 2*h).T

    pyramid = pyramid[:-1] + [Y, Xd]

    return pyramid


def lp_quantise(pyramid: List[np.ndarray], init_step: float =20.1, r: float=1/2) -> List[np.ndarray]:

    return [quantise(y, init_step*(r**i)) for i, y in enumerate(pyramid)]


def lp_dec(pyramid: List[np.ndarray], h: np.ndarray=np.array([1, 2, 1])) -> np.ndarray:

    Zn = pyramid[-1]
    Ysr = reversed(pyramid[:-1])

    for y in Ysr:

        # interpolate X to upscale and add to each y, for every y
        Zn = y + rowint(rowint(Zn, 2*h).T, 2*h).T

    return Zn



