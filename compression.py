
from typing import List
import numpy as np
from cued_sf2_lab.laplacian_pyramid import rowdec,rowint, quantise, bpp
from cued_sf2_lab.dct import colxfm, dct_ii, regroup
from cued_sf2_lab.lbt import pot_ii, dct_ii
from cued_sf2_lab.dwt import dwt, idwt


# Image size func

def img_size(img):
    return bpp(img) * img.shape[0] * img.shape[1]


# --- Laplacian Pyramid ---

class LPCompression:

    def __init__(self, n: int = 3, h: np.ndarray = np.array([1, 2, 1])/4, **kwargs) -> None:
        
        self.n = n
        self.h = h


    def compress(self, X: np.ndarray) -> List[np.ndarray]:

        Y = [X]

        for _ in range(self.n):

            Xn = Y[-1]

            # Decimate X 2:1
            Xd = rowdec(rowdec(Xn, self.h).T, self.h).T

            # Interpolate Xd to upscale and subtract from Xn to obtain Y
            Yn = Xn - rowint(rowint(Xd, 2*self.h).T, 2*self.h).T

            Y = Y[:-1] + [Yn, Xd]

        return Y

   
    def decompress(self, Y: List[np.ndarray]) -> np.ndarray:

        Zn = Y[-1]
        Ysr = reversed(Y[:-1])

        for y in Ysr:

            # interpolate X to upscale and add to each y, for every y
            Zn = y + rowint(rowint(Zn, 2*self.h).T, 2*self.h).T

        return Zn


    # Adjust quantise structure

    def quantise(self, Y: List[np.ndarray], init_step: float = 20.1, r: float = 1/2) -> List[np.ndarray]:
    
        return [quantise(y, init_step*(r**i)) for i, y in enumerate(Y)] 


# --- DCT --- 

class DCTCompression:

    def __init__(self, N: int = 8):

        self.N = N
        self.C = dct_ii(N)

        
    def compress(self, X: np.ndarray) -> np.ndarray:
        
        return colxfm(colxfm(X, self.C).T, self.C).T


    def decompress(self, Y: np.ndarray) -> np.ndarray:

        return colxfm(colxfm(Y.T, self.C.T).T, self.C.T)


    def regroup(self, Y: np.ndarray) -> np.ndarray:

        return regroup(Y, self.N)


    # quantise


# --- LBT ---

def pre_filter(X, Pf):
    N = Pf.shape[0]
    t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
    Xp = X.copy()  # copy the non-transformed edges directly from X
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
    return Xp

def post_filter(Z, Pr):
    N = Pr.shape[0]
    t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
    Zp = Z.copy()  # copy the non-transformed edges directly from X
    Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
    Zp[t,:] = colxfm(Zp[t,:], Pr.T)
    return Zp


class LBTCompression:

    def __init__(self, s: float = 1.6, N: int = 8):
        
        self.Pf, self.Pr = pot_ii(N, s=s)
        self.DCT = DCTCompression(N)

    def pre_filter(self, X: np.ndarray) -> np.ndarray:

        return pre_filter(X, self.Pf)

    def post_filter(self, Z: np.ndarray) -> np.ndarray:

        return post_filter(Z, self.Pr)

    
    def compress(self, X: np.ndarray) -> np.ndarray:

        Xp = self.pre_filter(X)

        return self.DCT.compress(Xp)


    def decompress(self, Y: np.ndarray) -> np.ndarray:

        Z = self.DCT.decompress(Y)

        return self.post_filter(Z)



# --- DWT ---

class DWTCompression:
    def __init__(self, n: int, steps: np.ndarray):
        self.n = n
        self.steps = steps
        self.ent

    def compress(self, X: np.ndarray):
        Y = X.copy()

        for i in range(self.n):
            m = 256//(2**i)
            Y[:m,:m] = dwt(Y[:m,:m])
            
        return Y

    def decompress(self, Y: np.ndarray):
        Yr = Y.copy()

        for i in range(self.n):
            m = 256//(2**(self.n - i - 1))
            Yr[:m,:m] = idwt(Yr[:m,:m])

        return Yr

    def quantise(self, Y):
        Yq = np.zeros_like(Y)
        dwtent = 0

        for i in range(self.n):

            m = 256//(2**i) # 256, 128, 64 ... 
            h = m//2 # Midpoint: 128, 64, 32 ...

            # Quantising
            Yq[:h, h:m] = quantise(Y[:h, h:m], self.steps[0, i]) # Top right 
            Yq[h:m, :h] = quantise(Y[h:m, :h], self.steps[1, i]) # Bottom left
            Yq[h:m, h:m] = quantise(Y[h:m, h:m], self.steps[2, i]) # Bottom right

            dwtent += img_size(Yq[:h, h:m]) + img_size(Yq[h:m, :h]) + img_size(Yq[h:m, h:m])

        # Final low pass image
        m = 256//(2**self.n)
        Yq[:m, :m] = quantise(Y[:m, :m], self.steps[0, self.n])
        self.ent = dwtent + img_size(Yq[:m, :m])

        return Yq


class SVDCompression:

    def __init__(self, K) -> None:

        self.K = K

    def encode(self, X: np.ndarray) -> List[np.ndarray]:

        U, s, V = np.linalg.svd(X)
        U = U[:, :self.K]
        s = s[:self.K]
        V = V[:self.K, :]

        return [U, s, V]

    def decode(self, Y: List[np.ndarray]) -> np.ndarray:

        return Y[0] @ np.diag(Y[1]) @ Y[2]


