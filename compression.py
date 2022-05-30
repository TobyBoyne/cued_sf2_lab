
from typing import List, Tuple, NamedTuple, Optional
import numpy as np
from cued_sf2_lab.laplacian_pyramid import rowdec,rowint, quantise, quant1, quant2, bpp
from cued_sf2_lab.dct import colxfm, dct_ii, regroup
from cued_sf2_lab.lbt import pot_ii, dct_ii
from cued_sf2_lab.dwt import dwt, idwt
from cued_sf2_lab.jpeg import diagscan, huffdflt, HuffmanTable, huffenc, huffgen, huffdes, runampl


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


    def quantise(self, Y):
        pass


# --- LBT ---

class LBTCompression:

    def __init__(self, s: float = 1.6, N: int = 8):
        
        self.Pf, self.Pr = pot_ii(N, s=s)
        self.DCT = DCTCompression(N)

    def pre_filter(self, X: np.ndarray) -> np.ndarray:

        N = self.Pf.shape[0]
        t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
        Xp = X.copy()  # copy the non-transformed edges directly from X
        Xp[t,:] = colxfm(Xp[t,:], Pf)
        Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
        return Xp

    def post_filter(self, Z: np.ndarray) -> np.ndarray:

        N = self.Pr.shape[0]
        t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
        Zp = Z.copy()  # copy the non-transformed edges directly from X
        Zp[:,t] = colxfm(Zp[:,t].T, self.Pr.T).T
        Zp[t,:] = colxfm(Zp[t,:], self.Pr.T)
        return Zp

    
    def compress(self, X: np.ndarray) -> np.ndarray:

        Xp = self.pre_filter(X)

        return self.DCT.compress(Xp)


    def decompress(self, Y: np.ndarray) -> np.ndarray:

        Z = self.DCT.decompress(Y)

        return self.post_filter(Z)



# --- DWT ---

class DWTCompression:
    
    def __init__(self, n: int, steps: np.ndarray, log: bool = True):
        self.n = n
        self.steps = steps
        self.bool = bool

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

    def encode(self, X: np.ndarray) -> Tuple[np.ndarray, HuffmanTable]:

        Y = self.compress(X)
        Yq = Y # quantise 

        # Below is from JPEG:
        # Bunch of other features from the JPEG encoding not included at this stage 

        scan = diagscan(M) # scan pattern
        dhufftab = huffdflt(1)  # Default huffman table
        huffcode, ehuf = huffgen(dhufftab)

        M = 16 # scanning block size 
        N = 16 # another one?
        dcbits = 8 # number of bits to encode 

        sy = Yq.shape
        huffhist = np.zeros(16 ** 2)
        vlc = []

        for r in range(0, sy[0], M):
            for c in range(0, sy[1], M):

                yq = Yq[r:r+M,c:c+M]

                if M > N:
                     yq = regroup(yq, N) # Possibly regroup

                yqflat = yq.flatten('F')
                
                dccoef = yqflat[0] + 2 ** (dcbits-1) # Encode DC coefficient first

                if dccoef not in range(2**dcbits):
                    raise ValueError('DC coefficients too large for desired number of bits')

                vlc.append(np.array([[dccoef, dcbits]]))
                
                ra1 = runampl(yqflat[scan]) # Encode the other AC coefficients in scan order
                vlc.append(huffenc(huffhist, ra1, ehuf)) # huffenc() also updates huffhist.

        # (0, 2) array makes this work even if `vlc == []`
        vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

        if self.log:
            print('Bits for coded image = {}'.format(sum(vlc[:, 1])))

        return (vlc, dhufftab) # "variable length code" and header (huffman table "hufftab" or other)

    def decode(self, vlc: np.ndarray, hufftab: HuffmanTable, log: bool = True) -> np.ndarray:

        scan = diagscan(M)
        hufftab = huffdflt(1)

        # Define starting addresses of each new code length in huffcode.
        huffstart = np.cumsum(np.block([0, hufftab.bits[:15]])) # 0-based indexing instead of 1
        huffcode, ehuf = huffgen(hufftab) # Set up huffman coding arrays.

        k = 2 ** np.arange(17) # Define array of powers of 2 from 1 to 2^16.

        # For each block in the image:

        # Decode the dc coef (a fixed-length word)
        # Look for any 15/0 code words.
        # Choose alternate code words to be decoded (excluding 15/0 ones).
        # and mark these with vector t until the next 0/0 EOB code is found.
        # Decode all the t huffman codes, and the t+1 amplitude codes.

        eob = ehuf[0]
        run16 = ehuf[15 * 16]
        i = 0
        Zq = np.zeros((H, W))

        M = 16 # scanning block size 
        N = 16 # another one?
        dcbits = 8 # number of bits to encode 
        qstep = 16 # another mystery parameter of this process

        for r in range(0, H, M):
            for c in range(0, W, M):

                yq = np.zeros(M**2)
                cf = 0 # Decode DC coef - assume no of bits is correctly given in vlc table.

                if vlc[i, 1] != dcbits:
                    raise ValueError('The bits for the DC coefficient does not agree with vlc table')

                yq[cf] = vlc[i, 0] - 2 ** (dcbits-1)
                i += 1

                while np.any(vlc[i] != eob): # Loop for each non-zero AC coef.
                    run = 0
                    
                    while np.all(vlc[i] == run16): # Decode any runs of 16 zeros first.
                        run += 16
                        i += 1

                    # Decode run and size (in bits) of AC coef.
                    start = huffstart[vlc[i, 1] - 1]
                    res = hufftab.huffval[start + vlc[i, 0] - huffcode[start]]
                    run += res // 16
                    cf += run + 1
                    si = res % 16
                    i += 1
                    
                    if vlc[i, 1] != si: # Decode amplitude of AC coef.
                        raise ValueError('Problem with decoding .. you might be using the wrong hufftab table')
                    ampl = vlc[i, 0]

                    thr = k[si - 1] # Adjust ampl for negative coef (i.e. MSB = 0).
                    yq[scan[cf-1]] = ampl - (ampl < thr) * (2 * thr - 1)
                    i += 1

                i += 1 # End-of-block detected, save block.

                yq = yq.reshape((M, M)).T

                if M > N: # Possibly regroup yq
                    yq = regroup(yq, M//N)

                Zq[r:r+M, c:c+M] = yq

        Zi = quant2(Zq, qstep, qstep)
        Z = self.decompress(Zi)

        return Z


class SVDCompression:

    def __init__(self, K) -> None:

        self.K = K

    def compress(self, X: np.ndarray) -> List[np.ndarray]:

        U, s, V = np.linalg.svd(X)
        U = U[:, :self.K]
        s = s[:self.K]
        V = V[:self.K, :]

        return [U, s, V]

    def decompress(self, Y: List[np.ndarray]) -> np.ndarray:

        return Y[0] @ np.diag(Y[1]) @ Y[2]

