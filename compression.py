
from typing import List, Tuple, NamedTuple, Optional
import numpy as np
from cued_sf2_lab.laplacian_pyramid import rowdec,rowint, quantise, quant1, quant2, bpp
from cued_sf2_lab.dct import colxfm, dct_ii, regroup
from cued_sf2_lab.lbt import pot_ii, dct_ii
from cued_sf2_lab.dwt import dwt, idwt
from scipy import optimize
from cued_sf2_lab.jpeg import diagscan, dwtgroup, huffdflt, HuffmanTable, huffenc, huffgen, huffdes, runampl
from encoding import huffblockhist, huffencopt
from quantisation import quant, inv_quant

from time import perf_counter


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

    def __init__(self, s: float = 1.3, N: int = 8):
        
        self.name = "LBT"
        self.N = N
        self.Pf, self.Pr = pot_ii(N, s=s)
        self.DCT = DCTCompression(N)
        self.grad_r = 1.039

    def pre_filter(self, X: np.ndarray) -> np.ndarray:

        N = self.Pf.shape[0]
        t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
        Xp = X.copy()  # copy the non-transformed edges directly from X
        Xp[t,:] = colxfm(Xp[t,:], self.Pf)
        Xp[:,t] = colxfm(Xp[:,t].T, self.Pf).T
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

    def quant(self, Y: np.ndarray, step: float) -> np.ndarray:

        N = self.N
        width = Y.shape[0]
        h = width//N

        Yr = regroup(Y, N)
        Yr[0:N, 0:N] = quant(Yr[0:N, 0:N], 1)

        for i in range(1, 2*h):
            for j in range(i+1):

                Yr[(i-j)*N:(i-j+1)*N, j*N:(j+1)*N] = quant(Yr[(i-j)*N:(i-j+1)*N, j*N:(j+1)*N], step*((1.3)**(max(1, i-12))))

        Yr = regroup(Yr, h) / h

        return Yr

    def inv_quant(self, Y: np.ndarray, step: float) -> np.ndarray:

        N = self.N
        width = Y.shape[0]
        h = width//N

        Yr = regroup(Y, N) * N
        Yr[0:N, 0:N] = inv_quant(Yr[0:N, 0:N], 1)

        for i in range(1, 2*h):
            for j in range(i+1):

                Yr[(i-j)*N:(i-j+1)*N, j*N:(j+1)*N] = inv_quant(Yr[(i-j)*N:(i-j+1)*N, j*N:(j+1)*N], step*((1.3)**(max(1, i-12))))

        Yr = regroup(Yr, h)

        return Yr

    def quant2(self, Y: np.ndarray, step: float) -> np.ndarray:
        
        N = self.N
        Yq = Y.copy()

        x = np.arange(N)
        steps = step * (self.grad_r ** np.add.outer(x, x))
        steps = np.maximum(steps, 1)

        # Yr = regroup(Y, N)

        for i in range(N):
            for j in range(N):

                if j == i == 0:

                    Yq[i::N, j::N] = quant(Y[i::N, j::N], 1)

                else:

                    Yq[i::N, j::N] = quant(Y[i::N, j::N], steps[i, j])           

        return Yq

    def inv_quant2(self, Y: np.ndarray, step: float) -> np.ndarray:

        N = self.N
        Yq = Y.copy()

        x = np.arange(N)
        steps = step * (self.grad_r ** np.add.outer(x, x))
        steps = np.maximum(steps, 1)

        for i in range(N):
            for j in range(N):

                if j == i == 0:

                    Yq[i::N, j::N] = inv_quant(Y[i::N, j::N], 1)

                else:

                    Yq[i::N, j::N] = inv_quant(Y[i::N, j::N],  steps[i, j])

        return Yq

    def encode(self, Y: np.ndarray, qstep: float, M: int = 8, dcbits: int = 16, log: bool = False, quant_grad=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pass in a transformed image, Y, and
         - regroup
         - quantise
         - generate huffman encoding"""
        if quant_grad:
            Yq = self.quant(Y, qstep).astype(int)
        else:
            Yq = self.quant2(Y, qstep).astype(int)

        # Generate zig-zag scan of AC coefs.
        scan = diagscan(M)

        huffhist = np.zeros(16 ** 2)
        t = perf_counter()
        # First pass to generate histogram
        for r in range(0, Yq.shape[0], M):
            for c in range(0, Yq.shape[1], M):
                yq = Yq[r:r+M,c:c+M]
                if M > self.N:
                    yq = regroup(yq, self.N)
                yqflat = yq.flatten('F')
                dccoef = yqflat[0] + 2 ** (dcbits-1)

                ra1 = runampl(yqflat[scan])
                huffhist += huffblockhist(ra1)

        # Use histogram
        vlc = []
        dhufftab = huffdes(huffhist)
        huffcode, ehuf = huffgen(dhufftab)
        t = perf_counter()
        for r in range(0, Yq.shape[0], M):
            for c in range(0, Yq.shape[1], M):
                yq = Yq[r:r+M,c:c+M]
                # Possibly regroup
                if M > self.N:
                    yq = regroup(yq, self.N)
                yqflat = yq.flatten('F')
                # Encode DC coefficient first
                dccoef = yqflat[0] + 2 ** (dcbits-1)
                if dccoef > 2**dcbits:
                    raise ValueError(
                        'DC coefficients too large for desired number of bits')
                vlc.append(np.array([[dccoef, dcbits]]))
                # Encode the other AC coefficients in scan order
                # huffenc() also updates huffhist.
                ra1 = runampl(yqflat[scan])
                vlc.append(huffencopt(ra1, ehuf))
        # (0, 2) array makes this work even if `vlc == []`
        vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

        return (vlc, dhufftab) # "variable length code" and header (huffman table "hufftab" or other)

    def decode(self, vlc: np.ndarray, qstep: float = 17, M: int = 8,
            hufftab: Optional[HuffmanTable] = None,
            dcbits: int = 16, W: int = 256, H: int = 256, log: bool = False,
            quant_grad=False) -> np.ndarray:
        '''
        Decodes a (simplified) JPEG bit stream to an image

        Parameters:

            vlc: variable length output code from jpegenc
            qstep: quantisation step to use in decoding
            N: width of the DCT block (defaults to 8)
            M: width of each block to be coded (defaults to N). Must be an
                integer multiple of N - if it is larger, individual blocks are
                regrouped.
            hufftab: if supplied, these will be used in Huffman decoding
                of the data, otherwise default tables are used
            dcbits: the number of bits to use to decode the DC coefficients
                of the DCT
            W, H: the size of the image (defaults to 256 x 256)

        Returns:

            Z: the output greyscale image
        '''

        opthuff = (hufftab is not None)
        if M % self.N != 0:
            raise ValueError('M must be an integer multiple of N!')

        # Set up standard scan sequence
        scan = diagscan(M)

        if opthuff:
            if len(hufftab.bits.shape) != 1:
                raise ValueError('bits.shape must be (len(bits),)')
            if log:
                print('Generating huffcode and ehuf using custom tables')
        else:
            if log:
                print('Generating huffcode and ehuf using default tables')
            hufftab = huffdflt(1)
        # Define starting addresses of each new code length in huffcode.
        # 0-based indexing instead of 1
        huffstart = np.cumsum(np.block([0, hufftab.bits[:15]]))
        # Set up huffman coding arrays.
        huffcode, ehuf = huffgen(hufftab)

        # Define array of powers of 2 from 1 to 2^16.
        k = 2 ** np.arange(17)

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

        if log:
            print('Decoding rows')
        for r in range(0, H, M):
            for c in range(0, W, M):
                yq = np.zeros(M**2)

                # Decode DC coef - assume no of bits is correctly given in vlc table.
                cf = 0
                if vlc[i, 1] != dcbits:
                    raise ValueError(
                        'The bits for the DC coefficient does not agree with vlc table')
                yq[cf] = vlc[i, 0] - 2 ** (dcbits-1)
                i += 1

                # Loop for each non-zero AC coef.
                while np.any(vlc[i] != eob):
                    run = 0

                    # Decode any runs of 16 zeros first.
                    while np.all(vlc[i] == run16):
                        run += 16
                        i += 1

                    # Decode run and size (in bits) of AC coef.
                    start = huffstart[vlc[i, 1] - 1]
                    res = hufftab.huffval[start + vlc[i, 0] - huffcode[start]]
                    run += res // 16
                    cf += run + 1
                    si = res % 16
                    i += 1

                    # Decode amplitude of AC coef.
                    if vlc[i, 1] != si:
                        raise ValueError(
                            'Problem with decoding .. you might be using the wrong hufftab table')
                    ampl = vlc[i, 0]

                    # Adjust ampl for negative coef (i.e. MSB = 0).
                    thr = k[si - 1]
                    yq[scan[cf-1]] = ampl - (ampl < thr) * (2 * thr - 1)

                    i += 1

                # End-of-block detected, save block.
                i += 1

                yq = yq.reshape((M, M)).T

                # Possibly regroup yq
                if M > self.N:
                    yq = regroup(yq, M//self.N)
                Zq[r:r+M, c:c+M] = yq

        if log:
            print('Inverse quantising to step size of {}'.format(qstep))

        if quant_grad:
            Zi = self.inv_quant(Zq, qstep)
        else:
            Zi = self.inv_quant2(Zq, qstep)

        

        if log:
            print('Inverse {} x {} DCT\n'.format(self.N, self.N))

        return Zi


    def opt_encode(self, Y: np.ndarray, size_lim=40960,  M: int = 8, quant_grad=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        def error(qstep: float) -> int:

            Z, h = self.encode(Y, qstep, M=M, quant_grad=quant_grad)
            size = Z[:, 1].sum()

            return np.sum((size - size_lim)**2)

        res = optimize.minimize_scalar(error, method="bounded", bounds=(1 if quant_grad else 16, 4 if quant_grad else 128))

        vlc, hufftab = self.encode(Y, res.x, M=M, quant_grad=quant_grad)

        return (vlc, hufftab), res.x


# --- DWT ---

class DWTCompression:
    
    def __init__(self, n: int, log: bool = True):
        self.name = "DWT"
        self.n = n
        self.log = log
        self.img_size = (256, 256)

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

    def estimate_entropy(self, Y):
        """Estimate the entropy of the image by considering coding each block"""
        dwtent = 0

        for i in range(self.n):

            m = 256//(2**i) # 256, 128, 64 ... 
            h = m//2 # Midpoint: 128, 64, 32 ...

            dwtent += img_size(Y[:h, h:m]) + img_size(Y[h:m, :h]) + img_size(Y[h:m, h:m])

        # Final low pass image
        dwtent += img_size(Y[:m, :m])

        return dwtent

    def constant_steps(self, step: float = 1.):
        dwtstep = np.ones((3, self.n)) * step
        dwtstep = np.concatenate((dwtstep, np.ones((3, 1))), axis=1)

        return dwtstep

    def equal_mse_steps(self, initial: float = 1.,ratio: float = 2., root2: bool = True):
        """Return a (3 x N+1) np array of step sizes with constant ratio"""

        if root2:
            const_ratio = np.logspace(start=self.n, stop=0, num=self.n, base=ratio) * initial
            dwtstep = np.stack((const_ratio, const_ratio, const_ratio * np.sqrt(2)))
            # append ones for DC componenta
            dwtstep = np.concatenate((dwtstep, np.ones((3, 1))), axis=1)
        else:
            dwtstep = np.array([np.ones((1, 3))[0]*initial*(0.5**i) for i in range(self.n + 1)]).T

        return dwtstep

    def quantise(self, Y, steps: np.ndarray, rise_ratio=None):
        """Quantise as integers"""
        Yq = np.zeros_like(Y)
        if rise_ratio is None: rise_ratio = 0.5
        for i in range(self.n):

            m = 256//(2**i) # 256, 128, 64 ... 
            h = m//2 # Midpoint: 128, 64, 32 ...

            # Quantising
            s_tr = max(steps[0, i], 1)
            s_bl = max(steps[1, i], 1)
            s_br = max(steps[2, i], 1)

            Yq[:h, h:m] = quant(Y[:h, h:m],s_tr, s_tr * rise_ratio) # Top right 
            Yq[h:m, :h] = quant(Y[h:m, :h], s_bl, s_bl * rise_ratio) # Bottom left
            Yq[h:m, h:m] = quant(Y[h:m, h:m], s_br, s_br * rise_ratio) # Bottom right

        # Final low pass image

        # Yq[128:, 128:] = 0

        m = 256//(2**self.n)
        s_tr = max(steps[0, self.n], 1)
        Yq[:m, :m] = quant(Y[:m, :m], s_tr, s_tr * rise_ratio)

        return Yq.astype(int)

    def inv_quantise(self, Y, steps: np.ndarray, rise_ratio=None):
        """Quantise as integers"""
        Yq = np.zeros_like(Y)
        if rise_ratio is None: rise_ratio = 0.5
        for i in range(self.n):
            m = 256//(2**i) # 256, 128, 64 ... 
            h = m//2 # Midpoint: 128, 64, 32 ...

            s_tr = max(steps[0, i], 1)
            s_bl = max(steps[1, i], 1)
            s_br = max(steps[2, i], 1)

            Yq[:h, h:m] = inv_quant(Y[:h, h:m],s_tr, s_tr * rise_ratio) # Top right 
            Yq[h:m, :h] = inv_quant(Y[h:m, :h], s_bl, s_bl * rise_ratio) # Bottom left
            Yq[h:m, h:m] = inv_quant(Y[h:m, h:m], s_br, s_br * rise_ratio) # Bottom right

        # Final low pass image
        m = 256//(2**self.n)
        s_tr = max(steps[0, self.n], 1)
        Yq[:m, :m] = inv_quant(Y[:m, :m], s_tr, s_tr * rise_ratio)
        return Yq.astype(int)

    def encode(self, Y: np.ndarray, qstep: Optional[int] = None, M: Optional[int] = None, dcbits: int = 16, rise_ratio=None, root2: bool=True) -> Tuple[np.ndarray, HuffmanTable]:
        """Pass in a transformed image, Y, and
         - regroup
         - quantise
         - generate huffman encoding"""

        if qstep is None:
            dwtsteps = self.constant_steps()
        else:
            dwtsteps=self.equal_mse_steps(qstep, root2=root2)

        Yq = self.quantise(Y, dwtsteps, rise_ratio=rise_ratio)

        Yq = dwtgroup(Yq, self.n)
        self.A = Yq


        N = np.round(2**self.n)
        if M is None:
            M = N

        # Generate zig-zag scan of AC coefs.
        scan = diagscan(M)

        huffhist = np.zeros(16 ** 2)
        t = perf_counter()
        # First pass to generate histogram
        for r in range(0, Yq.shape[0], M):
            for c in range(0, Yq.shape[1], M):
                yq = Yq[r:r+M,c:c+M]
                if M > N:
                    yq = regroup(yq, N)
                yqflat = yq.flatten('F')
                dccoef = yqflat[0] + 2 ** (dcbits-1)

                ra1 = runampl(yqflat[scan])
                huffhist += huffblockhist(ra1)

        # Use histogram
        vlc = []
        dhufftab = huffdes(huffhist)
        huffcode, ehuf = huffgen(dhufftab)

        for r in range(0, Yq.shape[0], M):
            for c in range(0, Yq.shape[1], M):
                yq = Yq[r:r+M,c:c+M]
                # Possibly regroup
                if M > N:
                    yq = regroup(yq, N)
                yqflat = yq.flatten('F')
                # Encode DC coefficient first
                dccoef = yqflat[0] + 2 ** (dcbits-1)
                if dccoef > 2**dcbits:
                    raise ValueError(
                        'DC coefficients too large for desired number of bits')
                vlc.append(np.array([[dccoef, dcbits]]))
                # Encode the other AC coefficients in scan order
                # huffenc() also updates huffhist.
                ra1 = runampl(yqflat[scan])
                vlc.append(huffencopt(ra1, ehuf))
        # (0, 2) array makes this work even if `vlc == []`
        vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

        return (vlc, dhufftab) # "variable length code" and header (huffman table "hufftab" or other)

    def decode(self, vlc: np.ndarray, qstep: Optional[int] = None, hufftab: Optional[HuffmanTable] = None, N: int = 8, M: int = 8, dcbits: int = 16, rise_ratio=None, root2: bool=True) -> np.ndarray:

        N = np.round(2**self.n)

        if M is None:
            M = N

        if qstep is None:
            dwtsteps = self.constant_steps()
        else:
            dwtsteps = self.equal_mse_steps(qstep, root2=root2)

        scan = diagscan(M)

        # Define starting addresses of each new code length in huffcode.
        huffstart = np.cumsum(np.block([0, hufftab.bits[:15]]))
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
        Zq = np.zeros(self.img_size)

        W, H = self.img_size
        for r in range(0, H, M):
            for c in range(0, W, M):
                yq = np.zeros(M**2)

                # Decode DC coef - assume no of bits is correctly given in vlc table.
                cf = 0 
                if vlc[i, 1] != dcbits:
                    raise ValueError('The bits for the DC coefficient does not agree with vlc table')

                yq[cf] = vlc[i, 0] - 2 ** (dcbits-1)
                i += 1

                while vlc[i, 0] != eob[0]: # Loop for each non-zero AC coef.
                    run = 0

                    while vlc[i, 0] == run16[0]: # Decode any runs of 16 zeros first.
                        run += 16
                        i += 1

                    # Decode run and size (in bits) of AC coef.
                    start = huffstart[vlc[i, 1] - 1]
                    res = hufftab.huffval[start + vlc[i, 0] - huffcode[start]]
                    run += res // 16
                    cf += run + 1
                    si = res % 16
                    i += 1
                    
                    # Assume no problem with Huffman table/decoding
                    ampl = vlc[i, 0]

                    thr = k[si - 1] # Adjust ampl for negative coef (i.e. MSB = 0).
                    yq[scan[cf-1]] = ampl - (ampl < thr) * (2 * thr - 1)
                    i += 1

                i += 1 # End-of-block detected, save block.

                yq = yq.reshape((M, M)).T

                if M > N: # Possibly regroup yq
                    yq = regroup(yq, M//N)

                Zq[r:r+M, c:c+M] = yq

        Zq = dwtgroup(Zq, -self.n)
        Z = self.inv_quantise(Zq, dwtsteps, rise_ratio=rise_ratio)
        return Z

    def opt_encode(self, Y: np.ndarray, size_lim=40960, M: int = 8, root2: bool=True, rise_ratio=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            
        def error(qstep: int) -> int:

            Z, h = self.encode(Y, qstep=qstep, M=M, rise_ratio=rise_ratio, root2=root2)
            size = Z[:, 1].sum()

            return np.sum((size - size_lim)**2)

        opt_step = optimize.minimize_scalar(error, method="bounded", bounds=(0.1, 64)).x
        vlc, hufftab = self.encode(Y, qstep=opt_step, M=M, rise_ratio=rise_ratio, root2=root2)

        return (vlc, hufftab), opt_step


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

