import numpy as np

def huffblockhist(rsa: np.ndarray):
    """Create a huffman histogram for a single block"""
    if max(rsa[:, 1]) > 10:
        print("Warning: Size of value in run-amplitude "
              "code is too large for Huffman table")
    rsa[rsa[:, 1] > 10, 1:3] = [10, (2 ** 10) - 1]

    huffhist = np.zeros(16 ** 2) 
    for i in range(rsa.shape[0]):
        run = rsa[i, 0]
        # If run > 15, use repeated codes for 16 zeros.
        run, count16 = run % 16, run // 16
        huffhist[15 * 16] += count16
        # Code the run and size.
        # Got rid off + 1 to suit python indexing.
        code = run * 16 + rsa[i, 1]
        huffhist[code] += 1

    return huffhist


def huffencopt(rsa: np.ndarray, ehuf: np.ndarray
    ) -> np.ndarray:
    """
    Convert a run-length encoded stream to huffman coding.

    Parameters:
        rsa: run-length information as provided by `runampl`.
        ehuf: the huffman codes and their lengths

    Returns:
        vlc: Variable-length codewords, consisting of codewords in ``vlc[:,0]``
            and corresponding lengths in ``vlc[:,1]``.
    """
    if max(rsa[:, 1]) > 10:
        print("Warning: Size of value in run-amplitude "
              "code is too large for Huffman table")
    rsa[rsa[:, 1] > 10, 1:3] = [10, (2 ** 10) - 1]

    vlc = []
    for i in range(rsa.shape[0]):
        run = rsa[i, 0]
        # If run > 15, use repeated codes for 16 zeros.
        run, count16 = run % 16, run // 16
        vlc += [ehuf[15 * 16] for _ in range(count16)]
        # Code the run and size.
        code = run * 16 + rsa[i, 1]
        vlc.append(ehuf[code])
        # If size > 0, add in the remainder (which is not coded).
        if rsa[i, 1] > 0:
            vlc.append(rsa[i, [2, 1]])
    return np.array(vlc)