import pytest
import numpy as np
import scipy.io as sio
from pathlib import Path

from cued_sf2_lab.jpeg import (
    diagscan, runampl,
    huffdes,
    dwtgroup,
    jpegenc, jpegdec,
    vlctest)


this_dir = Path(__file__).parent

@pytest.fixture(scope="module")
def X():
    '''Return the lighthouse image X'''
    return sio.loadmat(str(this_dir / '../lighthouse.mat'))['X'].astype(float)


@pytest.fixture(scope="module")
def jpegout():
    '''Return the lighthouse image X'''
    return sio.loadmat(str(this_dir / 'jpegout.mat'))


@pytest.fixture(scope="module")
def jpegout_custom():
    '''Return the lighthouse image X'''
    return sio.loadmat(str(this_dir / 'jpegout_custom.mat'))


def test_jpegenc(X, jpegout):
    '''Test jpegenc with the lighthouse image and qstep=17'''
    vlc, (_bits, _huffval) = jpegenc(X-128, 17)
    vlctest(vlc)
    diff = vlc - jpegout['vlc'].astype(int)  # index 17548 off by one on Mac
    assert (np.array_equal(vlc, jpegout['vlc'].astype(int)) or
    (np.where(diff != 0)[0] == np.array(17548) and diff[17548, 0] == -1))


def test_jpegenc_custom(X, jpegout_custom):
    '''Test jpegenc with the lighthouse image and qstep=17 and custom optimised tables'''
    vlc, (_bits, _huffval) = jpegenc(X-128, 17, opthuff=True)
    diff = vlc - jpegout_custom['vlc'].astype(int)  # index 17548 off by one on Mac
    assert (np.array_equal(vlc, jpegout_custom['vlc'].astype(int)) or
    (np.where(diff != 0)[0] == np.array(17548) and diff[17548, 0] == -1))


def test_jpegdec(X, jpegout):
    vlc = jpegout['vlc'].astype(int)
    Z = jpegdec(vlc, 17)
    assert np.allclose(Z, jpegout['Z'].astype(float))


def test_dwtgroup(X, jpegout):
    test = jpegout['test'].astype(float)
    tested = dwtgroup(test, 2)
    assert np.array_equal(tested, jpegout['test_dwtgrouped'].astype(float))
    test_reverse = dwtgroup(tested, -2)
    assert np.array_equal(test_reverse, test)


def test_huffdes():
    # from lighthouse, but not important
    huffhist = np.array([[
        1024, 2258, 1525,  811,  359,  112,    3,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,  865,  404,  116,   37,    8,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
         404,  169,   27,    6,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,  251,   67,   28,   12,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  188,
          62,   16,    2,    1,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,  163,   47,    7,    5,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,  135,   25,
          11,    4,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,  120,   34,    8,    1,    1,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,   93,   23,    7,
           9,    2,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,   80,   35,    4,    5,    1,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,   71,   33,   13,    2,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,   51,   20,   14,    1,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,   39,   13,    4,    2,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          40,   10,    2,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,   18,    1,    1,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,   69,   10,
           1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0]], dtype=np.intp).T
    expected_bits = np.array([
         0,  1,  2,  3,  3,  6,  7, 10,  6, 12,  5,  6,  8,  2, 91,  0])
    expected_huffval = np.array([
          1,   0,   2,   3,  17,  18,   4,  33,  49,  19,  34,  65,  81,
         97, 113,   5,  50,  66, 129, 145, 161, 240,  20,  35,  51,  82,
        114, 146, 162, 177, 193, 209,  67,  98, 130, 178, 179, 225,  21,
         36,  52,  83,  99, 115, 131, 132, 163, 194, 210, 241,  84, 100,
        147, 148, 195,   6,  68, 133, 164, 196, 211,  69, 116, 117, 149,
        180, 226, 227, 242,   7,   8,   9,  10,  22,  23,  24,  25,  26,
         37,  38,  39,  40,  41,  42,  53,  54,  55,  56,  57,  58,  70,
         71,  72,  73,  74,  85,  86,  87,  88,  89,  90, 101, 102, 103,
        104, 105, 106, 118, 119, 120, 121, 122, 134, 135, 136, 137, 138,
        150, 151, 152, 153, 154, 165, 166, 167, 168, 169, 170, 181, 182,
        183, 184, 185, 186, 197, 198, 199, 200, 201, 202, 212, 213, 214,
        215, 216, 217, 218, 228, 229, 230, 231, 232, 233, 234, 243, 244,
        245, 246, 247, 248, 249, 250], dtype=np.uint8)
    bits, huffval = huffdes(huffhist)
    np.testing.assert_equal(expected_bits, bits)
    np.testing.assert_equal(expected_huffval, huffval)


class TestDiagscan:

    @pytest.mark.xfail(exception=ValueError)
    def test_0(self):
        assert diagscan(0).shape == (0,)

    @pytest.mark.xfail(exception=ValueError)
    def test_1(self):
        assert diagscan(1).shape == (0,)

    def test_small(self):
        np.testing.assert_equal(diagscan(2), [2, 1, 3])

    def test_order(self):
        d4 = diagscan(4)
        np.testing.assert_equal(d4,
            [4,  1,  2,  5,  8, 12,  9,  6,  3,  7, 10, 13, 14, 11, 15])

        # check `diagscan` fills out the flattened matrix in the order we expect
        x = np.full((4, 4), -1, dtype=int)
        x.ravel()[d4] = np.arange(15)
        np.testing.assert_equal(x, [
            [-1,  1,  2,  8],
            [ 0,  3,  7,  9],
            [ 4,  6, 10, 13],
            [ 5, 11, 12, 14]])


class TestRunampl:

    def test_no_zeros(self):
        ret = runampl(np.array([1, 2, 3, 4, 5]))
        np.testing.assert_equal(ret,
            [[0, 1, 1],
             [0, 2, 2],
             [0, 2, 3],
             [0, 3, 4],
             [0, 3, 5],
             [0, 0, 0]])

    def test_leading_zeros(self):
        ret = runampl(np.array([0, 0, 1, 2, 3]))
        np.testing.assert_equal(ret,
            [[2, 1, 1],  # two leading zeros
             [0, 2, 2],
             [0, 2, 3],
             [0, 0, 0]])

    def test_trailing_zeros(self):
        ret = runampl(np.array([1, 2, 3, 0, 0]))
        np.testing.assert_equal(ret,
            [[0, 1, 1],
             [0, 2, 2],
             [0, 2, 3],
             [0, 0, 0]])  # doesn't record the trailing zeros at all!

        # Is this a bug? The encoding is lossy!
        np.testing.assert_equal(
            runampl(np.array([1, 2, 3, 0, 0])),
            runampl(np.array([1, 2, 3])))

    def test_all_zeros(self):
        ret = runampl(np.array([0, 0, 0]))
        np.testing.assert_equal(ret,
            [[0, 0, 0]])

    def test_mid_zeros(self):
        ret = runampl(np.array([-4, -3, -2, -1, 0, 0, 1, 2, 3, 4]))
        np.testing.assert_equal(ret,
            [[0, 3, 3],
             [0, 2, 0],
             [0, 2, 1],
             [0, 1, 0],
             [2, 1, 1],  # two zeros
             [0, 2, 2],
             [0, 2, 3],
             [0, 3, 4],
             [0, 0, 0]])
