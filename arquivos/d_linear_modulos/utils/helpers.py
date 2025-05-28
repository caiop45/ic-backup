import numpy as np

def decode_hour(sin_s, cos_s):
    """
    Converte seno/cosseno da hora em inteiro de 0-23 h.
    """
    ang = np.mod(np.arctan2(sin_s, cos_s), 2 * np.pi)
    return np.rint(ang * 24 / (2 * np.pi)).astype(int) % 24
