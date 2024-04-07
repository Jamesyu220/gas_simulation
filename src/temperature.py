import numpy as np
def cal_temperature(m: float, R: float, v: np.ndarray) -> float:
    v_square = np.sum(np.square(v), axis=1)
    v_rms = np.sqrt(np.mean(v_square)).item()
    T = (m * v_rms ** 2) / (3 * R)
    return T