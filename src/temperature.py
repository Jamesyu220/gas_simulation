import numpy as np

def get_v_abs(v: np.ndarray) -> np.ndarray:
    v_abs = np.sqrt(np.sum(np.square(v), axis=1, keepdims=True))
    return v_abs

def cal_temperature(m: float, R: float, v: np.ndarray) -> float:
    v_abs = get_v_abs(v)
    v_rms = np.sqrt(np.mean(np.square(v_abs))).item()
    T = (m * v_rms ** 2) / (3 * R)
    return T