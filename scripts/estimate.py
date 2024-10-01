# %%
import numpy as np
import matplotlib.pyplot as plt

from rpsnumerics import L1_residual_min, L1_residual_min_cupy

def optmize_s_gs(blurs, depths, fs, fnums):
    assert len(blurs) == len(depths) == len(fs) == len(fnums)
    Ls = [f/fnum for f, fnum in zip(fs, fnums)]

    A = np.array([])
    for idx, (blur, depth, L) in enumerate(zip(blurs, depths, Ls)):
        a = [
            np.zeros_like(depth)
            if i != idx
            else depth.reshape(-1, 1) * (blur.reshape(-1, 1) + L)
            for i in range(len(blurs))
        ]
        a.append(-np.ones_like(blur).reshape(-1, 1) * L)
        a = np.array(a).T
        a = a[0]
        A = np.vstack((A, a)) if A.size else a

    b = np.array([])
    for idx, (blur, depth, f) in enumerate(zip(blurs, depths, fs)):
        b_ = (blur * depth) / f
        b = np.vstack((b, b_)) if b.size else b_
    # print(A)
    # print(b)
    try:
        result_ = L1_residual_min_cupy(A, b)
    except Exception as e:
        print("cupy error -> np")
        result_ = L1_residual_min(A, b)
    g_, s_ = result_[:-1], result_[-1]
    g_ = g_.flatten()
    g_, s_ = 1 / g_, 1 / s_
    s_ = s_[0]
    return s_, g_


def fn(g, Z, focal_length, fnum, s=1):
    L = focal_length / fnum
    b = L * focal_length / (1 - focal_length / g) * (1 / g - 1 / Z / s)
    return b


def calc_s_and_g(blur, depth, focal_length, fnum):
    L = focal_length / fnum
    A = np.hstack(
        (
            depth.reshape(-1, 1) * (blur.reshape(-1, 1) + L),
            -np.ones_like(blur).reshape(-1, 1) * L,
        )
    )
    b = (blur * depth) / focal_length

    result = np.linalg.lstsq(A, b, rcond=None)[0]
    g, s_ = result
    g, s_ = 1 / g, 1 / s_
    print(f"s = {s_}, g={g}")
    return s_, g

def calc_s_and_gs_cupy(blurs, depths, focal_length, fnum, way="L1"):
    import cupy as cp
    import cupyx.scipy as cpx
    print(f"f: {focal_length}, fnum: {fnum}")
    if way in ["L1"]:
        pass
    else:
        raise ValueError("way must be L1")

    L = focal_length / fnum
    A = np.array([])
    for idx, (blur, depth) in enumerate(zip(blurs, depths)):
        a = [
            np.zeros_like(depth)
            if i != idx
            else depth.reshape(-1, 1) * (blur.reshape(-1, 1) + L)
            for i in range(len(blurs))
        ]
        a.append(-np.ones_like(blur).reshape(-1, 1) * L)
        a = np.array(a).T
        a = a[0]
        A = np.vstack((A, a)) if A.size else a

    b = np.array([])
    for idx, (blur, depth) in enumerate(zip(blurs, depths)):
        b_ = (blur * depth) / focal_length
        b = np.vstack((b, b_)) if b.size else b_
    # print(A)
    # print(b)
    if way == "L1":
        try:
            result_ = L1_residual_min_cupy(A, b)
        except Exception as e:
            print("cupy error -> np")
            result_ = L1_residual_min(A, b)
        g_, s_ = result_[:-1], result_[-1]
        g_ = g_.flatten()
        g_, s_ = 1 / g_, 1 / s_
        s_ = s_[0]
        return s_, g_

def calc_s_and_gs(blurs, depths, focal_length, fnum, way="L1"):
    print(f"f: {focal_length}, fnum: {fnum}")
    L = focal_length / fnum
    A = np.array([])
    for idx, (blur, depth) in enumerate(zip(blurs, depths)):
        a = [
            np.zeros_like(depth)
            if i != idx
            else depth.reshape(-1, 1) * (blur.reshape(-1, 1) + L)
            for i in range(len(blurs))
        ]
        a.append(-np.ones_like(blur).reshape(-1, 1) * L)
        a = np.array(a).T
        a = a[0]
        A = np.vstack((A, a)) if A.size else a

    b = np.array([])
    for idx, (blur, depth) in enumerate(zip(blurs, depths)):
        b_ = (blur * depth) / focal_length
        b = np.vstack((b, b_)) if b.size else b_
    # print(A)
    # print(b)
    if way == "L1":
        try:
            result_ = L1_residual_min_cupy(A, b)
        except Exception as e:
            print("cupy error -> np")
            result_ = L1_residual_min(A, b)
        g_, s_ = result_[:-1], result_[-1]
        g_ = g_.flatten()
        g_, s_ = 1 / g_, 1 / s_
        s_ = s_[0]
        return s_, g_
 