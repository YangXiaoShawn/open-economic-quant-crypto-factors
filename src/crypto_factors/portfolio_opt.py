# -*- coding: utf-8 -*-
"""
约束优化组合层 —— 在"因子收益流"上做带约束的权重分配 (因子配置, 非个券)。

输入: R = DataFrame(index=日, columns=各因子已定向的多空日收益)。
方法 (scipy SLSQP / 解析):
  max_sharpe   : 最大化组合夏普, 约束 sum(w)=1, 0<=w<=w_max  (long-only 因子配置)
  min_var      : 最小方差
  risk_parity  : 等风险贡献 (ERC), 抗单因子主导
全部支持 **walk-forward**: 每个再平衡点只用"过去 lookback 天"估计 mu/cov, 杜绝前视。
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ANN = 365


# ----------------------------------------------------------- 单期权重求解
def max_sharpe_weights(mu, cov, w_max=0.6):
    n = len(mu)
    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -ret / (vol + 1e-12)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, w_max)] * n
    res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                   bounds=bnds, constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    return res.x if res.success else np.ones(n) / n


def min_var_weights(cov, w_max=0.6):
    n = cov.shape[0]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, w_max)] * n
    res = minimize(lambda w: w @ cov @ w, np.ones(n) / n, method="SLSQP",
                   bounds=bnds, constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    return res.x if res.success else np.ones(n) / n


def risk_parity_weights(cov):
    """等风险贡献: 最小化各因子风险贡献的离散度。"""
    n = cov.shape[0]
    def obj(w):
        port_var = w @ cov @ w
        mrc = cov @ w                      # 边际风险贡献
        rc = w * mrc                       # 风险贡献
        target = port_var / n
        return np.sum((rc - target) ** 2)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(1e-4, 1.0)] * n
    res = minimize(obj, np.ones(n) / n, method="SLSQP",
                   bounds=bnds, constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    return res.x if res.success else np.ones(n) / n


def solve(R_win, method, w_max=0.6):
    mu = R_win.mean().values * ANN
    cov = R_win.cov().values * ANN
    if method == "max_sharpe":
        return max_sharpe_weights(mu, cov, w_max)
    if method == "min_var":
        return min_var_weights(cov, w_max)
    if method == "risk_parity":
        return risk_parity_weights(cov)
    raise ValueError(method)


# ----------------------------------------------------------- walk-forward 组合
def walk_forward(R, method="max_sharpe", lookback=120, rebal=20, w_max=0.6):
    """
    返回 (组合日收益 Series, 权重轨迹 DataFrame)。
    每 rebal 天用过去 lookback 天重估权重; 不足 lookback 时等权。
    """
    R = R.dropna()
    cols = list(R.columns)
    n = len(cols)
    dates = R.index
    w = np.ones(n) / n
    ret = pd.Series(0.0, index=dates, dtype=float)
    wtraj = pd.DataFrame(0.0, index=dates, columns=cols)
    for i in range(len(dates)):
        if i >= lookback and (i - lookback) % rebal == 0:
            try:
                w = solve(R.iloc[i - lookback:i], method, w_max)
            except Exception:
                pass
        ret.iloc[i] = float(R.iloc[i].values @ w)
        wtraj.iloc[i] = w
    return ret, wtraj


def equal_weight(R):
    R = R.dropna()
    return R.mean(axis=1)


def inverse_vol(R):
    R = R.dropna()
    iv = 1.0 / (R.std() + 1e-12)
    iv = iv / iv.sum()
    return (R * iv).sum(axis=1)
