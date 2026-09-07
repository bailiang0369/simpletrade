#!/usr/bin/env python3
"""缩周期(10/15/20/30min)数据可行性评估——纯数据、无模型。

回答"缩周期是否值得认真做"：
- 样本量: 周期越短样本越多(降方差)
- 标签难度: 正样本比例偏离 50% 程度
- edge 天花板: 简单动量(沿用过去K根收益符号)能否显著跑赢 50%?
        若动量在短周期崩到~50%, 则"edge摊薄"成立, 短周期不可学。

口径与现 30min label 一致: label[t]=1 iff close[t+K] > close[t]  (K 根=K 分钟)。
用法:
  python experiment_horizon_feas.py            # ETH+BTC
  python experiment_horizon_feas.py BTC
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

K_SET = (10, 15, 20, 30)


def run_symbol(symbol):
    print(f"\n########## {symbol} 缩周期数据可行性 ##########")
    ctx = AssetContext(symbol)
    c = ctx.c.astype(np.float64)          # close (N_raw)
    raw_ts = ctx.raw_ts
    n = len(c)

    # test 段(raw时间轴): META_VAL_END 之后
    import datetime
    def _ep(s):
        return int(datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc).timestamp())
    t_ep = _ep(config.META_VAL_END)
    is_test = raw_ts >= t_ep

    print(f"{'K(min)':<8}{'样本数':>12}{'P(up)':>8}{'动量win(全)':>14}{'动量win(test)':>15}"
          f"{'test样本':>10}")
    for K in K_SET:
        rK = np.diff(c, n=K) / c[:-K]                 # 过去K根收益 (N-K), rK[-...]对应close[t+K]-close[t]
        lbl = rK > 0.0                                # 未来K根方向 & 过去K根方向同轴
        # 动量: 用 t-K 时刻的过去K根收益符号预测 t 时刻未来K根方向
        # 简化 self-consistency: P(sign(rK[t])==sign(rK[t-K])), t>=K
        r = rK[K:]
        rprev = rK[:-K]
        m_all = float(((r > 0) == (rprev > 0)).mean())
        # test 段
        tk = is_test[K:]
        tk = tk[K:]
        if tk.sum() > 0:
            m_te = float(((r > 0) == (rprev > 0))[tk].mean())
        else:
            m_te = float("nan")
        p_up = float((rK[K:] > 0).mean())
        n_samp = int(len(r) )
        print(f"{K:<8}{n_samp:>12}{p_up:>8.4f}{m_all:>14.4f}{m_te:>15.4f}{int(tk.sum()):>10}")
    del ctx
    print("-" * 70 + "\n说明: 动量win≈50%→随机游走(无可学); >52%→存在短期动量可榨。P(up)偏离0.5越小越难。")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default=None)
    ap.add_argument("--symbols", default="ETH,BTC")
    a = ap.parse_args()
    for s in ([a.symbol] if a.symbol else a.symbols.split(",")):
        run_symbol(s)


if __name__ == "__main__":
    main()