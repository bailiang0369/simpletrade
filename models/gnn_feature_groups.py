"""GNN 特征子集系统化消融与评测 (GNN Feature Ablation & Comprehensive Evaluation)

为不影响已保存的初始 10 维版特征 (`models/gnn_features.py`)，本脚本独立定义并测试不同特征组合：
1. Group A (Core 10-dim): 初始 10 维核心特征 (lr_15, lr_120, rvol_30, rvol_60, z_30, z_120, pos_30, cvd_30, hour_sin, hour_cos)
2. Group B (Pure Momentum & Trend 10-dim): 纯动量与趋势特征 (lr_15, lr_120, lr_240, mom_60, z_30, z_60, z_120, pos_30, pos_120, pos_240)
3. Group C (Regime & Volatility 10-dim): 市场状态与波动率特征 (rvol_30, rvol_60, rvol_z_60, rvol_dir, dn_run_len, dn_net_240, dn_accel_240, low_break_cnt_120, regime_dn_bear, dn_rvol_ratio)
4. Group D (OrderFlow & CVD 8-dim): 主动买卖量与微观结构特征 (tb_act_60, ts_act_60, cvd_30, cvd_60, ret_day, max_range_30, hh_dd_60, ll_ru_60)
5. Group E (Full Set 20-dim): 扩展 20 维全能组合
"""

FEATURE_GROUPS = {
    "Group_A_Core10": [
        "lr_15", "lr_120", "rvol_30", "rvol_60", "z_30", "z_120", "pos_30", "cvd_30", "hour_sin", "hour_cos"
    ],
    "Group_B_Momentum10": [
        "lr_15", "lr_120", "lr_240", "mom_60", "z_30", "z_60", "z_120", "pos_30", "pos_120", "pos_240"
    ],
    "Group_C_Regime10": [
        "rvol_30", "rvol_60", "rvol_z_60", "rvol_dir", "dn_run_len", "dn_net_240", "dn_accel_240", "low_break_cnt_120", "regime_dn_bear", "dn_rvol_ratio"
    ],
    "Group_D_OrderFlow8": [
        "tb_act_60", "ts_act_60", "cvd_30", "cvd_60", "ret_day", "max_range_30", "hh_dd_60", "ll_ru_60"
    ],
    "Group_E_Full20": [
        "lr_15", "lr_120", "lr_240", "mom_60", "rvol_30", "rvol_60", "rvol_z_60", "z_30", "z_120", "pos_30", "pos_120",
        "cvd_30", "cvd_60", "dn_run_len", "dn_net_240", "dn_accel_240", "low_break_cnt_120", "regime_dn_bear", "hour_sin", "hour_cos"
    ]
}
