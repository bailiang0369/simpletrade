"""simpletrade 模型包: 单模型 + 集成。

已实现模型:
- stat_signal.StatSignal    统计信号基线(原始K线统计特征 + 逻辑回归)
- gbdt.GBDTModel            LightGBM 梯度提升
- seq_lstm.SeqGBDLSTM       LSTM 时序模型(60分钟窗口)
- seq_alt.SeqAltModel       GRU/AttentionLSTM/TransformerEncoder 替代时序模型
"""
