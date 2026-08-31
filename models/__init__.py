"""simpletrade 模型包: 单模型 + 集成。

已实现:
- stat_signal.StatSignal  统计信号基线(原始K线统计特征 + 逻辑回归)

待实现(规划中):
- base.BaseModel          模型基类(ensemble 依赖其类型注解)
- gbdt.GBDTModel          LightGBM 梯度提升
- dl_seq.GRUModel         PyTorch GRU 序列模型
- pattern.CNNImageModel / DTWKNN / FAISSNN / TimeNormKNN
"""
