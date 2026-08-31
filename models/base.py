"""模型统一接口: 每个模型在 train 上 fit, 在任意切分上输出"上涨概率"(对齐该切分的行)。"""
import abc


class BaseModel(abc.ABC):
    name = "base"

    def __init__(self, seed=None):
        self.seed = seed or 42
        self.fitted = False

    @abc.abstractmethod
    def fit(self, ctx):
        """ctx: AssetContext。用 ctx 的 train 切分训练, early_stop 切分做早停/验证。"""

    @abc.abstractmethod
    def predict(self, ctx, split):
        """返回长度为该切分行数的一维数组, 值为上涨概率 [0,1]。"""
