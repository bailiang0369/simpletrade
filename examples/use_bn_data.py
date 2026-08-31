"""下游项目复用 bn_data 共享加载器的用法示例(完全自动, 零手动操作)。

公用 "从 GitHub Release 下载并拼接 parquet" 的能力封装在数据侧仓库
bn_data 的 bn_load.py(单个自包含文件)。本脚本在运行时自动完成接入,
**无需手动拷贝文件、无需 pip install**: 首次运行自动把 bn_load.py 从 GitHub
下载到本地缓存目录并加载, 之后命中缓存直接使用。

然后一行调用即可(自动: 解析 latest 版本 -> 按需下载到 dest_dir -> sha256 校验 -> 拼接):

    from bn_load import load_symbol
    df = load_symbol("BTC")                        # 全年份 2020-2026
    df = load_symbol("ETH", years=[2024, 2025])    # 只取 2024+2025
    df = load_symbol("BTCUSDT", dest_dir=".data", tag="data-2026-08")  # 固定版本 + 指定落地目录

列固定为 [ts, open, high, low, close, buy_vol, sell_vol, funding] (schema_version=1),
ts = 下一根K开始时间(秒, UTC)。详见 bn_data/schema.md。

运行本示例:
    python examples/use_bn_data.py --symbol ETH --years 2024 2025
"""
import argparse
import os
import sys
import urllib.request

# bn_load.py 在数据仓库里的存储路径(自动拉取用)
_BN_REPO = "bailiang0369/bn_data"
_BN_LOAD_URL = f"https://raw.githubusercontent.com/{_BN_REPO}/main/bn_load.py"


def _ensure_bn_load():
    """运行期自动接入 bn_load: 首次下载到本地缓存, 之后复用。返回可用的 bn_load 模块。"""
    # 缓存目录: 本项目根下的 .bn_data/ (非 parquet 下载目录, 仅存 bn_load.py 本身)
    here = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.abspath(os.path.join(here, "..", ".bn_data"))
    os.makedirs(cache_dir, exist_ok=True)
    local_file = os.path.join(cache_dir, "bn_load.py")

    if not os.path.exists(local_file):
        print(f"[bn_load] 首次运行, 自动下载共享库 -> {local_file}")
        with urllib.request.urlopen(_BN_LOAD_URL, timeout=60) as r:
            data = r.read()
        with open(local_file, "wb") as f:
            f.write(data)
    # 把缓存目录加入 import 搜索路径, 然后导入
    if cache_dir not in sys.path:
        sys.path.insert(0, cache_dir)
    import bn_load
    return bn_load


def main(symbol, years, dest_dir, tag):
    bn_load = _ensure_bn_load()
    df = bn_load.load_symbol(symbol, years=years, dest_dir=dest_dir, tag=tag)
    _dt = __import__("datetime").datetime.utcfromtimestamp
    print(f"{symbol}: rows={df.height}, ts 区间 "
          f"{_dt(df['ts'][0])} -> {_dt(df['ts'][df.height-1])} (UTC)")
    print(df.head(5))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--dest-dir", default=".data")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    main(a.symbol, a.years, a.dest_dir, a.tag)