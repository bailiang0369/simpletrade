"""fetch_data: 从 GitHub Release 拉取 bn_data 合并数据, 落地为 raw_{symbol}.parquet。

替代已删除的旧下载管线(refresh_data/download_data/prepare_data):
数据不再从币安逐个下载 zip 再解析, 而是直接用 bn_data 发布的 parquet
(现货+合约已合并 + funding 列), 经共享库 bn_load 一键获取。

流程:
  _ensure_bn_load()   运行期自动接入 bn_load(首次自动下载到 .bn_data/, 之后复用, 零手动)
  fetch(symbols)      对每个币: bn_load.load_symbol() -> 写 {DS_DIR}/raw_{SYM}.parquet
  tag 打点            .fetched_tag.json 记录已拉取的数据版本; 同版本且 raw 存在则跳过(增量)

这样 features/build_dataset/data_store 等下游无需改动即可消费 raw parquet。

用法:
  python fetch_data.py                 # 拉取全部 config.SYMBOLS
  python fetch_data.py --symbols BTC   # 只拉某币
  python fetch_data.py --force         # 忽略打点, 强制重拉当前版本
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import config

REPO = "bailiang0369/bn_data"
_BN_LOAD_URL = f"https://raw.githubusercontent.com/{REPO}/main/bn_load.py"
_LOADER_DIR = os.path.join(config.PROJECT_DIR, ".bn_data")   # 放共享库 bn_load.py
_BN_CACHE = os.path.join(config.DATA_DIR, "bn_cache")        # 年度 parquet 缓存(非 DS_DIR)
_STAMP = os.path.join(config.DS_DIR, ".fetched_tag.json")    # 已拉取版本打点


def _ensure_bn_load():
    """运行期自动接入 bn_load: 首次自动下载到 .bn_data/, 之后复用。"""
    os.makedirs(_LOADER_DIR, exist_ok=True)
    lp = os.path.join(_LOADER_DIR, "bn_load.py")
    if not os.path.exists(lp):
        print(f"[fetch] 首次运行, 自动下载共享库 -> {lp}")
        with urllib.request.urlopen(_BN_LOAD_URL, timeout=60) as r:
            with open(lp, "wb") as f:
                f.write(r.read())
    if _LOADER_DIR not in sys.path:
        sys.path.insert(0, _LOADER_DIR)
    import bn_load
    return bn_load


def fetch(symbols=None, tag=None, force=False):
    symbols = symbols or config.SYMBOLS
    bn_load = _ensure_bn_load()
    tag = tag or bn_load._latest_tag(REPO)          # None -> 仓库 latest 数据版本

    stamp = {}
    if os.path.exists(_STAMP):
        try:
            stamp = json.load(open(_STAMP))
        except Exception:
            stamp = {}
    symbols_stamp = stamp.setdefault("symbols", {})
    stamp["tag"] = tag

    t0 = time.time()
    for s in symbols:
        short = s.rstrip("USDT")
        if not force and os.path.exists(f"{config.DS_DIR}/raw_{s}.parquet") \
           and symbols_stamp.get(s) == tag:
            print(f"[fetch] {s}: 已是版本 {tag}, 跳过")
            continue
        df = bn_load.load_symbol(short, dest_dir=_BN_CACHE, tag=tag)
        out = f"{config.DS_DIR}/raw_{s}.parquet"
        df.write_parquet(out, compression="zstd")
        symbols_stamp[s] = tag
        json.dump(stamp, open(_STAMP, "w"), ensure_ascii=False)
        print(f"[fetch] {s}: {df.height} rows -> {out} "
              f"ts[{df['ts'][0]}..{df['ts'][df.height-1]}] ({time.time()-t0:.0f}s)", flush=True)
    print("[fetch] done")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    fetch(symbols=a.symbols, tag=a.tag, force=a.force)