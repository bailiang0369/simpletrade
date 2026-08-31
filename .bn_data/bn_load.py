"""bn_load: 从 bn_data 仓库的 GitHub Release 直接下载并拼接年度 parquet 的共享库。

这是数据侧仓库 bn_data 提供的公用能力, 供任意下游项目复用(无需各自复制代码)。
所有请求走纯直链( github.com/{repo}/releases/latest 重定向 + releases/download/{tag}/{file} ),
不依赖 GitHub API, 避免匿名限流。下载后按 manifest.json 的 sha256 校验并缓存复用。

典型用法(下游项目):
    pip install git+https://github.com/bailiang0369/bn_data.git
    from bn_load import load_symbol

    df = load_symbol("BTC", dest_dir="data/bn_data")            # 全年份 2020-2026
    df = load_symbol("ETH", years=[2024, 2025], dest_dir="data/bn_data")
    df = load_symbol("BTCUSDT")                                  # 短名/完整名均兼容

返回 DataFrame 固定列: [ts, open, high, low, close, buy_vol, sell_vol, funding] (schema_version=1)
  ts = 该K线"下一根K的开始时间"(秒, UTC), 语义见 bn_data/schema.md
"""
import hashlib
import json
import os

import polars as pl

REPO = "bailiang0369/bn_data"
COLS = ["ts", "open", "high", "low", "close", "buy_vol", "sell_vol", "funding"]
_DEF_DEST = os.path.join(os.path.expanduser("~"), ".cache", "bn_data")


def _dest_dir(dest_dir):
    return dest_dir or os.environ.get("BN_DATA_DIR") or _DEF_DEST


def _latest_tag(repo):
    import requests
    r = requests.get(f"https://github.com/{repo}/releases/latest",
                     allow_redirects=True, timeout=30)
    r.raise_for_status()
    return r.url.rstrip("/").rsplit("/", 1)[-1]


def _base(repo=REPO, tag=None):
    return f"https://github.com/{repo}/releases/download/{tag or _latest_tag(repo)}"


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def _download(name, dest_dir, base, sha256_hex=None):
    d = _dest_dir(dest_dir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    if os.path.exists(path) and sha256_hex and _sha256(path) == sha256_hex:
        return path
    import requests
    with requests.get(f"{base}/{name}", stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = path + f".{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if sha256_hex and _sha256(tmp) != sha256_hex:
        os.remove(tmp)
        raise RuntimeError(f"sha256 不匹配 {name}")
    os.replace(tmp, path)
    return path


def load_symbol(symbol, years=None, repo=REPO, tag=None, dest_dir=None):
    """下载并拼接某币(全部或指定年份)年度 parquet, 按 ts 升序返回 DataFrame。

    参数:
        symbol:   'BTC' 或 'BTCUSDT' 均可
        years:    [2020, 2024] 只取指定年份; None=全部
        repo:     数据仓库, 默认 bailiang0369/bn_data
        tag:      数据版本标签(如 data-2026-08); None=仓库 latest Release
        dest_dir: 数据落地目录(文件存为 {dest_dir}/{SYM}_{YYYY}.parquet)
    返回:
        polars DataFrame, 列固定为 bn_load.COLS
    """
    base, tag = _base(repo, tag), (tag or _latest_tag(repo))
    man = json.load(open(_download("manifest.json", dest_dir, base)))
    short = symbol.rstrip("USDT")
    sy = man["symbols"][short]["years"]
    years = [str(y) for y in (years or [])] or sorted(sy.keys())
    frames = []
    for y in years:
        if y not in sy:
            raise FileNotFoundError(f"{short} 无 {y} 年度 (available={sorted(sy)})")
        info = sy[y]
        frames.append(pl.read_parquet(_download(info["path"], dest_dir, base, info.get("sha256"))))
    return pl.concat(frames).sort("ts").unique(subset="ts", keep="first")


def available(repo=REPO, tag=None):
    """返回该数据版本可用的 币种 -> 年度 映射(仅需下载 manifest, 很轻)。"""
    base, tag = _base(repo, tag), (tag or _latest_tag(repo))
    man = json.load(open(_download("manifest.json", None, base)))
    return {s: sorted(e["years"].keys()) for s, e in man["symbols"].items()}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--dest-dir", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    df = load_symbol(args.symbol, years=args.years, tag=args.tag, dest_dir=args.dest_dir)
    _dt = __import__("datetime").datetime.utcfromtimestamp
    print(f"{args.symbol}: rows={df.height}, cols={df.columns}")
    print(f"  ts 区间 {_dt(df['ts'][0])} -> {_dt(df['ts'][df.height-1])} (UTC)")
    print(df.head(5))