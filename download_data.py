"""从 data.binance.vision 下载月度1分钟K线zip（现货），支持断点续传与完整性校验。

数据列（无表头CSV）:
0 open_time 1 open 2 high 3 low 4 close 5 volume(总成交量,不使用)
6 close_time 7 quote_asset_volume(不使用) 8 number_of_trades(不使用)
9 taker_buy_base_asset_volume(主动买量,使用) 10 taker_buy_quote_asset_volume
11 ignore
主动卖量 = volume - taker_buy_base（由程序在特征阶段推导，直接用其差值，不引入总成交量指标本身）。
"""
import concurrent.futures as cf
import os
import sys
import time
import zipfile

import pandas as pd
import requests

import config

BASE_URL = {
    "spot": "https://data.binance.vision/data/spot/monthly/klines",
    "futures/um": "https://data.binance.vision/data/futures/um/monthly/klines",
}
BASE_URL_DAILY = {
    "spot": "https://data.binance.vision/data/spot/daily/klines",
    "futures/um": "https://data.binance.vision/data/futures/um/daily/klines",
}


def months_range(start_y, start_m, end_y, end_m):
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def month_path(symbol, market, y, m, dest):
    return os.path.join(dest, f"{symbol}-1m-{y:04d}-{m:02d}.zip")


def _verify(path):
    try:
        with zipfile.ZipFile(path) as z:
            if z.testzip() is None and len(z.namelist()) >= 1:
                return True
    except Exception:
        pass
    return False


def download_month(symbol, market, y, m, dest):
    """下载单个月度文件，返回路径；404（无该月数据）返回 None。"""
    path = month_path(symbol, market, y, m, dest)
    if os.path.exists(path) and _verify(path):
        return path
    mm = f"{y:04d}-{m:02d}"
    url = f"{BASE_URL[market]}/{symbol}/1m/{symbol}-1m-{mm}.zip"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 404:
                return None
            if r.status_code == 200 and len(r.content) > 500:
                tmp = path + ".part"
                with open(tmp, "wb") as f:
                    f.write(r.content)
                if _verify(tmp):
                    os.replace(tmp, path)
                    return path
                os.remove(tmp)
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    return None


def download_current_month_daily(symbols=None, market=None, workers=6):
    """当月月度文件通常延迟几天发布, 用日线文件补齐当月数据。
    日线文件命名: {SYMBOL}-1m-YYYY-MM-DD.zip
    """
    symbols = symbols or config.SYMBOLS
    market = market or config.MARKET
    now = pd.Timestamp.now()
    tasks = []
    # 回退到最近可用日(数据源通常滞后1-2天)
    for day in range(now.day, 0, -1):
        d = now.replace(day=day)
        if d > now:
            continue
        for sym in symbols:
            dest = os.path.join(config.RAW_DIR, market, sym, "1m")
            os.makedirs(dest, exist_ok=True)
            tasks.append((sym, market, d, dest))
    total = len(tasks)
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download_daily, *t): t for t in tasks}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            if fut.result():
                ok += 1
            if i % 10 == 0 or i == total:
                print(f"[daily-fill] {i}/{total} ok={ok}", flush=True)
    print(f"[daily-fill] done ok={ok}/{total}")
    return ok


def _download_daily(symbol, market, d, dest):
    path = os.path.join(dest, f"{symbol}-1m-{d:%Y-%m-%d}.zip")
    if os.path.exists(path) and _verify(path):
        return True
    url = f"{BASE_URL_DAILY[market]}/{symbol}/1m/{symbol}-1m-{d:%Y-%m-%d}.zip"
    try:
        r = requests.get(url, timeout=120)
        if r.status_code != 200 or len(r.content) < 500:
            return False
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(r.content)
        if _verify(tmp):
            os.replace(tmp, path)
            return True
        os.remove(tmp)
    except Exception:
        pass
    return False


def download_all(symbols=None, market=None, start=None, end=None, workers=8):
    symbols = symbols or config.SYMBOLS
    market = market or config.MARKET
    start = start or (config.START_YEAR, config.START_MONTH)
    end = end or (config.END_YEAR, config.END_MONTH)
    months = list(months_range(*start, *end))
    tasks = []
    for sym in symbols:
        dest = os.path.join(config.RAW_DIR, market, sym, "1m")
        os.makedirs(dest, exist_ok=True)
        for (y, m) in months:
            tasks.append((sym, market, y, m, dest))
    total = len(tasks)
    ok = 0
    missing = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_month, *t): t for t in tasks}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            t = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = None
            if res:
                ok += 1
            else:
                missing.append((t[0], t[2], t[3]))
            if i % 20 == 0 or i == total:
                print(f"[download] {i}/{total} ok={ok} missing={len(missing)} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[download] done. ok={ok}/{total} missing={len(missing)}")
    for sym, y, m in missing:
        print(f"  missing: {sym}-1m-{y:04d}-{m:02d}")
    return missing


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    download_all(workers=workers)
    download_current_month_daily(workers=workers)
