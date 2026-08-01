import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

START = "2010-01-01"
END = "2026-08-01"
NAVER_END = "20260731"
INITIAL_SHARES = 125
TACTICAL_LIMIT = 10

# Model costs: assumed brokerage 0.015% on both sides and 2026 KOSPI
# securities transaction tax + rural special tax of 0.20% on sale.
BUY_COST = 0.00015
SELL_COST = 0.00215


def download_data():
    meta = {}
    naver = yahoo = None

    url = (
        "https://api.finance.naver.com/siseJson.naver"
        f"?symbol=000660&requestType=1&startTime=20100101"
        f"&endTime={NAVER_END}&timeframe=day"
    )
    try:
        response = requests.get(
            url, timeout=60, headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()
        rows = ast.literal_eval(response.text.strip())
        naver = pd.DataFrame(
            rows[1:], columns=[str(value).strip() for value in rows[0]]
        ).rename(
            columns={
                "날짜": "date",
                "시가": "open",
                "고가": "high",
                "저가": "low",
                "종가": "close",
                "거래량": "volume",
            }
        )
        naver = naver[["date", "open", "high", "low", "close", "volume"]]
        naver["date"] = pd.to_datetime(
            naver["date"].astype(str), format="%Y%m%d"
        )
        for column in naver.columns[1:]:
            naver[column] = pd.to_numeric(naver[column], errors="coerce")
        naver = (
            naver.dropna()
            .drop_duplicates("date")
            .set_index("date")
            .sort_index()
        )
        meta["naver_rows"] = len(naver)
    except Exception as exc:
        meta["naver_error"] = repr(exc)

    try:
        yahoo = yf.download(
            "000660.KS",
            start=START,
            end=END,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if isinstance(yahoo.columns, pd.MultiIndex):
            yahoo.columns = yahoo.columns.get_level_values(0)
        yahoo = yahoo.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )[["open", "high", "low", "close", "volume"]].dropna()
        meta["yahoo_rows"] = len(yahoo)
    except Exception as exc:
        meta["yahoo_error"] = repr(exc)

    if naver is None and yahoo is None:
        raise RuntimeError(meta)

    if naver is not None and yahoo is not None:
        overlap = naver[["close"]].join(
            yahoo[["close"]], how="inner", lsuffix="_naver", rsuffix="_yahoo"
        )
        difference = (overlap["close_naver"] / overlap["close_yahoo"] - 1).abs()
        meta.update(
            overlap_rows=len(overlap),
            median_abs_close_difference_pct=float(difference.median() * 100),
            p95_abs_close_difference_pct=float(difference.quantile(0.95) * 100),
            latest_naver_close=float(overlap["close_naver"].iloc[-1]),
            latest_yahoo_close=float(overlap["close_yahoo"].iloc[-1]),
        )

    data = naver if naver is not None and len(naver) > 500 else yahoo
    meta["primary_source"] = "Naver" if data is naver else "Yahoo"
    return data, meta


def add_indicators(data, high_window=120):
    frame = data.copy()
    frame["return_5d"] = frame["close"].pct_change(5)
    frame["ma20"] = frame["close"].rolling(20).mean()
    frame["ma60"] = frame["close"].rolling(60).mean()
    frame["ma200"] = frame["close"].rolling(200).mean()
    frame["volume20"] = frame["volume"].rolling(20).mean()
    frame["volume_ratio"] = frame["volume"] / frame["volume20"]
    frame["recent_high"] = frame["high"].rolling(high_window).max()
    frame["ma20_gap"] = frame["close"] / frame["ma20"] - 1
    frame["high_gap"] = frame["close"] / frame["recent_high"] - 1
    return frame


def simulate(raw, start, end, rule=None, cost_multiplier=1.0):
    # Indicators are calculated on all history before slicing so the first
    # day of an evaluation period has a genuine warm-up.
    all_data = add_indicators(raw, (rule or {}).get("high_window", 120))
    data = all_data.loc[start:end].copy()
    if len(data) < 3:
        raise ValueError(f"Insufficient data for {start} to {end}")

    start_price = float(data["open"].iloc[0])
    end_price = float(data["close"].iloc[-1])

    if rule is None:
        equity = INITIAL_SHARES * data["close"]
        years = (data.index[-1] - data.index[0]).days / 365.25
        return {
            "name": "buy_and_hold",
            "start": start,
            "end": end,
            "ending_shares": INITIAL_SHARES,
            "ending_cash": 0,
            "equivalent_shares": float(INITIAL_SHARES),
            "share_gain": 0.0,
            "ending_value": float(INITIAL_SHARES * end_price),
            "excess_return_pct": 0.0,
            "completed_cycles": 0,
            "open_cycle": 0,
            "sell_orders": 0,
            "buy_orders": 0,
            "cash_days": 0,
            "max_drawdown_pct": float(
                (equity / equity.cummax() - 1).min() * 100
            ),
            "cagr_pct": float(
                ((end_price / start_price) ** (1 / years) - 1) * 100
            ),
        }

    shares = INITIAL_SHARES
    cash = 0.0
    sale_lots = []
    completed_cycles = sell_orders = buy_orders = cash_days = 0
    last_sale_index = -99
    equity_curve = []

    for index, row in enumerate(data.itertuples()):
        equity_curve.append(shares * row.close + cash)
        cash_days += int(bool(sale_lots))

        if index + 1 >= len(data):
            continue

        required = [
            row.return_5d,
            row.ma20,
            row.ma60,
            row.ma200,
            row.volume_ratio,
            row.recent_high,
            row.ma20_gap,
            row.high_gap,
        ]
        if any(pd.isna(value) for value in required):
            continue

        next_open = float(data["open"].iloc[index + 1])
        ma60_20_days_ago = float(
            all_data.loc[: row.Index, "ma60"].iloc[-21]
            if len(all_data.loc[: row.Index]) >= 21
            else row.ma60
        )

        if sale_lots:
            sold_quantity = sum(lot["quantity"] for lot in sale_lots)
            net_sale_per_share = (
                sum(lot["net_proceeds"] for lot in sale_lots) / sold_quantity
            )
            affordable = int(
                cash // (next_open * (1 + BUY_COST * cost_multiplier))
            )
            stable_two_days = (
                index >= 2
                and data["close"].iloc[index]
                >= data["close"].iloc[index - 1]
                >= data["close"].iloc[index - 2]
            )
            near_average = (
                abs(row.close / row.ma20 - 1) <= rule["reentry_band"]
                or abs(row.close / row.ma60 - 1) <= rule["reentry_band"]
                or row.close <= row.ma20
            )
            trend_intact = (
                row.close >= row.ma200 or row.ma60 >= ma60_20_days_ago
            )
            if (
                row.close
                <= net_sale_per_share * (1 - rule["reentry_drop"])
                and affordable >= sold_quantity + 1
                and stable_two_days
                and near_average
                and trend_intact
            ):
                payment = (
                    affordable
                    * next_open
                    * (1 + BUY_COST * cost_multiplier)
                )
                cash -= payment
                shares += affordable
                buy_orders += 1
                completed_cycles += 1
                sale_lots = []
                continue

        outstanding = sum(lot["quantity"] for lot in sale_lots)
        if TACTICAL_LIMIT - outstanding < 5:
            continue

        conditions = [
            row.return_5d >= rule["return_5d"],
            row.high_gap >= -rule["near_high"],
            row.ma20_gap >= rule["ma20_gap"],
            row.volume_ratio >= rule["volume_ratio"],
        ]
        uptrend = row.close >= row.ma200 and row.ma60 >= ma60_20_days_ago

        if sum(conditions) >= rule["required_conditions"] and uptrend:
            # The second five-share sale is not allowed immediately, nor below
            # the first sale price.
            if sale_lots and (
                index - last_sale_index < 2
                or row.close < sale_lots[0]["execution_price"]
            ):
                continue
            net_proceeds = (
                5
                * next_open
                * (1 - SELL_COST * cost_multiplier)
            )
            shares -= 5
            cash += net_proceeds
            sale_lots.append(
                {
                    "quantity": 5,
                    "net_proceeds": net_proceeds,
                    "execution_price": next_open,
                }
            )
            sell_orders += 1
            last_sale_index = index

    ending_value = shares * end_price + cash
    hold_value = INITIAL_SHARES * end_price
    equivalent_shares = ending_value / end_price
    equity = pd.Series(equity_curve, index=data.index[: len(equity_curve)])
    years = (data.index[-1] - data.index[0]).days / 365.25

    return {
        "name": rule.get("name", "rule"),
        "start": start,
        "end": end,
        "ending_shares": int(shares),
        "ending_cash": float(round(cash)),
        "equivalent_shares": float(equivalent_shares),
        "share_gain": float(equivalent_shares - INITIAL_SHARES),
        "ending_value": float(round(ending_value)),
        "excess_return_pct": float((ending_value / hold_value - 1) * 100),
        "completed_cycles": int(completed_cycles),
        "open_cycle": int(bool(sale_lots)),
        "sell_orders": int(sell_orders),
        "buy_orders": int(buy_orders),
        "cash_days": int(cash_days),
        "max_drawdown_pct": float(
            (equity / equity.cummax() - 1).min() * 100
        ),
        "cagr_pct": float(
            (
                (ending_value / (INITIAL_SHARES * start_price))
                ** (1 / years)
                - 1
            )
            * 100
        ),
    }


def main():
    data, metadata = download_data()
    data = data.loc[START:"2026-07-31"]

    current_proxy = {
        "name": "current_observable_4of4",
        "return_5d": 0.25,
        "near_high": 0.03,
        "ma20_gap": 0.15,
        "volume_ratio": 2.0,
        "required_conditions": 4,
        "reentry_drop": 0.10,
        "reentry_band": 0.05,
        "high_window": 120,
    }
    moderate = {
        "name": "moderate_3of4",
        "return_5d": 0.20,
        "near_high": 0.05,
        "ma20_gap": 0.12,
        "volume_ratio": 1.5,
        "required_conditions": 3,
        "reentry_drop": 0.10,
        "reentry_band": 0.05,
        "high_window": 120,
    }

    candidates = []
    for return_5d in [0.15, 0.20, 0.25]:
        for ma20_gap in [0.10, 0.15]:
            for volume_ratio in [1.5, 2.0]:
                for required_conditions in [3, 4]:
                    for reentry_drop in [0.08, 0.10, 0.12]:
                        candidates.append(
                            {
                                "name": "grid",
                                "return_5d": return_5d,
                                "near_high": 0.05,
                                "ma20_gap": ma20_gap,
                                "volume_ratio": volume_ratio,
                                "required_conditions": required_conditions,
                                "reentry_drop": reentry_drop,
                                "reentry_band": 0.05,
                                "high_window": 120,
                            }
                        )

    training_windows = [
        ("2013-01-01", "2016-12-31"),
        ("2017-01-01", "2020-12-31"),
        ("2021-01-01", "2022-12-31"),
    ]
    ranking = []
    for candidate in candidates:
        window_results = [
            simulate(data, start, end, candidate)
            for start, end in training_windows
        ]
        gains = np.array(
            [result["share_gain"] for result in window_results]
        )
        excess = np.array(
            [result["excess_return_pct"] for result in window_results]
        )
        open_cycles = sum(
            result["open_cycle"] for result in window_results
        )
        cycles = sum(
            result["completed_cycles"] for result in window_results
        )
        score = (
            gains.mean() * 2
            + np.median(gains) * 1.5
            + (gains > 0).sum()
            - open_cycles * 1.5
            + np.minimum(excess, 0).mean() * 0.25
            - gains.std() * 0.35
        )
        ranking.append(
            (score, cycles, np.median(gains), candidate, window_results)
        )

    ranking.sort(key=lambda item: item[0], reverse=True)
    selected = next(
        (
            item[3]
            for item in ranking
            if item[1] >= 2 and item[2] > 0
        ),
        ranking[0][3],
    ).copy()
    selected["name"] = "robust_selected"

    periods = {
        "full_2013_2026": ("2013-01-01", "2026-07-31"),
        "out_of_sample_2023_2026": ("2023-01-01", "2026-07-31"),
        "ai_cycle_2024_2026": ("2024-01-01", "2026-07-31"),
    }
    results = []
    for period_name, (start, end) in periods.items():
        for rule in [None, current_proxy, moderate, selected]:
            result = simulate(data, start, end, rule)
            result["period"] = period_name
            results.append(result)

    cost_sensitivity = []
    for multiplier in [0.75, 1.0, 1.5, 2.0]:
        result = simulate(
            data,
            "2023-01-01",
            "2026-07-31",
            selected,
            multiplier,
        )
        result["cost_multiplier"] = multiplier
        cost_sensitivity.append(result)

    output = {
        "generated_at": str(pd.Timestamp.utcnow()),
        "data": {
            **metadata,
            "rows": len(data),
            "first_date": str(data.index.min().date()),
            "last_date": str(data.index.max().date()),
        },
        "assumptions": {
            "initial_shares": INITIAL_SHARES,
            "tactical_limit": TACTICAL_LIMIT,
            "buy_cost_pct": BUY_COST * 100,
            "sell_cost_pct": SELL_COST * 100,
            "signal_timing": "close",
            "execution_timing": "next_open",
        },
        "current_proxy": current_proxy,
        "moderate": moderate,
        "selected": selected,
        "results": results,
        "cost_sensitivity": cost_sensitivity,
        "limitations": [
            "Historical daily consensus EPS revisions are not freely reproducible, so EPS divergence was not backtested.",
            "The original five-condition rule is represented by its four reproducible price-volume conditions.",
            "Signals use close data and execute at the next open to avoid look-ahead bias.",
            "Official quarterly results remain a fundamental veto.",
            "Dividends and loan interest are excluded because they are common to the compared variants.",
        ],
    }

    output_dir = Path("analysis/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(results).to_csv(
        output_dir / "summary.csv", index=False
    )
    print("BACKTEST_JSON_START")
    print(json.dumps(output, ensure_ascii=False))
    print("BACKTEST_JSON_END")


if __name__ == "__main__":
    main()
