from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9


def score_factor_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=["factor_name", "score", "direction"])
    scored = summary_df.copy()
    for col in ["mean_ic", "ic_ir", "spread_mean", "spread_ir"]:
        if col not in scored.columns:
            scored[col] = np.nan
    scored["direction"] = np.where(
        scored["spread_mean"].fillna(0.0).abs() > 0.0,
        np.sign(scored["spread_mean"].fillna(0.0)),
        np.sign(scored["mean_ic"].fillna(0.0)),
    )
    scored["direction"] = scored["direction"].replace(0.0, 1.0)
    scored["score"] = 0.7 * scored["spread_ir"].abs().fillna(0.0) + 0.3 * scored["ic_ir"].abs().fillna(0.0)
    scored = scored.sort_values("score", ascending=False).reset_index(drop=True)
    return scored


def select_diversified_factors(
    scored_summary: pd.DataFrame,
    factor_df: pd.DataFrame,
    top_k: int,
    corr_threshold: float = 0.85,
    pool_size: int = 120,
    max_rows_for_corr: int = 50000,
) -> tuple[list[str], pd.DataFrame]:
    if top_k <= 0 or scored_summary.empty:
        empty = pd.DataFrame(columns=["factor_name", "selected", "max_abs_corr_to_selected", "reason"])
        return [], empty

    scored = score_factor_summary(scored_summary)
    candidates_df = scored.dropna(subset=["factor_name"]).copy()
    candidates_df["factor_name"] = candidates_df["factor_name"].astype(str)
    candidates_df = candidates_df.drop_duplicates(subset=["factor_name"]).head(max(int(pool_size), int(top_k)))
    candidate_names = candidates_df["factor_name"].tolist()
    if not candidate_names:
        empty = pd.DataFrame(columns=["factor_name", "selected", "max_abs_corr_to_selected", "reason"])
        return [], empty

    subset = factor_df[factor_df["factor_name"].isin(candidate_names)][["timestamp", "symbol", "factor_name", "factor_value"]]
    corr_matrix = pd.DataFrame()
    if not subset.empty:
        matrix = subset.pivot_table(
            index=["timestamp", "symbol"],
            columns="factor_name",
            values="factor_value",
            aggfunc="last",
        )
        if max_rows_for_corr > 0 and len(matrix) > max_rows_for_corr:
            matrix = matrix.sample(n=max_rows_for_corr, random_state=42)
        corr_matrix = matrix.corr(method="spearman", min_periods=200)

    selected: list[str] = []
    diagnostics: list[dict[str, object]] = []
    for name in candidate_names:
        if len(selected) >= top_k:
            diagnostics.append(
                {
                    "factor_name": name,
                    "selected": False,
                    "max_abs_corr_to_selected": np.nan,
                    "reason": "over_top_k",
                }
            )
            continue

        if not selected:
            selected.append(name)
            diagnostics.append(
                {
                    "factor_name": name,
                    "selected": True,
                    "max_abs_corr_to_selected": 0.0,
                    "reason": "seed",
                }
            )
            continue

        max_abs_corr = 0.0
        if not corr_matrix.empty and name in corr_matrix.index:
            corr_vals = corr_matrix.loc[name, selected].abs().replace([np.inf, -np.inf], np.nan).dropna()
            if not corr_vals.empty:
                max_abs_corr = float(corr_vals.max())
        if max_abs_corr <= float(corr_threshold):
            selected.append(name)
            diagnostics.append(
                {
                    "factor_name": name,
                    "selected": True,
                    "max_abs_corr_to_selected": max_abs_corr,
                    "reason": "corr_ok",
                }
            )
        else:
            diagnostics.append(
                {
                    "factor_name": name,
                    "selected": False,
                    "max_abs_corr_to_selected": max_abs_corr,
                    "reason": "corr_too_high",
                }
            )

    if len(selected) < top_k:
        for name in candidate_names:
            if name in selected:
                continue
            selected.append(name)
            diagnostics.append(
                {
                    "factor_name": name,
                    "selected": True,
                    "max_abs_corr_to_selected": np.nan,
                    "reason": "backfill_to_top_k",
                }
            )
            if len(selected) >= top_k:
                break

    diag_df = pd.DataFrame(diagnostics)
    if not diag_df.empty:
        diag_df = (
            diag_df.sort_values(["selected", "factor_name"], ascending=[False, True])
            .drop_duplicates(subset=["factor_name"], keep="first")
            .reset_index(drop=True)
        )
    return selected[:top_k], diag_df


def _safe_spearman_corr(group: pd.DataFrame, min_assets: int) -> float:
    if len(group) < min_assets:
        return np.nan
    if group["factor_value"].nunique() < 2 or group["future_return"].nunique() < 2:
        return np.nan
    return float(group["factor_value"].corr(group["future_return"], method="spearman"))


def _evaluate_against_future_returns(
    factor_df: pd.DataFrame,
    future_return_long: pd.DataFrame,
    min_assets_per_timestamp: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = factor_df.merge(future_return_long, on=["timestamp", "symbol"], how="inner")
    merged = merged.dropna(subset=["factor_value", "future_return"]).reset_index(drop=True)

    group_cols = ["factor_name", "timestamp"]
    grp_size = merged.groupby(group_cols)["factor_value"].transform("size")
    valid = merged[grp_size >= min_assets_per_timestamp].copy()

    if valid.empty:
        empty_summary = pd.DataFrame(
            columns=[
                "factor_name",
                "mean_ic",
                "std_ic",
                "obs",
                "ic_ir",
                "hit_rate",
                "t_stat",
                "spread_mean",
                "spread_std",
                "spread_ir",
            ]
        )
        return empty_summary, pd.DataFrame(columns=["factor_name", "timestamp", "ic"]), pd.DataFrame(
            columns=["factor_name", "timestamp", "q5_q1"]
        )

    valid["factor_rank"] = valid.groupby(group_cols)["factor_value"].rank(method="average")
    valid["future_rank"] = valid.groupby(group_cols)["future_return"].rank(method="average")
    valid["xy"] = valid["factor_rank"] * valid["future_rank"]
    valid["x2"] = valid["factor_rank"] * valid["factor_rank"]
    valid["y2"] = valid["future_rank"] * valid["future_rank"]

    ic_agg = (
        valid.groupby(group_cols)
        .agg(
            n=("factor_rank", "size"),
            sx=("factor_rank", "sum"),
            sy=("future_rank", "sum"),
            sxy=("xy", "sum"),
            sx2=("x2", "sum"),
            sy2=("y2", "sum"),
            fx_unique=("factor_value", "nunique"),
            fr_unique=("future_return", "nunique"),
        )
        .reset_index()
    )
    num = ic_agg["n"] * ic_agg["sxy"] - ic_agg["sx"] * ic_agg["sy"]
    den_left = ic_agg["n"] * ic_agg["sx2"] - ic_agg["sx"] * ic_agg["sx"]
    den_right = ic_agg["n"] * ic_agg["sy2"] - ic_agg["sy"] * ic_agg["sy"]
    den = np.sqrt((den_left.clip(lower=0.0)) * (den_right.clip(lower=0.0)))
    ic_agg["ic"] = num / (den + EPS)
    ic_agg.loc[(ic_agg["fx_unique"] < 2) | (ic_agg["fr_unique"] < 2) | (den <= 0), "ic"] = np.nan
    ic_ts = ic_agg[["factor_name", "timestamp", "ic"]].copy()

    summary = (
        ic_ts.groupby("factor_name")
        .agg(mean_ic=("ic", "mean"), std_ic=("ic", "std"), obs=("ic", "count"))
        .reset_index()
    )
    summary["ic_ir"] = summary["mean_ic"] / (summary["std_ic"] + EPS)
    summary["hit_rate"] = (
        ic_ts.assign(hit=(ic_ts["ic"] > 0).astype(float)).groupby("factor_name")["hit"].mean().values
    )
    summary["t_stat"] = summary["mean_ic"] / (summary["std_ic"] / np.sqrt(summary["obs"].clip(lower=1)) + EPS)

    # Approximate equal-frequency buckets via percentile ranks (much faster than per-group qcut).
    valid["bucket_rank"] = valid.groupby(group_cols)["factor_value"].rank(method="first", pct=True)
    valid["bucket"] = np.ceil(valid["bucket_rank"] * 5.0).clip(1.0, 5.0).astype(int)
    bucket_df = (
        valid.groupby(["factor_name", "timestamp", "bucket"], as_index=False)["future_return"]
        .mean()
        .reset_index(drop=True)
    )

    if not bucket_df.empty:
        spread = bucket_df.pivot_table(
            index=["factor_name", "timestamp"],
            columns="bucket",
            values="future_return",
            aggfunc="mean",
        )
        spread = spread.reset_index()
        spread["q5_q1"] = spread.get(5, np.nan) - spread.get(1, np.nan)

        spread_summary = (
            spread.groupby("factor_name")["q5_q1"]
            .agg(spread_mean="mean", spread_std="std")
            .reset_index()
        )
        spread_summary["spread_ir"] = spread_summary["spread_mean"] / (spread_summary["spread_std"] + EPS)
        summary = summary.merge(spread_summary, on="factor_name", how="left")
    else:
        spread = pd.DataFrame(columns=["factor_name", "timestamp", "q5_q1"])
        summary["spread_mean"] = np.nan
        summary["spread_std"] = np.nan
        summary["spread_ir"] = np.nan

    summary = summary.sort_values(["mean_ic", "ic_ir"], ascending=False).reset_index(drop=True)
    return summary, ic_ts, spread


def aggregate_horizon_summaries(summary_by_horizon_df: pd.DataFrame) -> pd.DataFrame:
    if summary_by_horizon_df.empty:
        return pd.DataFrame(
            columns=[
                "factor_name",
                "mean_ic",
                "std_ic",
                "obs",
                "ic_ir",
                "hit_rate",
                "t_stat",
                "spread_mean",
                "spread_std",
                "spread_ir",
                "horizon_count",
                "ic_sign_consistency",
            ]
        )

    agg = (
        summary_by_horizon_df.groupby("factor_name", as_index=False)
        .agg(
            mean_ic=("mean_ic", "mean"),
            std_ic=("std_ic", "mean"),
            obs=("obs", "sum"),
            ic_ir=("ic_ir", "mean"),
            hit_rate=("hit_rate", "mean"),
            t_stat=("t_stat", "mean"),
            spread_mean=("spread_mean", "mean"),
            spread_std=("spread_std", "mean"),
            spread_ir=("spread_ir", "mean"),
            horizon_count=("horizon_bars", "nunique"),
        )
    )
    sign_df = summary_by_horizon_df[["factor_name", "horizon_bars", "mean_ic"]].copy()
    avg_sign = np.sign(agg.set_index("factor_name")["mean_ic"]).replace(0.0, np.nan)
    sign_df["avg_sign"] = sign_df["factor_name"].map(avg_sign)
    sign_df["same_sign"] = np.where(
        sign_df["avg_sign"].isna() | sign_df["mean_ic"].isna(),
        np.nan,
        (np.sign(sign_df["mean_ic"]) == sign_df["avg_sign"]).astype(float),
    )
    consistency = sign_df.groupby("factor_name")["same_sign"].mean().rename("ic_sign_consistency").reset_index()
    agg = agg.merge(consistency, on="factor_name", how="left")
    return agg.sort_values(["mean_ic", "ic_ir"], ascending=False).reset_index(drop=True)


def evaluate_factors(
    factor_df: pd.DataFrame,
    market_df: pd.DataFrame,
    horizon_bars: int,
    min_assets_per_timestamp: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market_df = market_df.drop_duplicates(subset=["timestamp", "symbol"])
    close = market_df.pivot_table(
        index="timestamp",
        columns="symbol",
        values="close",
        aggfunc="last",
    ).sort_index()
    future_return = close.pct_change(horizon_bars).shift(-horizon_bars)
    future_return_long = future_return.stack(future_stack=True).rename("future_return").reset_index()
    future_return_long.columns = ["timestamp", "symbol", "future_return"]
    return _evaluate_against_future_returns(
        factor_df=factor_df,
        future_return_long=future_return_long,
        min_assets_per_timestamp=min_assets_per_timestamp,
    )


def evaluate_factors_multi_horizon(
    factor_df: pd.DataFrame,
    market_df: pd.DataFrame,
    horizons: list[int],
    min_assets_per_timestamp: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_horizons = sorted({int(h) for h in horizons if int(h) > 0})
    if not unique_horizons:
        raise ValueError("horizons must contain at least one positive integer")

    market_df = market_df.drop_duplicates(subset=["timestamp", "symbol"])
    close = market_df.pivot_table(
        index="timestamp",
        columns="symbol",
        values="close",
        aggfunc="last",
    ).sort_index()

    summary_frames: list[pd.DataFrame] = []
    ic_frames: list[pd.DataFrame] = []
    spread_frames: list[pd.DataFrame] = []
    for horizon in unique_horizons:
        future_return = close.pct_change(horizon).shift(-horizon)
        future_return_long = future_return.stack(future_stack=True).rename("future_return").reset_index()
        future_return_long.columns = ["timestamp", "symbol", "future_return"]
        summary_df, ic_ts_df, spread_df = _evaluate_against_future_returns(
            factor_df=factor_df,
            future_return_long=future_return_long,
            min_assets_per_timestamp=min_assets_per_timestamp,
        )
        summary_df["horizon_bars"] = horizon
        ic_ts_df["horizon_bars"] = horizon
        spread_df["horizon_bars"] = horizon
        summary_frames.append(summary_df)
        ic_frames.append(ic_ts_df)
        spread_frames.append(spread_df)

    summary_by_horizon = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    ic_by_horizon = pd.concat(ic_frames, ignore_index=True) if ic_frames else pd.DataFrame()
    spread_by_horizon = pd.concat(spread_frames, ignore_index=True) if spread_frames else pd.DataFrame()
    summary_agg = aggregate_horizon_summaries(summary_by_horizon)
    return summary_agg, summary_by_horizon, ic_by_horizon, spread_by_horizon


def build_composite_signal(
    factor_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    top_k: int = 5,
    selected_factors: list[str] | None = None,
    corr_threshold: float = 0.85,
    diversify_pool_size: int = 120,
) -> pd.DataFrame:
    if top_k <= 0:
        return pd.DataFrame(columns=["timestamp", "symbol", "composite_signal"])

    scored = score_factor_summary(summary_df)
    if scored.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "composite_signal"])

    if selected_factors:
        ordered = [str(x).strip() for x in selected_factors if str(x).strip()]
        chosen = [name for name in ordered if name in set(scored["factor_name"])]
    else:
        chosen, _ = select_diversified_factors(
            scored_summary=scored,
            factor_df=factor_df,
            top_k=top_k,
            corr_threshold=corr_threshold,
            pool_size=diversify_pool_size,
        )
    if not chosen:
        return pd.DataFrame(columns=["timestamp", "symbol", "composite_signal"])

    eligible = scored[scored["factor_name"].isin(chosen)].copy()
    eligible = eligible.set_index("factor_name").loc[chosen].reset_index()
    weights = eligible.set_index("factor_name")["score"] * eligible.set_index("factor_name")["direction"]
    weights = weights / (weights.abs().sum() + EPS)

    tmp = factor_df[factor_df["factor_name"].isin(chosen)].copy()
    tmp["weight"] = tmp["factor_name"].map(weights)
    tmp["weighted"] = tmp["factor_value"] * tmp["weight"]

    out = (
        tmp.groupby(["timestamp", "symbol"], as_index=False)["weighted"]
        .sum()
        .rename(columns={"weighted": "composite_signal"})
    )
    return out
