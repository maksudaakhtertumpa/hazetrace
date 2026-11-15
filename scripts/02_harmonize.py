"""Step 2: learn VIIRS -> MODIS-equivalent counts, validate honestly, and build
one consistent hotspot record from 2003 to today.

    python scripts/02_harmonize.py

What it does
1. Pairs MODIS and VIIRS S-NPP counts for the same grid cell and month
   (overlap 2012-2026, standard-quality months only).
2. Baseline: one MODIS/VIIRS ratio per region, learned on training years.
3. Model: LightGBM on VIIRS features -> log(1 + MODIS count).
4. Validation: leave-one-YEAR-out (no random split: neighbouring months leak).
   Scored at region x month level, the level the calendar shows.
5. Transfer check: does the S-NPP-trained model work on NOAA-20 (2018+)?
6. Final record: MODIS observed before 2012, model-from-VIIRS after
   (S-NPP until Nov 2026, NOAA-20 afterwards), with an approximate 10-90% band.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hazetrace import config as C
from hazetrace.regions import assign_region as _assign_region  # noqa: E402

# Only SCALE-FREE features: what kind of fire (mean FRP, confidence, night share),
# where (region, land cover) and when (season). Raw counts are left out on purpose:
# with them the model learns cell-level noise (regression to the mean) and then
# under-predicts record fire years. Scale enters only through the VIIRS offset.
# One activity feature is added at REGION level (total VIIRS hotspots in the region that
# month): the MODIS/VIIRS ratio drops in big burning seasons, when many small peat fires
# burn that only VIIRS sees. A regional total is far less noisy than a cell count.
FEATURES = ["v_frac_high", "v_frac_night", "log_v_mean_frp", "log_v_region_month",
            "month_sin", "month_cos", "region_code", "lc_code"]
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, random_state=0)


def assign_region(lat, lon):
    """Real province boundaries (assets/provinces.geojson); falls back to config boxes."""
    return _assign_region(lat, lon)


def load():
    cm = pd.read_parquet(C.PROCESSED_DIR / "cell_month.parquet")
    cells = cm[["cell_lat", "cell_lon"]].drop_duplicates()
    cells["region"] = assign_region(cells.cell_lat.values, cells.cell_lon.values)
    lc_path = C.ROOT / "data" / "landcover_cells.csv"   # optional: cell_lat, cell_lon, landcover
    if lc_path.exists():
        lc = pd.read_csv(lc_path)
        cells = cells.merge(lc, on=["cell_lat", "cell_lon"], how="left")
    else:
        print("No data/landcover_cells.csv yet: land-cover split will show as 'unknown'.")
        cells["landcover"] = "unknown"
    cells["landcover"] = cells["landcover"].fillna("unknown")
    return cm.merge(cells, on=["cell_lat", "cell_lon"]), cells


def viirs_features(df, prefix="v"):
    d = pd.DataFrame(index=df.index)
    cnt = df[f"{prefix}_count"]
    d["log_v_count"] = np.log1p(cnt)
    d["log_v_frp"] = np.log1p(df[f"{prefix}_frp_sum"])
    d["v_frac_high"] = np.where(cnt > 0, df[f"{prefix}_n_high"] / cnt.clip(lower=1), 0)
    d["v_frac_night"] = np.where(cnt > 0, df[f"{prefix}_n_night"] / cnt.clip(lower=1), 0)
    d["log_v_mean_frp"] = np.log1p(np.where(cnt > 0, df[f"{prefix}_frp_sum"] / cnt.clip(lower=1), 0))
    d["log_v_region_month"] = np.log1p(df.groupby(["region", "ym"])[f"{prefix}_count"].transform("sum"))
    month = pd.PeriodIndex(df["ym"], freq="M").month
    d["month_sin"] = np.sin(2 * np.pi * month / 12)
    d["month_cos"] = np.cos(2 * np.pi * month / 12)
    d["region_code"] = pd.Categorical(df["region"], categories=C.REGION_ORDER + ["Other"]).codes
    d["lc_code"] = pd.Categorical(df["landcover"]).codes
    return d[FEATURES]


def pair(cm, viirs_sensor):
    keys = ["cell_lat", "cell_lon", "ym", "region", "landcover"]
    m = cm[cm.sensor == "modis"].set_index(keys)
    v = cm[cm.sensor == viirs_sensor].set_index(keys)
    both_months = sorted(set(m.index.get_level_values("ym")) & set(v.index.get_level_values("ym")))
    cols = ["count", "frp_sum", "n_high", "n_night", "n_nrt"]
    p = m[cols].add_prefix("m_").join(v[cols].add_prefix("v_"), how="outer").fillna(0).reset_index()
    first_full = C.SNPP_START if viirs_sensor == "snpp" else C.NOAA20_START
    p = p[p.ym.isin(both_months) & (p.ym >= first_full) & (p.ym <= C.PAIR_END)].copy()
    p["year"] = pd.PeriodIndex(p.ym, freq="M").year
    p["standard"] = (p.m_n_nrt == 0) & (p.v_n_nrt == 0)
    return p


def fit(X, y, objective="regression", alpha=None):
    kw = dict(PARAMS, objective=objective)
    if alpha is not None:
        kw["alpha"] = alpha
    return lgb.LGBMRegressor(**kw).fit(X, y)


OFFSET_EPS = 0.5


def fit_ratio(X, m, v):
    model = lgb.LGBMRegressor(**dict(PARAMS, objective="poisson"))
    return model.fit(X, m, init_score=np.log(v + OFFSET_EPS))


def predict_ratio(model, X, v):
    return np.exp(model.predict(X, raw_score=True) + np.log(np.asarray(v, float) + OFFSET_EPS))


def region_scores(df, pred_col):
    g = df.groupby(["region", "ym"]).agg(obs=("m_count", "sum"), pred=(pred_col, "sum")).reset_index()
    return g


def metrics(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    mae = float(np.mean(np.abs(obs - pred)))
    ss_res, ss_tot = np.sum((obs - pred) ** 2), np.sum((obs - obs.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    bias = float((pred.sum() - obs.sum()) / obs.sum() * 100) if obs.sum() > 0 else float("nan")
    return {"MAE": round(mae, 2), "R2": round(r2, 3), "bias_pct": round(bias, 1), "n": int(len(obs))}


def main():
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    C.MODEL_DIR.mkdir(exist_ok=True)
    cm, cells = load()
    p = pair(cm, "snpp")
    train_pool = p[p.standard]
    years = sorted(y for y in train_pool.year.unique() if y >= C.FIRST_FULL_OVERLAP_YEAR)
    # the last year may be partly NRT; only score years with >= 10 standard months
    years = [y for y in years if train_pool[train_pool.year == y].ym.nunique() >= 10]
    print(f"Pairs: {len(p):,} cell-months, standard: {len(train_pool):,}; validation years: {[int(y) for y in years]}")

    # The model predicts the MODIS/VIIRS RATIO, not the MODIS count. Tree models cannot
    # predict beyond the range they saw in training, so a count model fails in record fire
    # years (2015, 2019...). We use a Poisson model with log(VIIRS count) as an offset:
    # MODIS_count ~ VIIRS_count x ratio(features). Scale comes from VIIRS, the trees learn
    # only how the ratio changes with region, season, land cover and fire type, and the
    # Poisson mean needs no back-transformation bias fix.
    X_all = viirs_features(train_pool)
    m_all, v_all = train_pool.m_count.to_numpy(), train_pool.v_count.to_numpy()
    cv_rows, oof = [], []
    for yr in years:
        tr, te = (train_pool.year != yr).to_numpy(), (train_pool.year == yr).to_numpy()
        # baseline: one ratio per region from training years
        r = train_pool[tr].groupby("region").apply(lambda g: g.m_count.sum() / max(g.v_count.sum(), 1),
                                                     include_groups=False)
        base = train_pool[te].v_count * train_pool[te].region.map(r).fillna(r.mean())
        model = fit_ratio(X_all[tr], m_all[tr], v_all[tr])
        pred = predict_ratio(model, X_all[te], v_all[te])
        pred = np.where(v_all[te] == 0, 0.0, pred)   # same rule as the final record: no VIIRS detection -> 0
        part = train_pool[te][["region", "landcover", "ym", "year", "m_count", "v_count"]].copy()
        part["pred_model"], part["pred_base"] = pred.clip(min=0), base.values
        oof.append(part)
        g = region_scores(part, "pred_model").merge(region_scores(part, "pred_base"), on=["region", "ym"],
                                                    suffixes=("", "_b"))
        cv_rows.append({"year": yr, **{f"model_{k}": v for k, v in metrics(g.obs, g.pred).items()},
                        **{f"base_{k}": v for k, v in metrics(g.obs, g.pred_b).items() if k != "n"}})
        print(f"  held out {yr}: model MAE {cv_rows[-1]['model_MAE']}, baseline MAE {cv_rows[-1]['base_MAE']}")

    oof = pd.concat(oof)
    by_year = pd.DataFrame(cv_rows)
    by_year.to_csv(C.OUTPUT_DIR / "validation_by_year.csv", index=False)

    rm = region_scores(oof, "pred_model").merge(region_scores(oof, "pred_base"), on=["region", "ym"],
                                                suffixes=("", "_b"))
    by_group = []
    for name, sub in [("All regions", rm)] + list(rm.groupby("region")):
        by_group.append({"group": name, "kind": "region",
                         **{f"model_{k}": v for k, v in metrics(sub.obs, sub.pred).items()},
                         **{f"base_{k}": v for k, v in metrics(sub.obs, sub.pred_b).items() if k != "n"}})
    lcm = oof.groupby(["landcover", "ym"]).agg(obs=("m_count", "sum"), pred=("pred_model", "sum"),
                                               pred_b=("pred_base", "sum")).reset_index()
    for name, sub in lcm.groupby("landcover"):
        by_group.append({"group": name, "kind": "landcover",
                         **{f"model_{k}": v for k, v in metrics(sub.obs, sub.pred).items()},
                         **{f"base_{k}": v for k, v in metrics(sub.obs, sub.pred_b).items() if k != "n"}})
    by_group = pd.DataFrame(by_group)
    by_group.to_csv(C.OUTPUT_DIR / "validation_by_group.csv", index=False)

    # Use whichever method did better on held-out years (honest, automatic choice)
    overall = by_group[by_group.group == "All regions"].iloc[0]
    use_model = bool(overall.model_MAE < overall.base_MAE)
    method = "LightGBM ratio model" if use_model else "Baseline regional ratio"
    print(f"Chosen method for the final record: {method}")
    ratio_all = train_pool.groupby("region").apply(lambda g: g.m_count.sum() / max(g.v_count.sum(), 1),
                                                     include_groups=False)

    # final models on all standard pairs
    final = fit_ratio(X_all, m_all, v_all)
    # approximate 10-90% band: quantiles of the log ratio
    y_ratio = np.log1p(m_all) - np.log1p(v_all)
    q10, q90 = fit(X_all, y_ratio, "quantile", 0.1), fit(X_all, y_ratio, "quantile", 0.9)
    final.booster_.save_model(str(C.MODEL_DIR / "harmonizer.txt"))

    # explainability
    try:
        import shap
        sample = X_all.sample(min(5000, len(X_all)), random_state=0)
        sv = shap.TreeExplainer(final).shap_values(sample)
        imp = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(sv).mean(0)})
    except Exception as e:  # keep the pipeline running even if SHAP fails
        print("SHAP skipped:", e)
        imp = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": final.feature_importances_})
    imp.sort_values("mean_abs_shap", ascending=False).to_csv(C.OUTPUT_DIR / "feature_importance.csv", index=False)

    # transfer check on NOAA-20
    transfer = None
    if (cm.sensor == "noaa20").any():
        p20 = pair(cm, "noaa20")
        p20 = p20[p20.standard]
        if len(p20):
            p20["pred_model"] = np.where(p20.v_count == 0, 0.0,   # same zero rule as the final record
                                         predict_ratio(final, viirs_features(p20), p20.v_count))
            p20["pred_base"] = p20.v_count * p20.region.map(ratio_all).fillna(ratio_all.mean())
            transfer = {"model": metrics(*region_scores(p20, "pred_model")[["obs", "pred"]].T.values),
                        "baseline": metrics(*region_scores(p20, "pred_base")[["obs", "pred"]].T.values)}
            print("NOAA-20 transfer (region-month):", transfer)

    # ---- consistent record -------------------------------------------------
    keys = ["cell_lat", "cell_lon", "ym", "region", "landcover"]
    wide = cm.pivot_table(index=keys, columns="sensor", values="count", aggfunc="sum").fillna(0)
    nrt = cm.pivot_table(index=keys, columns="sensor", values="n_nrt", aggfunc="sum").fillna(0).sum(axis=1) > 0
    wide = wide.reset_index()
    wide["nrt"] = nrt.values
    for col in ["modis", "snpp", "noaa20"]:
        if col not in wide:
            wide[col] = 0.0

    def predict_from(sensor, rows):
        sub = cm[cm.sensor == sensor].set_index(keys)[["count", "frp_sum", "n_high", "n_night"]].add_prefix("v_")
        sub = rows.set_index(keys).join(sub).fillna(0).reset_index()
        X, off = viirs_features(sub), np.log1p(sub.v_count.values)
        if use_model:
            mid = predict_ratio(final, X, sub.v_count.values)
        else:
            mid = (sub.v_count * sub.region.map(ratio_all).fillna(ratio_all.mean())).to_numpy(dtype=float, copy=True)
        lo, hi = np.array(np.expm1(off + q10.predict(X))), np.array(np.expm1(off + q90.predict(X)))
        zero = sub.v_count.values == 0
        mid[zero], lo[zero], hi[zero] = 0, 0, 0   # no VIIRS detection -> 0
        return mid.clip(0), np.minimum(lo, mid).clip(0), np.maximum(hi, mid).clip(0)

    wide["value"], wide["lo"], wide["hi"] = wide.modis, wide.modis, wide.modis
    wide["source"] = "MODIS observed"
    # Use S-NPP for every month it covers; otherwise fall back to NOAA-20 (after S-NPP
    # ended on 2 Nov 2026, or for months whose S-NPP files were not downloaded).
    snpp_months = set(cm.loc[cm.sensor == "snpp", "ym"])
    n20_months = set(cm.loc[cm.sensor == "noaa20", "ym"])
    in_snpp = wide.ym.isin(snpp_months) & (wide.ym >= C.SNPP_START) & (wide.ym < C.SNPP_END)
    snpp_rows = in_snpp
    n20_rows = ~in_snpp & wide.ym.isin(n20_months) & (wide.ym >= C.SNPP_START)
    tag = "Model" if use_model else "Baseline ratio"
    for mask, sensor, label in [(snpp_rows, "snpp", f"{tag} from VIIRS S-NPP"),
                                (n20_rows, "noaa20", f"{tag} from VIIRS NOAA-20")]:
        if mask.any():
            mid, lo, hi = predict_from(sensor, wide.loc[mask, keys])
            wide.loc[mask, ["value", "lo", "hi"]] = np.c_[mid, lo, hi]
            wide.loc[mask, "source"] = label
    wide["viirs_raw"] = np.where(n20_rows, wide.noaa20, wide.snpp)
    cell_out = wide[["cell_lat", "cell_lon", "ym", "region", "landcover", "value", "lo", "hi",
                     "modis", "viirs_raw", "source", "nrt"]]
    cell_out = cell_out[(cell_out.value > 0) | (cell_out.modis > 0) | (cell_out.viirs_raw > 0)]
    cell_out = cell_out[cell_out.ym >= C.RECORD_START]
    cell_out.to_parquet(C.OUTPUT_DIR / "harmonized_cells.parquet", index=False)

    # ---- independent out-of-time check: Malaysia 2026 (never used in training) ----
    check_2026 = None

    def file_cells(pattern):
        out = set()
        for f in C.RAW_DIR.glob(pattern):
            d = pd.read_csv(f, usecols=["latitude", "longitude"])
            g = C.GRID_DEG
            out |= set(zip((np.floor(d.latitude / g) * g + g / 2).round(4), (np.floor(d.longitude / g) * g + g / 2).round(4)))
        return out
    pure = file_cells(C.CHECK_INCLUDE_GLOB) - file_cells(C.CHECK_EXCLUDE_GLOB)
    # Saved for the app: cells whose 2026 MODIS count is complete (Malaysia only, no Indonesian part).
    pd.DataFrame(sorted(pure), columns=["cell_lat", "cell_lon"]).to_csv(
        C.OUTPUT_DIR / "malaysia_2026_modis_cells.csv", index=False)
    my = pd.Series(list(zip(cell_out.cell_lat.round(4), cell_out.cell_lon.round(4)))).isin(pure).to_numpy()
    c26 = cell_out[my & (cell_out.ym >= C.MODIS_PARTIAL_FROM) & (cell_out.ym < C.SNPP_END)]
    if len(c26) and c26.modis.sum() > 0:
        m26 = c26.groupby("ym").agg(obs=("modis", "sum"), pred=("value", "sum"), viirs=("viirs_raw", "sum"))
        check_2026 = {"area": f"Malaysia ({len(pure)} grid cells)", "months": f"{m26.index.min()} to {m26.index.max()}",
                      "modis_observed": int(m26.obs.sum()), "harmonized": round(float(m26.pred.sum()), 0),
                      "viirs_raw": int(m26.viirs.sum()),
                      "monthly_MAE": round(float((m26.obs - m26.pred).abs().mean()), 1),
                      "total_error_pct": round(float((m26.pred.sum() - m26.obs.sum()) / m26.obs.sum() * 100), 1)}
        print("Malaysia 2026 independent check (MODIS never seen in training):", check_2026)

    meta = {
        "validation": "leave-one-year-out, region x month",
        "regions": "provinces" if C.PROVINCES_FILE.exists() else "boxes",
        "check_malaysia_2026": check_2026,
        "method_used": method,
        "years": [int(y) for y in years],
        "overall": by_group[by_group.group == "All regions"].iloc[0].to_dict(),
        "noaa20_transfer": transfer,
        "first_month": cell_out.ym.min(), "last_month": cell_out.ym.max(),
        "nrt_months": sorted(cell_out[cell_out.nrt].ym.unique().tolist()),
        "band_note": "10-90% band summed over cells: approximate, not an exact interval for the region total",
    }
    (C.OUTPUT_DIR / "metrics.json").write_text(json.dumps(meta, indent=2, default=str))
    o = meta["overall"]
    print(f"\nOverall (held-out years): model MAE {o['model_MAE']} vs baseline {o['base_MAE']}, "
          f"model R2 {o['model_R2']} vs baseline {o['base_R2']} -> using {method}")
    print("Saved outputs/ harmonized_cells.parquet, validation_*.csv, feature_importance.csv, metrics.json")


if __name__ == "__main__":
    main()
