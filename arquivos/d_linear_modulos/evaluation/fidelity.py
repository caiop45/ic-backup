import json
from pathlib import Path
from typing import Callable, Tuple, Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import distance
from fastdtw import fastdtw
import ot
import geopandas as gpd
import contextily as ctx

__all__ = [
    "sample_same_counts",
    "compute_metrics",
    "make_all_plots",
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def sample_same_counts(real_df: pd.DataFrame, synth_df: pd.DataFrame, *, date_col: str = "tpep_pickup_datetime") -> pd.DataFrame:
    """Sample synthetic rows to match real counts per day."""
    if date_col not in real_df.columns or date_col not in synth_df.columns:
        raise ValueError(f"Column '{date_col}' must be present in both dataframes")

    real_df = real_df.copy()
    synth_df = synth_df.copy()

    real_df[date_col] = pd.to_datetime(real_df[date_col])
    synth_df[date_col] = pd.to_datetime(synth_df[date_col])

    real_counts = real_df[date_col].dt.date.value_counts().sort_index()
    samples = []
    for day, count in real_counts.items():
        mask = synth_df[date_col].dt.date == day
        if mask.sum() == 0:
            continue
        samples.append(synth_df[mask].sample(n=min(count, mask.sum()), replace=False, random_state=0))
    if not samples:
        return pd.DataFrame(columns=synth_df.columns)
    balanced_syn = pd.concat(samples, ignore_index=True)
    return balanced_syn


def _bootstrap_ci(metric_func: Callable[[pd.DataFrame, pd.DataFrame], float],
                  real_df: pd.DataFrame,
                  synth_df: pd.DataFrame,
                  *,
                  date_col: str = "tpep_pickup_datetime",
                  n_boot: int = 1000,
                  seed: int | None = None) -> Tuple[float, Tuple[float, float]]:
    """Internal helper to compute metric mean and bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    days = real_df[date_col].dt.normalize().unique()
    boot_vals = []
    for _ in range(n_boot):
        chosen = rng.choice(days, size=len(days), replace=True)
        real_sample = pd.concat([real_df[real_df[date_col].dt.normalize() == d] for d in chosen])
        synth_sample = pd.concat([synth_df[synth_df[date_col].dt.normalize() == d] for d in chosen])
        boot_vals.append(metric_func(real_sample, synth_sample))
    mean_val = metric_func(real_df, synth_df)
    lower, upper = np.percentile(boot_vals, [2.5, 97.5])
    return float(mean_val), (float(lower), float(upper))

# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def compute_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame, *, seed: int | None = None,
                    date_col: str = "tpep_pickup_datetime") -> Dict[str, Any]:
    """Compute all fidelity metrics with bootstrap 95% CI."""

    def city_hourly_mae(real: pd.DataFrame, synth: pd.DataFrame) -> float:
        r = real.groupby(real[date_col].dt.hour).size().reindex(range(24), fill_value=0)
        s = synth.groupby(synth[date_col].dt.hour).size().reindex(range(24), fill_value=0)
        return float(np.mean(np.abs(r - s)))

    def city_hourly_mape(real: pd.DataFrame, synth: pd.DataFrame) -> float:
        r = real.groupby(real[date_col].dt.hour).size().reindex(range(24), fill_value=0)
        s = synth.groupby(synth[date_col].dt.hour).size().reindex(range(24), fill_value=0)
        mask = r > 0
        if not mask.any():
            return float("nan")
        return float(np.mean(np.abs((r[mask] - s[mask]) / r[mask])) * 100)

    def city_hourly_wass(real: pd.DataFrame, synth: pd.DataFrame) -> float:
        r = real.groupby(real[date_col].dt.hour).size().reindex(range(24), fill_value=0)
        s = synth.groupby(synth[date_col].dt.hour).size().reindex(range(24), fill_value=0)
        return float(stats.wasserstein_distance(r, s))

    def per_zone_dtw(real: pd.DataFrame, synth: pd.DataFrame) -> float:
        zone_col = "PULocationID"
        if zone_col not in real.columns or zone_col not in synth.columns:
            return float("nan")
        zones = np.intersect1d(real[zone_col].unique(), synth[zone_col].unique())
        dists = []
        for z in zones:
            r = real[real[zone_col]==z].groupby(real[date_col].dt.hour).size().reindex(range(24), fill_value=0)
            s = synth[synth[zone_col]==z].groupby(synth[date_col].dt.hour).size().reindex(range(24), fill_value=0)
            d, _ = fastdtw(r.values, s.values)
            dists.append(d)
        return float(np.mean(dists)) if dists else float("nan")

    def per_zone_pearson(real: pd.DataFrame, synth: pd.DataFrame) -> float:
        zone_col = "PULocationID"
        if zone_col not in real.columns or zone_col not in synth.columns:
            return float("nan")
        zones = np.intersect1d(real[zone_col].unique(), synth[zone_col].unique())
        cors = []
        for z in zones:
            r = real[real[zone_col]==z].groupby(real[date_col].dt.hour).size().reindex(range(24), fill_value=0)
            s = synth[synth[zone_col]==z].groupby(synth[date_col].dt.hour).size().reindex(range(24), fill_value=0)
            if r.std() == 0 or s.std() == 0:
                continue
            cors.append(np.corrcoef(r, s)[0,1])
        return float(np.mean(cors)) if cors else float("nan")

    def _hist2d(df, x, y, bins):
        H, xedges, yedges = np.histogram2d(df[x], df[y], bins=bins)
        return H / H.sum(), (xedges, yedges)

    def spatial_wass(real: pd.DataFrame, synth: pd.DataFrame) -> float:
        bins = 20
        r_hist, (rx, ry) = _hist2d(real, "pickup_longitude", "pickup_latitude", bins)
        s_hist, _ = _hist2d(synth, "pickup_longitude", "pickup_latitude", bins)
        xs = 0.5*(rx[:-1] + rx[1:])
        ys = 0.5*(ry[:-1] + ry[1:])
        grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
        M = ot.dist(grid, grid)
        return float(ot.emd2(r_hist.flatten(), s_hist.flatten(), M))

    def spatial_js(real: pd.DataFrame, synth: pd.DataFrame) -> float:
        bins = 20
        r_hist, _ = _hist2d(real, "pickup_longitude", "pickup_latitude", bins)
        s_hist, _ = _hist2d(synth, "pickup_longitude", "pickup_latitude", bins)
        return float(distance.jensenshannon(r_hist.flatten(), s_hist.flatten()))

    def trip_len_ks(real: pd.DataFrame, synth: pd.DataFrame) -> float:
        if "trip_distance" not in real.columns or "trip_distance" not in synth.columns:
            return float("nan")
        r = real["trip_distance"].dropna()
        s = synth["trip_distance"].dropna()
        if len(r)==0 or len(s)==0:
            return float("nan")
        return float(stats.ks_2samp(r, s).statistic)

    def od_frobenius(real: pd.DataFrame, synth: pd.DataFrame) -> float:
        req = {"PULocationID", "DOLocationID"}
        if not req.issubset(real.columns) or not req.issubset(synth.columns):
            return float("nan")
        matrixes = []
        for df in (real, synth):
            pivot = pd.pivot_table(df, index="PULocationID", columns="DOLocationID", values=date_col, aggfunc="count", fill_value=0)
            matrixes.append(pivot)
        real_mat, synth_mat = matrixes
        diff = real_mat.reindex_like(synth_mat, fill_value=0) - synth_mat.reindex_like(real_mat, fill_value=0)
        return float(np.linalg.norm(diff.to_numpy(), ord="fro"))

    def od_sliced_emd(real: pd.DataFrame, synth: pd.DataFrame) -> float:
        req = {"PULocationID", "DOLocationID"}
        if not req.issubset(real.columns) or not req.issubset(synth.columns):
            return float("nan")
        pivot_real = pd.pivot_table(real, index="PULocationID", columns="DOLocationID", values=date_col, aggfunc="count", fill_value=0)
        pivot_syn  = pd.pivot_table(synth, index="PULocationID", columns="DOLocationID", values=date_col, aggfunc="count", fill_value=0)
        r = pivot_real.reindex_like(pivot_syn, fill_value=0).to_numpy().flatten()
        s = pivot_syn.reindex_like(pivot_real, fill_value=0).to_numpy().flatten()
        return float(ot.sliced.sliced_wasserstein_distance(r, s, n_projections=200))

    def hotspot_hausdorff(real: pd.DataFrame, synth: pd.DataFrame) -> float:
        def centroids(df):
            grp = df.groupby(df[date_col].dt.hour)
            return grp[["pickup_longitude", "pickup_latitude"]].mean().dropna()
        r = centroids(real).to_numpy()
        s = centroids(synth).to_numpy()
        if len(r)==0 or len(s)==0:
            return float("nan")
        d1 = distance.directed_hausdorff(r, s)[0]
        d2 = distance.directed_hausdorff(s, r)[0]
        return float(max(d1, d2))

    metrics = {
        "temporal_demand": {
            "MAE": _bootstrap_ci(city_hourly_mae, real_df, synth_df, seed=seed),
            "MAPE": _bootstrap_ci(city_hourly_mape, real_df, synth_df, seed=seed),
            "Wasserstein": _bootstrap_ci(city_hourly_wass, real_df, synth_df, seed=seed),
            "DTW": _bootstrap_ci(per_zone_dtw, real_df, synth_df, seed=seed),
            "Pearson_r": _bootstrap_ci(per_zone_pearson, real_df, synth_df, seed=seed),
        },
        "spatial_density": {
            "2D_Wasserstein": _bootstrap_ci(spatial_wass, real_df, synth_df, seed=seed),
            "Jensen-Shannon": _bootstrap_ci(spatial_js, real_df, synth_df, seed=seed),
            "KS-distance": _bootstrap_ci(trip_len_ks, real_df, synth_df, seed=seed),
        },
        "spatio_temporal": {
            "Frobenius": _bootstrap_ci(od_frobenius, real_df, synth_df, seed=seed),
            "Sliced-EMD": _bootstrap_ci(od_sliced_emd, real_df, synth_df, seed=seed),
            "Hausdorff": _bootstrap_ci(hotspot_hausdorff, real_df, synth_df, seed=seed),
        },
    }
    return metrics

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_all_plots(real_df: pd.DataFrame, synth_df: pd.DataFrame, out_dir: str | Path,
                    *, date_col: str = "tpep_pickup_datetime") -> None:
    """Generate evaluation plots comparing real and synthetic data."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_palette(["gray", "magenta"])

    real_df[date_col] = pd.to_datetime(real_df[date_col])
    synth_df[date_col] = pd.to_datetime(synth_df[date_col])

    # Overlayed hourly curves with bootstrap ribbon
    def hourly_counts(df):
        return df.groupby(df[date_col].dt.hour).size().reindex(range(24), fill_value=0)

    real_counts = hourly_counts(real_df)
    synth_counts = hourly_counts(synth_df)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(real_counts.index, real_counts.values, label="Real", color="gray")
    ax.plot(synth_counts.index, synth_counts.values, label="Sintético", color="magenta")
    ax.fill_between(real_counts.index, real_counts.values, synth_counts.values, color="magenta", alpha=0.2)
    ax.set_xlabel("Hora do dia"); ax.set_ylabel("Total de corridas")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir/"overlay_hourly.png")
    plt.close(fig)

    # Violin plot erro horário por zona
    if "PULocationID" in real_df.columns:
        real_hour_zone = real_df.groupby([real_df[date_col].dt.hour, "PULocationID"]).size()
        synth_hour_zone = synth_df.groupby([synth_df[date_col].dt.hour, "PULocationID"]).size()
        df_err = (synth_hour_zone - real_hour_zone).reset_index(name="err").fillna(0)
        fig, ax = plt.subplots(figsize=(10,4))
        sns.violinplot(x="PULocationID", y="err", data=df_err, ax=ax, color="magenta")
        ax.axhline(0, color="gray", lw=1)
        ax.set_title("Erro horário por zona")
        fig.tight_layout(); fig.savefig(out_dir/"violin_zone_error.png"); plt.close(fig)

    # Heat maps side-by-side pickups (4 horas-chave)
    if {"pickup_longitude", "pickup_latitude"}.issubset(real_df.columns):
        key_hours = [0,6,12,18]
        for h in key_hours:
            r = real_df[real_df[date_col].dt.hour==h]
            s = synth_df[synth_df[date_col].dt.hour==h]
            fig, axes = plt.subplots(1,2,figsize=(10,4),sharex=True,sharey=True)
            sns.kdeplot(x=r["pickup_longitude"], y=r["pickup_latitude"], fill=True, ax=axes[0], cmap="Greys")
            sns.kdeplot(x=s["pickup_longitude"], y=s["pickup_latitude"], fill=True, ax=axes[1], cmap="magma")
            axes[0].set_title(f"Real h={h}"); axes[1].set_title(f"Sintético h={h}")
            fig.tight_layout(); fig.savefig(out_dir/f"pickup_heatmap_h{h}.png"); plt.close(fig)

    # OD-matrix Δ-heat map
    if {"PULocationID","DOLocationID"}.issubset(real_df.columns):
        pivot_real = pd.pivot_table(real_df, index="PULocationID", columns="DOLocationID", values=date_col, aggfunc="count", fill_value=0)
        pivot_syn  = pd.pivot_table(synth_df, index="PULocationID", columns="DOLocationID", values=date_col, aggfunc="count", fill_value=0)
        delta = pivot_real.reindex_like(pivot_syn, fill_value=0) - pivot_syn.reindex_like(pivot_real, fill_value=0)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(delta, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("OD-matrix Δ (real – synth)")
        fig.tight_layout(); fig.savefig(out_dir/"od_delta_heatmap.png"); plt.close(fig)

    # Hotspot migration sobre mapa base
    if {"pickup_longitude", "pickup_latitude"}.issubset(real_df.columns):
        def traj(df):
            cent = df.groupby(real_df[date_col].dt.hour)[["pickup_longitude", "pickup_latitude"]].mean().reset_index()
            gdf = gpd.GeoDataFrame(cent, geometry=gpd.points_from_xy(cent.pickup_longitude, cent.pickup_latitude), crs="EPSG:4326")
            return gdf.to_crs(epsg=3857)
        g_real = traj(real_df); g_syn = traj(synth_df)
        base = g_real.plot(color="gray", markersize=50)
        g_syn.plot(ax=base, color="magenta", markersize=20)
        ctx.add_basemap(base, source=ctx.providers.Stamen.TonerLite)
        base.figure.savefig(out_dir/"hotspot_migration.png")
        plt.close(base.figure)

