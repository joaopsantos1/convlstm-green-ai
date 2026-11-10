"""Organize the IOPS results and draw the comparison figures.

Models:
  convlstm     ConvLSTM, 2 layers (campaign 1)
  convlstm6l   ConvLSTM, 6 layers, same depth as MS-ConvLSTM (campaign 2, *_convlstm6)
  ms_convlstm  MS-ConvLSTM, 6 layers at full/half/quarter resolution (campaign 1)

Inputs (copied from plafrim):
  iops_runs/<stage>[_convlstm6]/run_001/{results.csv,__iops_resource_summary.csv}
  experiments_results/training_*/metrics/loss_epoch_standard_*.csv

Outputs:
  tables/runs_<stage>.csv          one row per execution, tidy columns
  tables/summary_<stage>.csv       mean/std per dataset x model (x host)
  tables/model_ratios.csv          pairwise ratios, overall and host-matched
  figures/*.png
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "iops_runs"
TAB = ROOT / "tables"
FIG = ROOT / "figures"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

KEYS = ["dataset", "model_architecture", "repetition"]
DATASETS = ["taasrad19", "moving_mnist", "meteonet"]
ARCHS = ["convlstm", "convlstm6l", "ms_convlstm"]
ARCH_LABEL = {"convlstm": "ConvLSTM 2L", "convlstm6l": "ConvLSTM 6L", "ms_convlstm": "MS-ConvLSTM 6L"}
ARCH_COLOR = {"convlstm": "#2a78d6", "ms_convlstm": "#eb6834", "convlstm6l": "#1baf7a"}
# (numerator, denominator) for the ratio table
COMPARISONS = [("ms_convlstm", "convlstm6l"), ("ms_convlstm", "convlstm"), ("convlstm6l", "convlstm")]
HOST_MARKER = {"sirocco21": "o", "sirocco23": "^", "sirocco24": "s"}

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 9,
    }
)


# ---------------------------------------------------------------- loading
def load_stage(stage: str, model: str | None = None) -> pd.DataFrame:
    run = RUNS / stage / "run_001"
    res = pd.read_csv(run / "results.csv").rename(
        columns={
            "vars.dataset": "dataset",
            "vars.model_architecture": "model_architecture",
            "execution.repetition": "repetition",
        }
    )
    res = res[res["metadata.executor_status"] == "SUCCEEDED"]
    gpu = pd.read_csv(run / "__iops_resource_summary.csv")
    df = res.merge(gpu, on=KEYS, suffixes=("", "_res"))
    if model:
        df["model_architecture"] = model

    df["host"] = df["metadata.sysinfo.hostname"].str.split(".").str[0]
    df["gpus_on_node"] = df["gpu_count"].astype(int)

    # The code uses one GPU. On 2-GPU nodes the second one sits idle (0% util,
    # ~32 W) and IOPS' gpu_energy_j adds it in. Keep only the busiest GPU.
    util = df[[c for c in ["gpu0_avg_utilization_pct", "gpu1_avg_utilization_pct"] if c in df]].fillna(-1)
    active = util.values.argmax(axis=1)
    for col in ["energy_j", "avg_power_w", "avg_utilization_pct", "mem_peak_mib"]:
        vals = df[[f"gpu0_{col}", f"gpu1_{col}"]].values
        df[f"active_gpu_{col}"] = vals[np.arange(len(df)), active]
    df["idle_gpu_energy_j"] = df["gpu_energy_j"] - df["active_gpu_energy_j"]

    metric_cols = [c for c in df.columns if c.startswith("metrics.")]
    keep = (
        KEYS
        + ["host", "gpus_on_node", "metadata.jobid"]
        + metric_cols
        + [
            "active_gpu_energy_j",
            "active_gpu_avg_power_w",
            "active_gpu_avg_utilization_pct",
            "active_gpu_mem_peak_mib",
            "idle_gpu_energy_j",
            "gpu_energy_j",
            "gpu_trace_duration_s",
            "cpu_avg_pct",
            "mem_peak_gb",
        ]
    )
    out = df[keep].rename(columns={c: c.replace("metrics.", "") for c in metric_cols})
    out = out.rename(columns={"metadata.jobid": "jobid", "gpu_energy_j": "all_gpus_energy_j"})
    return out.sort_values(KEYS).reset_index(drop=True)


def load_all(stage: str) -> pd.DataFrame:
    df = pd.concat([load_stage(stage), load_stage(f"{stage}_convlstm6", "convlstm6l")], ignore_index=True)
    return df.sort_values(KEYS).reset_index(drop=True)


def load_loss_curves() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for arch in ARCHS:
            for rep in range(1, 5):
                f = (
                    ROOT
                    / "experiments_results"
                    / f"training_{ds}_standard_{arch}_{rep}"
                    / "metrics"
                    / f"loss_epoch_standard_{ds}.csv"
                )
                if f.exists():
                    d = pd.read_csv(f)[["epoch", "loss"]]
                    d[KEYS] = ds, arch, rep
                    rows.append(d)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- tables
def summarize(df: pd.DataFrame, cols: list[str], by_host: bool) -> pd.DataFrame:
    g = ["dataset", "model_architecture"] + (["host"] if by_host else [])
    s = df.groupby(g)[cols].agg(["mean", "std", "count"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    return s.reset_index()


def ratios(df: pd.DataFrame, cols: list[str], stage: str, host_matched: bool = True) -> list[dict]:
    """num/den mean ratio per dataset for each pair in COMPARISONS: naive (all
    runs) and host-matched (geometric mean of per-host ratios, hosts that ran both)."""
    out = []
    for num, den in COMPARISONS:
        for ds in DATASETS:
            d = df[(df.dataset == ds) & df.model_architecture.isin([num, den])]
            for c in cols:
                m = d.groupby("model_architecture")[c].mean()
                per_host = d.groupby(["host", "model_architecture"])[c].mean().unstack()
                per_host = per_host.dropna() if {num, den} <= set(per_host.columns) else per_host.iloc[0:0]
                hm = (
                    float(np.exp(np.log(per_host[num] / per_host[den]).mean()))
                    if len(per_host) and host_matched
                    else np.nan
                )
                out.append(
                    {
                        "stage": stage,
                        "comparison": f"{num} / {den}",
                        "dataset": ds,
                        "metric": c,
                        "numerator_mean": m.get(num),
                        "denominator_mean": m.get(den),
                        "ratio": m.get(num) / m.get(den),
                        "ratio_host_matched": hm,
                        "hosts_with_both": ",".join(per_host.index) if host_matched else "",
                    }
                )
    return out


# ---------------------------------------------------------------- plotting
def strip_panel(ax, d: pd.DataFrame, col: str, scale: float = 1.0, zero: bool = True):
    rng = np.random.default_rng(0)
    for i, arch in enumerate(ARCHS):
        a = d[d.model_architecture == arch]
        if a.empty:
            continue
        y = a[col].values * scale
        ax.hlines(y.mean(), i - 0.28, i + 0.28, color=INK2, linewidth=1.5, zorder=2)
        for (_, row), yy in zip(a.iterrows(), y):
            ax.scatter(
                i + rng.uniform(-0.15, 0.15),
                yy,
                s=46,
                marker=HOST_MARKER.get(row.host, "o"),
                color=ARCH_COLOR[arch],
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=3,
            )
    ax.set_xticks(range(len(ARCHS)), [ARCH_LABEL[a].replace(" ", "\n", 1) for a in ARCHS])
    ax.set_xlim(-0.6, len(ARCHS) - 0.4)
    if zero:
        ax.set_ylim(bottom=0)
    ax.margins(y=0.15)


def legend_handles(include_hosts=True):
    h = [
        Line2D([], [], marker="o", linestyle="", color=ARCH_COLOR[a], markersize=8, label=ARCH_LABEL[a])
        for a in ARCHS
    ]
    if include_hosts:
        h += [
            Line2D([], [], marker=m, linestyle="", color=INK2, markerfacecolor="none", markersize=8, label=host)
            for host, m in HOST_MARKER.items()
        ]
    h.append(Line2D([], [], color=INK2, linewidth=1.5, label="mean"))
    return h


def grid_figure(df, rows, title, fname, note=None):
    fig, axes = plt.subplots(len(rows), len(DATASETS), figsize=(11, 2.7 * len(rows) + 0.9), squeeze=False)
    for r, (col, label, scale, zero) in enumerate(rows):
        for c, ds in enumerate(DATASETS):
            ax = axes[r][c]
            strip_panel(ax, df[df.dataset == ds], col, scale, zero)
            if r == 0:
                ax.set_title(ds, loc="left", fontweight="bold")
            if c == 0:
                ax.set_ylabel(label)
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.12 / h, ha="left", va="top", fontsize=13, color=INK, fontweight="bold")
    fig.legend(handles=legend_handles(), loc="upper left", ncol=len(ARCHS) + len(HOST_MARKER) + 1, frameon=False, bbox_to_anchor=(0.005, 1 - 0.45 / h))
    if note:
        fig.text(0.01, 0.08 / h, textwrap.fill(note, 170), fontsize=8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, (0.45 if note else 0.05) / h, 1, 1 - 0.85 / h))
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def loss_figure(loss: pd.DataFrame):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 4.0))
    for ax, ds in zip(axes, DATASETS):
        d = loss[loss.dataset == ds]
        ax.set_yscale("log")
        ax.set_ylim(d.loss.min() * 0.85, d.loss.max() * 1.1)
        ends = []
        for arch in ARCHS:
            a = d[d.model_architecture == arch]
            for _, r in a.groupby("repetition"):
                ax.plot(r.epoch, r.loss, color=ARCH_COLOR[arch], alpha=0.25, linewidth=1)
            m = a.groupby("epoch").loss.mean()
            ax.plot(m.index, m.values, color=ARCH_COLOR[arch], linewidth=2, label=ARCH_LABEL[arch])
            ends.append((m.values[-1], m.index[-1]))
        # end-of-line value labels, nudged apart (in points) when they would overlap
        ends.sort()
        offsets, prev = [], None
        for v, _ in ends:
            pos = ax.transData.transform((0, v))[1] * 72 / fig.dpi
            off = 0 if prev is None or pos - prev >= 9 else prev + 9 - pos
            offsets.append(off)
            prev = pos + off
        for (v, x), off in zip(ends, offsets):
            ax.annotate(f"{v:.4f}", (x, v), xytext=(4, off), textcoords="offset points", fontsize=8, color=INK2, va="center")
        ax.set_title(ds, loc="left", fontweight="bold")
        ax.set_xlabel("epoch")
        ax.grid(axis="both", color=GRID)
    axes[0].set_ylabel("training loss (log)")
    fig.suptitle("Training loss per epoch (thin: each repetition, thick: mean)", x=0.01, y=0.97, ha="left", va="top", fontsize=13, fontweight="bold")
    fig.legend(handles=legend_handles(include_hosts=False)[: len(ARCHS)], loc="upper left", ncol=len(ARCHS), frameon=False, bbox_to_anchor=(0.005, 0.89))
    fig.text(
        0.01,
        0.01,
        "Teacher forcing decays linearly from 0.5 to 0 over the 100 epochs, so later epochs are harder; "
        "a rising training loss reflects that schedule, not divergence.",
        fontsize=8,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.82))
    fig.savefig(FIG / "fig4_training_loss.png", dpi=150)
    plt.close(fig)


def summary_figure(train: pd.DataFrame, ev: pd.DataFrame):
    """One-row overview for sharing: training GPU energy (cost) vs test MAE (error).

    Energy uses only sirocco21/24 runs (sirocco23 is underfed and not comparable);
    MAE uses every trained model, since quality does not depend on the node.
    """
    fast = train[train.host.isin(["sirocco21", "sirocco24"])]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(12, 4.4))
    for ax, ds in zip(axes, DATASETS):
        pts, ylo, yhi = {}, [], []
        for arch in ARCHS:
            e = fast[(fast.dataset == ds) & (fast.model_architecture == arch)].active_gpu_energy_j / 3.6e6
            q = ev[(ev.dataset == ds) & (ev.model_architecture == arch)].MAE
            pts[arch] = (e.mean(), q.mean())
            ylo.append(q.mean() - q.std())
            yhi.append(q.mean() + q.std())
            ax.errorbar(
                e.mean(), q.mean(), xerr=e.std() if len(e) > 1 else None, yerr=q.std(),
                fmt="o", markersize=10, color=ARCH_COLOR[arch], ecolor=ARCH_COLOR[arch],
                elinewidth=1.5, capsize=3, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
            )
            ax.annotate(
                ARCH_LABEL[arch], pts[arch], xytext=(10, -16) if arch == "ms_convlstm" else (10, 6),
                textcoords="offset points",
                fontsize=9, color=INK, fontweight="bold" if arch == "ms_convlstm" else "normal",
            )
        # MS vs same-depth ConvLSTM
        (e6, q6), (ems, qms) = pts["convlstm6l"], pts["ms_convlstm"]
        ax.annotate(
            "", xy=(ems, qms), xytext=(e6, q6),
            arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=1.2, shrinkA=9, shrinkB=9, linestyle="--"),
            zorder=2,
        )
        ax.text(
            0.98, 0.97,
            f"MS vs same-depth ConvLSTM:\n{(ems / e6 - 1) * 100:+.0f}% energy, {(qms / q6 - 1) * 100:+.0f}% MAE",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=GRID),
        )
        ax.set_title(ds, loc="left", fontweight="bold")
        ax.set_xlabel("training GPU energy per run [kWh]  (lower = cheaper)")
        ax.set_xlim(0, max(p[0] for p in pts.values()) * 1.45)
        lo, hi = min(ylo), max(yhi)
        ax.set_ylim(lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.4)
        ax.grid(axis="both", color=GRID)
    axes[0].set_ylabel("test MAE  (lower = better)")
    fig.suptitle(
        "MS-ConvLSTM: about half the training energy of a same-depth ConvLSTM, with lower error",
        x=0.01, y=0.98, ha="left", va="top", fontsize=13, fontweight="bold", color=INK,
    )
    fig.text(
        0.01, 0.9,
        "Bottom-left is best. MAE: mean of the 4 trained models; energy: mean of the runs on sirocco21/24; bars: standard deviation. "
        "ConvLSTM 2L is cheaper but clearly less accurate.",
        fontsize=9, color=INK2, ha="left",
    )
    fig.text(
        0.01, 0.01,
        "100 epochs, batch 10, 64 channels, NVIDIA A100-PCIE-40GB (plafrim). Energy = active GPU, from IOPS 1 s nvidia-smi "
        "sampling, runs on sirocco21/24 only (n = 1 to 4 per point). MAE on the test split, all 4 repetitions.",
        fontsize=7.5, color=INK2, ha="left",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.savefig(FIG / "fig5_summary.png", dpi=200)
    fig.savefig(FIG / "fig5_summary.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    train = load_all("train")
    pred = load_all("predict")
    ev = load_all("predict_evaluation")
    loss = load_loss_curves()

    for name, df in [("train", train), ("predict", pred), ("predict_evaluation", ev)]:
        df.to_csv(TAB / f"runs_{name}.csv", index=False)

    train_cols = [
        "total_training_time_sec",
        "average_epoch_time_sec",
        "min_epoch_loss",
        "active_gpu_energy_j",
        "active_gpu_avg_power_w",
        "active_gpu_avg_utilization_pct",
        "active_gpu_mem_peak_mib",
    ]
    pred_cols = ["total_predict_time_sec", "active_gpu_energy_j", "active_gpu_avg_power_w"]
    qual_cols = ["MAE", "MSE", "RMSE", "SSIM", "PSNR", "CSI_avg", "POD_avg", "FAR_avg", "HSS_avg"]

    summarize(train, train_cols, False).to_csv(TAB / "summary_train.csv", index=False)
    summarize(train, train_cols, True).to_csv(TAB / "summary_train_by_host.csv", index=False)
    summarize(pred, pred_cols, False).to_csv(TAB / "summary_predict.csv", index=False)
    summarize(pred, pred_cols, True).to_csv(TAB / "summary_predict_by_host.csv", index=False)
    summarize(ev, qual_cols + ["total_predict_time_with_evaluation_sec"], False).to_csv(
        TAB / "summary_quality.csv", index=False
    )

    rat = (
        ratios(train, ["total_training_time_sec", "active_gpu_energy_j", "min_epoch_loss"], "train")
        + ratios(pred, ["total_predict_time_sec", "active_gpu_energy_j"], "predict")
        + ratios(ev, qual_cols, "predict_evaluation", host_matched=False)
    )
    pd.DataFrame(rat).to_csv(TAB / "model_ratios.csv", index=False)

    host_note = (
        "Marker shape = node. sirocco23 runs its A100 at ~70-77% utilization and ~90-110 W (vs ~93% and ~190-210 W on "
        "sirocco21/24), so it is slower but draws less energy; compare shapes, not only colors. ConvLSTM 6L ran on sirocco21/24 only."
    )
    grid_figure(
        train,
        [
            ("total_training_time_sec", "training time [h]", 1 / 3600, True),
            ("active_gpu_energy_j", "GPU energy [kWh]", 1 / 3.6e6, True),
            ("active_gpu_avg_power_w", "avg GPU power [W]", 1, True),
        ],
        "Training (100 epochs): three models on A100",
        "fig1_training_cost.png",
        host_note,
    )
    grid_figure(
        pred,
        [
            ("total_predict_time_sec", "prediction time [s]", 1, True),
            ("active_gpu_energy_j", "GPU energy [kJ]", 1 / 1e3, True),
        ],
        "Inference on the test split (no evaluation)",
        "fig2_inference_cost.png",
        "Marker shape = node. GPU energy covers the whole job trace (model load, data load, forecasting).",
    )
    grid_figure(
        ev,
        [
            ("MAE", "MAE (lower is better)", 1, False),
            ("SSIM", "SSIM (higher is better)", 1, False),
            ("CSI_avg", "CSI avg (higher is better)", 1, False),
            ("HSS_avg", "HSS avg (higher is better)", 1, False),
        ],
        "Forecast quality on the test split",
        "fig3_quality.png",
        "Each dot is one trained model (repetition). Quality does not depend on the node.",
    )
    loss_figure(loss)
    summary_figure(train, ev)

    pd.set_option("display.width", 200)
    print(pd.DataFrame(rat).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
