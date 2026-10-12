"""B200 counterparts of the Hopper paper's Figures 2-4; optional Matplotlib."""
import argparse
import csv
import json
from pathlib import Path

from ..campaign import validate_completed
from ..common.artifacts import sha256
from ..common.reporting import _collect
from .publication import DEVICE_FIELDS, load


def generate(results, output, *, published=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    if published:
        provenance, rows, samples = load(results)
        manifest = {"identity": provenance["campaign_identity"], "device": provenance["device"]}
        input_hashes = provenance["file_hashes"]
    else:
        manifest = json.loads((results / "manifest.json").read_text())
        state = json.loads((results / "state.json").read_text())
        validate_completed(results, manifest, state)
        rows, samples = _collect(results, manifest, state)
        input_hashes = {name: sha256((results/name).read_bytes()) for name in ("manifest.json", "state.json")}
    if (len(rows) != 216
            or any(r["status"] != "passed" or r["recorded_samples"] != 5 for r in rows)
            or len(samples) != 1080 or "B200" not in manifest["device"]["name"]):
        raise ValueError("These figures require the complete 216-case, five-sample B200 campaign")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.linewidth": .75,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.hashsalt": manifest["identity"], "savefig.facecolor": "white"})
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "D", "x", "^", "v"]
    outputs, points = {}, []

    def series(ax, selected, xs, label, color, marker, metric, figure):
        if not selected:
            raise ValueError(f"Missing series: {label}")
        ax.plot(xs, [r[metric] for r in selected], color=color, marker=marker,
                linewidth=1.5, markersize=4.5, label=label)
        points.extend({"figure": figure, "series": label, "job_id": r["job_id"],
                       "input_bytes": r["input_bytes"], "transfer_bytes": r["transfer_bytes"],
                       "blocks": r["blocks"], "shape": "x".join(map(str, r["shape"])),
                       "metric": metric, "median": r[metric], "samples": r["recorded_samples"]}
                      for r in selected)

    def save(fig, stem, caption, note):
        fig.text(.5, .067, caption, ha="center", va="center", fontfamily="DejaVu Serif", fontsize=12)
        fig.text(.5, .018, note, ha="center", va="center", fontsize=9, color="#424242")
        for extension in ("png", "svg", "pdf"):
            path = output / f"{stem}.{extension}"
            fig.savefig(path, dpi=200)
            outputs[path.name] = sha256(path.read_bytes())
        plt.close(fig)

    def throughput_axis(ax):
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.tick_params(direction="in")

    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    fig.subplots_adjust(left=.075, right=.985, bottom=.32, top=.90)
    ax.set_facecolor("#eeeeee")
    for family, label, color in (("memory", "B200 (ordinary LDG)", colors[0]),
                                 ("tma", "B200 (TMA)", "#548235")):
        selected = sorted((r for r in rows if r["kind"] == "latency" and r["family"] == family),
                          key=lambda r: r["input_bytes"])
        series(ax, selected, [r["input_bytes"] for r in selected], label, color, ".",
               "cycles_per_iteration_median", 2)
    ax.set_xscale("log", base=2)
    ticks = [2**n for n in range(10, 29, 2)]
    ax.set_xticks(ticks, [f"{v // 2**20} MiB" if v >= 2**20 else f"{v // 1024} KiB" for v in ticks],
                  rotation=30, ha="right")
    ax.set_ylim(0, 820)
    ax.set_yticks(range(0, 801, 200))
    ax.set_xlabel("Chain data volume", labelpad=1)
    ax.set_ylabel("Latency (cycles)")
    ax.grid(True, linestyle=":", color="#777777", linewidth=.55)
    ax.legend(loc="upper left", framealpha=.85)
    save(fig, "figure2_b200_latency", "Figure 2. B200 dependent-access latency with ordinary loads and TMA.",
         "Median of 5 samples; 21 measured sizes per curve. Node spacing and warm-up differ; includes loop/completion overhead.")

    cases = ["tma_bulk_throughput", "tma_tensor_1d_throughput", "tma_tensor_2d_throughput", "tma_tensor_3d_throughput"]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.8), sharey=True)
    fig.subplots_adjust(left=.067, right=.99, bottom=.30, top=.90, wspace=.08)
    for ax, case, title in zip(axes, cases, ("Non-Tensor", "1D Tensor", "2D Tensor", "3D Tensor")):
        subset = [r for r in rows if r["case"] == case]
        for i in range(6):
            selected = sorted((r for r in subset if r["sm_multiplier"] == i+1), key=lambda r: r["transfer_bytes"])
            label = f"{selected[0]['blocks']} blocks ({i+1}S)"
            series(ax, selected, range(len(selected)), label, colors[i], markers[i], "requested_gbps_median", 3)
        sizes = sorted({r["transfer_bytes"] for r in subset})
        ax.set_xticks(range(len(sizes)), [f"{v/1024:g}" for v in sizes])
        ax.set_title(title)
        throughput_axis(ax)
    axes[0].set_ylabel("Requested throughput (GB/s)")
    fig.text(.53, .215, "Loaded size (KiB) per TMA copy", ha="center", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(.53, .145), ncol=6, frameon=True,
               columnspacing=1.0, handletextpad=.5)
    save(fig, "figure3_b200_throughput", "Figure 3. B200 TMA throughput across copy sizes and thread-block counts.",
         "Median of 5 samples; S = 148 SMs. Overlapping streams and cached data: requested transfer rate, not peak HBM bandwidth.")

    fig, ax = plt.subplots(figsize=(9, 5.7))
    fig.subplots_adjust(left=.12, right=.98, bottom=.30, top=.94)
    shapes = [(16,16,16), (64,64,1), (256,16,1), (16,256,1), (4,4,256)]
    for i, shape in enumerate(shapes):
        selected = sorted((r for r in rows if r["case"] == "tma_equal_bytes_shape_sweep" and tuple(r["shape"]) == shape),
                          key=lambda r: r["blocks"])
        series(ax, selected, [r["blocks"] for r in selected], " × ".join(map(str, shape)),
               colors[i], ["^", "s", "D", "x", "+"][i], "requested_gbps_median", 4)
    sm = manifest["device"]["sm_count"]
    ax.set_xticks([sm*i for i in range(1, 7)])
    ax.set_xlabel("Thread blocks")
    ax.set_ylabel("Requested throughput (GB/s)")
    throughput_axis(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc="center", bbox_to_anchor=(.55, .155), ncol=3, frameon=True)
    save(fig, "figure4_b200_shapes", "Figure 4. B200 throughput for 16-KiB TMA copies with different tensor shapes.",
         "Median of 5 samples; 148-888 blocks. Overlapping streams: requested transfer rate, not peak HBM bandwidth.")

    if [sum(p["figure"] == n for p in points) for n in (2,3,4)] != [42,144,30]:
        raise ValueError("Figure coverage does not match the campaign")
    with (output / "plotted_points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=points[0].keys())
        writer.writeheader(); writer.writerows(points)
    outputs["plotted_points.csv"] = sha256((output / "plotted_points.csv").read_bytes())
    provenance = {"campaign_identity": manifest["identity"],
                  "device": {key: manifest["device"][key] for key in DEVICE_FIELDS},
                  "input_kind": "published_measurements" if published else "raw_campaign",
                  "input_hashes": input_hashes,
                  "plotter_sha256": sha256(Path(__file__).read_bytes()), "matplotlib_version": matplotlib.__version__,
                  "points_by_figure": {"2": 42, "3": 144, "4": 30}, "statistic": "median of 5 samples",
                  "performance_claim_eligible": False, "output_hashes": outputs}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(output), "files": list(outputs), "points": 216}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--published", action="store_true", help="Use the portable published measurements, not the full raw archive")
    args = parser.parse_args()
    generate(args.results, args.output, published=args.published)
