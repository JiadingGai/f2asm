"""Offline campaign reports and SVG plots; standard library only.

Only authenticated, fully passed attempts contribute to statistics or plots.
Failed/partial samples remain available in samples.csv, explicitly labelled.
"""
import csv
from html import escape
import io
import json
import math
from pathlib import Path
import tempfile

from .metrics import summarize
from .artifacts import validate_result


DISCLAIMER = "Experimental GPU measurements; not publication qualified. Missing/failed cases are not zeros."
LATENCY_NOTE = "TMA: 16-byte nodes; LDG: 4-byte loads / 32-byte nodes. Warmup differs; curves do not isolate TMA overhead."
THROUGHPUT_NOTE = "Requested decimal GB/s, not distinct HBM bandwidth; overlapping CTA streams retained."
COLORS = ("#1565c0", "#d05a00", "#00866a", "#9147b8", "#a22c45", "#637521")


def _inside(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Report path must stay within campaign directory")
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Report path escapes campaign directory") from exc
    return path


def _atomic(root, name, text):
    path = _inside(root, name)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                     prefix=".report-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _read_attempt(root, manifest, job, record):
    attempts = record.get("attempts", [])
    if not attempts:
        return None, "No attempt recorded"
    attempt = attempts[-1]
    try:
        directory = _inside(root, attempt["path"])
        path = _inside(root, directory.relative_to(root) / "run.json")
        blob = path.read_bytes()
        def finite_float(value):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Nonfinite JSON number")
            return number
        run = json.loads(blob, parse_float=finite_float, parse_constant=finite_float)
        if not isinstance(run, dict):
            raise ValueError("run.json must be an object")
    except FileNotFoundError:
        return None, "run.json missing"
    except (KeyError, TypeError, ValueError, OSError):
        return None, "Unsafe path or unreadable run.json"
    if record.get("status") != "passed":
        return run, None
    if attempt.get("status") != "passed":
        return run, "Latest attempt did not pass"
    try:
        validate_result(root, manifest, job, attempt, verify_hash=True)
    except (KeyError, TypeError, AttributeError, ValueError, OSError) as exc:
        return run, str(exc)
    return run, None


def _collect(root, manifest, state):
    rows, raw_rows = [], []
    sm_count = manifest.get("device", {}).get("sm_count", 0)
    for job in manifest["jobs"]:
        record = state.get("jobs", {}).get(job["id"], {})
        state_status = record.get("status", "pending")
        run, validation_error = _read_attempt(root, manifest, job, record)
        status = state_status
        if state_status == "passed" and validation_error:
            status = "missing" if run is None and validation_error == "run.json missing" else "invalid"
        parameters, launch = job["variant"]["parameters"], job["launch"]
        row = {"job_id": job["id"], "family": job["family"], "kind": job["kind"],
               "case": job["variant"]["case"], "variant": job["variant"]["id"],
               "status": status, "state_status": state_status,
               "error": validation_error if state_status == "passed" else record.get("error"),
               "input_bytes": job["memory"]["input_bytes"],
               "transfer_bytes": parameters.get("transfer_bytes"),
               "shape": parameters.get("shape", []), "blocks": launch["blocks"],
               "sm_multiplier": launch["blocks"] / sm_count if sm_count else None,
               "count": launch["count"], "repetitions": launch["repetitions"],
               "requested_samples": job["samples"], "recorded_samples": 0,
               "performance_claim_eligible": False}
        samples = run.get("raw_samples", []) if isinstance(run, dict) else []
        if not isinstance(samples, list):
            samples = []
        samples = [sample for sample in samples if isinstance(sample, dict)]
        row["recorded_samples"] = len(samples)
        for index, sample in enumerate(samples):
            raw_rows.append({"job_id": job["id"], "status": status,
                             "sample_index": index, "numerical_check": sample.get("numerical_check"),
                             "event_ms": sample.get("event_ms"),
                             "cycles": sample.get("cycles"),
                             "cycles_per_iteration": sample.get("cycles_per_iteration"),
                             "work": sample.get("work"), "rates": sample.get("rates"),
                             "cpu_check_seconds": sample.get("cpu_check_seconds"),
                             "output_sha256": sample.get("output_sha256"),
                             "telemetry_before": sample.get("telemetry_before"),
                             "telemetry_after": sample.get("telemetry_after")})
        if status == "passed":
            for name, values in (("event_ms", [s["event_ms"] for s in samples]),
                                 ("cycles_per_iteration", [value for s in samples
                                                           for value in s["cycles_per_iteration"]])):
                for key, value in summarize(values).items():
                    row[name + "_" + key] = value
            if job["kind"] == "throughput":
                volume = launch["blocks"] * launch["repetitions"] * parameters["transfer_bytes"]
                for key, value in summarize([volume / s["event_ms"] / 1e6 for s in samples]).items():
                    row["requested_gbps_" + key] = value
        rows.append(row)
    return rows, raw_rows


def _csv(rows, fixed_fields):
    fields = list(fixed_fields)
    fields.extend(sorted({key for row in rows for key in row} - set(fields)))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, separators=(",", ":"), allow_nan=False)
                         if isinstance(value, (dict, list)) else value for key, value in row.items()})
    return output.getvalue()


def _text(x, y, value, size=12, anchor="start", color="#253348"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}">{escape(str(value))}</text>')


def _line(x1, y1, x2, y2, color="#cbd2dc", width=1, dash=""):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')


def _bytes(value):
    for divisor, unit in ((1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")):
        if value >= divisor:
            return f"{value / divisor:g} {unit}"
    return f"{value:g} B"


def _svg(width, height, title, subtitle, content):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title, quote=True)}">'
            '<style>text{font-family:Arial,Helvetica,sans-serif}</style>'
            f'<rect width="{width}" height="{height}" fill="white"/>'
            + _text(24, 30, title, 20) + _text(24, 53, subtitle, 12)
            + "".join(content) + '</svg>\n')


def _counts(rows):
    passed = sum(row["status"] == "passed" for row in rows)
    failed = sum(row["status"] in {"failed", "timeout", "invalid", "missing"} for row in rows)
    return f"{passed}/{len(rows)} passed; {failed} failed/invalid; {len(rows) - passed - failed} incomplete"


def _panel(rows, x, y, width, height, title, metric, xlabel, ylabel,
           xvalues, xlabels, key, log=False, categorical=False):
    """Plot only adjacent passing configurations; missing points break lines."""
    output = [_text(x, y, title, 15)]
    left, right, top, bottom = x + 62, x + width - 16, y + 42, y + height - 58
    values = [row[metric] for row in rows if row["status"] == "passed"]
    ymax = max(values, default=1) * 1.12
    unique_x = sorted(set(xvalues))
    if not unique_x:
        output.append(_text((left + right) / 2, (top + bottom) / 2, "No configurations", anchor="middle"))
        return output
    transformed = [math.log2(v) if log else v for v in unique_x]
    lo, hi = min(transformed), max(transformed)
    if hi == lo:
        lo, hi = lo - 1, hi + 1

    def xp(value):
        value = math.log2(value) if log else value
        return left + (value - lo) / (hi - lo) * (right - left)

    def yp(value):
        return bottom - value / ymax * (bottom - top)

    for fraction in (0, .25, .5, .75, 1):
        py = yp(ymax * fraction)
        output.extend((_line(left, py, right, py),
                       _text(left - 8, py + 4, f"{ymax * fraction:.3g}", 10, "end")))
    output.extend((_line(left, top, left, bottom), _line(left, bottom, right, bottom),
                   _text(left, top - 10, ylabel, 10),
                   _text((left + right) / 2, bottom + 44, xlabel, 11, "middle")))
    tick_step = max(1, math.ceil(len(unique_x) / (5 if categorical else 7)))
    for i, value in enumerate(unique_x):
        if i % tick_step and i != len(unique_x) - 1:
            continue
        output.extend((_line(xp(value), bottom, xp(value), bottom + 4),
                       _text(xp(value), bottom + 18, xlabels[value], 10, "middle")))
    groups = sorted(set(key(row) for row in rows))
    for index, group in enumerate(groups):
        color = COLORS[index % len(COLORS)]
        output.append(_text(left + index * (155 if len(groups) <= 2 else 66), y + 23,
                            group, 11, color=color))
        previous = None
        for row in sorted((r for r in rows if key(r) == group), key=lambda r: r["plot_x"]):
            if row["status"] != "passed":
                previous = None
                continue
            px, py = xp(row["plot_x"]), yp(row[metric])
            if previous is not None:
                output.append(_line(*previous, px, py, color, 1.6))
            output.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.4" fill="{color}">'
                          f'<title>{escape(row["variant"])}: {row[metric]:.6g}; '
                          f'{escape(row["job_id"])}</title></circle>')
            previous = (px, py)
    if not values:
        output.append(_text((left + right) / 2, (top + bottom) / 2,
                            "No validated measurements yet", 14, "middle"))
    output.append(_text(x, y + height - 3, _counts(rows), 10))
    return output


def _latency(rows, manifest):
    rows = [dict(row, plot_x=row["input_bytes"]) for row in rows if row["kind"] == "latency"]
    values = [row["plot_x"] for row in rows]
    content = _panel(rows, 24, 85, 1100, 450, "Dependent-access latency",
                     "cycles_per_iteration_median", "Input working set (log2 scale)",
                     "Median cycles / iteration", values, {v: _bytes(v) for v in values},
                     lambda r: "TMA completion" if r["family"] == "tma" else "Ordinary LDG", log=True)
    content.extend((_text(24, 563, LATENCY_NOTE, 12), _text(24, 587, DISCLAIMER, 12)))
    return _svg(1150, 610, "Blackwell latency campaign",
                f'{manifest.get("device", {}).get("name", "Device not measured")} | {_counts(rows)}', content)


def _throughput(rows, manifest):
    rows = [row for row in rows if row["kind"] == "throughput"]
    cases = sorted(set(row["case"] for row in rows))
    content = []
    for index, case in enumerate(cases):
        subset = [dict(row) for row in rows if row["case"] == case]
        categorical = case == "tma_equal_bytes_shape_sweep"
        shapes = sorted(set(tuple(row["shape"]) for row in subset))
        for row in subset:
            row["plot_x"] = shapes.index(tuple(row["shape"])) if categorical else row["transfer_bytes"]
        values = [row["plot_x"] for row in subset]
        labels = {v: "x".join(map(str, shapes[v])) if categorical else _bytes(v) for v in values}
        def grid(row):
            multiplier = row["sm_multiplier"]
            return f"{multiplier:g}S" if multiplier is not None else f'{row["blocks"]} CTAs'
        content.extend(_panel(subset, 24 + (index % 2) * 600, 88 + (index // 2) * 320,
                              574, 304, case.replace("tma_", "").replace("_", " "),
                              "requested_gbps_median", "Tensor shape (equal bytes)" if categorical
                              else "Bytes / transfer", "Median requested GB/s", values, labels, grid,
                              categorical=categorical))
    panel_rows = max(1, math.ceil(len(cases) / 2))
    if not cases:
        content.append(_text(600, 160, "No throughput configurations", 16, "middle"))
    footer = 95 + panel_rows * 320
    content.extend((_text(24, footer, THROUGHPUT_NOTE, 12), _text(24, footer + 24, DISCLAIMER, 12)))
    return _svg(1230, footer + 48, "Blackwell TMA throughput campaign",
                f'{manifest.get("device", {}).get("name", "Device not measured")} | {_counts(rows)}', content)


def write_reports(root: Path, manifest: dict, state: dict) -> None:
    """Write replaceable derived reports; never modify manifest, state, or runs."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows, raw_rows = _collect(root, manifest, state)
    counts = {status: sum(row["status"] == status for row in rows)
              for status in sorted(set(row["status"] for row in rows))}
    summary = {"schema": "blackwell-campaign-summary-v1", "campaign_status": state.get("status"),
               "device": manifest.get("device"), "status_counts": counts,
               "measurement_class": "experimental_not_publication_qualified",
               "performance_claim_eligible": False, "notes": [DISCLAIMER, LATENCY_NOTE, THROUGHPUT_NOTE],
               "configurations": rows}
    _atomic(root, "summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    _atomic(root, "summary.csv", _csv(rows, ("job_id", "family", "kind", "variant", "status", "state_status")))
    _atomic(root, "samples.csv", _csv(raw_rows, ("job_id", "status", "sample_index", "numerical_check")))
    _atomic(root, "latency.svg", _latency(rows, manifest))
    _atomic(root, "throughput.svg", _throughput(rows, manifest))
