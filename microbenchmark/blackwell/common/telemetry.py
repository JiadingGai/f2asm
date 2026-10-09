"""Optional point-in-time telemetry; not steady-state power measurement."""
import subprocess
import time


def telemetry(device):
    fields = "timestamp,index,name,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,clocks_event_reasons.active"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={device}", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True)
        return {"host_monotonic": time.monotonic(), "fields": fields,
                "csv": result.stdout.strip(), "error": None}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"host_monotonic": time.monotonic(), "error": str(exc)}
