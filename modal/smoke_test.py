"""Modal end-to-end smoke test for the Secret Loyalties hackathon.

Verifies Jack's Modal account can (a) run a CPU function and (b) schedule a GPU
container and see the accelerator. Deliberately cheap: the GPU function only
shells out to `nvidia-smi` (no torch install, no model load) so it finishes in
seconds and costs cents.

Run (from WSL, using the modal venv):
    ~/venvs/modal/bin/modal run \
      /mnt/c/Users/HighOrder/prog/multi-agent/secret-loyalties-detection/modal/smoke_test.py

The first run pays a one-time image cold-start (build + push). Subsequent runs
reuse the cached image and start in a few seconds.
"""

import subprocess
import sys

import modal

app = modal.App("sl-smoke-test")

# Slim image; no torch. Debian slim is ~small and fast to build/push.
image = modal.Image.debian_slim(python_version="3.11")

# Cheapest GPU on Modal. A10G (24GB) is the real-workload target; T4 (16GB) is
# only used here because the check is trivial and we want minimum cost.
GPU_TYPE = "T4"


@app.function(image=image)
def hello_cpu() -> dict:
    """CPU function: report the Python version inside the container."""
    return {
        "where": "cpu",
        "python_version": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
    }


@app.function(image=image, gpu=GPU_TYPE)
def gpu_check() -> dict:
    """GPU function: confirm a CUDA device is visible via nvidia-smi.

    No torch — we just parse `nvidia-smi` so the container is light and quick.
    """
    result = {"where": f"gpu:{GPU_TYPE}"}
    try:
        # Compact, parseable query: name, total memory, driver, CUDA version.
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        result["nvidia_smi_rc"] = out.returncode
        result["gpu_info"] = out.stdout.strip()
        if out.stderr.strip():
            result["nvidia_smi_stderr"] = out.stderr.strip()
    except Exception as exc:  # noqa: BLE001 - surface any failure in the report
        result["error"] = f"{type(exc).__name__}: {exc}"

    # Also report the CUDA runtime version line from the full nvidia-smi header.
    try:
        header = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=30
        )
        for line in header.stdout.splitlines():
            if "CUDA Version" in line:
                result["cuda_version_line"] = line.strip()
                break
    except Exception:  # noqa: BLE001
        pass

    return result


@app.local_entrypoint()
def main():
    """Drive both functions and print their results locally."""
    print("== CPU function ==")
    cpu = hello_cpu.remote()
    print(cpu)

    print(f"\n== GPU function (gpu={GPU_TYPE}) ==")
    gpu = gpu_check.remote()
    print(gpu)

    ok_cpu = bool(cpu.get("python_version"))
    ok_gpu = bool(gpu.get("gpu_info")) and gpu.get("nvidia_smi_rc") == 0
    print(f"\nCPU ok: {ok_cpu}   GPU ok: {ok_gpu}")
    if not (ok_cpu and ok_gpu):
        raise SystemExit("smoke test FAILED — see output above")
    print("smoke test PASSED")
