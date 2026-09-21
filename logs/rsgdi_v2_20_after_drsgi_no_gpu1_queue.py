#!/usr/bin/env python3
import os
import subprocess
import sys
import time
from datetime import datetime


ROOT = "/mnt/sdb1/lm/MixerCSeg"
PYTHON = "/home/lm/miniconda3/envs/MixerCSeg/bin/python"
PREV_PID_FILE = os.path.join(
    ROOT, "logs", "20260730_223536_CrackMap_drsgi8_resume_remaining6_no_gpu1.pid"
)
ALLOWED_GPUS = {"0", "2"}
MEM_LIMIT_MIB = 1000
UTIL_LIMIT_PERCENT = 10
POLL_SECONDS = 60

MODES = [
    "topology_bridge_gate",
    "width_adaptive_router",
    "endpoint_bilateral_completion",
    "ridge_token_memory",
    "hessian_laplacian_consensus",
    "frequency_gap_residual",
    "orientation_frequency_coupler",
    "boundary_entropy_hardening",
    "precision_recall_balance_gate",
    "anisotropic_scale_router",
    "subpixel_ridge_alignment",
    "local_global_ridge_fusion",
    "sparse_centerline_router",
    "strip_token_gap_attention",
    "multi_order_gap_refiner",
    "topology_preserving_suppressor",
    "directional_width_calibrator",
    "ridge_gap_transformer_lite",
    "uncertainty_gap_bridge",
    "ensemble_rsgdi_gate",
]


def log(message):
    print(f"[{datetime.now().strftime('%F %T')}] {message}", flush=True)


def read_previous_pid():
    env_pid = os.environ.get("PREV_PID", "").strip()
    if env_pid:
        return env_pid
    try:
        with open(PREV_PID_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def wait_for_previous_queue():
    pid = read_previous_pid()
    if not pid:
        log("no previous queue pid found; continue to GPU wait")
        return
    while pid_alive(pid):
        log(f"waiting for previous DRSGI queue pid={pid}")
        time.sleep(POLL_SECONDS)
    log(f"previous DRSGI queue pid={pid} has exited")


def gpu_rows():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        idx, mem, util = parts
        rows.append((idx, int(mem), int(util)))
    return rows


def select_gpu():
    while True:
        try:
            rows = gpu_rows()
        except Exception as exc:
            log(f"nvidia-smi failed: {exc}; sleep {POLL_SECONDS}s")
            time.sleep(POLL_SECONDS)
            continue
        snapshot = "; ".join(f"{idx}:mem={mem}MiB,util={util}%" for idx, mem, util in rows)
        log(f"gpu snapshot: {snapshot}")
        for idx, mem, util in rows:
            if idx == "1":
                log(f"skip GPU 1 mem={mem}MiB util={util}%")
                continue
            if idx in ALLOWED_GPUS and mem < MEM_LIMIT_MIB and util < UTIL_LIMIT_PERCENT:
                return idx
        log(f"no allowed idle GPU; sleep {POLL_SECONDS}s")
        time.sleep(POLL_SECONDS)


def run_mode(gpu, mode):
    cmd = [
        PYTHON,
        "-u",
        "main.py",
        "--dataset_path",
        "dataset/CrackMap",
        "--nbins",
        "180",
        "--seed",
        "42",
        "--epochs",
        "50",
        "--output_dir",
        "work_dirs",
        "--use_rsgdi_v2",
        "--rsgdi_v2_mode",
        mode,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log(f"START mode={mode}")
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode == 0:
        log(f"DONE mode={mode}")
        return 0
    log(f"FAILED mode={mode} rc={result.returncode}")
    return 1


def main():
    os.chdir(ROOT)
    log("rsgdi-v2 20 after-drsgi no-gpu1 queue started")
    wait_for_previous_queue()
    gpu = select_gpu()
    log(f"selected GPU {gpu}")
    log(f"python -> {PYTHON}")
    failures = 0
    for mode in MODES:
        failures += run_mode(gpu, mode)
    log(f"rsgdi-v2 20 after-drsgi no-gpu1 queue finished with failures={failures}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
