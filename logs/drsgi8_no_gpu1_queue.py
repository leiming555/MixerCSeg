import datetime as _dt
import os
import subprocess
import sys
import time


PROJECT = "/mnt/sdb1/lm/MixerCSeg"
PYTHON = "/home/lm/miniconda3/envs/MixerCSeg/bin/python"

MODES = [
    "pre_gate",
    "parallel_gate",
    "sandwich_gate",
    "pre_scale_ridge_post_gap",
    "direction_ridge_cross",
    "direction_gap_cross",
    "full_cross_gate",
    "full_cross_gate_post_rgd",
]


def log(message):
    print(f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def gpu_rows():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            rows.append((parts[0], int(parts[1]), int(parts[2])))
    return rows


def select_gpu():
    for idx, mem, util in gpu_rows():
        if idx == "1":
            log(f"skip GPU 1 mem={mem}MiB util={util}%")
            continue
        if mem < 1000 and util < 10:
            return idx
    return None


def main():
    log("drsgi8 no-gpu1 queue started")
    gpu = None
    while gpu is None:
        try:
            snapshot = "; ".join(f"{idx},{mem},{util}" for idx, mem, util in gpu_rows())
            log(f"gpu snapshot: {snapshot}")
            gpu = select_gpu()
        except Exception as exc:
            log(f"gpu polling error: {exc}")
        if gpu is None:
            log("no allowed idle GPU; sleep 60s")
            time.sleep(60)

    log(f"selected GPU {gpu}")
    log(f"python -> {PYTHON}")
    failures = 0
    for mode in MODES:
        log(f"START mode={mode}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
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
            "--use_drsgi",
            "--drsgi_mode",
            mode,
        ]
        rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
        if rc == 0:
            log(f"DONE mode={mode}")
        else:
            failures += 1
            log(f"FAILED mode={mode} rc={rc}")
    log(f"drsgi8 no-gpu1 queue finished with failures={failures}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
