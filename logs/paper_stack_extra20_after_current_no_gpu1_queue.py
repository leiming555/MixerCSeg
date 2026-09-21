import datetime as _dt
import os
import subprocess
import sys
import time


PROJECT = "/mnt/sdb1/lm/MixerCSeg"
PYTHON = "/home/lm/miniconda3/envs/MixerCSeg/bin/python"

MODES = [
    "saf_rgp_heb",
    "saf_rgp_vsr",
    "saf_rgp_ros",
    "saf_rgp_ccg",
    "saf_rgp_egc",
    "saf_rgp_cwa",
    "saf_rgp_tcc",
    "saf_rgp_srt",
    "saf_rgp_lpq",
    "saf_rgp_adr",
    "saf_rgp_cfc",
    "saf_rgp_mlf",
    "saf_rgp_dsm",
    "saf_rgp_hsc",
    "saf_rgp_pcr",
    "saf_rgp_fhr",
    "saf_eut_vsr",
    "saf_eut_cwa",
    "saf_ons_mlf",
    "saf_stc_pcr",
]


def log(message):
    print(f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def pid_alive(pid_text):
    if not pid_text:
        return False
    try:
        os.kill(int(pid_text), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except ValueError:
        return False


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
    prev_pid = os.environ.get("PREV_PID", "").strip()
    log("paper stack extra20 after-current no-gpu1 queue started")
    log(f"previous queue pid={prev_pid}")
    while pid_alive(prev_pid):
        log(f"waiting for previous queue pid={prev_pid} to finish")
        time.sleep(300)

    log("previous queue finished; start idle GPU polling")
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
            "--use_paper_stack",
            "--paper_stack_mode",
            mode,
        ]
        rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
        if rc == 0:
            log(f"DONE mode={mode}")
        else:
            failures += 1
            log(f"FAILED mode={mode} rc={rc}")
    log(f"paper stack extra20 after-current no-gpu1 queue finished with failures={failures}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
