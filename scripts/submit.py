import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from subprocess import Popen

parser = ArgumentParser()
parser.add_argument("--name", "-n", default="")
name = parser.parse_args().name

log_dir = Path(__file__).parent / "logs"
log_path = log_dir / ("batch" + (f"_{name}" if name else "") + ".log")

cmd = [sys.executable, "-u", "-m", "scripts.batch_run"]
os.makedirs(log_dir, exist_ok=True)

os.chdir(log_dir.parents[1])
with open(log_dir / log_path, "a") as log:
    process = Popen(cmd, stdout=log, stderr=log, start_new_session=True)
    print(f"Launched batch (PID {process.pid})\nLogging to {log_path}")
