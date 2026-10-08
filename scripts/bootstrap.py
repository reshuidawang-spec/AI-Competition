#!/usr/bin/env python3
"""Create a reproducible local Python environment and verify the clone.

Usage on Ubuntu: python3.10 scripts/bootstrap.py
Usage on Windows: py -3.10 scripts/bootstrap.py
Use --core-only when only the simulator is needed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_DIR / ".venv"


def run(*args: str, env=None) -> None:
    print("+", " ".join(map(str, args)), flush=True)
    clean_env = (env if env is not None else os.environ).copy()
    # A sourced ROS workspace can prepend external site-packages to PYTHONPATH.
    # Keep the bootstrap isolated so a passing check proves this clone works alone.
    if env is None:
        clean_env.pop("PYTHONPATH", None)
        clean_env.pop("PYTHONHOME", None)
    subprocess.run(args, cwd=PROJECT_DIR, check=True, env=clean_env)


def create_environment(vpy: Path) -> None:
    if vpy.is_file() and subprocess.run(
            [str(vpy), "-m", "pip", "--version"],
            cwd=PROJECT_DIR, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL).returncode == 0:
        return
    print("+", sys.executable, "-m venv", VENV_DIR, flush=True)
    if subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)],
                      cwd=PROJECT_DIR).returncode == 0:
        return
    print("系统缺少 ensurepip，尝试 virtualenv 创建隔离环境。", flush=True)
    command = [sys.executable, "-m", "virtualenv", str(VENV_DIR)]
    if subprocess.run(command, cwd=PROJECT_DIR).returncode == 0:
        return
    tools_dir = PROJECT_DIR / ".bootstrap-tools"
    run(sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "--target", str(tools_dir), "virtualenv==21.6.1")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tools_dir) + os.pathsep + env.get("PYTHONPATH", "")
    run(*command, env=env)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-only", action="store_true",
                        help="install and check the simulation without AI inference")
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 10):
        parser.error("请使用 Python 3.10；该版本已在本工程验证")

    vpy = VENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    create_environment(vpy)
    if not vpy.is_file():
        parser.error(f"虚拟环境不完整：{vpy}；请检查或重建 .venv")

    run(str(vpy), "-m", "pip", "install", "--disable-pip-version-check",
        "--no-cache-dir", "--upgrade", "pip")
    if not args.core_only:
        run(str(vpy), "-m", "pip", "install", "--disable-pip-version-check",
            "--no-cache-dir", "--index-url", "https://download.pytorch.org/whl/cpu",
            "torch==2.9.1+cpu", "torchvision==0.24.1+cpu")
    requirements = "requirements.txt" if args.core_only else "requirements-vision.txt"
    run(str(vpy), "-m", "pip", "install", "--disable-pip-version-check",
        "--no-cache-dir", "-r", requirements)
    check = [str(vpy), "scripts/verify_portable.py"]
    if args.core_only:
        check.append("--core-only")
    run(*check)
    print(f"可用环境：{vpy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
