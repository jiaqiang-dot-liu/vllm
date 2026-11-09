# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import io
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import pytest

from vllm.utils import system_utils
from vllm.utils.system_utils import (
    _maybe_force_spawn,
    suppress_stdout,
    unique_filepath,
)


def test_unique_filepath():
    temp_dir = tempfile.mkdtemp()
    path_fn = lambda i: Path(temp_dir) / f"file_{i}.txt"
    paths = set()
    for _ in range(10):
        path = unique_filepath(path_fn)
        path.write_text("test")
        paths.add(path)
    assert len(paths) == 10
    assert len(list(Path(temp_dir).glob("*.txt"))) == 10


@pytest.mark.parametrize("start_method", [None, "fork"])
@pytest.mark.parametrize(
    "numa_args, force_spawn",
    [
        (["--numa-bind"], True),
        (["--numa-bind-nodes", "0"], True),
        (["--numa-bind-nodes=0"], True),
        (["--numa-bind-cpus", "0-3"], True),
        (["--numa-bind-cpus=0-3"], True),
        (["--numa_bind"], True),
        (["--numa_bind_nodes", "0"], True),
        (["--numa_bind_cpus=0-3"], True),
        ([], False),
        (["--no-numa-bind"], False),
    ],
)
def test_numa_bind_cli_start_method(monkeypatch, start_method, numa_args, force_spawn):
    if start_method is None:
        monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    else:
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", start_method)
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", *numa_args])
    for check in (
        "is_in_ray_actor",
        "cuda_is_initialized",
        "xpu_is_initialized",
        "in_wsl",
    ):
        monkeypatch.setattr(system_utils, check, lambda: False)

    _maybe_force_spawn()
    expected_method = "spawn" if force_spawn else start_method
    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == expected_method


def test_suppress_stdout_when_sys_stdout_has_no_fd(capfd):
    with contextlib.redirect_stdout(io.StringIO()), suppress_stdout():
        os.write(1, b"c library output\n")
    assert "c library output" not in capfd.readouterr().out


def test_suppress_stdout_keeps_stderr_when_sys_stdout_is_stderr(capfd, monkeypatch):
    monkeypatch.setattr(sys, "stdout", sys.stderr)
    with suppress_stdout():
        os.write(1, b"c library output\n")
        os.write(sys.stderr.fileno(), b"error message\n")
    out, err = capfd.readouterr()
    assert "c library output" not in out
    assert "error message" in err


def test_kill_process_tree_skips_child_that_exits_mid_walk(monkeypatch):
    sleep = [sys.executable, "-c", "import time; time.sleep(60)"]
    target, *children = procs = [subprocess.Popen(sleep) for _ in range(4)]
    try:
        # List the children while they are alive, then let one exit and be
        # reaped before the walk reaches it, so psutil's name() raises.
        listed = [psutil.Process(p.pid) for p in children]
        children[1].terminate()
        children[1].wait(timeout=10)
        monkeypatch.setattr(
            psutil.Process, "children", lambda self, recursive=False: listed
        )

        system_utils.kill_process_tree(target.pid)

        # The exited child died from our SIGTERM; the rest were SIGKILLed.
        assert children[1].returncode == -signal.SIGTERM
        for p in (children[0], children[2], target):
            assert p.wait(timeout=10) == -signal.SIGKILL
    finally:
        for p in procs:
            p.kill()
            p.wait()


def test_forked_child_can_use_openmp_after_parent_did():
    """A forked child deadlocks in libgomp if its parent already ran a parallel
    region, unless get_mp_context() releases the parent's OpenMP workers."""
    script = """
import torch
from vllm.utils.system_utils import get_mp_context

torch.set_num_threads(4)
x = torch.randn(1024, 1024)
torch.einsum("in,jn->ij", x, x)
proc = get_mp_context().Process(target=torch.zeros, args=(64 * 1024 * 1024,))
proc.start()
proc.join(60)
hung = proc.is_alive()
proc.kill()
raise SystemExit(1 if hung else proc.exitcode)
"""
    env = os.environ | {"VLLM_WORKER_MULTIPROC_METHOD": "fork"}
    result = subprocess.run([sys.executable, "-c", script], env=env, timeout=120)
    assert result.returncode == 0


# A process that arms the parent-death signal and then idles. `mode` controls
# which thread spawns it and whether it arms before or after its parent dies.
_PARENT_DEATH_LAUNCHER = """
import os
import subprocess
import sys
import threading
import time

child_script = '''
import signal, sys, time
from vllm.utils.system_utils import set_parent_death_signal
time.sleep(float(sys.argv[1]))
set_parent_death_signal(signal.SIGTERM)
print("armed", flush=True)
time.sleep(600)
'''

mode = sys.argv[1]
pid_file = sys.argv[2]
arm_delay = "2" if mode == "late_arm" else "0"


def spawn():
    proc = subprocess.Popen(
        [sys.executable, "-c", child_script, arm_delay], stdout=subprocess.PIPE
    )
    if mode != "late_arm":
        # Wait until the child has armed the signal.
        proc.stdout.readline()
    with open(pid_file, "w") as f:
        f.write(str(proc.pid))


if mode == "thread":
    thread = threading.Thread(target=spawn)
    thread.start()
    thread.join()
else:
    spawn()

if mode == "late_arm":
    # Exit while the child is still starting up, before it arms the signal.
    raise SystemExit(0)
time.sleep(600)
"""


def _wait_gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return True
        with contextlib.suppress(psutil.Error):
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        time.sleep(0.05)
    return False


@contextlib.contextmanager
def _parent_death_child(mode: str, tmp_path: Path):
    """Run the launcher in `mode` and yield (launcher, child pid)."""
    pid_file = tmp_path / "child.pid"
    child_pid = None
    launcher = subprocess.Popen(
        [sys.executable, "-c", _PARENT_DEATH_LAUNCHER, mode, str(pid_file)]
    )
    try:
        deadline = time.monotonic() + 120
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists(), "child never reported its pid"
        child_pid = int(pid_file.read_text())
        yield launcher, child_pid
    finally:
        for pid in (launcher.pid, child_pid):
            if pid is not None:
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)
        launcher.wait(timeout=10)


@pytest.mark.skipif(sys.platform != "linux", reason="prctl is Linux only")
def test_set_parent_death_signal_terminates_orphan(tmp_path):
    """SIGKILLing a process takes down a child that armed the signal."""
    with _parent_death_child("direct", tmp_path) as (launcher, child_pid):
        assert psutil.pid_exists(child_pid)
        launcher.kill()
        assert _wait_gone(child_pid, timeout=30), (
            "child survived the death of its parent"
        )


@pytest.mark.skipif(sys.platform != "linux", reason="prctl is Linux only")
def test_set_parent_death_signal_handles_parent_that_already_died(tmp_path):
    """The parent dying *while* the child arms the signal is still caught."""
    with _parent_death_child("late_arm", tmp_path) as (launcher, child_pid):
        # The launcher exits on its own here, orphaning the child before it
        # gets to call prctl().
        launcher.wait(timeout=30)
        assert _wait_gone(child_pid, timeout=30), (
            "child survived being orphaned during start-up"
        )


@pytest.mark.skipif(sys.platform != "linux", reason="prctl is Linux only")
def test_set_parent_death_signal_survives_spawning_thread_exit(tmp_path):
    """PR_SET_PDEATHSIG tracks the spawning thread, which may exit long before
    the parent process does; that must not take the child down."""
    with _parent_death_child("thread", tmp_path) as (launcher, child_pid):
        assert not _wait_gone(child_pid, timeout=5), (
            "child was killed when the thread that spawned it exited"
        )
        assert launcher.poll() is None
