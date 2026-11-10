# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import ctypes
import logging
import multiprocessing
import os
import signal
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import psutil

import vllm.envs as envs
from vllm.logger import init_logger, set_vllm_process_name
from vllm.platforms import current_platform
from vllm.platforms.interface import in_wsl
from vllm.ray.lazy_utils import is_in_ray_actor

from .platform_utils import cuda_is_initialized, xpu_is_initialized

logger = init_logger(__name__)

# Environment variable utilities


def update_environment_variables(envs_dict: dict[str, str]):
    """Update multiple environment variables with logging."""
    for k, v in envs_dict.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s from '%s' to '%s'",
                k,
                os.environ[k],
                v,
            )
        os.environ[k] = v


@contextlib.contextmanager
def set_env_var(key: str, value: str) -> Iterator[None]:
    """Temporarily set an environment variable."""
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


@contextlib.contextmanager
def suppress_stdout():
    """Suppress stdout from C libraries at the file descriptor level.

    Only suppresses stdout, not stderr, to preserve error messages.
    Suppression is disabled when vLLM debug logging is enabled.

    Example:
        with suppress_stdout():
            # C library calls that would normally print to stdout
            torch.distributed.new_group(ranks, backend="gloo")

    """
    # Don't suppress if logging level is DEBUG
    if logger.isEnabledFor(logging.DEBUG):
        yield
        return

    # C libraries write to fd 1, whatever sys.stdout is bound to.
    stdout_fd = 1
    stdout_dup = os.dup(stdout_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        sys.stdout.flush()
        os.dup2(devnull_fd, stdout_fd)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(stdout_dup, stdout_fd)
        os.close(stdout_dup)
        os.close(devnull_fd)


# File path utilities


def unique_filepath(fn: Callable[[int], Path]) -> Path:
    """Generate a unique file path by trying incrementing integers.

    Note: This function has a TOCTOU race condition.
    Caller should use atomic operations (e.g., open with 'x' mode)
    when creating the file to ensure thread safety.
    """
    i = 0
    while True:
        p = fn(i)
        if not p.exists():
            return p
        i += 1


# Process management utilities


def _sync_visible_devices_env_vars():
    """Sync HIP/CUDA visibility env vars before spawning (ROCm only)."""
    if not current_platform.is_rocm():
        return

    from vllm.platforms.rocm import _sync_hip_cuda_env_vars

    _sync_hip_cuda_env_vars()


def _maybe_force_spawn():
    """Check if we need to force the use of the `spawn` multiprocessing start
    method.
    """
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn":
        return

    reasons = []
    if is_in_ray_actor():
        # even if we choose to spawn, we need to pass the ray address
        # to the subprocess so that it knows how to connect to the ray cluster.
        # env vars are inherited by subprocesses, even if we use spawn.
        import ray

        os.environ["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
        reasons.append("In a Ray actor and can only be spawned")

    # NUMA binding uses executable hijacking, which requires spawn.
    if any(
        arg.split("=", 1)[0].replace("_", "-")
        in ("--numa-bind", "--numa-bind-nodes", "--numa-bind-cpus")
        for arg in sys.argv
    ):
        reasons.append("NUMA binding requires spawn method")

    if cuda_is_initialized():
        reasons.append("CUDA is initialized")
    elif xpu_is_initialized():
        reasons.append("XPU is initialized")

    if in_wsl():
        reasons.append("WSL is detected and NVML is not compatible with fork")

    if reasons:
        logger.warning(
            "We must use the `spawn` multiprocessing start method. "
            "Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. "
            "See https://docs.vllm.ai/en/latest/usage/"
            "troubleshooting.html#python-multiprocessing "
            "for more information. Reasons: %s",
            "; ".join(reasons),
        )
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


_OMP_PAUSE_HARD = 2  # omp_pause_resource_t, OpenMP 5.0


def _release_libgomp_threads() -> None:
    """Join libgomp's idle worker threads so a forked child can use OpenMP.

    libgomp has no fork handler: once this process has run a parallel region
    (e.g. any multi-threaded torch CPU op), a forked child deadlocks in its
    own first parallel region. Pausing the runtime (OpenMP 5.0) joins the
    workers; this process recreates them on its next parallel region.
    """
    try:
        paths = find_loaded_libraries("libgomp")
    except OSError:
        return
    for path in paths:
        with contextlib.suppress(OSError, AttributeError):
            ctypes.CDLL(path).omp_pause_resource_all(_OMP_PAUSE_HARD)


def get_mp_context():
    """Get a multiprocessing context with a particular method (spawn or fork).
    By default we follow the value of the VLLM_WORKER_MULTIPROC_METHOD to
    determine the multiprocessing method (default is fork). However, under
    certain conditions, we may enforce spawn and override the value of
    VLLM_WORKER_MULTIPROC_METHOD.
    """
    _maybe_force_spawn()
    # (ROCm): Sync GPU visibility env vars so spawned children inherit
    # consistent values. Must run after _maybe_force_spawn and regardless
    # of whether spawn was already set.
    _sync_visible_devices_env_vars()
    mp_method = envs.VLLM_WORKER_MULTIPROC_METHOD
    if mp_method == "fork":
        _release_libgomp_threads()
    return multiprocessing.get_context(mp_method)


def set_process_title(
    name: str,
    suffix: str = "",
    prefix: str = envs.VLLM_PROCESS_NAME_PREFIX,
) -> None:
    """Set the current process title with optional suffix."""
    try:
        import setproctitle
    except ImportError:
        return

    if suffix:
        name = f"{name}_{suffix}"

    setproctitle.setproctitle(f"{prefix}::{name}")


def decorate_logs(
    process_name: str | None = None, *, skip_if_decorated: bool = False
) -> None:
    """Add vLLM process metadata to subsequent log records."""
    if process_name is None:
        process_name = multiprocessing.current_process().name

    set_vllm_process_name(process_name, skip_if_set=skip_if_decorated)


def kill_process_tree(pid: int):
    """Kills all descendant processes of the given pid by sending SIGKILL.

    Args:
        pid (int): Process ID of the parent process

    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Get all children recursively
    children = parent.children(recursive=True)

    # Send SIGKILL to all children first
    for child in children:
        with contextlib.suppress(ProcessLookupError, psutil.NoSuchProcess):
            logger.warning(
                "[shutdown] force killing sub process %s pid %d",
                child.name(),
                child.pid,
            )
            os.kill(child.pid, signal.SIGKILL)

    # Finally kill the parent
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


# PR_SET_PDEATHSIG, from <linux/prctl.h>. See prctl(2).
_PR_SET_PDEATHSIG = 1

# Signal the kernel uses to tell a child that its parent died. A real-time
# signal keeps this notification distinguishable from a shutdown request sent
# by a user or by vLLM itself; glibc reserves the two signals below SIGRTMIN
# for its threading implementation, so SIGRTMIN is the first one free for
# applications.
_PARENT_DEATH_NOTIFY_SIG = getattr(signal, "SIGRTMIN", None)


def _parent_is_gone(original_ppid: int) -> bool:
    """Whether we have been reparented away from `original_ppid`."""
    # A parent pid of 1 is not evidence of anything: inside a container the
    # real parent may well be the init process.
    return original_ppid != 1 and os.getppid() != original_ppid


def set_parent_death_signal(sig: int = signal.SIGTERM) -> None:
    """Send `sig` to this process when its parent process dies.

    `prctl(PR_SET_PDEATHSIG)` is the only parent-death notification that still
    works when the parent is killed with SIGKILL - by the OOM killer, by a
    `kill -9`, or by a job scheduler tearing down a run - which is precisely
    the case in which the parent cannot clean up after itself. Without it the
    child is reparented to init and keeps running, holding its GPU memory, its
    device handles and its shared-memory segments until someone notices and
    kills it by hand.

    Call this from the child process once it is running, on its main thread:
    the setting is cleared by fork(), and `signal.signal` is main-thread only.

    The kernel notification is delivered to a dedicated signal rather than to
    `sig` directly, because `PR_SET_PDEATHSIG` tracks the *thread* that created
    this process, not the parent process: it also fires when that thread exits
    while the parent lives on, which happens if an embedding application builds
    its engine from a short-lived thread. The handler therefore checks whether
    we really were reparented and ignores the notification if we were not. The
    kernel only notifies once, so after such a spurious wake-up this process is
    no longer protected - no worse than not arming the signal at all.

    No-op on platforms other than Linux.
    """
    if sys.platform != "linux" or _PARENT_DEATH_NOTIFY_SIG is None:
        return

    original_ppid = os.getppid()

    def handle_parent_death(signum, frame):
        if not _parent_is_gone(original_ppid):
            logger.debug(
                "Ignoring parent-death notification: the thread that spawned "
                "this process exited but parent %d is still alive.",
                original_ppid,
            )
            return
        logger.warning(
            "Parent process %d exited; terminating self (pid %d).",
            original_ppid,
            os.getpid(),
        )
        os.kill(os.getpid(), sig)

    try:
        signal.signal(_PARENT_DEATH_NOTIFY_SIG, handle_parent_death)
    except ValueError:
        # Not the main thread; leaving the signal unarmed is the safe choice.
        logger.warning(
            "set_parent_death_signal() must be called from the main thread; "
            "this process may outlive its parent."
        )
        return

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, _PARENT_DEATH_NOTIFY_SIG, 0, 0, 0) != 0:
        logger.warning(
            "prctl(PR_SET_PDEATHSIG) failed: %s. This process may outlive its "
            "parent if the parent dies abnormally.",
            os.strerror(ctypes.get_errno()),
        )
        signal.signal(_PARENT_DEATH_NOTIFY_SIG, signal.SIG_DFL)
        return

    # prctl() only covers parent deaths that happen after the call above, so
    # re-check: a parent that died while we were starting up is missed.
    if _parent_is_gone(original_ppid):
        logger.warning(
            "Parent process %d exited while starting up; terminating self "
            "(pid %d) rather than leaking an orphan holding device memory.",
            original_ppid,
            os.getpid(),
        )
        os.kill(os.getpid(), sig)


# Resource utilities


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L630
def set_ulimit(target_soft_limit: int = 65535):
    if sys.platform.startswith("win"):
        logger.debug("Windows detected, skipping ulimit adjustment.")
        return

    import resource

    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase "
                "with error %s. This can cause fd limit errors like "
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n",
                current_soft,
                e,
            )


def find_loaded_libraries(lib_name: str) -> list[str]:
    """
    According to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    loaded library.
    """  # noqa
    # Match the mapped file's name, not the whole line: an unrelated library
    # whose name merely contains lib_name (e.g. TileLang's libcudart_stub.so
    # when looking for libcudart) or a directory component containing it must
    # not win. Legitimate filenames are {lib_name}.so[.*] or a name-mangled
    # {lib_name}-<hash>.so[.*], and /proc/self/maps is ordered by mapping
    # address rather than load order, so a substring hit is a
    # nondeterministic hijack.
    paths: dict[str, None] = {}
    with open("/proc/self/maps") as f:
        for line in f:
            start = line.find("/")
            if start == -1:
                continue
            path = line[start:].strip()
            filename = path.rsplit("/", maxsplit=1)[-1]
            if filename.startswith((f"{lib_name}.", f"{lib_name}-")):
                paths[path] = None
    return list(paths)


def find_loaded_library(lib_name: str) -> str | None:
    paths = find_loaded_libraries(lib_name)
    return paths[0] if paths else None
