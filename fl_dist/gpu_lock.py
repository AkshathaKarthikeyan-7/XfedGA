"""
Cross-process GPU lock for fl_dist (18 Oct 2026). With 5 client processes + the server on one 6 GB GPU, concurrent
local training runs out of memory, so each client holds this lock while it uses the GPU (load the global model, local
training, q_i) and frees its cached GPU memory before releasing it. An OS file lock (msvcrt on Windows, fcntl
elsewhere) is released automatically if the holding process dies. Each client's computation is independent and
seeded, so serialising them changes only the timing, never the results. path=None disables the lock.
"""
import contextlib
import os
import time


@contextlib.contextmanager
def gpu_lock(path, poll_s=0.2):
    """Context manager; yields the seconds spent waiting for the lock."""
    if not path:
        yield 0.0
        return
    f = open(path, "a+b")
    t0 = time.time()
    try:
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(poll_s)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield time.time() - t0
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        f.close()


def release_cached_gpu_memory():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
