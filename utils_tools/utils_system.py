def limit_cpu_cores(n_cores: int = 4, *, pin_process: bool = True, verbose: bool = True):
    """
    Restrict this Python process (and most math libraries inside it) to
    `n_cores` logical CPU threads.

    Works for:
      • BLAS back-ends (OpenBLAS, MKL, Accelerate)
      • Numexpr, SciPy, scikit-learn, pandas, etc.
      • PyTorch  (if already imported)
      • TensorFlow 2.x  (if already imported)
      • Optional: OS-level CPU affinity on Linux/WSL

    Parameters
    ----------
    n_cores : int
        Target number of logical cores/threads.
    pin_process : bool, default True
        On Linux/WSL, pin the entire Python process to cores 0…n_cores-1
        using `psutil`. Ignored on macOS/Windows.
    verbose : bool, default True
        Print a confirmation banner.
    """
    import os, platform, sys

    # ─── 1) Low-level environment switches (must be set before imports) ──────
    for var in (
        "OMP_NUM_THREADS",        # OpenMP kernels
        "MKL_NUM_THREADS",        # Intel MKL (NumPy/SciPy on many installs)
        "OPENBLAS_NUM_THREADS",   # OpenBLAS (NumPy/SciPy on others)
        "NUMEXPR_NUM_THREADS",    # numexpr, pandas eval, …
        "VECLIB_MAXIMUM_THREADS"  # macOS Accelerate/vecLib
    ):
        os.environ[var] = str(n_cores)

    # ─── 2) Library-specific knobs (if lib is already in memory) ─────────────
    if "torch" in sys.modules:
        import torch
        torch.set_num_threads(n_cores)
        torch.set_num_interop_threads(max(1, n_cores // 2))

    if "tensorflow" in sys.modules:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(n_cores)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, n_cores // 2))

    # ─── 3) Pin the *whole process* (Linux/WSL only) ─────────────────────────
    if pin_process and platform.system() == "Linux":
        try:
            import psutil
            psutil.Process().cpu_affinity(list(range(n_cores)))
        except (ImportError, AttributeError):
            pass  # psutil not installed or not supported

    if verbose:
        print(f"🔧  Limited Python process to {n_cores} logical CPU cores.")

# ─── Example usage ───────────────────────────────────────────────────────────
# limit_cpu_cores(4)