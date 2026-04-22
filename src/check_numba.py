try:
    import numba
    print(f"Numba version: {numba.__version__}")
    from numba_kernels import step_pde_numba
    print("Import successful.")
except Exception as e:
    print(f"Import failed: {e}")
