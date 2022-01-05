from kernel_matrix_benchmarks.main import main
from multiprocessing import freeze_support

if __name__ == "__main__":
    # Freeze_support is a Windows-only function that ensures compatibility
    # with a ".py -> .exe" packaging method. On Linux, this has zero consequences.
    freeze_support()

    # The actual code:
    main()
