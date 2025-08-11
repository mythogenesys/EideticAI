import torch
import mlx.core as mx
import platform

def verify_environment():
    """
    Checks and prints the status of critical hardware acceleration backends.
    """
    print("--- Eidetic AI Environment Verification ---")
    print(f"Python Version: {platform.python_version()}")
    print("-" * 35)

    try:
        if torch.backends.mps.is_available():
            print("PyTorch Backend: \033[92mMPS (Metal) is available.\033[0m")
            mps_device = torch.device("mps")
            x = torch.ones(1, device=mps_device)
            print("MPS Test Tensor: \033[92mSuccess.\033[0m")
        elif torch.cuda.is_available():
            print("PyTorch Backend: \033[96mCUDA is available.\033[0m")
        else:
            print("PyTorch Backend: \033[93mUsing CPU. MPS/CUDA not available.\033[0m")
    except Exception as e:
        print(f"\033[91mAn error occurred during PyTorch verification: {e}\033[0m")

    print("-" * 35)

    try:
        mx.set_default_device(mx.gpu)
        a = mx.array([1, 2, 3])
        b = a * 2
        mx.eval(b)
        print("Apple MLX Backend: \033[92mGPU is available and working.\033[0m")
        print(f"Default MLX Device: {mx.default_device()}")
    except Exception as e:
        print(f"\033[91mAn error occurred during MLX verification: {e}\033[0m")
        print("\033[93mMLX may be falling back to CPU.\033[0m")

    print("-" * 35)
    print("Environment setup verification complete.")


if __name__ == "__main__":
    verify_environment()