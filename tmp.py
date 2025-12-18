import torch
import torch_geometric

def check_env():
    print(f"--- Environment Report ---")
    print(f"PyTorch Version: {torch.__version__}")
    
    # 1. Check GPU availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_available}")
    
    if cuda_available:
        print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version (Torch): {torch.version.cuda}")
        # Check the compiled version of PyG dependencies
        try:
            print(f"PyG Version:     {torch_geometric.__version__}")
            print(f"Status:          All systems go! 🚀")
        except ImportError:
            print(f"Status:          ⚠️ PyTorch is fine, but PyG dependencies (scatter/sparse) are missing!")
    else:
        print(f"Status:          🚨 GPU NOT DETECTED. Check your NVIDIA drivers/Docker flags.")

if __name__ == "__main__":
    check_env()