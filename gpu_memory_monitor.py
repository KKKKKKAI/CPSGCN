import pynvml
import time

def get_gpu_memory_info(gpu_index=0):
    # Initialize NVML
    pynvml.nvmlInit()

    # Get the handle for the specified GPU
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    # Get memory information
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    return mem_info

def print_memory_info(mem_info):
    print(f"Used memory: {mem_info.used / 1024 ** 2:.2f} MB")

if __name__ == "__main__":
    gpu_index = 0  # Change this if you have multiple GPUs and want to monitor a different one

    try:
        while True:
            mem_info = get_gpu_memory_info(gpu_index)
            print_memory_info(mem_info)
            time.sleep(1)  # Update every second
    except KeyboardInterrupt:
        print("Exiting monitoring...")
    finally:
        # Shutdown NVML
        pynvml.nvmlShutdown()
