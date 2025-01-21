import torch

# Function to calculate GFLOPS


def calculate_gflops(n, elapsed_time):
    # GFLOPS = (2 * n^3) / time in seconds
    gflops = (2.0 * n**3) / (elapsed_time * 1e9)
    return gflops


# List of matrix sizes
sizes = [256, 512, 1024, 2048, 4096]

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for n in sizes:
    # Generate random matrices A and B
    A = torch.rand(n, n, dtype=torch.float32, device=device)
    B = torch.rand(n, n, dtype=torch.float32, device=device)

    # Record start time using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    # Perform matrix multiplication on the GPU
    C = torch.mm(A, B)

    # Record end time using CUDA events
    end_event.record()

    # Wait for the events to finish
    torch.cuda.synchronize()

    # Calculate elapsed time
    elapsed_time = start_event.elapsed_time(
        end_event) / 1000.0  # Convert to seconds

    # Calculate GFLOPS
    gflops = calculate_gflops(n, elapsed_time)

    # Print the results
    print(
        f"Matrix size: {n}x{n} | Time: {elapsed_time:.4f} seconds | GFLOPS: {gflops:.4f}")

