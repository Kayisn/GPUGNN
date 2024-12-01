from dataclasses import dataclass

import pycuda.driver as cuda


@dataclass
class OccupancyStats:
    active_blocks_per_sm: int
    max_blocks_per_sm: int
    active_warps_per_sm: int
    max_warps_per_sm: int
    occupancy_percentage: float
    theoretical_occupancy: float


class OccupancyTracker:
    def __init__(self):
        self.device = cuda.Device(0)
        self.attributes = {
            "max_threads_per_block": self.device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK),
            "max_block_dims": (
                self.device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_X),
                self.device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_Y),
                self.device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_Z),
            ),
            "max_grid_dims": (
                self.device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_X),
                self.device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Y),
                self.device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Z),
            ),
            "warp_size": self.device.get_attribute(cuda.device_attribute.WARP_SIZE),
            "max_shared_memory_per_block": self.device.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK),
            "multiprocessor_count": self.device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT),
        }

    def calculate_occupancy(self, kernel_function, block_size):
        """
        Calculate occupancy statistics for a given kernel and block size.

        Args:
            kernel_function: The PyCUDA kernel function
            block_size: Tuple of block dimensions (x, y, z)

        Returns:
            OccupancyStats object containing occupancy information
        """
        threads_per_block = block_size[0] * block_size[1] * block_size[2]

        # Calculate theoretical max active blocks per SM
        max_blocks_per_sm = self.device.get_attribute(cuda.device_attribute.MAX_BLOCKS_PER_MULTIPROCESSOR)

        # Get actual active blocks per SM using CUDA occupancy calculator
        active_blocks_per_sm = (
            cuda.func_get_attribute(cuda.function_attribute.MAX_THREADS_PER_BLOCK, kernel_function) // threads_per_block
        )

        # Ensure we don't exceed device limits
        active_blocks_per_sm = min(active_blocks_per_sm, max_blocks_per_sm)

        # Calculate warp-related statistics
        warps_per_block = (threads_per_block + self.attributes["warp_size"] - 1) // self.attributes["warp_size"]
        max_warps_per_sm = self.device.get_attribute(cuda.device_attribute.MAX_WARPS_PER_MULTIPROCESSOR)
        active_warps_per_sm = warps_per_block * active_blocks_per_sm

        # Calculate occupancy percentages
        theoretical_occupancy = min(1.0, active_warps_per_sm / max_warps_per_sm)

        # Account for shared memory and register usage
        shared_mem_per_block = kernel_function.shared_size_bytes
        regs_per_thread = kernel_function.num_regs

        # Adjust active blocks based on shared memory constraints
        if shared_mem_per_block > 0:
            blocks_by_shared_mem = self.attributes["max_shared_memory_per_block"] // shared_mem_per_block
            active_blocks_per_sm = min(active_blocks_per_sm, blocks_by_shared_mem)

        # Calculate actual occupancy percentage
        occupancy_percentage = (active_blocks_per_sm * warps_per_block) / max_warps_per_sm

        return OccupancyStats(
            active_blocks_per_sm=active_blocks_per_sm,
            max_blocks_per_sm=max_blocks_per_sm,
            active_warps_per_sm=active_warps_per_sm,
            max_warps_per_sm=max_warps_per_sm,
            occupancy_percentage=occupancy_percentage * 100,
            theoretical_occupancy=theoretical_occupancy * 100,
        )

    def suggest_block_size(self, kernel_function, min_blocks=1):
        """
        Suggest an optimal block size for the given kernel.

        Args:
            kernel_function: The PyCUDA kernel function
            min_blocks: Minimum number of blocks required

        Returns:
            tuple: Suggested block dimensions (x, y, z)
        """
        max_threads = min(self.attributes["max_threads_per_block"], kernel_function.max_threads_per_block)

        best_occupancy = 0
        best_block_size = (16, 16, 1)  # Default

        # Test different block sizes that are multiples of warp size
        for block_x in range(self.attributes["warp_size"], max_threads + 1, self.attributes["warp_size"]):
            for block_y in [1, 2, 4, 8, 16, 32]:
                if block_x * block_y > max_threads:
                    continue

                block_size = (block_x, block_y, 1)
                stats = self.calculate_occupancy(kernel_function, block_size)

                if stats.occupancy_percentage > best_occupancy and stats.active_blocks_per_sm >= min_blocks:
                    best_occupancy = stats.occupancy_percentage
                    best_block_size = block_size

        return best_block_size

    def log_statistics(self, kernel_function, block_size, grid_size=None):
        """
        Log detailed occupancy and execution statistics.

        Args:
            kernel_function: The PyCUDA kernel function
            block_size: Tuple of block dimensions
            grid_size: Optional tuple of grid dimensions
        """
        stats = self.calculate_occupancy(kernel_function, block_size)

        print("\nKernel Occupancy Statistics:")
        print(f"Block Size: {block_size}")
        if grid_size:
            print(f"Grid Size: {grid_size}")
        print(f"Active Blocks per SM: {stats.active_blocks_per_sm}/{stats.max_blocks_per_sm}")
        print(f"Active Warps per SM: {stats.active_warps_per_sm}/{stats.max_warps_per_sm}")
        print(f"Occupancy: {stats.occupancy_percentage:.2f}%")
        print(f"Theoretical Max Occupancy: {stats.theoretical_occupancy:.2f}%")
        print(f"Shared Memory per Block: {kernel_function.shared_size_bytes/1024:.2f} KB")
        print(f"Registers per Thread: {kernel_function.num_regs}")
