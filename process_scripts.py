import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Check if running as administrator
if (os.name == "nt" and not ctypes.windll.shell32.IsUserAnAdmin()) or (os.name == "posix" and os.geteuid() != 0):
    print("Please run as root/administrator.")
    exit()

# Need to add the Nsight Compute Python package to the path before importing ncu_report
ncu_path = subprocess.run(["where", "ncu"], stdout=subprocess.PIPE)
sys.path.append(str(Path(ncu_path.stdout.decode().strip()).parent / "extras" / "python"))
import ncu_report

# Add command line argument parsing
parser = argparse.ArgumentParser(description="Profile GNN experiments with different implementations")
parser.add_argument("--methods", type=str, default="all", help="List of methods to run")
parser.add_argument("--profile", default=False, action="store_true", help="Enable profiling")
parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")
parser.add_argument("--test-runs", type=int, default=1, help="Number of test runs for timing")
parser.add_argument("--nvtx", type=str, default="main", help="Which NVTX range to profile")
parser.add_argument("--graphs", type=str, default=None, help="Index pattern of graphs to process")
args = parser.parse_args()

# List of methods to run
methods_cupy = ["chatgpt_cupy_sparse"]
methods_pytorch = ["chatgpt_pytorch_dense", "chatgpt_pytorch_sparse"]
methods_pycuda = [
    "chatgpt_pycuda_decomposed",
    "chatgpt_pycuda_dense",
    "chatgpt_pycuda_sparse",
    "claude_pycuda_sparse_csr_csc",
    "claude_pycuda_sparse_instrumented",
    # "claude_pycuda_sparse_tiled_coalesced",
    # "claude_pycuda_sparse_tiled",
    "claude_pycuda_sparse",
]
methods = []
if args.methods == "all":
    methods = methods_cupy + methods_pytorch + methods_pycuda
elif args.methods == "cupy":
    methods = methods_cupy
elif args.methods == "pytorch":
    methods = methods_pytorch
elif args.methods == "pycuda":
    methods = methods_pycuda
else:
    methods = args.methods.replace(" ", "").split(",")


# Run each script sequentially
report_dir = Path("reports")
report_dir.mkdir(exist_ok=True)
for method in methods:
    try:
        if args.profile:
            # NVTX filtering is not possible using the `subprocess` module, for reasons that are yet unknown to me
            print(f"Profiling {method}...")
            nvtx_pattern = f'"regex:{method}@{args.nvtx}*/"'
            exit_code = os.system(
                f"ncu --nvtx --nvtx-include {nvtx_pattern} -f -o {str(report_dir / f'report_{method}')} python executer.py --method {method} --warmup 1 --test-runs 1 {'--graphs ' +  args.graphs if args.graphs else ''}"
            )
            print(f"Completed {method} with exit code {exit_code}")
        else:
            print(f"Running {method}...")
            result = subprocess.run(
                [
                    "python",
                    "executer.py",
                    "--method",
                    method,
                    "--warmup",
                    str(args.warmup),
                    "--test-runs",
                    str(args.test_runs),
                ]
                + (["--graphs", args.graphs] if args.graphs else []),
                check=True,
            )
            print(f"Completed {method} with return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {method}: {e}")

    if args.profile:
        # Generate a report
        metrics2names = {
            "device__attribute_display_name": "device_name",  # the name of the device
            "gpu__time_duration.avg": "time_duration",  # the duration of the kernel
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "compute_memory_throughput",  # compute memory pipeline throughput
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput",  # GPU DRAM throughput
            "dram__cycles_active.avg": "dram_cycles_active",  # of cycles where DRAM was active
            "dram__cycles_elapsed.avg.per_second": "dram_cycles_elapsed",  # of elapsed DRAM clock cycles per second
            "fbpa__dram_sectors.avg.pct_of_peak_sustained_elapsed": "dram_sectors",  # of of DRAM sectors accessed
            "lts__cycles_active.avg": "lts_cycles_active",  # of cycles where LTS was active
            "lts__d_sectors.avg.pct_of_peak_sustained_elapsed": "lts_sectors_accessed",  # of LTS sectors accessed
            "lts__d_sectors_fill_device.avg.pct_of_peak_sustained_elapsed": "lts_sectors_filled",  # of LTS sectors filled by device
            "gpc__cycles_elapsed.avg.per_second": "gpc_cycles_elapsed",  # of elapsed GPC clock cycles per second
            "idc__request_cycles_active.avg.pct_of_peak_sustained_elapsed": "idc_request_cycles_active",  # of cycles where IDC processed requests from SM
            "l1tex__cycles_active.avg": "l1tex_cycles_active",  # of cycles where L1TEX was active
            "l1tex__data_bank_reads.avg.pct_of_peak_sustained_elapsed": "l1tex_data_bank_reads",  # the number of data bank reads
            "l1tex__data_bank_writes.avg.pct_of_peak_sustained_elapsed": "l1tex_data_bank_writes",  # the number of data bank writes
            "sm__cycles_active.avg": "sm_cycles_active",  # of cycles with at least one warp in flight
            "sm__inst_executed.avg.pct_of_peak_sustained_elapsed": "inst_count",  # of instructions executed
            "sm__maximum_warps_avg_per_active_cycle": "max_warps",  # the maximum number of warps per active cycle=
            "sm__memory_throughput_internal_activity.avg.pct_of_peak_sustained_elapsed": "sm_memory_throughput",  # SM memory throughput
            "sm__warps_active.avg.per_cycle_active": "warps_active",  # the average number of active warps per active cycle
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput",  # the overall throughput of the SMs
            "launch__sm_count": "sm_count",  # of SMs launched
            "launch__thread_count": "thread_count",  # of threads launched
            "launch__occupancy_limit_blocks": "occupancy_limit_blocks",  # the occupancy limit in blocks
            "launch__occupancy_limit_registers": "occupancy_limit_registers",  # the occupancy limit in registers
            "launch__occupancy_limit_shared_memory": "occupancy_limit_shared_memory",  # the occupancy limit in shared memory
            "launch__occupancy_limit_warps": "occupancy_limit_warps",  # the occupancy limit in warps
            "launch__occupancy_per_block_size": "occupancy_per_block_size",  # the occupancy per block size
            "launch__occupancy_per_register_count": "occupancy_per_register_count",  # the occupancy per register count
            "launch__occupancy_per_shared_mem_size": "occupancy_per_shared_mem_size",  # the occupancy per shared memory size
            "launch__registers_per_thread": "registers_per_thread",  # the number of registers per thread
            "launch__registers_per_thread_allocated": "registers_per_thread_allocated",  # the number of registers per thread allocated
            "launch__shared_mem_per_block": "shared_mem_per_block",  # the amount of shared memory per block
            "launch__shared_mem_per_block_allocated": "shared_mem_per_block_allocated",  # the amount of shared memory per block allocated
            "smsp__cycles_active.avg": "smsp_cycles_active",  # of cycles with at least one warp in flight on SMSP
            "smsp__maximum_warps_avg_per_active_cycle": "smsp_max_warps",  # the maximum number of warps per active cycle on SMSP
        }

        with open("gnn_results.json", "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                raise ("Invalid result file generated.")

        avg_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        report = ncu_report.load_report(report_dir / f"report_{method}.ncu-rep")
        for rng_id in range(report.num_ranges()):
            rng = report.range_by_idx(rng_id)
            for action_id in range(rng.num_actions()):
                action = rng.action_by_idx(action_id)
                nvtx_state = action.nvtx_state()
                for domain_id in nvtx_state.domains():
                    domain = nvtx_state.domain_by_id(domain_id)
                    if domain.name() in methods:
                        method = domain.name()
                        graph_index = re.search(r"\d+", domain.push_pop_ranges()[0]).group()
                        for metric in metrics2names.keys():
                            if metric_data := action.metric_by_name(metric):
                                if metric_data.value() is not None:
                                    avg_metrics[method][graph_index][metric].setdefault("values", []).append(
                                        metric_data.value()
                                    )
                                    avg_metrics[method][graph_index][metric]["unit"] = metric_data.unit()

        for method, graphs in avg_metrics.items():
            for graph_index, metrics in graphs.items():
                results[method][graph_index]["metrics"] = {}
                for metric, data in metrics.items():
                    results[method][graph_index]["metrics"][metrics2names[metric]] = {
                        "value": (
                            sum(data["values"]) / len(data["values"])
                            if (type(data["values"][0]) == float or type(data["values"][0]) == int)
                            else data["values"][0]
                        ),
                        "unit": data["unit"],
                    }

        with open("gnn_results.json", "w") as f:
            json.dump(results, f, indent=4)

print("All scripts have been executed.")
