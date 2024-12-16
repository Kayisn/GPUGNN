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
sys.path.append(str(Path(subprocess.check_output(["where", "ncu"], text=True).strip()).parent / "extras" / "python"))
import ncu_report

# Add command line argument parsing
parser = argparse.ArgumentParser(description="Profile GNN experiments with different implementations")
parser.add_argument("--methods", "-m", type=str, default="all", help="List of methods to run")
parser.add_argument("--list-methods", default=False, action="store_true", help="List available methods")
parser.add_argument("--warmup", "-w", type=int, default=1, help="Number of warmup runs")
parser.add_argument("--test-runs", "-r", type=int, default=1, help="Number of test runs for timing")
parser.add_argument("--verify", default=False, action="store_true", help="Verify the result")
parser.add_argument("--profile", default=False, action="store_true", help="Enable profiling")
parser.add_argument("--nvtx", "-n", type=str, default="main", help="Which NVTX range to profile")
parser.add_argument("--graphs", "-g", type=str, default=None, help="Index pattern of graphs to process")
args = parser.parse_args()

# List of methods to run
methods = [path.stem for path in Path("scripts").glob("*.py") if path.stem != "__init__"]

if args.methods == "cupy":
    methods = [method for method in methods if "cupy" in method]
elif args.methods == "pytorch":
    methods = [method for method in methods if "pytorch" in method]
elif args.methods == "pycuda":
    methods = [method for method in methods if "pycuda" in method]
elif args.methods != "all":
    methods = [method for method in methods if method in args.methods.split(",")]

if args.list_methods:
    print("Available methods:")
    print("\n".join(methods))
    exit()


report_dir = Path("reports")
if args.profile:
    report_dir.mkdir(exist_ok=True)

# Run each script sequentially
for method in methods:
    cmd = f"python executer.py --method {method} --warmup {args.warmup} --test-runs {args.test_runs} {'--graphs ' + args.graphs if args.graphs else ''} {'--verify' if args.verify else ''}"

    try:
        if args.profile:
            print(f"Profiling {method}...")
            nvtx_pattern = f'"regex:{method}@{args.nvtx}*/"'
            metrics = [
                "gpu__time_duration_measured_wallclock",  # The wall-clock time duration.
                "gpu__cycles_active",  # Number of cycles where the GPU is actively processing.
                "gpu__cycles_elapsed",  # Total elapsed cycles, including idle periods. Helps in comparing against active cycles to detect inefficiencies.
                "dram__bytes",  # Total bytes accessed in DRAM (global memory). Indicates how much memory bandwidth the program consumes.
                "dram__bytes_read",  # Total bytes read from DRAM. Useful to identify memory access patterns.
                "dram__bytes_write",  # Total bytes written to DRAM. Useful to identify memory access patterns.
                "lts__t_bytes",  # Total bytes requested from L2 cache. Indicates data dependency and cache usage efficiency.
                "lts__t_request_hit_rate",  # Hit rate for L2 cache requests. A low hit rate suggests suboptimal memory access patterns and potential need for memory locality optimization.
                "l1tex__t_bytes",  # Total bytes requested from L1 texture cache. Indicates data dependency and cache usage efficiency.
                "l1tex__t_bytes_lookup_hit",  # Tracks bytes that hit in L1 texture cache. A low hit rate signals inefficient data locality.
                "l1tex__t_bytes_lookup_miss",  # Tracks bytes that missed in L1 texture cache. A high miss rate signals inefficient data locality.
                "gr__ctas_launched",  # Number of Cooperative Thread Arrays (CTAs) launched. Reflects parallel workload distribution across the GPU.
                "gpc__cycles_active",  # Tracks activity in the Graphics Processing Clusters. Helps analyze how effectively compute units are utilized.
                "idc__request_hit_rate",  # Hit rate for intermediate data cache (IDC). Useful for understanding performance of inter-thread data sharing.
            ]
            subprocess.check_call(
                f"ncu --nvtx --metrics {','.join(metrics)} --nvtx-include {nvtx_pattern} -f -o {str(report_dir / f'report_{method}')} {cmd}",
                stderr=subprocess.STDOUT,
                shell=True,
            )
            print(f"Completed profiling {method}.")
        else:
            print(f"Running {method}...")
            subprocess.check_call(cmd, stderr=subprocess.STDOUT)
            print(f"Completed running {method}.")
    except subprocess.CalledProcessError as e:
        print(f"Error running {method}. Exit code: {e.returncode}")
        exit()

    if args.profile:
        """
        What's happening here?

        Nvidia Nsight Compute (ncu) iterates over each kernel funcion 8 times when recording a metric.
        Then, we have the option of choosing how we want to aggregate the data.
        We may choose to sum or take the average/minimum/maximum of the data across all iterations.
        In some cases, we are able to take average per second or put all the data in relation to their
        corresponding peak values (e.g., metric.(sum|min|max|avg).pct_of_peak_sustained_elapsed).

        Most libraries (e.g., PyTorch, Cupy) invoke multiple kernels in a single function call. As a result, after
        deciding what to do with the information gathered from different iterations of the same kernel, we need to
        decide how to collect the data across all the kernels in a single function call. This is where the
        `sum_metrics`, `max_metrics`, and `avg_metrics` dictionaries come into play. For some metrics, for example,
        the time duration, we need to add up the values across all kernels to get the total time spent in that function.
        For others, for example memory throughput, taking the average is more meaningful.

        How we choose to collect the data is merely a matter of preference. This is how I chose to do it:
        """
        # Map metric names to human-readable names
        nan_metrics = {
            "device__attribute_display_name": "device_name",  # the name of the device
        }  # metrics that are non-numeric
        sum_metrics = {
            "gpu__time_duration_measured_wallclock.avg": "time_duration",  # the wall-clock time duration
            "gpu__cycles_active.avg": "cycles_active",  # of cycles where the GPU was active
            "gpu__cycles_elapsed.avg": "cycles_elapsed",  # of elapsed GPU clock cycles
            "dram__bytes.avg": "dram_bytes",  # of bytes accessed in DRAM
            "dram__bytes_read.avg": "dram_bytes_read",  # of bytes read from DRAM
            "dram__bytes_write.avg": "dram_bytes_write",  # of bytes written to DRAM
            "lts__t_bytes.avg": "lts_bytes",  # of bytes requested from L2 cache
            "l1tex__t_bytes.avg": "l1tex_bytes",  # of bytes requested from L1 texture cache
            "l1tex__t_bytes_lookup_hit.avg": "l1tex_bytes_lookup_hit",  # of bytes that hit in L1 texture cache
            "l1tex__t_bytes_lookup_miss.avg": "l1tex_bytes_lookup_miss",  # of bytes that missed in L1 texture cache
            "l1tex__data_bank_reads.avg": "l1tex_data_bank_reads",  # of data bank reads
            "l1tex__data_bank_writes.avg": "l1tex_data_bank_writes",  # of data bank writes
            "gpc__cycles_active.avg": "gpc_cycles_active",  # of cycles where GPC was active
            "idc__request_cycles_active.avg": "idc_request_cycles_active",  # of cycles where IDC processed requests from SM
            "gr__ctas_launched.avg": "ctas_launched",  # of CTAs launched
            # "dram__cycles_active.avg": "dram_cycles_active",  # of cycles where DRAM was active
            # "lts__cycles_active.avg": "lts_cycles_active",  # of cycles where LTS was active
            # "l1tex__cycles_active.avg": "l1tex_cycles_active",  # of cycles where L1TEX was active
            # "sm__cycles_active.avg": "sm_cycles_active",  # of cycles with at least one warp in flight
            # "smsp__cycles_active.avg": "smsp_cycles_active",  # of cycles with at least one warp in flight on SMSP
            # "launch__sm_count": "sm_count",  # of SMs launched
            # "launch__thread_count": "thread_count",  # of threads launched
        }  # metrics we will add up
        max_metrics = {
            "sm__maximum_warps_avg_per_active_cycle": "max_warps",  # the maximum number of warps per active cycle
            # "launch__occupancy_limit_blocks": "occupancy_limit_blocks",  # the occupancy limit in blocks
            # "launch__occupancy_limit_registers": "occupancy_limit_registers",  # the occupancy limit in registers
            # "launch__occupancy_limit_shared_memory": "occupancy_limit_shared_memory",  # the occupancy limit in shared memory
            # "launch__occupancy_limit_warps": "occupancy_limit_warps",  # the occupancy limit in warps
        }  # metrics we will take the maximum of
        avg_metrics = {
            "fbpa__dram_sectors.avg": "dram_sectors",  # of DRAM sectors accessed
            "lts__d_sectors.avg": "lts_sectors_accessed",  # of LTS sectors accessed
            "lts__d_sectors_fill_device.avg": "lts_sectors_filled",  # of LTS sectors filled by device
            "sm__throughput.avg": "sm_throughput",  # the overall throughput of the SMs
            "sm__memory_throughput_internal_activity.avg": "sm_memory_throughput",  # SM memory throughput
            # "sm__inst_executed.avg": "inst_count",  # of instructions executed
            "gpu__dram_throughput.avg": "dram_throughput",  # GPU DRAM throughput
            "gpu__compute_memory_throughput.avg": "compute_memory_throughput",  # compute memory pipeline throughput
            "lts__t_request_hit_rate.avg": "lts_request_hit_rate",  # the hit rate for L2 cache requests
            "gpc__cycles_elapsed.avg.per_second": "gpc_cycles_elapsed",  # of elapsed GPC clock cycles per second
            "idc__request_hit_rate.avg": "idc_request_hit_rate",  # the hit rate for IDC requests
            "sm__warps_active.avg.per_cycle_active": "warps_active",  # of active warps per active cycle
            # "dram__cycles_elapsed.avg.per_second": "dram_cycles_elapsed",  # of elapsed DRAM clock cycles per second
            # "launch__occupancy_per_block_size": "occupancy_per_block_size",  # the occupancy per block size
            # "launch__occupancy_per_register_count": "occupancy_per_register_count",  # the occupancy per register count
            # "launch__occupancy_per_shared_mem_size": "occupancy_per_shared_mem_size",  # the occupancy per shared memory size
            # "launch__registers_per_thread": "registers_per_thread",  # of registers per thread
            # "launch__registers_per_thread_allocated": "registers_per_thread_allocated",  # of registers per thread allocated
            # "launch__shared_mem_per_block": "shared_mem_per_block",  # the amount of shared memory per block
            # "launch__shared_mem_per_block_allocated": "shared_mem_per_block_allocated",  # the amount of shared memory per block allocated
            # "smsp__maximum_warps_avg_per_active_cycle": "smsp_max_warps",  # the maximum number of warps per active cycle on SMSP
        }  # metrics we will take the average of
        all_metrics = {**nan_metrics, **sum_metrics, **max_metrics, **avg_metrics}

        with open("results.json", "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format in results.json")

        # metrics structure: method -> graph_index -> metric: {values: [], unit: ""}
        metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
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
                        graph_idx = re.search(r"\d+", domain.push_pop_ranges()[0]).group()
                        for metric in all_metrics.keys():
                            if metric_data := action.metric_by_name(metric):
                                if metric_data.value() is not None:
                                    metrics[method][graph_idx][metric].setdefault("values", []).append(
                                        metric_data.value()
                                    )
                                    metrics[method][graph_idx][metric]["unit"] = metric_data.unit()

        for method, graphs in metrics.items():
            for graph_idx, graph_metrics in graphs.items():
                results[method][graph_idx]["metrics"] = dict()
                for metric, data in graph_metrics.items():
                    print(f"Processing {metric} for {method} on graph {graph_idx}")
                    print(data)
                    if metric in sum_metrics:
                        metric_value = sum(data["values"])
                    elif metric in max_metrics:
                        metric_value = max(data["values"])
                    elif metric in avg_metrics:
                        metric_value = sum(data["values"]) / len(data["values"])
                    else:
                        metric_value = data["values"][0]

                    if metric == "gpu__time_duration_measured_wallclock.avg":
                        metric_value = metric_value / 1e9  # convert to seconds

                    results[method][graph_idx]["metrics"][all_metrics[metric]] = {
                        "value": metric_value,
                        "unit": data["unit"],
                    }

        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)

print("All scripts have been executed.")
