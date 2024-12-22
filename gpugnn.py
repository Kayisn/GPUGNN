import argparse
import ctypes
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Add command line argument parsing
parser = argparse.ArgumentParser(description="Profile GNN experiments with different implementations")
parser.add_argument("--methods", "-m", type=str, default="all", help="List of methods to run")
parser.add_argument("--list-methods", default=False, action="store_true", help="List available methods")
parser.add_argument("--warmup", "-w", type=int, default=1, help="Number of warmup runs")
parser.add_argument("--verify", default=False, action="store_true", help="Verify the result")
parser.add_argument("--profile", default=False, action="store_true", help="Enable profiling")
parser.add_argument("--nvtx", "-n", type=str, default="main", help="Comma-separated list of NVTX ranges to profile")
parser.add_argument("--graphs", "-g", type=str, default=None, help="Index pattern of graphs to process")
args = parser.parse_args()

if args.profile:
    # Check if running as administrator
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("Please run as root/administrator.")
        exit(1)

    # Need to add the Nsight Compute Python package to the path before importing ncu_report
    sys.path.append(
        str(Path(subprocess.check_output(["where", "ncu"], text=True).strip()).parent / "extras" / "python")
    )
    import ncu_report

# List of methods to run
methods = all_methods = [path.stem for path in Path("scripts").glob("*.py") if path.stem != "__init__"]

if args.methods != "all":
    methods = []
    for method in args.methods.split(","):
        methods.extend(method for method in all_methods if args.methods in method)

if args.list_methods:
    print("Available methods:")
    print("\n".join(methods))
    exit()


results_path = Path("results") / "results.json"
report_dir = Path("reports")
if args.profile:
    report_dir.mkdir(exist_ok=True)

# Run each script sequentially
for method in methods:
    cmd = f"python executer.py --method {method} --warmup {args.warmup} {'--graphs ' + args.graphs if args.graphs else ''} {'--verify' if args.verify else ''}"

    try:
        if args.profile:
            print(f"Profiling {method}...")
            nvtx_patterns = [f'"regex:{method}@{nvtx}*/"' for nvtx in args.nvtx.split(",")]
            nvtx_include = "--nvtx-include " + " --nvtx-include ".join(nvtx_patterns)
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
                f"ncu --nvtx --metrics {','.join(metrics)} {nvtx_include} -f -o {str(report_dir / f'report_{method}')} {cmd}",
                stderr=subprocess.STDOUT,
                shell=True,
            )
            print(f"Completed profiling {method}.")
        else:
            print(f"Running {method}...")
            subprocess.check_call(cmd, stderr=subprocess.STDOUT)
            print(f"Completed running {method}.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {method}. Exit code: {e.returncode}")
        exit(1)

    if args.profile and (report_dir / f"report_{method}.ncu-rep").exists():
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

        with open(results_path, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in {results_path}")

        # metrics structure: method -> graph_index -> nvtx_range -> metric: {values: [], unit: ""}
        metrics = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"values": [], "unit": ""})))
        )
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
                        for nvtx_section in domain.push_pop_ranges():
                            nvtx_range = re.search(r"[A-Za-z]+", nvtx_section).group()
                            graph_idx = re.search(r"\d+", nvtx_section).group()
                            for metric in all_metrics.keys():
                                if metric_data := action.metric_by_name(metric):
                                    if metric_data.value() is not None:
                                        metrics[method][graph_idx][nvtx_range][metric]["values"].append(
                                            metric_data.value()
                                        )
                                        metrics[method][graph_idx][nvtx_range][metric]["unit"] = metric_data.unit()

        for method, graphs in metrics.items():
            for graph_idx, nvtx_ranges in graphs.items():
                results[method][graph_idx]["metrics"] = dict()
                for nvtx_range, graph_metrics in nvtx_ranges.items():
                    results[method][graph_idx]["metrics"][nvtx_range] = dict()
                    for metric, data in graph_metrics.items():
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

                        results[method][graph_idx]["metrics"][nvtx_range][all_metrics[metric]] = {
                            "value": metric_value,
                            "unit": data["unit"],
                        }

                    # metrics computed manually
                    results[method][graph_idx]["metrics"][nvtx_range]["active_cycles_ratio"] = {
                        "value": results[method][graph_idx]["metrics"][nvtx_range]["cycles_active"]["value"]
                        / results[method][graph_idx]["metrics"][nvtx_range]["cycles_elapsed"]["value"],
                        "unit": "%",
                    }
                    results[method][graph_idx]["metrics"][nvtx_range]["dram_throughput"] = {
                        "value": results[method][graph_idx]["metrics"][nvtx_range]["dram_bytes"]["value"]
                        / results[method][graph_idx]["metrics"][nvtx_range]["time_duration"]["value"],
                        "unit": "B/s",
                    }
                    results[method][graph_idx]["metrics"][nvtx_range]["l1tex_hit_rate"] = {
                        "value": results[method][graph_idx]["metrics"][nvtx_range]["l1tex_bytes_lookup_hit"]["value"]
                        / (
                            results[method][graph_idx]["metrics"][nvtx_range]["l1tex_bytes_lookup_hit"]["value"]
                            + results[method][graph_idx]["metrics"][nvtx_range]["l1tex_bytes_lookup_miss"]["value"]
                        ),
                        "unit": "%",
                    }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

print("All scripts have been executed.")
