import os
from multiprocessing import get_context
from vanilla_simulator import simulation

def run_multi_cpu(
    N_total,
    n_procs=None,
    chunk_size=10000,
    out_dir="simulation_output",
):
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if n_procs is None:
        n_procs = max(1, os.cpu_count() - 2)

    base = N_total // n_procs
    rem  = N_total % n_procs

    jobs = []
    for wid in range(n_procs):
        Ni = base + (1 if wid < rem else 0)
        jobs.append((Ni, chunk_size, out_dir, wid))

    ctx = get_context("spawn")
    with ctx.Pool(processes=n_procs) as pool:
        pool.starmap(simulation, jobs)

if __name__ == "__main__":
    run_multi_cpu(1_000_000, n_procs=100, chunk_size=10000, out_dir="vanilla_convertibles_simulation")
