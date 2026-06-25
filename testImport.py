from adamsknee import AdamsKnee
import os
import multiprocessing




# ── PATHS ──────────────────────────────────────────────────────────────────
study     = 'S:\\BiomechanicsResearch\\groupImhauser\\OREF TKA\\Modeling'
data_raw  = study + '\\Data_Raw' 
data_reduced = study + '\\Data_Reduced'

subs = ['S026']
def build(args):
    sub, study = args
    print(f"[PID {os.getpid()}] {sub} - Building model...")
    subcase = AdamsKnee(sub, 'L', study)
    run_adams = True
    # subcase.import_geometries(verbose=False, run_adams=run_adams)
    
    subcase.create_elements(verbose=False, run_adams=run_adams)
    subcase.smcl(verbose=False, run_adams=run_adams)
    subcase.distcomp(verbose=False, run_adams=run_adams)
    subcase.updateligs(verbose=False, run_adams=run_adams)
    
if __name__ == "__main__":
    isparallel = False  # Set to True to enable multiprocessing, False for sequential execution
    if not isparallel:
        for sub in subs:
            print(f"Building model for {sub}...")
            subcase = AdamsKnee(sub, 'L', study)
            run_adams = True
            # subcase.import_geometries(verbose=False, run_adams=run_adams)
            subcase.create_elements(verbose=False, run_adams=True)
            subcase.smcl(verbose=False, run_adams=True)
            subcase.distcomp(verbose=False, run_adams=True)
            subcase.updateligs(verbose=False, run_adams=True)
                    ####
                    # subcase.create_elements(verbose=False, run_adams=run_adams)
                    # subcase.smcl(verbose=False, run_adams=run_adams)
                    # subcase.distcomp(verbose=False, run_adams=run_adams)
                    # subcase.updateligs(verbose=False, run_adams=run_adams)
    elif isparallel:
        tasks = [
                (sub, thecase, coltension, study)
                for sub in subs
                for thecase in cases
                for coltension in coltensions
            ]
        # Use at most one worker per task, or cap at CPU count - 1
        n_workers = min(len(tasks), max(1, multiprocessing.cpu_count() - 1))
        print(f"Spawning {n_workers} workers for {len(tasks)} task groups...")

        with multiprocessing.Pool(processes=n_workers) as pool:
            all_results = pool.map(build, tasks)

        print("All simulations complete.")
    else:
        print("Invalid parallelization option. Set isparallel to True or False.")
            