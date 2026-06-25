from adamsknee import AdamsKnee

# ── PATHS ──────────────────────────────────────────────────────────────────
study     = 'S:\\BiomechanicsResearch\\groupImhauser\\OREF TKA\\Modeling'
data_raw  = study + '\\Data_Raw'
data_reduced = study + '\\Data_Reduced'

subs = ['S026']

for sub in subs:
    subcase = AdamsKnee(sub, 'L', study)
    print(f'Optimizing pre-optimization variables for {subcase.subject}...')
    result, best_vars = subcase.optimize_preoptimization(
        method='coordinate',      # your rule: force>target -> raise percent, force<target -> lower it
        bounds=(0.8, 1.2),        # safe range; tighten to (0.9, 1.05) for fewer/faster sims
        diff_step=0.03,           # sensitivity-probe step (one extra sim at the start)
        force_tol=0.5,            # stop once every force is within 0.5 N of target
        max_step=0.06,            # cap on percent change per iteration
        relax=0.8,                # damping to avoid overshoot on coupled ligaments
        max_nfev=60,              # cap on the number of Adams forward simulations
        verbose=True,
        # method='least_squares', # alternative: more robust to coupling, ~9x more sims
    )
    print(f'Done. Best variables for {subcase.subject}:')
    for name, value in best_vars.items():
        print(f'  {name:24s} {value:.4f}')
