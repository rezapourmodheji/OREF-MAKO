from adamsknee import AdamsKnee

# ── PATHS ──────────────────────────────────────────────────────────────────
study     = 'S:\\BiomechanicsResearch\\groupImhauser\\OREF TKA\\Modeling'
data_raw  = study + '\\Data_Raw' 
data_reduced = study + '\\Data_Reduced'

subs = ['S026']

for sub in subs:
    subcase = AdamsKnee(sub, 'L', study)
    print(f'Running pre-optimization for {subcase.subject}...')
    subcase.preoptimization(verbose=False, run_adams=True)