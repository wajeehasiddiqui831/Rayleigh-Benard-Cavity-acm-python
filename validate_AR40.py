"""Validation at AR = 40 against Lartigue et al., the cases of Table 2 in the FEDSM 2022 paper.
Grids 1:40 and 1:50, dt = 0.004, total time 200 (50,000 steps), as used for the README table."""
from rb_cavity_acm import run_case

Ra_list = [3550, 6800, 10102, 14200, 17750]
ref     = {3550: 1.0640, 6800: 1.1670, 10102: 1.2920, 14200: 1.3880, 17750: 1.4840}   # Lartigue et al.
paper   = {3550: 1.0910, 6800: 1.1860, 10102: 1.3147, 14200: 1.4160, 17750: 1.4987}   # FEDSM 2022 (Nx scaling)
DT, MAXIT = 0.004, 50000

for PTS in (40, 50):                                   # grid points per unit length
    print(f"\nGrid 1:{PTS}, dt = {DT}")
    print(f"{'Ra':>6} {'Lartigue':>9} {'paper':>8} {'Python':>8} {'diff %':>7} {'converged':>10}")
    for Ra in Ra_list:
        r = run_case(Ra, 40, dt=DT, pts_per_unit=PTS, max_iter=MAXIT, verbose=False)
        nu = r["Nu_avg_alt"]                           # final-step average Nu, consistent (Nx - 1) scaling
        print(f"{Ra:>6} {ref[Ra]:>9.4f} {paper[Ra]:>8.4f} {nu:>8.4f} {100*(nu-ref[Ra])/ref[Ra]:>7.2f} {str(r['converged']):>10}")
