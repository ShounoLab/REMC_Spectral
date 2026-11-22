"""free_energy.csv から，隣接格子点間の ⟨nE⟩ の変化と −∫Var(nE)dβ を比べる．
Var は log β–log Var で線形補間（局所的にべき関数）して積分する．"""
import csv, sys, math
rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(sys.argv[1]))]
bmin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
Ks = sorted({int(r['K']) for r in rows})
for K in Ks:
    g = [r for r in rows if r['K'] == K]
    print(f"K={K}")
    print(f"  {'beta_l':>8} {'beta_l+1':>8} {'exch':>5} {'dm_obs':>10} {'dm_pred':>10} {'ratio':>6} {'λ_l=β²Var':>9}")
    worst = []
    for a, b in zip(g, g[1:]):
        r = b['beta'] / a['beta']
        p = math.log(b['var_nE'] / a['var_nE']) / math.log(r)
        integ = a['var_nE'] * a['beta'] * ((r ** (p + 1) - 1) / (p + 1) if abs(p + 1) > 1e-9 else math.log(r))
        obs, pred = b['mean_nE'] - a['mean_nE'], -integ
        ratio = obs / pred
        worst.append((abs(math.log(ratio)), a['beta'], ratio, a['exchange_rate']))
        if a['beta'] >= bmin:
            print(f"  {a['beta']:8.1f} {b['beta']:8.1f} {a['exchange_rate']:5.2f} {obs:10.5f} {pred:10.5f} {ratio:6.2f} {a['beta']**2*a['var_nE']:9.2f}")
    w = max(worst)
    print(f"  全区間で ratio が 1 から最も離れた所: β={w[1]:.1f}, ratio={w[2]:.2f}, 交換率={w[3]:.2f}")
