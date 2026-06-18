"""pareto_report.py — merge the accuracy axis (elastic_pareto.py) with the
measured per-topology latency (tb_topo_sweep) into the elastic-Pareto table,
plot, and consolidated JSON.

One bitstream, many operating points: same runtime-reconfigurable RTL runs every
topology bit-exact (tb_topo_sweep 5/5 PASS); accuracy from QAT power-of-2 INT8.

Latency is MEASURED (tb_topo_sweep, weight-invariant, deterministic FSM).
Energy is first-order: E = P * latency using the Phase-C PowerPlay numbers for the
baseline (4,4,8,8) bitstream. Dynamic power is held at the baseline value for all
topologies — a CONSERVATIVE upper bound for the smaller ones, since masked CP
lanes draw less; per-topology PowerPlay would only push the cheap points lower.
"""

import os
import json
import argparse

# ── measured latency (tb_topo_sweep, cycles @ 100 MHz → us) ─────────────────
LATENCY_CY = {
    (2, 2, 2, 2): 3844,
    (2, 2, 4, 4): 3894,
    (4, 4, 4, 4): 5114,
    (4, 4, 8, 8): 5214,   # ~= documented baseline 5216 cy / 52.16 us
    (8, 8, 8, 8): 7654,   # == documented endpoint 76.54 us
}
CLK_MHZ = 100.0

# ── Phase-C PowerPlay (baseline 4,4,8,8 bitstream) ──────────────────────────
P_TOTAL_W = 0.623
P_DYN_W = 0.198
P_STATIC_W = 0.413


def energy_uJ(latency_s, power_w):
    return latency_s * power_w * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acc_json', default='./results/elastic_pareto/deployment_pareto.json',
                    help='deployment_pareto.json (headline) or pareto_accuracy.json '
                         '(from-scratch capacity-isolation)')
    ap.add_argument('--fair_4488', default='',
                    help='(from-scratch mode only) replace the production anchor with the '
                         'from-scratch (4,4,8,8) to isolate capacity; empty in deployment mode')
    ap.add_argument('--output_dir', default='./results/elastic_pareto')
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.acc_json) as f:
        acc = json.load(f)
    deployment = acc.get('mode') == 'deployment' or any('method' in p for p in acc['points'])

    # (from-scratch mode only) swap (4,4,8,8) production anchor for the from-scratch
    # point so the frontier isolates capacity from training effort; the production
    # result becomes an annotation (+2.8pp recipe lever orthogonal to capacity).
    production_anchor = None
    if not deployment and args.fair_4488 and os.path.exists(args.fair_4488):
        with open(args.fair_4488) as f:
            fair_pt = json.load(f)['points'][0]
        for i, pt in enumerate(acc['points']):
            if tuple(pt['topology']) == (4, 4, 8, 8):
                production_anchor = {'int8_acc': pt['int8_p2']['acc'],
                                     'float_acc': pt['float']['acc']}
                fair_pt['anchor'] = False
                acc['points'][i] = fair_pt

    rows = []
    for pt in acc['points']:
        ch = tuple(pt['topology'])
        cy = LATENCY_CY[ch]
        t_s = cy / (CLK_MHZ * 1e6)
        rows.append({
            'topology': list(ch),
            'method': pt.get('method', 'from-scratch'),
            'float_acc': pt['float']['acc'],
            'int8_acc': pt['int8_p2']['acc'],
            'int8_f1': pt['int8_p2']['f1'],
            'conv_weights': pt['conv_weights'],
            'latency_cy': cy,
            'latency_us': round(cy / CLK_MHZ, 2),
            'energy_total_uJ': round(energy_uJ(t_s, P_TOTAL_W), 2),
            'energy_dyn_uJ': round(energy_uJ(t_s, P_DYN_W), 2),
        })
    rows.sort(key=lambda r: r['latency_cy'])

    # ── markdown table ──────────────────────────────────────────────────────
    head = ("# Elastic-Pareto (deployment): one bitstream, best-deployable per "
            "operating point\n" if deployment else
            "# Elastic-Pareto (capacity-isolation): one bitstream, consistent "
            "from-scratch recipe\n")
    md = [head,
          "All points run **bit-exact on ONE bitstream** (tb_topo_sweep 5/5 PASS); "
          "INT8 = power-of-2 QAT on Chapman test. Latency MEASURED; energy = P×latency "
          "(Phase-C PowerPlay, baseline power held constant — conservative for small "
          "topologies).\n"]
    if deployment:
        md += ["| Topology | INT8 acc | INT8 F1 | Conv w | Latency (µs) | E_total (µJ) | "
               "E_dyn (µJ) | Method |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {tuple(r['topology'])} | {r['int8_acc']:.4f} | {r['int8_f1']:.4f} | "
                      f"{r['conv_weights']} | {r['latency_us']:.2f} | {r['energy_total_uJ']:.2f} | "
                      f"{r['energy_dyn_uJ']:.2f} | {r['method']} |")
        md.append("\nEach point = **best-deployable** model at that topology via the same "
                  "prune-transfer recipe as the shipped (4,4,8,8)=94.65%. (8,8,8,8) is "
                  "from-scratch (conv1=8 exceeds the (4,8,8,16) parent → not prunable). "
                  "INT8 ≈ float at every point (power-of-2 0-DSP rescale generalises). "
                  "(8,8,8,8) costs 2× weights + 47% latency/energy for ≤(4,4,8,8) accuracy "
                  "→ (4,4,8,8) is the knee; (2,2,*) are low-power screening points "
                  "(38 µs, ~24 µJ, −26% energy).")
    else:
        md += ["| Topology | Float acc | INT8 acc | INT8 F1 | Conv w | Latency (µs) | "
               "E_total (µJ) | E_dyn (µJ) |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            dagger = production_anchor is not None and tuple(r['topology']) == (4, 4, 8, 8)
            tag = str(tuple(r['topology'])) + (' †' if dagger else '')
            md.append(f"| {tag} | {r['float_acc']:.4f} | {r['int8_acc']:.4f} | "
                      f"{r['int8_f1']:.4f} | {r['conv_weights']} | {r['latency_us']:.2f} | "
                      f"{r['energy_total_uJ']:.2f} | {r['energy_dyn_uJ']:.2f} |")
        md.append("\nAll points use ONE consistent **from-scratch** recipe (50 float + "
                  "30 QAT ep, seed 42) so accuracy reflects capacity, not training effort. "
                  "Frontier monotone in capacity; (8,8,8,8) is NOT worse than (4,4,8,8) "
                  "at equal recipe. INT8 ≈ float at every point.")
        if production_anchor:
            fs = [r for r in rows if tuple(r['topology']) == (4, 4, 8, 8)][0]['int8_acc']
            md.append(f"\n† **Production-recipe lever** (orthogonal to capacity): the shipped "
                      f"(4,4,8,8) (train-full → prune-transfer → finetune + tuned QAT) reaches "
                      f"**{production_anchor['int8_acc']*100:.2f}% INT8** vs {fs*100:.2f}% "
                      f"from-scratch — +{(production_anchor['int8_acc']-fs)*100:.1f}pp available "
                      f"at any operating point, NOT a capacity effect.")
    md_text = "\n".join(md) + "\n"
    md_path = os.path.join(args.output_dir, 'PARETO.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)

    # ── consolidated JSON ───────────────────────────────────────────────────
    out = {
        'claim': 'single runtime-reconfigurable bitstream, all points bit-exact '
                 '(tb_topo_sweep 5/5 PASS)',
        'power_model': {'P_total_W': P_TOTAL_W, 'P_dyn_W': P_DYN_W,
                        'P_static_W': P_STATIC_W,
                        'note': 'baseline power held constant across topologies; '
                                'conservative upper bound for small topologies'},
        'mode': 'deployment' if deployment else 'capacity-isolation',
        'recipe': acc.get('recipe'),
        'production_anchor_4488': production_anchor,
        'points': rows,
    }
    json_path = os.path.join(args.output_dir, 'pareto_full.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)

    # ── plot ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        lat = [r['latency_us'] for r in rows]
        en = [r['energy_total_uJ'] for r in rows]
        acc8 = [r['int8_acc'] * 100 for r in rows]
        labels = [str(tuple(r['topology'])) for r in rows]
        for ax, x, xlabel in [(ax1, lat, 'Latency (µs/inference)'),
                              (ax2, en, 'Energy (µJ/inference, total)')]:
            ax.plot(x, acc8, 'o-', color='#1f77b4')
            for xi, yi, lb in zip(x, acc8, labels):
                ax.annotate(lb, (xi, yi), textcoords='offset points',
                            xytext=(5, 5), fontsize=8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('INT8 accuracy (%)')
            ax.grid(True, alpha=0.3)
        fig.suptitle('Elastic accelerator: accuracy vs latency / energy '
                     '(one bitstream, ' +
                     ('best-deployable per point)' if deployment
                      else 'consistent from-scratch recipe)'))
        fig.tight_layout()
        fig_path = os.path.join(args.output_dir, 'pareto.png')
        fig.savefig(fig_path, dpi=130)
        print(f"  plot  -> {fig_path}")
    except Exception as e:
        print(f"  [WARN] plot skipped: {e}")

    print(md_text)
    print(f"  table -> {md_path}\n  json  -> {json_path}")


if __name__ == '__main__':
    main()
