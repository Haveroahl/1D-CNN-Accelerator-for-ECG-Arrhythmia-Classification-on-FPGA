"""gen_pareto_golden.py — golden + manifest for the 5 elastic-Pareto topologies.

Thin wrapper over gen_topo_golden (imports its verified reference pipeline, does
NOT modify it). Emits a topo_manifest.txt containing exactly the Pareto operating
points so tb_topo_sweep measures their latency (weight-invariant) and re-confirms
bit-exact for these specific points on the single bitstream.

Tags are distinct (t<c1c2c3c4>) so generated dirs do not collide with the 48-case
coverage golden. Caller backs up / restores the canonical manifest around the run.

Usage (from software/python):
  python gen_pareto_golden.py \
    --ecg ../../hardware/fpga/simulation/questa/ecg_sample0.hex \
    --output_dir ../../hardware/fpga/simulation/questa/topo_golden
"""

import os
import argparse

from gen_topo_golden import gen_topology, int8_forward, write_topology, read_ecg_hex

PARETO_TOPOS = [(2, 2, 2, 2), (2, 2, 4, 4), (4, 4, 4, 4), (4, 4, 8, 8), (8, 8, 8, 8)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ecg', required=True)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    x_raw = read_ecg_hex(args.ecg)
    order = ['conv1', 'conv2', 'conv3', 'conv4']
    rows = []
    seed = 7000
    for ch in PARETO_TOPOS:
        tag = 't' + ''.join(str(c) for c in ch)
        w, b, in_ch, out_ch = gen_topology(ch, seed=seed)
        seed += 1
        logits, _ = int8_forward(x_raw, w, b, ch)
        cfg = write_topology(args.output_dir, tag, ch, w, b, in_ch, logits)
        ic = [1, ch[0], ch[1], ch[2]]
        ce = [cfg['cp_en'][n] for n in order]
        nb = [cfg['nb'][n] for n in order]
        bs = [cfg['base'][n] for n in order]
        rows.append([tag] + list(ch) + ic + ce + nb + bs + [cfg['expected_argmax']])

    mpath = os.path.join(args.output_dir, 'topo_manifest.txt')
    with open(mpath, 'w') as f:
        f.write(f"# {len(rows)} elastic-Pareto topologies: "
                f"tag c1..c4 ic0..3 ce0..3 nb0..3 bs0..3 argmax\n")
        for row in rows:
            f.write(' '.join(str(x) for x in row) + '\n')
    print(f"[INFO] wrote Pareto manifest {mpath} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
