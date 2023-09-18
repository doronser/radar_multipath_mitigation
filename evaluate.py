import os
import re
import cv2
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import trange, tqdm
from scipy.stats import linregress

sys.path.insert(0, "/home/doronser/workspace")
from radar_multipath_mitigation.utils import *


#vis
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run multipath mitgation algorithm and save results")
    parser.add_argument("--scn", type=int, help="scenario to run on")
    parser.add_argument("--seq", type=int, help="sequence to run on")
    parser.add_argument("--side", type=str, default="right", help="sequence to run on")
    parser.add_argument("--agg", type=int, default=3, help="How many radar frames to aggregate")
    parser.add_argument("--start", type=int, default=0, help="start frame")
    parser.add_argument("--end", type=int, default=None, help="end frame")
    parser.add_argument("-d", "--debug", action="store_true", default=False, help="use label to get main obj")
    args = parser.parse_args()

    print("getting data")
    data_root = Path('/media/oldrrtammyfs/Users/doronser/radar_ghost_dataset')
    ds = RadarGhostDataset(data_root=data_root)

    radar_df = ds.get_radar_pc(scenario=args.scn, sequence=args.seq, sensor=args.side, raw=False)
    min_idx_for_eval = args.start
    max_idx_for_eval = args.end or radar_df.index.max() // 2  # target leaves FOV and returns. we evaluate only the first half
    seq_class = radar_df.query('is_main_obj==1')['class'].iloc[0]

    metrics = []
    agg = args.agg
    plots_dir = Path("/home/doronser/workspace/radar_multipath_mitigation/plots/")
    dir_name = f"results_agg{args.agg}_debug" if args.debug else f"results_agg{args.agg}"
    subdir = plots_dir / f"scene{args.scn:02d}_seq{args.seq:02d}_{args.side}" / dir_name
    os.makedirs(subdir, exist_ok=True)
    for idx in trange(min_idx_for_eval, max_idx_for_eval, agg+1,
                      desc=f"Multipath Mitigation Scn{args.scn:02d}Seq{args.seq:02d}_{args.side}"):
        try:
            clst0, clst1, stat = get_clusters(radar_df, idx=idx, agg=agg, debug=args.debug)
            if clst0 is None:
                continue
            r, a, b, beta = estimate_reflector_points(clst0, clst1, debug=True)
            res = linregress(r[:, 0], r[:, 1])
            metrics.append(dict(
                idx=idx,
                msd=MSD(r, res.slope, res.intercept),
                ang=ANG(r, a),
                per=0
            ))
            fig = plot_results(radar_pc=(clst0, clst1, stat), algo_res=(r, a, b, beta), linreg_res=res, debug=True,
                               title=f"Frame #{idx} | ", xlim=(0, 40), ylim=(-30, 10))

            plt.savefig(f'{subdir}/frame{idx:03d}.png')
            plt.close()
        except Exception as e:
            print("priblem with frame", idx)
            print(e)
            continue

    metrics_df = pd.DataFrame(metrics)
    if len(metrics_df):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        metrics_df.msd.plot(ax=axes[0], title=f'MSD={metrics_df.msd.mean():.2f}', xlabel='frame group')
        metrics_df.ang.plot(ax=axes[1], title=f'ANG={metrics_df.ang.mean():.2f}', xlabel='frame group')
        plt.suptitle(f"Scenario #{args.scn:02d} | Sequence #{args.seq:02d} | Side: {args.side} | {seq_class}")
        plt.savefig(plots_dir.parent / "results" /  f"scene{args.scn:02d}_seq{args.seq:02d}_{args.side}.png")
        plt.close()

        csv_filename = plots_dir.parent / "results" /  f"scene{args.scn:02d}_seq{args.seq:02d}_{args.side}_agg{args.agg}.csv"
        metrics_df.to_csv(str(csv_filename), index=False)

