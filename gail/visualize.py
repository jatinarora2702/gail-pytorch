import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def smooth_fn(a, window=100):
    smooth_a = []
    running_sum = 0
    for i in range(window):
        running_sum += a[i]
        smooth_a.append(running_sum / (i + 1))
    for i in range(window, len(a)):
        running_sum += a[i] - a[i - window]
        smooth_a.append(running_sum / window)
    return smooth_a


def main(args):
    root_dir = os.path.join(args.out_dir, args.env_id, args.model_name)

    record_map = dict()
    max_cnt = 0
    for i in range(1, 6):
        pickle_path = os.path.join(root_dir, "record{}.pkl".format(i))
        with open(pickle_path, "rb") as handle:
            record = pickle.load(handle)
        max_cnt = max(max_cnt, len(record))
        episode_rewards = smooth_fn([tup[1] for tup in record])
        for it in range(len(episode_rewards)):
            if it not in record_map:
                record_map[it] = []
            record_map[it].append(episode_rewards[it])

    values = []
    errors = []
    for i in range(max_cnt):
        m = np.mean(record_map[i])
        e = 2.0 * stats.sem(record_map[i])  # 95% confidence interval
        # e = 2.0 * np.std(record_map[i])  # 95% confidence interval
        values.append(m)
        errors.append(e)

    iteration_counts = list(range(max_cnt))

    fig = plt.figure()
    markers, caps, bars = plt.errorbar(iteration_counts, values, errors, ecolor='lightskyblue')
    [bar.set_alpha(0.1) for bar in bars]
    [cap.set_alpha(0.1) for cap in caps]

    plt.ylabel("Episode Rewards")
    plt.xlabel("Training Iteration")
    fig.suptitle(args.model_name.upper())
    fig.savefig(os.path.join(root_dir, "rewards.png"), dpi=600)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Visualize Training")
    ap.add_argument("--env_id", type=str, default="CartPole-v0", help="gym env identifier")
    ap.add_argument("--model_name", type=str, default="ppo", help="algorithm name (ppo|gail)")
    ap.add_argument("--out_dir", type=str, default="../out", help="outputs parent directory")
    ap = ap.parse_args()
    main(ap)
