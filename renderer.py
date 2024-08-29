import matplotlib
matplotlib.use('Agg')  # For environments without graphical interfaces

import pickle
import matplotlib.pyplot as plt
from matplotlib import animation
for i in range(0,20):
    rollout = pickle.load(open(f"/home/aistudio/rollout/gns_san2d_20240228-180012/best/rollout_{i}.pkl", "rb"))
    print(f"rollout of shape {rollout['predicted_rollout'].shape}(steps, nodes, xy pos)")

    fig, ax = plt.subplots(1, 2)
    # ax[0].set_xlim([0, 5.59])
    # ax[0].set_ylim([0, 2.22])
    # ax[1].set_xlim([0, 5.59])
    # ax[1].set_ylim([0, 2.22])

    ax[0].set_xlim([0.1, 0.9])
    ax[0].set_ylim([0.2, 0.8])
    ax[1].set_xlim([0.1, 0.9])
    ax[1].set_ylim([0.2, 0.8])
    fig.set_size_inches(10, 5, forward=True)
    ax[0].set_title("GNS")
    ax[1].set_title("Ground Truth")

    rollout_len = rollout["predicted_rollout"].shape[0] - 1

    scat0 = ax[0].scatter(
        rollout["predicted_rollout"][0, :, 0], rollout["predicted_rollout"][0, :, 1], s=40, animated=True
    )
    scat1 = ax[1].scatter(
        rollout["ground_truth_rollout"][0, :, 0], rollout["ground_truth_rollout"][0, :, 1], s=40, animated=True
    )


    def animate(i):
        scat0.set_offsets(rollout["predicted_rollout"][i])
        scat1.set_offsets(rollout["ground_truth_rollout"][i])
        return scat0, scat1


    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=rollout_len, interval=50, blit=True
    )

    plt.close(fig)

    writer = animation.PillowWriter(fps=10, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(f"/home/aistudio/gif/sand/{i}.gif", writer=writer)
