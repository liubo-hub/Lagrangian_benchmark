import matplotlib
matplotlib.use('Agg')  # For environments without graphical interfaces

import pickle
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
for i in range(0,20):
    # load datasets of real and predict
    rollout = pickle.load(
        open(f"/home/aistudio/rollout/gns_rpf3d_20240228-143146/best/rollout_{i}.pkl", "rb"))
    print(f"rollout of shape {rollout['predicted_rollout'].shape} (steps, nodes, xyz pos)")

    fig = plt.figure(figsize=(10, 5))
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')

    # set axis range
    ax0.set_xlim([0, 0.8])
    ax0.set_ylim([0, 0.8])
    ax0.set_zlim([0, 0.8])
    ax1.set_xlim([0, 0.8])
    ax1.set_ylim([0, 0.8])
    ax1.set_zlim([0, 0.8])

    # set titlr
    ax0.set_title("GNS Prediction")
    ax1.set_title("Ground Truth")

    rollout_len = rollout["predicted_rollout"].shape[0] - 1

    scat0 = ax0.scatter(
        rollout["predicted_rollout"][0, :, 0], rollout["predicted_rollout"][0, :, 2], rollout["predicted_rollout"][0, :, 1], s=5, animated=True
    )
    scat1 = ax1.scatter(
        rollout["ground_truth_rollout"][0, :, 0], rollout["ground_truth_rollout"][0, :, 2], rollout["ground_truth_rollout"][0, :, 1], s=5, animated=True
    )

    # define update animate function
    def animate(i):
        scat0._offsets3d = (rollout["predicted_rollout"][i, :, 0], rollout["predicted_rollout"][i, :, 2], rollout["predicted_rollout"][i, :, 1])
        scat1._offsets3d = (rollout["ground_truth_rollout"][i, :, 0], rollout["ground_truth_rollout"][i, :, 2], rollout["ground_truth_rollout"][i, :, 1])
        return scat0, scat1

    # create animate
    ani = animation.FuncAnimation(fig, animate, frames=rollout_len, interval=50, repeat=True)

    plt.close(fig)

    # save
    writer = animation.PillowWriter(fps=10)
    ani.save(f"/home/aistudio/gif/rpf/{i}.gif", writer=writer)

