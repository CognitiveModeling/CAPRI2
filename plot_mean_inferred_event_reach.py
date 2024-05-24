# Plot of (mean) inferred event probabilities during reaching for hand and claw agent

import numpy as np

# Define path to result files

foldername = "Experiments/Test1"

#a_shapes = ["hand", "claw"]
a_shapes = [0.2, 0.8]

num_runs = 12

num_sims = 20

f = "adaptation_" + str(num_runs) + "runs"


# Index of event probabilities is 3, 4, 5, 6
e_prob = 3

def reach_estimation_one_run(data):
    reach_end = 100  # end of reaching event
    reach_estimates = np.zeros((4))
    for t in range(reach_end):
        for e in range(4):
            reach_estimates[e] += data[t, e + e_prob]
    reach_estimates = reach_estimates / reach_end
    return reach_estimates


# Read in hand-agent and claw-agent testing data and compute mean inferred event during reaching

# Find event estimates for reaching for hand
reach_event_estimates_hand = np.zeros((num_sims, 30, 4))
for sim in range(num_sims):
    for epoch in range(30):
        for run in range(num_runs):
            log_file_name = foldername + "/" + test_name + str(sim) + "/log_files/"
            filename = log_file_name + "res_tau_2_sim" + str(sim) + "_epoch" + str(epoch) + "_" + str(
                a_shapes[0]) + "_run" + str(run) + ".txt"
            data = np.loadtxt(filename, dtype='float64', skiprows=1, delimiter=', ')
            estimates = reach_estimation_one_run(data)
            reach_event_estimates_hand[sim, epoch, :] = estimates[:]
    print("Processed hand-agent data of simulation " + str(sim))

# Find event estimates for reaching for claw
reach_event_estimates_claw = np.zeros((num_sims, 30, 4))
for sim in range(num_sims):
    for epoch in range(30):
        for run in range(num_runs):
            log_file_name = foldername + "/" + test_name + str(sim) + "/log_files/"
            filename = log_file_name + "res_tau_2_sim" + str(sim) + "_epoch" + str(epoch) + "_" + str(
                a_shapes[1]) + "_run" + str(run) + ".txt"
            data = np.loadtxt(filename, dtype='float64', skiprows=1, delimiter=', ')
            estimates = reach_estimation_one_run(data)
            reach_event_estimates_claw[sim, epoch, :] = estimates[:]
    print("Processed claw-agent data of simulation " + str(sim))

# Take the mean
reach_estimates_hand_over_time = np.mean(reach_event_estimates_hand, axis=0)
reach_estimates_claw_over_time = np.mean(reach_event_estimates_claw, axis=0)

# Nice color definitions
colors = [(0.368, 0.507, 0.71), (0.881, 0.611, 0.142),
          (0.56, 0.692, 0.195), (0.923, 0.386, 0.209),
          (0.528, 0.471, 0.701), (0.772, 0.432, 0.102),
          (0.364, 0.619, 0.782), (0.572, 0.586, 0.)]

import matplotlib.pyplot as plt
#import tikzplotlib

epochs = range(30)

line1 = plt.plot([0], [0], color=colors[1], alpha=1)
line2 = plt.plot([0], [0], color=colors[5], alpha=1)
line3 = plt.plot([0], [0], color=colors[2], alpha=1)
line4 = plt.plot([0], [0], color=colors[4], alpha=1)

plt.fill_between(epochs, 0, reach_estimates_hand_over_time[:, 0], alpha=1.0, facecolor=colors[1])
plt.fill_between(epochs, reach_estimates_hand_over_time[:, 0],
                 reach_estimates_hand_over_time[:, 0] + reach_estimates_hand_over_time[:, 1], alpha=1.0,
                 facecolor=colors[5])
plt.fill_between(epochs, reach_estimates_hand_over_time[:, 0] + reach_estimates_hand_over_time[:, 1],
                 reach_estimates_hand_over_time[:, 0] + reach_estimates_hand_over_time[:,
                                                        1] + reach_estimates_hand_over_time[:, 2], alpha=1.0,
                 facecolor=colors[2])
plt.fill_between(epochs, reach_estimates_hand_over_time[:, 0] + reach_estimates_hand_over_time[:,
                                                                1] + + reach_estimates_hand_over_time[:, 2],
                 reach_estimates_hand_over_time[:, 0] + reach_estimates_hand_over_time[:,
                                                        1] + reach_estimates_hand_over_time[:,
                                                             2] + + reach_estimates_hand_over_time[:, 3], alpha=1.0,
                 facecolor=colors[4])

# line1 = plot.errorbar(x=epochs, y=looking_ts_hand_mean, yerr=looking_ts_hand_sd, color='b')
plt.legend((line1[0], line2[0], line3[0], line4[0]), ('still', 'random', 'reach', 'transport'))

plt.xlim([0, 29])
plt.ylim([0, 1])
plt.xlabel('training phases')
plt.ylabel('P(e(t)|O(t), Pi(t))')

plt.savefig(foldername + "/" + f + "_event_probability_reach_hand_" + str(num_sims) + "sims.png", bbox_inches="tight")
#tikzplotlib.save(foldername + "/" + f + "_event_probability_reach_hand_" + str(num_sims) + "sims.tex")

plt.show()


line1 = plt.plot([0], [0], color=colors[1], alpha=1)
line2 = plt.plot([0], [0], color=colors[5], alpha=1)
line3 = plt.plot([0], [0], color=colors[2], alpha=1)
line4 = plt.plot([0], [0], color=colors[4], alpha=1)

plt.fill_between(epochs, 0, reach_estimates_claw_over_time[:, 0], alpha=1.0, facecolor=colors[1])
plt.fill_between(epochs, reach_estimates_claw_over_time[:, 0],
                 reach_estimates_claw_over_time[:, 0] + reach_estimates_claw_over_time[:, 1], alpha=1.0,
                 facecolor=colors[5])
plt.fill_between(epochs, reach_estimates_claw_over_time[:, 0] + reach_estimates_claw_over_time[:, 1],
                 reach_estimates_claw_over_time[:, 0] + reach_estimates_claw_over_time[:,
                                                        1] + reach_estimates_claw_over_time[:, 2], alpha=1.0,
                 facecolor=colors[2])
plt.fill_between(epochs, reach_estimates_claw_over_time[:, 0] + reach_estimates_claw_over_time[:,
                                                                1] + + reach_estimates_claw_over_time[:, 2],
                 reach_estimates_claw_over_time[:, 0] + reach_estimates_claw_over_time[:,
                                                        1] + reach_estimates_claw_over_time[:,
                                                             2] + + reach_estimates_claw_over_time[:, 3], alpha=1.0,
                 facecolor=colors[4])

plt.legend((line1[0], line2[0], line3[0], line4[0]), ('still', 'random', 'reach', 'transport'))

plt.xlim([0, 29])
plt.ylim([0, 1])
plt.xlabel('training phases')
plt.ylabel('P(e(t)|O(t), Pi(t))')

plt.savefig(foldername + "/" + f + "_event_probability_reach_claw_" + str(num_sims) + "sims.png", bbox_inches="tight")
#tikzplotlib.save(foldername + "/"+ f + "_event_probability_reach_claw_" + str(num_sims) + "sims.tex")

plt.show()

