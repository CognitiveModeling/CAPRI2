# Plot the gaze arrival times as comparison with/without action effects / with/without adaptation

import numpy as np
import matplotlib.pyplot as plt


# with and without adaptation, with action effects(?)
folder_nonad = "Experiments/ResAblationTimeHorizon-test_old_0_4v.1.1-3-adashape-t0-0.4-t34-0-1"
folder_adapt = "Experiments/ResAblationTimeHorizon-test_old_0_4v.1.1-3-adashape-t0-0.4-t34-1"
v = "t0-3401-341-t1-comp-adapt-h0.4_c0.8"
#v = "t0-3401-341-t1-comp-adapt-h0.5_c1.0"

# # with and without action effects, without shape adaptation
# folder_nonad = "Experiments/ResAblationTimeHorizon-test_old_0_4v.1.1-3-adashape-t0-0.4-t35-c-0"
# folder_adapt = "Experiments/ResAblationTimeHorizon-test_old_0_4v.1.1-3-adashape-t0-0.4-t34-0-1"
# v = "t0-3401-35c0-t1-comp-acteff-noadapt-h0.4_c0.8"
# #v = "t0-3401-35c0-t1-comp-acteff-noadapt-h0.5_c1.0"

# # # with and without action effects, with shape adaptation
folder_nonad = "Experiments/ResAblationTimeHorizon-test_old_0_4v.1.1-3-adashape-t0-0.4-t35-c-1"
folder_adapt = "Experiments/ResAblationTimeHorizon-test_old_0_4v.1.1-3-adashape-t0-0.4-t34-1"
v = "t0-3401-351-t1-comp-acteff-wadapt-h0.4_c0.8"
# #v = "t0-3401-351-t1-comp-acteff-wadapt-h0.5_c1.0"

# # # with and without action effects, with shape adaptation
folder_nonad = "Experiments/ResAblationTimeHorizon-test_old_0_4v.1.1-3-adashape-t0-0.4-t35-c-2"
folder_adapt = "Experiments/ResAblationTimeHorizon-test_old_0_4v.1.1-3-adashape-t0-0.4-t34-1"
v = "t0-341-35c2-t1-comp-acteff-wadapt-h0.4_c0.8"

folder_nonad =  "Experiments/Test3" # same without action effects ...
folder_adapt = "Experiments/Test2" # with action effects ...
v = "test2-test3-comp-acteff-withoutadapt-h0.2_c0.8"

folder_names = [folder_nonad, folder_adapt]

label_add = ""

labels="hand (no action effect)", "claw (no action effect)", "hand (action effect)", "claw (action effect)" #(??)
#labels="hand (no adaptation)", "claw (no adaptation)", "hand (adaptation)", "claw (adaptation)"

test_name = "res_tau_2_sim"

# Hand/claw agents to plot
shape_claw = 0.8
shape_hand = 0.4

a_shapes = [shape_hand, shape_claw]

num_agents = len(folder_names) * len(a_shapes)


num_sims = 20
n_sims_1 = 20

runs = 12

#Define indices of data, where to find which information

event_t = 1 # index of e(t)
policy_t = 2 # index of pi(t)

epochs = 30

#For one result file determine the point t when pi(t) = pi_p

def find_t_look_policy(data):
    for t in range(len(data)):
        if data[t, policy_t] == 1:
            return t
    return 270

#We can also determine gaze at the AOI of the reaching target. This can be done for example by determining if the system
# looks at the patient or at the agent, when it is already transporting the patient, and is, thus, closer than a
# predefined distance threshold

def find_t_look(data):
    for t in range(len(data)):
        if data[t, policy_t] == 1 or (data[t, event_t] == 3 and data[t, policy_t] == 0):
            return t
    return 270

def normalize_t(data, t_look):
    return  (270 - t_look)/100

#Read in hand-agent and claw-agent testing data and compute time
#t of first goal-predictive gaze

looking_ts_all_1 = np.zeros((int(num_agents/2), n_sims_1, epochs, runs), dtype='float64')
looking_ts_all_policy_1 = np.zeros((int(num_agents/2), n_sims_1, epochs, runs), dtype='float64')
for m_num, folder in enumerate([folder_names[0]]):
    for s_num, a_s in enumerate(a_shapes):

        for sim in range(n_sims_1):
            for epoch in range(epochs):
                for run in range(runs):
                    log_file_name = folder + "/" + test_name + str(sim) + "/log_files/"
                    filename = log_file_name + "res_tau_2_sim" + str(sim) + "_epoch" + str(epoch) + "_" + str(a_s) +"_run" + str(
                        run) + ".txt"
                    data = np.loadtxt(filename, dtype='float64', skiprows=1, delimiter=', ')
                    t_look = find_t_look(data)
                    t_look_policy = find_t_look_policy(data)
                    looking_ts_all_1[m_num*len(folder_names)+s_num, sim, epoch, run] = normalize_t(data, t_look) #( indexing?)
                    looking_ts_all_policy_1[m_num*len(folder_names)+s_num, sim, epoch, run] = normalize_t(data, t_look_policy) #( indexing?)
            print("Processed agent data of simulation " + str(sim))
looking_ts_all_2 = np.zeros((int(num_agents/2), num_sims, epochs, runs), dtype='float64')
looking_ts_all_policy_2 = np.zeros((int(num_agents/2), num_sims, epochs, runs), dtype='float64')
for m_num, folder in enumerate([folder_names[1]]):
    for s_num, a_s in enumerate(a_shapes):
        #
        for sim in range(num_sims):
            for epoch in range(epochs):
                for run in range(runs):
                    log_file_name = folder + "/" + test_name + str(sim) + "/log_files/"
                    filename = log_file_name + "res_tau_2_sim" + str(sim) + "_epoch" + str(epoch) + "_" + str(a_s) +"_run" + str(
                        run) + ".txt"
                    data = np.loadtxt(filename, dtype='float64', skiprows=1, delimiter=', ')
                    t_look = find_t_look(data)
                    t_look_policy = find_t_look_policy(data)
                    looking_ts_all_2[m_num*len(folder_names)+s_num, sim, epoch, run] = normalize_t(data, t_look) #( indexing?)
                    looking_ts_all_policy_2[m_num*len(folder_names)+s_num, sim, epoch, run] = normalize_t(data, t_look_policy) #( indexing?)
            print("Processed agent data of simulation " + str(sim))


# Means etc.

# looking_ts_all_mean = np.mean(np.mean(looking_ts_all_policy, axis= 3), axis= 1)
# looking_ts_all_sd = np.std(np.mean(looking_ts_all_policy, axis= 3), axis= 1)

looking_ts_all_1_mean = np.mean(np.mean(looking_ts_all_policy_1, axis= 3), axis= 1)
looking_ts_all_1_sd = np.std(np.mean(looking_ts_all_policy_1, axis= 3), axis= 1)
looking_ts_all_2_mean = np.mean(np.mean(looking_ts_all_policy_2, axis= 3), axis= 1)
looking_ts_all_2_sd = np.std(np.mean(looking_ts_all_policy_2, axis= 3), axis= 1)
# (combine the two ...)
looking_ts_all_mean = np.concatenate((looking_ts_all_1_mean, looking_ts_all_2_mean), axis=0)
looking_ts_all_sd = np.concatenate((looking_ts_all_1_sd, looking_ts_all_2_sd), axis=0)


# Visualization
# Plot the gaze-behavior

# Nice color definitions
colors = [(0.368, 0.507, 0.71), (0.881, 0.611, 0.142),
          (0.56, 0.692, 0.195), (0.923, 0.386, 0.209),
          (0.528, 0.471, 0.701), (0.772, 0.432, 0.102),
          (0.364, 0.619, 0.782), (0.572, 0.586, 0.) ]

epochs_ = range(epochs)

for i in range(num_agents):
    x_ = epochs_
    y_ = looking_ts_all_mean[i]
    err_ = looking_ts_all_sd[i]
    line_ = plt.plot(x_, y_, marker='.' if i < 2 else 'd', color=colors[0 if i % 2 == 0 else 3], label=labels[i],
                      linestyle='dashed' if i > 1 else '-')

    # (with sd/shading as well)
    plt.fill_between(x_, y_ - err_, y_ + err_, alpha=0.5, facecolor=colors[0 if i%2==0 else 3])

    # create legend for the four lines
    plt.legend()


# Event boundaries
plt.plot([0, epochs], [0.0, 0.0], 'k')
plt.plot([0, epochs], [0.69, 0.69], 'k')
plt.plot([0, epochs], [1.7, 1.7], 'k')
plt.plot([0, epochs], [2.7, 2.7], 'k')
plt.xlim([0, epochs])
plt.ylim([-0.5, 3.0])
plt.yticks([0.35, 1.25, 2.25], ('e_random', 'e_transport', 'e_reach'))

plt.title('Time t of first activation of pi_patient')
plt.xlabel('# training phases')

# where to save
if num_sims != n_sims_1:
    plt.savefig(folder_names[0] + "/" + v + "_gaze_tau2_" + label_add + str(n_sims_1) + "_" + str(num_sims) +
                "sims_n.png", bbox_inches='tight')
else:
    plt.savefig(folder_names[0] + "/" + v + "_gaze_tau2_" + label_add + str(num_sims) + "sims_n.png",
                bbox_inches='tight')


# Use this to just visualize the plot
plt.show()


