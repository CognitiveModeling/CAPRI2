# Plot of gaze behavior (gaze arrival times) for all agents

import numpy as np

# Define path to result files
# Folder names to various conditions, with and without adaptation or action effects

test_name = "res_tau_2_sim"

#foldername = "Experiments/ResAblationTimeHorizon-test_old_0_4v.1.1-3-adashape-t0-0.4-t35-0"

foldername = "Experiments/ResAblationTimeHorizon-test_old_0_4v.1.1-3-adashape-t0-0.4-t35-c-2"

foldername = "Experiments/Test1"

v = "Test1"

# Number and shapes (and initial agency estimates) of agents
a_shapes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
num_agents = len(a_shapes)

nr_runs = 12

sims = range(20)
num_sims = len(sims)

# Define indices of data, where to find which information
event_t = 1 # index of e(t)
policy_t = 2 # index of pi(t)

epochs = 30

# For one result file determine the point t when pi(t) = pi_p

def find_t_look_policy(data):
    for t in range(len(data)):
        if data[t, policy_t] == 1:
            return t
    return 270

# We can also determine gaze at the AOI of the reaching target. This can be done for example by determining if the system
# looks at the patient or at the agent, when it is already transporting the patient, and is, thus, closer than a
# predefined distance threshold

def find_t_look(data):
    for t in range(len(data)):
        if data[t, policy_t] == 1 or (data[t, event_t] == 3 and data[t, policy_t] == 0):
            return t
    return 270

def normalize_t(data, t_look):
    return  (270 - t_look)/100

# Read in agents testing data and compute time t of first goal-predictive gaze

looking_ts_all = np.zeros((num_agents, num_sims, epochs, nr_runs), dtype='float64')
looking_ts_all_policy = np.zeros((num_agents, num_sims, epochs, nr_runs), dtype='float64')
for s_num, a_s in enumerate(a_shapes):
    for sim, simulation in enumerate(sims):
        for epoch in range(epochs):
            for run in range(nr_runs):
                log_file_name = foldername + "/" + test_name + str(simulation) + "/log_files/"
                filename = log_file_name + "res_tau_2_sim" + str(simulation) + "_epoch" + str(epoch) + "_" + str(a_s) +"_run" + str(
                    run) + ".txt"

                data = np.loadtxt(filename, dtype='float64', skiprows=1, delimiter=', ')
                t_look = find_t_look(data)
                t_look_policy = find_t_look_policy(data)
                looking_ts_all[s_num , sim, epoch, run] = normalize_t(data, t_look)
                looking_ts_all_policy[s_num ,sim, epoch, run] = normalize_t(data, t_look_policy)
        print("Processed agent data of simulation " + str(simulation))


# (as before) 0.2 and 0.8 hand and claw with sd, etc.
looking_ts_hand_policy = looking_ts_all_policy[2]
looking_ts_claw_policy = looking_ts_all_policy[8]

looking_ts_all_mean = np.mean(np.mean(looking_ts_all_policy, axis= 3), axis= 1)

# Compute mean
looking_ts_hand_mean = np.mean(np.mean(looking_ts_hand_policy, axis=2), axis = 0)
looking_ts_hand_sd = np.std(np.mean(looking_ts_hand_policy, axis=2), axis=0)
looking_ts_claw_mean = np.mean(np.mean(looking_ts_claw_policy, axis=2), axis=0)
looking_ts_claw_sd = np.std(np.mean(looking_ts_claw_policy, axis=2), axis=0)

# Plot the gaze-behavior

# Nice color definitions
colors = [(0.368, 0.507, 0.71), (0.881, 0.611, 0.142),
          (0.56, 0.692, 0.195), (0.923, 0.386, 0.209),
          (0.528, 0.471, 0.701), (0.772, 0.432, 0.102),
          (0.364, 0.619, 0.782), (0.572, 0.586, 0.) ]

import matplotlib.pyplot as plt

epochs_ = range(epochs)

# # hand-agent data
# x1 = epochs_
# y1 = looking_ts_hand_mean
# err1 = looking_ts_hand_sd
# line1= plt.plot(x1, y1, color=colors[0], linewidth=8)
# plt.fill_between(x1, y1-err1, y1+err1, alpha=0.5, facecolor=colors[0])
#
# # claw-agent data
# x2 = epochs_
# y2 = looking_ts_claw_mean
# err2= looking_ts_claw_sd
# line2 = plt.plot(x2, y2, color=colors[3], linewidth=8)
# plt.fill_between(x2, y2-err2, y2+err2, alpha=0.5, facecolor=colors[3])
#
# # (Todo: colors and legend etc.? ...)
#
# # (copied from before ...)
# colormap = plt.cm.get_cmap('viridis', num_agents)  # Choose a colormap
#
# sm = plt.cm.ScalarMappable(cmap=colormap)
# sm.set_clim(min(a_shapes), max(a_shapes)) #(??)
# plt.colorbar(sm, label='Agents')
#
# # Todo: new: new lines, etc.? ...
# for i in range(num_agents):
#     x_ = epochs_
#     y_ = looking_ts_all_mean[i]
#
#     # (?)
#     color = colormap(i / num_agents)  # Assign color based on column number
#
#     #line_ = plt.plot(x_, y_, linewidth=8) # (color??)
#     line_ = plt.plot(x_, y_, linewidth=8, color=color)  # (color??)
#
# #for ep in range(num_agents):
# #    color = colormap(ep / num_agents)  # Assign color based on column number
# #    plt.plot(s_c_mean[ep,:], color=color, label=f"Epoch {ep}")
#
# ##sm = cm.ScalarMappable(cmap=colormap)
# ##sm.set_clim(0, num_epochs)
# #plt.colorbar(sm, label='Agents') #(??)
#
#
# # Event boundaries
# plt.plot([0, epochs], [0.0, 0.0], 'k')
# plt.plot([0, epochs], [0.69, 0.69], 'k')
# plt.plot([0, epochs], [1.7, 1.7], 'k')
# plt.plot([0, epochs], [2.7, 2.7], 'k')
# plt.xlim([0, epochs])
# plt.ylim([-0.5, 3.0])
# plt.yticks([0.35, 1.25, 2.25], ('e_random', 'e_transport', 'e_reach'))
# # plt.legend((line1[0], line2[0]), ('Hand', 'Claw'))
# # plt.title('Time t of first activation of pi_patient')
# # plt.xlabel('# training phases')
# #
# # plt.savefig(foldername + "/gaze_tau2.png", bbox_inches='tight')
# # plt.savefig(foldername + "/" + v + "_gaze_tau2_" + str(num_sims) + "sims.png", bbox_inches='tight')
# #
# # # Use this to just visualize the plot
# # plt.show()
#
# # Uncomment this, to store plot in tikz formatz
# #import tikzplotlib
# #tikzplotlib.save("resultPlots/tikz_gaze_tau2.tex")
#
# # Todo: new: without the old legend ...
#
# plt.title('Time t of first activation of pi_patient')
# plt.xlabel('# training phases')
#
# #plt.savefig(foldername + "/gaze_tau2.png", bbox_inches='tight')
# plt.savefig(foldername + "/" + v + "_gaze_tau2_" + str(num_sims) + "sims_n.png", bbox_inches='tight')
#
# # Use this to just visualize the plot
# plt.show()
#
# #plt.legend((line1[0], line2[0]), ('Hand', 'Claw'))
#
# #plt.savefig(foldername + "/" + v + "_gaze_tau2_" + str(num_sims) + "sims_0.png", bbox_inches='tight')
# #plt.show()
#
#
#
# New, other figure, without the sd-shadow, etc.?

colormap = plt.cm.get_cmap('viridis', num_agents)  # Choose a colormap

sm = plt.cm.ScalarMappable(cmap=colormap)
sm.set_clim(min(a_shapes), max(a_shapes))
plt.colorbar(sm, label='Agents')

# Lines for each agent
for i in range(num_agents):
    x_ = epochs_
    y_ = looking_ts_all_mean[i]

    color = colormap(i / num_agents)  # Assign color based on column number

    line_ = plt.plot(x_, y_, linewidth=8, color=color)

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

plt.savefig(foldername + "/" + v + "_gaze_tau2_" + str(num_sims) + "sims_v2.png", bbox_inches='tight')

# also save it as tikz ...
import tikzplotlib
tikzplotlib.save(foldername + "/" + v + "_gaze_" + str(num_sims) + "sims_v2.tex")

# Use this to just visualize the plot
plt.show()


# # New, other figure, without axis label etc.? ...
#
# colormap = plt.cm.get_cmap('viridis', num_agents)  # Choose a colormap
#
# sm = plt.cm.ScalarMappable(cmap=colormap)
# sm.set_clim(min(a_shapes), max(a_shapes)) #(??)
# plt.colorbar(sm, label='Agents')
#
# # Todo: new: new lines, etc.? ...
# for i in range(num_agents):
#     x_ = epochs_
#     y_ = looking_ts_all_mean[i]
#
#     # (?)
#     color = colormap(i / num_agents)  # Assign color based on column number
#
#     #line_ = plt.plot(x_, y_, linewidth=8) # (color??)
#     line_ = plt.plot(x_, y_, linewidth=8, color=color)  # (color??)
#
# # Event boundaries
# plt.plot([0, epochs], [0.0, 0.0], 'k')
# plt.plot([0, epochs], [0.69, 0.69], 'k')
# plt.plot([0, epochs], [1.7, 1.7], 'k')
# plt.plot([0, epochs], [2.7, 2.7], 'k')
# plt.xlim([0, epochs])
# plt.ylim([-0.5, 3.0])
# plt.yticks([0.35, 1.25, 2.25], ('e_random', 'e_transport', 'e_reach'))
#
# #plt.title('Time t of first activation of pi_patient')
# plt.xlabel('# training phases')
#
# plt.savefig(foldername + "/" + v + "_gaze_tau2_" + str(num_sims) + "sims_v_notitle.png", bbox_inches='tight')
#
# # todo: new: also save it as tikz? ...
# #import tikzplotlib
# #tikzplotlib.save(foldername + "/" + v + "_gaze_" + str(num_sims) + "sims_v_notitle.tex")
#
# plt.show()
#
#
# colormap = plt.cm.get_cmap('viridis', num_agents)  # Choose a colormap
#
# sm = plt.cm.ScalarMappable(cmap=colormap)
# sm.set_clim(min(a_shapes), max(a_shapes)) #(??)
# plt.colorbar(sm, label='Agents')
#
# # Todo: new: new lines, etc.? ...
# for i in range(num_agents):
#     x_ = epochs_
#     y_ = looking_ts_all_mean[i]
#
#     # (?)
#     color = colormap(i / num_agents)  # Assign color based on column number
#
#     #line_ = plt.plot(x_, y_, linewidth=8) # (color??)
#     line_ = plt.plot(x_, y_, linewidth=8, color=color)  # (color??)
#
# # Event boundaries
# plt.plot([0, epochs], [0.0, 0.0], 'k')
# plt.plot([0, epochs], [0.69, 0.69], 'k')
# plt.plot([0, epochs], [1.7, 1.7], 'k')
# plt.plot([0, epochs], [2.7, 2.7], 'k')
# plt.xlim([0, epochs])
# plt.ylim([-0.5, 3.0])
# plt.yticks([0.35, 1.25, 2.25], ('e_random', 'e_transport', 'e_reach'))
#
# plt.title('Time t of first activation of pi_patient')
# #plt.xlabel('# training phases')
#
# plt.savefig(foldername + "/" + v + "_gaze_tau2_" + str(num_sims) + "sims_v_noxlabel.png", bbox_inches='tight')
#
# # todo: new: also save it as tikz? ...
# #import tikzplotlib
# #tikzplotlib.save(foldername + "/" + v + "_gaze_" + str(num_sims) + "sims_v_noxlabel.tex")
#
# plt.show()

# # TODO: (Check version etc.? - plot ...)
# # TODO: Currently still a problem, plots come out looking pretty bad? ...
#
# # New, with a figure/axis to make sure it works? ...
# fig, ax = plt.subplots()
# #fig = plt.figure()
# #ax = fig.add_subplot(111)
#
# colormap = plt.cm.get_cmap('viridis', num_agents)  # Choose a colormap
#
# sm = plt.cm.ScalarMappable(cmap=colormap)
# sm.set_clim(min(a_shapes), max(a_shapes)) #(??)
# plt.colorbar(sm, label='Agents', ax=ax) #(??)
#
# # Todo: new: new lines, etc.? ...
# for i in range(num_agents):
#     x_ = epochs_
#     y_ = looking_ts_all_mean[i]
#
#     # (?)
#     color = colormap(i / num_agents)  # Assign color based on column number
#
#     #line_ = plt.plot(x_, y_, linewidth=8) # (color??)
#     line_ = plt.plot(x_, y_, linewidth=8, color=color)  # (color??)
#
# # Event boundaries
# plt.plot([0, epochs], [0.0, 0.0], 'k')
# plt.plot([0, epochs], [0.69, 0.69], 'k')
# plt.plot([0, epochs], [1.7, 1.7], 'k')
# plt.plot([0, epochs], [2.7, 2.7], 'k')
# plt.xlim([0, epochs])
# plt.ylim([-0.5, 3.0])
# plt.yticks([0.35, 1.25, 2.25], ('e_random', 'e_transport', 'e_reach'))
#
# plt.title('Time t of first activation of pi_patient')
# plt.xlabel('# training phases')
#
# #plt.savefig(foldername + "/gaze_tau2.png", bbox_inches='tight')
# plt.savefig(foldername + "/" + v + "_gaze_tau2_" + str(num_sims) + "sims_v2.png", bbox_inches='tight')
#
#
# # todo: new: also save it as tikz? ...
# #import tikzplotlib
# #tikzplotlib.save(foldername + "/" + v + "_gaze_" + str(num_sims) + "sims_v2.tex")
#
#
# # Use this to just visualize the plot
# plt.show()