# Plot the infant data and model predictions

import numpy as np
import matplotlib.pyplot as plt
#import tikzplotlib

# actual data (infant data: Hand: Adam & Elsner (2020) PlosOne, https://doi.org/10.1371/journal.pone.0240165,
# Claw: Adam et al. (2021) Frontiers, https://doi.org/10.3389/fpsyg.2021.695550)

# 	                6 mo Mean	6 mo SE	7 mo Mean	7 mo SE	11 mo Mean	11 mo SE	18 mo Mean	18 mo SE
# hand with effect	-40,91	72	252,09	55	434,6	51
# hand w/o effect	6,53	64	9,92	80	352,38	68
# claw with effect			-2,85	83,7	265,8	92,8	241,2	85,5
# claw w/o effect			-392,2	68,5	-20,7	58,6	221,06	74,8
data = np.array([[-40.91, 72, 252.09, 55, 434.6, 51, np.nan, np.nan],
				 [6.53, 64, 9.92, 80, 352.38, 68, np.nan, np.nan],
				 [np.nan, np.nan, -2.85, 83.7, 265.8, 92.8, 241.2, 85.5],
				 [np.nan, np.nan, -392.2, 68.5, -20.7, 58.6, 221.06, 74.8]])
print(data)

# Comparison plot, with infant data and model

# (copied from other script)
def find_t_look_policy(data):
    for t in range(len(data)):#(270):
        if data[t, policy_t] == 1:
            return t
    return 270
def find_t_look(data):
    for t in range(len(data)):#(270):
        if data[t, policy_t] == 1 or (data[t, event_t] == 3 and data[t, policy_t] == 0):
            return t
    return 270

def normalize_t(data, t_look):
    return  (270 - t_look)/100

def normalize_t_to_one(data, t_look):
    return  (270 - t_look)/270 #(todo: ??)


# models
# - with and without action effects
# - with or without shape adaptation

test_name = "res_tau_2_sim"
folder_name_nonad_full = "Experiments/ResAblationTimeHorizon-test_old_0_4v.1.1-3-adashape-t0-0.4-t34-0-1"
folder_name_adapt_full = "Experiments/ResAblationTimeHorizon-test_old_0_4v.1.1-3-adashape-t0-0.4-t34-1"
folder_name_nonad_cut = "Experiments/ResAblationTimeHorizon-test_old_0_4v.1.1-3-adashape-t0-0.4-t35-c-0"
folder_name_adapt_cut = "Experiments/ResAblationTimeHorizon-test_old_0_4v.1.1-3-adashape-t0-0.4-t35-c-2"
model_folders = [folder_name_nonad_full, folder_name_adapt_full, folder_name_nonad_cut, folder_name_adapt_cut]

# either with or without shape adaptation
model_folders = [folder_name_adapt_full, folder_name_adapt_cut]

epochs = [2, 4, 13, 26]

runs = range(12)
num_runs = len(runs)
sims = range(20)
num_sims = len(sims)

# Processing
a_shapes = ["hand","claw"]
a_shapes = [0.4, 0.8]
event_t = 1 # index of e(t)
policy_t = 2 # index of pi(t)
looking_ts_all = np.zeros((len(model_folders), num_sims, len(a_shapes), len(epochs), num_runs))
looking_ts_all_policy = np.zeros((len(model_folders), num_sims, len(a_shapes), len(epochs), num_runs))
for model, foldername in enumerate(model_folders):
	for sim_idx, sim in enumerate(sims):
		for agent, a_shape in enumerate(a_shapes):
			for ep, epoch in enumerate(epochs):
				for run_idx, run in enumerate(runs):
					log_file_name = foldername + "/" + test_name + str(sim) + "/log_files/"
					filename = log_file_name + "res_tau_2_sim" + str(sim) + "_epoch" + str(epoch) + "_" + str(
						a_shape) + "_run" + str(run) + ".txt"
					data_0 = np.loadtxt(filename, dtype='float64', skiprows = 1, delimiter= ', ')
					t_look = find_t_look(data_0)
					t_look_policy = find_t_look_policy(data_0)
					looking_ts_all[model, sim_idx, agent, ep, run_idx] = normalize_t(data, t_look)
					looking_ts_all_policy[model, sim_idx, agent, ep, run_idx] = normalize_t(data, t_look_policy)
looking_ts_all_mean = np.mean(np.mean(looking_ts_all_policy, axis= 3), axis= 1)
looking_ts_hand_policy = looking_ts_all_policy[:,:,0,:,:]
looking_ts_claw_policy = looking_ts_all_policy[:,:,1,:,:]
looking_ts_hand_mean = np.mean(np.mean(looking_ts_hand_policy, axis=3), axis = 1)
looking_ts_hand_sd = np.std(np.mean(looking_ts_hand_policy, axis=3), axis=1)
looking_ts_claw_mean = np.mean(np.mean(looking_ts_claw_policy, axis=3), axis=1)
looking_ts_claw_sd = np.std(np.mean(looking_ts_claw_policy, axis=3), axis=1)

# (hand and claw, each, for two models - with or without adaptation/action effects)
looking_ts_all_1 = np.zeros((num_sims, len(model_folders)*len(a_shapes), len(epochs), num_runs))
looking_ts_all_policy_1 = np.zeros((num_sims, len(model_folders)*len(a_shapes), len(epochs), num_runs))
for sim_idx, sim in enumerate(sims):
	i = -1
	for agent, a_shape in enumerate(a_shapes):
		for model, foldername in enumerate(model_folders):
			i = i+1
			for ep, epoch in enumerate(epochs):
				for run_idx, run in enumerate(runs):#range(num_runs):
					log_file_name = foldername + "/" + test_name + str(sim) + "/log_files/"
					filename = log_file_name + "res_tau_2_sim" + str(sim) + "_epoch" + str(epoch) + "_" + str(
						a_shape) + "_run" + str(run) + ".txt"
					data_0 = np.loadtxt(filename, dtype='float64', skiprows = 1, delimiter= ', ')
					t_look = find_t_look(data_0)
					t_look_policy = find_t_look_policy(data_0)
					looking_ts_all_1[sim_idx, i, ep, run_idx] = normalize_t(data, t_look)
					looking_ts_all_policy_1[sim_idx, i, ep, run_idx] = normalize_t(data, t_look_policy)

looking_ts_all_mean = np.mean(np.mean(looking_ts_all_policy_1, axis= 3), axis=0)
print(looking_ts_all_mean)
print(looking_ts_all_mean.shape)

# Generate the plot

fig, ax = plt.subplots()
ax.plot([0,1,2], data[0,0:6:2], label='hand with effect', marker='o')
ax.plot([0,1,2], data[1,0:6:2], label='hand w/o effect', marker='o')
ax.plot([1,2,3], data[2,2:8:2], label='claw with effect', marker='o')
ax.plot([1,2,3], data[3,2:8:2], label='claw w/o effect', marker='o')
ax.set_ylabel('Gaze arrival times (ms)')
ax.set_xticks([0,1,2,3], ['6 mo', '7 mo', '11 mo', '18 mo'])
#ax.grid(True, which='major', axis='y')
fig.legend(loc='lower right', bbox_to_anchor=(0.89,0.12))
ax2 = ax.twinx()
for i in range(len(model_folders)*len(a_shapes)):
	ax2.plot([0, 1, 2, 3], looking_ts_all_mean[i], marker='d', linestyle='--')
ax2.set_ylim([0.7,2.7]) #(?)
ax2.set_yticks([1.25, 1.7, 2.25, 2.7], ('e_transport', '', 'e_reach', ''))
#plt.title('Gaze Arrival Times for Agents/Actions')
plt.savefig("gaze_arrival_times_comparison_infants_n4_epochs" + str(epochs) + ".png", bbox_inches="tight")
#tikzplotlib.save("gaze_arrival_times_comparison_infants_epochs" + str(epochs) + ".tex")
plt.show()



