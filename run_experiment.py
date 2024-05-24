import gym
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import os

import interaction_gym
import event_inference as event

import torch
import torch.optim as optim
from torch.autograd import Variable


# Global parameter settings
epsilon_start = 0.01
epsilon_end = 0.001
epsilon_dynamics = 0.001
random_Colors = True
percentage_reaching = 1.0/3.0


start_pos = [[-1,-1],[-1,-1],[-1,-1],
             [1,-1],[1,-1],[1,-1],
             [-1,1],[-1,1],[-1,1],
             [1,1],[1,1],[1,1]]
transport_goal_pos_signs = [[-1,1],[1,-1],[-1,-1],
                            [-1,-1],[1,1],[1,-1],
                            [1,1],[-1,-1],[-1,1],
                            [1,-1],[-1,1],[1,1]]
nr_run = 12 # 4 corners, 3 directions each
patient_colors = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5, 0.15, 0.95] #


folder_name = "Test01"

last = 100
cut = False

model_folder_name = 'Experiments/...'


a_shapes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

nr_epoch = 30
n_sequence = 100
epochs = range(0, nr_epoch)

simulations = range(0,20)
nr_sim = len(simulations)

saved_model = False # True

l_r = 0.00001
m_r = 0.3
nr_update_steps = 0 #5


def tst_run(directory_name, setting_name, event_system, interaction_env,
             a_shape_basis, simulation_num, epoch_num, run_num, time_horizon,
             a_shape, file_name_addition='', learning_rate = 0.0001):
    """
    Performs one test run and logs the results
    :param directory_name: name of target folder for log files
    :param setting_name: name of this simulation setting
    :param event_system: instance of trained event inference system
    :param interaction_env: instance of agent-patient interaction gym
    :param a_shape_basis: agent in this run
    :param a_shape: agent shape/agency estimate of the system
    :param simulation_num: number of this simulation
    :param epoch_num: number of training phase
    :param run_num: number of runs in this testing phase
    :param time_horizon: tau
    :param file_name_addition: extra string added at the end of file name
    :param learning_rate: learning rate for the agency estimation adaptation (gradient descent)
    """

    filename = directory_name + '/' + setting_name + str(simulation_num) + '_epoch' + str(
        epoch_num) + "_" + str(a_shape_basis) + file_name_addition + "_run" + str(run_num) + '.txt'
    file = open(filename, 'w')
    file.write('t, Event, Policy, P(still), P(random), P(reach), P(transport), o(t), s_a \n')

    claw = True if a_shape_basis > 0.5 else False
    o_t = interaction_env.reset_to_grasping(claw=claw, agent_color=a_shape * 10 ,
                                            patient_color=patient_colors[run_num],
                                            start_pos_signs=start_pos[run_num],
                                            transport_goal_sign=transport_goal_pos_signs[run_num])


    pi_t = np.array([0.0, 0.0, 1.0])  # During testing the system starts with no fixation
    event_system.reset()

    s_a = torch.autograd.Variable(torch.from_numpy(np.array(o_t[3])), requires_grad = True)
    s_p = torch.autograd.Variable(torch.from_numpy(np.array(o_t[14])), requires_grad=False)

    context_optimizer = optim.SGD([s_a], lr=learning_rate, momentum=m_r)

    done_t = False
    t = -1
    while not done_t:
        t = t + 1

        o_t, r_t, done_t, info_t, event_over = interaction_env.step(pi_t)

        # stop after the reach?
        if cut and t == last-1:
            event_over = True
            done_t = True

        # 2. step: Infer event model and next action
        pi_t, probs, s_a, s_p, loss = event_system.step(o_t=o_t, pi_t=pi_t,
                                       training=False,
                                       P_e_i_t = model.P_ei,
                                       o_t_minus_1 = model.o_t_minus_1,
                                       done=done_t,
                                       e_i=info_t,
                                       tau=time_horizon,
                                       event_over = event_over,
                                       context_optimizer = context_optimizer,
                                       x_pi_sa = s_a, x_pi_sp = s_p,
                                       nr_update_steps = nr_update_steps)

        if s_a <= 0.01: #< 1e-5:
             s_a.data = torch.tensor(0.01)#1e-5)

        # 3. step: Log data

        # save observations/env state
        observations[sim, epoch_num, run_num, t, :] = o_t

        obs_str = ', '.join(map(str, o_t))
        file.write(
           str(t) + ', ' + str(info_t) + ', ' + str(np.argmax(pi_t)) + ', ' + str(probs[0].detach().numpy()) +
           ', ' + str(probs[1].detach().numpy()) + ', ' + str(probs[2].detach().numpy()) + ', ' +
           str(probs[3].detach().numpy()) + ', ' + obs_str +
           ', ' + str(s_a.detach().numpy()) +
           '\n')

    file.close()
    interaction_env.close()

    return s_a.detach().numpy()

# varying training
conditions = [(random.gauss,2,1),
                    (random.gauss,3,1.5),
                    (random.uniform,0.2,6),
                    (random.uniform,0,8),
                    (random.uniform,0,10)]

tau = 2
test_name = 'res_tau_2_sim'

s_a_all = np.zeros((nr_sim, len(epochs), nr_run, len(a_shapes)))


observations = np.zeros((nr_sim,nr_epoch,nr_run,301,18))
if not saved_model:
    observations_training = np.zeros((nr_sim,len(epochs), n_sequence,500,18))
    parameters_training = np.zeros((nr_sim,len(epochs), n_sequence,500,3))

for sim, simulation in enumerate(simulations):
    seed = simulation
    #seed = seeds[simulation]
    model = event.CAPRI(epsilon_start=epsilon_start, epsilon_dynamics=epsilon_dynamics,
                        epsilon_end=epsilon_end, no_transition_prior=0.9, dim_observation=18,
                        num_policies=3, num_models=4, r_seed=seed, sampling_rate=2)
    env = interaction_gym.InteractionEventGym(sensory_noise_base=1.0, sensory_noise_focus=0.01,
                                              r_seed=seed, randomize_colors=random_Colors,
                                              percentage_reaching=percentage_reaching)#,
    log_file_name = folder_name + '/' + test_name + str(simulation) + '/log_files/'
    os.makedirs(log_file_name, exist_ok=True)

    for ep, epoch in enumerate(epochs):

        # training a new model or using a pretrained model
        if not saved_model:

            if conditions is not None and len(conditions) >= 2:
                # conditions for varying training
                cond_n = int(epoch / (nr_epoch / len(conditions)))
                cond = conditions[cond_n]
                print("Condition: ", cond_n, " ", str(cond))
            # TRAINING PHASE:
            # do 100 training event sequences per phase
            for sequence in range(n_sequence):
                # reset environment to new event sequence
                try:
                    observation = env.reset(cond)
                except:
                    observation = env.reset()
                # sample one-hot-encoding of policy pi(0)
                policy_t = np.array([0.0, 0.0, 0.0])
                policy_t[random.randint(0, 2)] = 1.0
                done = False
                t = -1
                while not done:
                    t = t + 1

                    # perform pi(t) and receive new observation o(t)
                    observation, reward, done, info, event_over = env.step(policy_t)
                    policy_t, P_ei = model.step(o_t=observation, pi_t=policy_t, training=True,
                                                done=done, e_i=info, P_e_i_t = model.P_ei,
                                                o_t_minus_1 = model.o_t_minus_1, \
                                                event_over=event_over)

                    # save observations
                    observations_training[sim, ep, sequence, t, :] = observation

                    parameters_training[sim, ep, sequence, t, :] = [done, info, event_over]

            # (save observations for this simulation and epoch)
            np.save(folder_name + "/observations_training_sim" + str(simulation) + "_epoch" + str(epoch),
                    observations_training[sim, ep, :, :, :])
            np.save(folder_name + "/parameters_training_sim" + str(simulation) + "_epoch" + str(epoch),
                    parameters_training[sim, ep, :, :, :])

            print("Epoch ", str(epoch), " training finished")

            model.save(folder_name + '/' + test_name + str(simulation), epoch)

        else:
            # load pretrained model
            model.load(directory=model_folder_name + '/' + test_name + str(simulation), epoch=epoch)


        # TESTING PHASE:
        shape_current = a_shapes.copy()
        s_a_basis = a_shapes.copy()

        for run in range(nr_run):

            print("Run: ", str(run))

            for s_num, s_a in enumerate(a_shapes):
                shape_current[s_num] = tst_run(directory_name=log_file_name,
                        setting_name=test_name, event_system=model,
                        interaction_env=env,
                        a_shape_basis=s_a_basis[s_num],
                        simulation_num=simulation,
                        epoch_num = epoch, run_num = run, time_horizon = tau,
                        a_shape = shape_current[s_num], learning_rate=l_r)
                s_a_all[sim, ep, run, s_num] = shape_current[s_num]

        print("Epoch ", str(epoch), " testing finished")

    print("Simulation ", str(simulation), " finished")


    # Save the shapes
    np.save(folder_name + '/s_a_all_' + str(simulation) + '.npy' , s_a_all[sim,:,:,:])


    # save observations
    np.save(folder_name + "/observations_sim" + str(simulation) + ".npy", observations[sim])

np.save(folder_name + "/observations.npy", observations)

