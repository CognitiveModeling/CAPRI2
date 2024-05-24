"""
Cognitive Action PRediction and Inference in Infants system (CAPRI²)

CAPRI learns probabilistic event schemata for observed sensorimotor interactions.
Each schema has a particular structure:
- It has a starting condition encoding where in the space of observation this event typically starts
- It has an event dynamics model, encoding how sensory information changes during the event
- It has an ending condition that encodes at which observation this event typically ends.

During training these event schemata are learned using supervised training. During testing, the
event schemata are inferred based on new observations and likelihoods produced by the event schema.

It can also infer desired policies. In this implementation the system infers its policy with
the aim to minimize predicted uncertainty about future events and event boundaries. We use this to
model the goal-predictive gaze in infants.

CAPRI² incorporates a retrospective inference mechanism to update the agency estimation of the model.

For more information, see our papers:
Emergent Goal-Anticipatory Gaze in Infants via Event-Predictive Learning and Inference
(2021), C. Gumbsch, M. Adam, B. Elsner, & M.V. Butz
Infants Infer And Predict Coherent Event Interactions: Modeling Cognitive Development (2024), J.K. Theuer, N.N. Koch,
C. Gumbsch, B. Elsner, & M.V. Butz
"""

import numpy as np
import gaussian_networks as gn
import torch
import torch.optim as optim
from scipy.stats import multivariate_normal
from sampling_buffer import SamplingBuffer
import random
import os
from torch.autograd import Variable


class CAPRI:
    
    # ------------- INITIALIZATION OF EVENT-PREDICTIVE INFERENCE SYSTEM -------------
    def __init__(self, epsilon_start, epsilon_dynamics, epsilon_end, no_transition_prior,
                 dim_observation, num_policies, num_models, r_seed,
                 sampling_rate=5, buffer_capacity=1000, tanh_loss=True):
        """
        Initializes the class
        :param epsilon_start: learning rate of start condition networks
        :param epsilon_dynamics: learning rate of event dynamics networks
        :param epsilon_end: learning rate of ending condition networks
        :param no_transition_prior: prior for having no transition in events,
            i.e., P(e_i|e_j) for e_i =/= e_j
        :param dim_observation: dimensionality of sensory observation
        :param num_policies: number of possible policies
        :param num_models: number of event models
        :param r_seed: random seed
        :param sampling_rate: sampling rate of the replay buffer
        :param buffer_capacity: capacity of the replay buffer
        :param tanh_loss: tanh or NLL loss
        """
        torch.set_default_tensor_type(torch.DoubleTensor)

        # Set the dimensionality of observations, policies and models
        self.dim_observation = dim_observation
        self.num_policies = num_policies
        self.num_models = num_models

        # Set the random seed
        np.random.seed(r_seed)
        torch.manual_seed(r_seed)
        random.seed(r_seed)

        # Prior for P(e_i|e_j) for e_i =/= e_j
        assert 0 < no_transition_prior < 1
        self.no_transition_prior = no_transition_prior

        # Parameters for the replay buffers
        self.sampling_rate = sampling_rate

        # Keep track of last event and observation
        self.current_event = -1  # -1 means no event
        self.o_t_minus_1 = None


        """
        Create the event schemata.
        Every event schemata e_i is composed of three likelihood distributions:
        1. Starting condition P^{start}_{e_i}
        2. Event dynamics model P^{event}_{e_i}
        3. Ending condition P^{end}_{e_i}
        For every likelihood distribution we create a Gaussian Mixture Density Network,
        an optimizer to train the network, and a replay sampling buffer to enlarge the
        training data
        """

        # 1. Starting conditions (P^{start}_{e_i})
        self.P_start_networks = np.empty(num_models, dtype=gn.Multivariate_Gaussian_Network)
        self.P_start_optimizers = np.empty(num_models, dtype=optim.Optimizer)
        self.P_start_buffer = np.empty(num_models, dtype=SamplingBuffer)
        for i in range(num_models):
            # P^{start}_{e_i) (o(t) | pi(t-1))
            # -> The network takes pi(t-1) as an input and predicts mu and Sigma in
            #    space of observations
            self.P_start_networks[i] = gn.Multivariate_Gaussian_Network(num_policies, dim_observation).double()
            self.P_start_optimizers[i] = self.P_start_networks[i].get_optimizer(learning_rate=epsilon_start,
                                                                                momentum_term=0.0)
            buffer_seed = random.randint(-1000, 1000)  # Sample a random seed for the sampling buffer
            self.P_start_buffer[i] = SamplingBuffer(num_policies, dim_observation,
                                                    buffer_capacity, buffer_seed)

        # 2. Event dynamics model (P^{event}_{e_i})
        self.P_ei_networks = np.empty(num_models, dtype=gn.Multivariate_Gaussian_Network)
        self.P_ei_optimizers = np.empty(num_models, dtype=optim.Optimizer)
        self.P_ei_buffer = np.empty(num_models, dtype=SamplingBuffer)
        for i in range(num_models):
            # P^{event}_{e_i} (o(t) | pi(t-1), o(t-1))
            # -> The network takes pi(t-1) and o(t-1) as inputs and predicts mu
            #    and Sigma in space of observations
            self.P_ei_networks[i] = gn.Multivariate_Gaussian_Network(dim_observation + num_policies,
                                                                     dim_observation).double()
            self.P_ei_optimizers[i] = self.P_ei_networks[i].get_optimizer(learning_rate=epsilon_dynamics,
                                                                          momentum_term=0.0)
            buffer_seed = random.randint(-1000, 1000)  # Sample a random seed for the sampling buffer
            self.P_ei_buffer[i] = SamplingBuffer(dim_observation + num_policies, dim_observation,
                                                 buffer_capacity, buffer_seed)

        # 3. Event end networks (P^{end}_{e_i})
        self.P_end_networks = np.empty(num_models, dtype=gn.Multivariate_Gaussian_Network)
        self.P_end_optimizers = np.empty(num_models, dtype=optim.Optimizer)
        self.P_end_buffer = np.empty(num_models, dtype=SamplingBuffer)
        for i in range(num_models):
            # P^{end}_{e_i} (o(t) | pi(t-1), o(t-1))
            # -> The network takes pi(t-1) and o(t-1) as inputs and predicts mu
            #    and Sigma in space of observations
            self.P_end_networks[i] = gn.Multivariate_Gaussian_Network(dim_observation + num_policies,
                                                                      dim_observation).double()
            self.P_end_optimizers[i] = self.P_end_networks[i].get_optimizer(learning_rate=epsilon_end,
                                                                            momentum_term=0.0)
            buffer_seed = random.randint(-1000, 1000)  # Sample a random seed for the sampling buffer
            self.P_end_buffer[i] = SamplingBuffer(dim_observation + num_policies, dim_observation,
                                                  buffer_capacity, buffer_seed)

        # Store the trajectory over the course of one event
        self.observation_trajectory = []
        self.policy_trajectory = []

        # Inferred event probabilities P(e_i(t) | O(t), Pi(t))
        self.P_ei = np.ones(self.num_models)
        # At the start of inference every event is equally likely
        self.P_ei = self.P_ei * (1.0/self.num_models)
        
        # For calculating the loss
        self.tanh_loss = tanh_loss
        self.pinv_matrix = True
        
        # Store observations and pi over all time steps
        self.o_t_tensor = None
        self.o_t_minus_1_tensor = None
        self.x_pi_tensor = None
        self.x_pi_sa = None
        self.log_LH = torch.zeros(1)
 
       

    # ------------- MAIN STEP OF THE SYSTEM -------------
    def step(self, o_t, pi_t, training, P_e_i_t, o_t_minus_1, done, event_over,  e_i=-1, tau=2, context_optimizer=None,
             nr_update_steps=0, x_pi_sa=None, x_pi_sp=None):
        """
        Main step of the sensorimotor loop of the system at time t
        :param o_t: new observation o(t)
        :param pi_t: last applied policy pi(t-1)
        :param training: training vs testing phase?

        :param P_e_i_t: last event probability (tensor)
        :param o_t_minus_1: last observation o(t-1)
   
        :param done: is the current event sequence over with this step?
        
        :param event_over: is the current event over?
        :param e_i: supervised label for current event (required for training)
        :param tau: time horizon for active inference
        
        :param context_optimizer: gives the optimizer for the adaptation
        :prarm nr_update_steps: number of steps it is updated
        :param x_pi_sa: shape of the agent/agency estimate
        :param x_pi_sp: shape of the patient                                                                              
        :return: next policy pi(t), probabilities P(e_i(t)|O(t), Pi(t)) for all e_i, agent estimate, patient shape, loss
        """
        if training:
            # if the system is in supervised training it requires a label
            # for the current event
            assert 0 <= e_i < self.num_models
            # the event schemata are updated based on the new sensorimotor information
            self.update(o_t=o_t, pi_t=pi_t, e_i=e_i, done=done)
            # here the event probabilities are known
            p_ei = np.zeros(self.num_models)
            p_ei[e_i] = 1.0
            # perform the same policy, i.e., pi_(t) = pi_(t-1)
            return pi_t, p_ei
        else:
            # save these as global variables
            self.x_pi_sa = x_pi_sa  
            self.x_pi_sp = x_pi_sp

            # conduct the inference 
            p_ei = self.inference_gradients(o_t=o_t, o_t_minus_1=o_t_minus_1, pi_t=pi_t, P_e_i_t=P_e_i_t) 
  
            # compute for every policy expected free energy
            expected_fe = np.zeros(self.num_policies)
            pis = np.identity(self.num_policies)
            for p in range(self.num_policies):
                one_hot_pi = pis[p, :]
                expected_fe[p] = self.estimate_fe_for_policy(pi=one_hot_pi, tau=tau)
            # choose the policy with lowest free energy as next policy pi(t)
            best_p = np.argmin(expected_fe)
            pi_t_plus_1 = pis[best_p, :]
            
            # during testing the event probabilities are inferred
            # we use the inference_gradients method to keep the gradients  
            loss = 0
            # if the current event is over, get the current loss
            if event_over == True: 
                loss = self.get_sum_neg_log_LH()   
            # if the event sequence is over, update the agency estimate of the agent
            if done == 1: 
                loss = self._update_context(context_optimizer, nr_update_steps)
            return pi_t_plus_1, p_ei, self.x_pi_sa, self.x_pi_sp, loss
            
    # ------------- SUPERVISED TRAINING OF EVENT SCHEMATA -------------
    def update(self, o_t, pi_t, e_i, done):
        """
        Update step during supervised training
        :param o_t: current observation o(t)
        :param pi_t: last policy pi(t-1)
        :param e_i: supervised label for current event e(t)
        :param done: flag if event sequence ends after this sample
        """
        assert 0 <= e_i < self.num_models

        """
        There are four cases for possible updates
        1. Start of a new event sequence
            -> Update starting condition of current event
        2. Transition in events from e_i to e_j
            -> Update ending condition of e_i and starting
               condition of e_j
        3. Last time step of an event sequence
            -> Update ending condition of current event
        4. Staying in same event
            -> Update event dynamics model of current event
        """
        if self.o_t_minus_1 is None:
            assert self.current_event == -1
            # 1. case: First call, update P^{start}_{e_i} for the current event e_i
            self._update_start(o_t, pi_t, e_i)

            # add (pi(t-1), o(t))-pair to sampling buffer
            self.P_start_buffer[e_i].add_sample(pi_t, o_t)

            # update P^{start}_{e_i} additionally based on past samples
            for _ in range(self.sampling_rate):
                sample = self.P_start_buffer[e_i].draw_sample()
                self._update_start(o_t=sample[1], pi_t=sample[0], e_i=e_i)

            # Store current event e_i, the last observation o(t)
            # and previous policy pi(t-1)
            self.o_t_minus_1 = o_t
            self.current_event = e_i
            self.observation_trajectory.append(o_t)
            self.policy_trajectory.append(pi_t)

        elif self.current_event != e_i:
            # 2. case: transition from e_j to e_i

            # update P^{end}_{e_j}, store sample and perform sampling:
            # (all handled by update_end_trajectory-method)
            self._update_end_trajectory(o_t, self.current_event)

            # update P^{start}_{e_i}
            self._update_start(o_t, pi_t, e_i)
            # add new sample to replay buffer
            self.P_start_buffer[e_i].add_sample(pi_t, o_t)
            # Update P^{start}_{e_i} based on sampling previous input
            for _ in range(self.sampling_rate):
                sample = self.P_start_buffer[e_i].draw_sample()
                self._update_start(o_t=sample[1], pi_t=sample[0], e_i=e_i)

            # Since e_j is over, clear the trajectory memory
            self.policy_trajectory.clear()
            self.observation_trajectory.clear()

            # Store current event e_i, the last observation o(t)
            # and previous policy pi(t-1)
            self.o_t_minus_1 = o_t
            self.current_event = e_i
            self.observation_trajectory.append(o_t)
            self.policy_trajectory.append(pi_t)

        elif done:
            # 3. case: end of event sequence
            # only update P^{end}_{e_i}
            assert self.current_event == e_i
            self._update_end_trajectory(o_t, self.current_event)

            # Since e_i is over, clear the trajectory memory
            self.policy_trajectory.clear()
            self.observation_trajectory.clear()

            # Reset observation and event knowledge
            self.o_t_minus_1 = None
            self.current_event = -1
            
        else:
            # 4. case: Staying in same event
            assert self.current_event == e_i

            # update P^{event}_{e_i}
            input = np.append(self.o_t_minus_1, pi_t)
            self._update_dynamics(o_t=o_t, input=input, e_i=e_i)

            # add input output pair to sampling buffer:
            self.P_ei_buffer[e_i].add_sample(input, o_t)

            # update P^{event}_{e_i} based on sampling
            for _ in range(self.sampling_rate):
                sample = self.P_ei_buffer[e_i].draw_sample()
                self._update_dynamics(o_t=sample[1], input=sample[0], e_i=e_i)

            # store observations and policies in trajectory of the current event
            self.observation_trajectory.append(o_t)
            self.policy_trajectory.append(pi_t)
            self.o_t_minus_1 = o_t

    def update_batch(self, inp_tensor, target_tensor, e_i, component):
        """
        Updates the subnetworks based on batches of inputs and targets
        :param inp_tensor: Two-dimensional tensor of inputs (first dim = batch size)
        :param target_tensor: Two-dimensional tensor of nominal outputs (first dim = batch size)
        :param e_i: which event to update
        :param component: which component of the event 'start', 'dynamics', or 'end
        """

        if component == 'start':
            subnetwork = self.P_start_networks[e_i]
            optimizer = self.P_start_optimizers[e_i]
        elif component == 'dynamics':
            subnetwork = self.P_ei_networks[e_i]
            optimizer = self.P_ei_optimizers[e_i]
        else:
            assert component == 'end'
            subnetwork = self.P_end_networks[e_i]
            optimizer = self.P_end_optimizers[e_i]

        optimizer.zero_grad()
        out_tensor = subnetwork.forward(inp_tensor)
        loss = subnetwork.batch_loss_criterion(out_tensor, target_tensor, tanh=self.tanh_loss)
        loss.backward()
        optimizer.step()
        return loss.detach().item()

    def _update_start(self, o_t, pi_t, e_i):
        """
        Update P^{start}_{e_i} (o(t)|pi(t))
        :param o_t: current observation o(t)
        :param pi_t: last policy pi(t-1)
        :param e_i: the starting event e_i
        """
        x = torch.from_numpy(pi_t)
        y = torch.from_numpy(o_t)
        self.P_start_optimizers[e_i].zero_grad()
        output = self.P_start_networks[e_i].forward(x)
        loss = self.P_start_networks[e_i].loss_criterion(output, y, tanh=self.tanh_loss)
        loss.backward()
        self.P_start_optimizers[e_i].step()

    def _update_dynamics(self, o_t, input, e_i):
        """
        Update P^{event}_{e_i} (o(t)|o(t-1), pi(t-1))
        :param o_t: current observation o(t)
        :param input: input pair [o(t-1), pi(t-1)]
        :param e_i: currently unfolding event e_i
        """
        x = torch.from_numpy(input)
        y = torch.from_numpy(o_t)
        self.P_ei_optimizers[e_i].zero_grad()
        output = self.P_ei_networks[e_i].forward(x)
        loss = self.P_ei_networks[e_i].loss_criterion(output, y, tanh=self.tanh_loss)
        loss.backward()
        self.P_ei_optimizers[e_i].step()

    def _update_end(self, o_t, input, e_i):
        """
        Update P^{end}_{e_i} (o(t)|o(t-1), pi(t-1))
        :param o_t: current observation o(t)
        :param input: input pair (o(t-1), pi(t-1))
        :param e_i: currently unfolding event e_i
        """
        x = torch.from_numpy(input)
        y = torch.from_numpy(o_t)
        self.P_end_optimizers[e_i].zero_grad()
        output = self.P_end_networks[e_i].forward(x)
        loss = self.P_end_networks[e_i].loss_criterion(output, y, tanh=self.tanh_loss)
        loss.backward()
        self.P_end_optimizers[e_i].step()

    def _update_end_trajectory(self, o_t, e_i):
        """
        Update P^{end}_{e_i} multiple times on previous trajectory
        to enlarge training data. P^{end}_{e_i} (o(t)|o(t-j), pi(t-j))
        is trained for all j time steps in event e_i
        Sampling is also handled automatically
        :param o_t: observation o(t) at end of event
        :param e_i: currently ending event e_i
        """
        length = len(self.observation_trajectory)
        assert length == len(self.policy_trajectory)
        for i in range(length):
            o_i = self.observation_trajectory[i]
            pi_i = self.policy_trajectory[i]
            input = np.append(o_i, pi_i)
            self._update_end(o_t, input, e_i)
            self.P_end_buffer[e_i].add_sample(input, o_t)
            for _ in range(self.sampling_rate):
                sample = self.P_end_buffer[e_i].draw_sample()
                self._update_end(sample[1], sample[0], e_i)
                
                
    def _update_context(self, context_optimizer, nr_update_steps):
        """
        Update a given parameter using the negative log likelihood
        :param context_optimizer: optimizer that is used to update
        :param nr_update_steps: number of updates
        """
        loss = self.get_sum_neg_log_LH()
        # for each optimization step
        for i in range(nr_update_steps):
            # loss is the sum over the neg log LH
            loss = self.get_sum_neg_log_LH()
            context_optimizer.zero_grad()										 
            loss.backward(retain_graph=True)
            # one update step
            context_optimizer.step()
        return loss
    
    
    # ------------- EVENT INFERENCE -------------
    def inference(self, o_t, pi_t, o_t_minus_1, P_ei):
        """
        Infer the probabilities of events in the absence of
        explicit labels
        :param o_t: current observation o(t)
        :param pi_t: last policy pi(t-1)
        :return: Array of probabilities P(e_i(t)|O(t), Pi(t)) for all e_i
        """

        """
        There are two cases for event inference:
        1. A new event sequence is starting
            -> only the starting conditions are required
        2. A event sequence is continuing
            -> all likelihood distributions are required
        """

        # 1. case: New event sequence or trial is starting
        if self.o_t_minus_1 is None:
            assert self.current_event == -1
            #if P_ei is None:
                #P_ei = torch.ones(self.num_models).float()
            # P(e_i | O(1), Pi(1)) = P^{start}_{e_i}(o(1)|pi(1))/( sum_{e_j}(P^{start}_{e_j}(o(1)|pi(1)))
            start_ei = np.zeros(self.num_models, dtype = np.float64)
            for i in range(self.num_models):
                x_pi = torch.from_numpy(pi_t)
                output = self.P_start_networks[i].forward(x_pi)
                mu_start = output[0].detach().numpy()
                sigma_start = torch.diag(output[1]).detach().numpy()
                start_ei[i] = self._compute_gauss_pdf(o_t, mu_start, sigma_start)

            sum_eis = np.sum(start_ei * self.P_ei)
            # Normalize probability in case of floatation errors
            P_ei_tplus1 = self.P_ei * (start_ei * 1.0/sum_eis)

            # Store observation and event estimation
            self.o_t_minus_1 = o_t
            self.P_ei = P_ei_tplus1

            return P_ei_tplus1

        # 2. case: Ongoing event sequence
        assert not (self.o_t_minus_1 is None)
        P_posterior = np.zeros((self.num_models, self.num_models))
        
        # Determine P(o(t) | o(t-1), pi(t-1), e_i(t), e_j(t)) for all i x j possible combinations
        for i in range(self.num_models):
            for j in range(self.num_models):
                if i == j:
                    # use P^{event}_{e_i} to estimate the likelihood
                    input = np.append(self.o_t_minus_1, pi_t)
                 
                    x = torch.from_numpy(input)
                    output = self.P_ei_networks[i].forward(x)
                    mu = output[0].detach().numpy()
                    sigma = torch.diag(output[1]).detach().numpy()
                    # likelihoods of staying in the same event
                    P_posterior[i][j] = self.no_transition_prior * self._compute_gauss_pdf(o_t, mu, sigma)
                else:
                    assert i != j
                    # e_j ended and e_i is starting now
                    # use P^{end}_{e_j} and P^{start}_{e_i} to estimate the likelihood

                    # P^{end}_{e_j}:
                    input = np.append(self.o_t_minus_1, pi_t)
                    x = torch.from_numpy(input)                                         
                    output_end = self.P_end_networks[j].forward(x)
                    mu_end = output_end[0].detach().numpy()
                    sigma_end = torch.diag(output_end[1]).detach().numpy()
                    P_j_end = self._compute_gauss_pdf(o_t, mu_end, sigma_end)

                    # P^{start}_{e_i}:
                    x_pi = torch.from_numpy(pi_t)
                    output_start = self.P_start_networks[i].forward(x_pi)
                    mu_start = output_start[0].detach().numpy()
                    sigma_start = torch.diag(output_start[1]).detach().numpy()
                    P_i_start = self._compute_gauss_pdf(o_t, mu_start, sigma_start)


                    # Likelihood for a transition, P(o(t) | e_i(t), e_j(t-1), o(t-1), pi(t-1)) =
                    # P(e_i | e_j) * P^{start}_{e_i}(o(t)| pi(t-1)) * P^{end}(o(t)| o(t-1), pi(t-1))
                    transition_prior = (1.0 - self.no_transition_prior)/(self.num_models - 1.0)
                    P_posterior[i][j] = transition_prior * P_i_start * P_j_end

        # use likelihoods to update inferred event probabilities
        # P(e_i(t)| O(t), Pi(t))
        P_ei_tplus1 = np.zeros(self.num_models)
        for i in range(self.num_models):
            for j in range(self.num_models):
                if np.sum(P_posterior[:, j]) > 0:
                    P_ei_tplus1[i] += (P_posterior[i][j] / np.sum(P_posterior[:, j])) * self.P_ei[j]

        # Normalize P(e_i(t) | O(t), Pi(t)) = P(e_i(t)| O(t), Pi(t))/( sum_{e_j}P(e_j(t) | O(t), Pi(t)))
        # Typically not necessary but sometimes required because of slight approximation errors
        P_ei_tplus1 /= np.sum(P_ei_tplus1)
  
        # Store observation and event estimation
        self.o_t_minus_1 = o_t
        self.P_ei = P_ei_tplus1

        return P_ei_tplus1

    def get_event_probabilities(self):
        """
        :return: P(e_i(t)| O(t), Pi(t)
        """
        return self.P_ei

    @staticmethod
    def _compute_gauss_pdf(x, mu, sigma):
        """
        Compute likelihood of Gaussian distribution
        :param x: variable
        :param mu: mean
        :param sigma: covariance matrix
        :return: N(x)(mu, Sigma)
        """
        
        var = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)

        return var.pdf(x)


    def inference_gradients(self, o_t, o_t_minus_1, pi_t, P_e_i_t, gradient_flow='all', event_entropy_transition = True):
        """
        Infer the probabilities of events in the absence of
        explicit labels
        :param o_t: current observation o(t)
        :param pi_t: last policy pi(t-1)
        :param P_e_i_t: last event probability (tensor)
        :param o_t_minus_1: last observation o(t-1)
        :return: Array of probabilities P(e_i(t)|O(t), Pi(t)) for all e_i
        """

        """
        There are two cases for event inference:
        1. A new event sequence is starting
            -> only the starting conditions are required
        2. A event sequence is continuing
            -> all likelihood distributions are required
        """
        # create tensors of o_t, pi and o_t together with x_pi (needed for networks)
        o_t_tensor, pi_tensor, x_pi_tensor = self.get_tensor(o_t, o_t_minus_1, pi_t)


        # 1. case: New event sequence or trial is starting
        if o_t_minus_1 is None:
            assert self.current_event == -1

            if P_e_i_t is None:
                P_e_i_t = torch.ones(self.num_models).float()

            start_ei = torch.zeros(self.num_models).float()
            # P(e_i | O(1), Pi(1)) = P^{start}_{e_i}(o(1)|pi(1))/( sum_{e_j}(P^{start}_{e_j}(o(1)|pi(1)))
            LH_t = 0
            for i in range(self.num_models):
                mu_start, sigma_start = self.P_start_networks[i].forward(pi_tensor)                    
                distribution = self._compute_gauss_distribution(mu_start, sigma_start)
                start_ei[i] = distribution.log_prob(o_t_tensor).exp()
                # P(o_t|O(t-1), Pi(t-1), e_j, e_i) =  P(o(t)|e_i(t), e_j(t-1), o(t-1), pi(t-1)) * 
                # P(e_i(t)|e_j(t-1), o(t), o(t-1), pt(t-1)) * P(e_j(t-1)|O(t-2) Pi(t-2)) 
                LH_t += distribution.log_prob(o_t_tensor).exp()*distribution.log_prob(o_t_tensor).exp()*P_e_i_t[i]#*P_e_i_t[i]
            sum_eis = torch.sum(torch.mul(start_ei, torch.from_numpy(P_e_i_t)))
            LH_t /= sum_eis
            
            # Normalize probability in case of floatation errors
            P_ei_tplus1 = torch.mul(torch.div(torch.mul(start_ei, 1.0), sum_eis), torch.from_numpy(P_e_i_t))

            # Store observation and event estimation
            self.o_t_minus_1 = o_t
            self.P_ei = P_ei_tplus1.detach().numpy()
            self.p_all = P_ei_tplus1
            self.log_LH =  torch.log(LH_t)
            return P_ei_tplus1

        # 2. case: Ongoing event sequence
        assert not (o_t_minus_1 is None)
        
        P_posterior = torch.zeros((self.num_models, self.num_models))
        LH_matrix = torch.zeros((self.num_models, self.num_models))
        # calculate the probability of changing the event
        transition_prior = (1.0 - self.no_transition_prior)/(self.num_models - 1.0)
        # Determine P(o(t) | o(t-1), pi(t-1), e_i(t), e_j(t)) for all i x j possible combinations
        for i in range(self.num_models):
            for j in range(self.num_models):
                if i == j:
                    # use P^{event}_{e_i} to estimate the likelihood
                    mu, sigma = self.P_ei_networks[i].forward(x_pi_tensor)
                    # likelihoods of staying in the same event
                    distribution_ei = self._compute_gauss_distribution(mu, sigma)
                    # likelihoods of staying in the same event
                    LH_matrix[i][j] = distribution_ei.log_prob(o_t_tensor).exp()
                    P_posterior[i][j] = self.no_transition_prior * LH_matrix[i][j]
                else:
                    assert i != j
                    # e_j ended and e_i is starting now
                    # use P^{end}_{e_j} and P^{start}_{e_i} to estimate the likelihood

                    # P^{end}_{e_j}:
                    mu_end, sigma_end = self.P_end_networks[j].forward(x_pi_tensor)
                    distribution_end = self._compute_gauss_distribution(mu_end, sigma_end)
                    # calculate the likelihood that the event e_j ends
                    LH_end = distribution_end.log_prob(o_t_tensor).exp()
                    # save it to a matrix
                    LH_matrix[i][j] = LH_end
                    P_j_end = LH_end 

                    # P^{start}_{e_i}:
                    mu_start, sigma_start  = self.P_start_networks[i].forward(pi_tensor)
                    distribution_start = self._compute_gauss_distribution(mu_start, sigma_start)
                    # calculate the likelihood that the event e_i starts
                    LH_start = distribution_start.log_prob(o_t_tensor).exp()
                    # multiply the LH of an event ending with the LH of an event starting
                    LH_matrix[i][j] *= LH_start
                    P_i_start = LH_start
                    # Likelihood for a transition, P(o(t) | e_i(t), e_j(t-1), o(t-1), pi(t-1)) =
                    # P(e_i | e_j) * P^{start}_{e_i}(o(t)| pi(t-1)) * P^{end}(o(t)| o(t-1), pi(t-1))
                    P_posterior[i][j] = transition_prior * P_i_start * P_j_end
                    mu, sigma = self._product_of_gaussians_gradients(mu_start, sigma_start, mu_end, sigma_end, self.pinv_matrix)
        # use likelihoods to update inferred event probabilities
        # P(e_i(t)| O(t), Pi(t))
        P_ei_tplus1 = torch.zeros(self.num_models)
        LH_t_matrix = torch.zeros((self.num_models,self.num_models))
        for i in range(self.num_models):
            for j in range(self.num_models):
                if torch.sum(P_posterior[:, j]) > 0:
                    P_ei_tplus1[i] += (P_posterior[i][j] / torch.sum(P_posterior[:, j])) * P_e_i_t[j]
                # P(o_t|O(t-1), Pi(t-1), e_j, e_i) =  P(o(t)|e_i(t), e_j(t-1), o(t-1), pi(t-1)) * P(e_i(t)|e_j(t-1), o(t), o(t-1), pt(t-1)) * P(e_j(t-1)|O(t-2) Pi(t-2)) 
                # P(e_j|e_i, o(t), o(t-1), pt(t-1)) = P_posterior[i][j] / torch.sum(P_posterior[:, j]) 
                # P_posterior[i][j] = (LH *transpition_prior/no_transition)                
                LH_t_matrix[i][j] = LH_matrix[i][j]*(P_posterior[i][j] / torch.sum(P_posterior[:, j])) * P_e_i_t[j]
 
        # Normalize P(e_i(t) | O(t), Pi(t)) = P(e_i(t)| O(t), Pi(t))/( sum_{e_j}P(e_j(t) | O(t), Pi(t)))
        # Typically not necessary but sometimes required because of slight approximation errors
        P_ei_tplus1 /= torch.sum(P_ei_tplus1)

        # Store observation and event estimation
        self.o_t_minus_1 = o_t
        
        # at each timepoint (t) the probability for each event at t + 1 and the neg. log LH at t + 1 is saved
        self.P_ei = P_ei_tplus1.detach().numpy()
        # append the data to the existing vectors
        if self.p_all.size() == torch.Size([4]):
            self.p_all = torch.cat((self.p_all.unsqueeze(0), P_ei_tplus1.unsqueeze(0)), dim = 0)
            self.log_LH = torch.cat((self.log_LH.unsqueeze(0),
                                     torch.log(torch.sum(LH_t_matrix)).unsqueeze(0)), dim = 0)
        else:
            self.p_all = torch.cat((self.p_all, P_ei_tplus1.unsqueeze(0)), dim = 0)
            self.log_LH = torch.cat((self.log_LH, torch.log(torch.sum(LH_t_matrix)).unsqueeze(0)), dim = 0)

        return P_ei_tplus1    
    

    @staticmethod
    def _compute_gauss_distribution(mu, sigma):
        """
        Compute likelihood of Gaussian distribution using only differentiable pytorch functions
        :param x: variable (tensor)
        :param mu: mean (tensor)
        :param sigma: vector of variances (tensor)
        :return: N(x)(mu, Sigma)
        """
        sigma_matrix = torch.diag(sigma)
        distr = torch.distributions.MultivariateNormal(mu, sigma_matrix)

        return distr

    # ------------- ACTIVE INFERENCE -------------
    def estimate_fe_for_policy(self, pi, tau):
        """
        Compute predicted uncertainty for given policy
        :param pi: policy
        :param tau: time horizon expanding into future
        :return: FE(pi, t, tau)
        """

        # For tau = 1 estimated FE is computed as entropy over event dynamics
        event_entropy = self._event_entropy(self.o_t_minus_1, pi)
        if tau <= 1:

            return np.sum(self.P_ei * event_entropy)

        # For tau = 2 estimated FE also contains entropy over next event boundary (start * end)
        end_entropy = self._event_boundary_entropy(self.o_t_minus_1, pi)
        if tau == 2:
            return np.sum(self.P_ei * event_entropy) + np.sum(self.P_ei * end_entropy)

        # For tau = 3 we go one event boundary (end * start) further into the future
        assert tau == 3  # Higher tau not implemented!
        end_end_entropy = self._event_boundary_boundary_entropy(self.o_t_minus_1, pi)

        return np.sum(self.P_ei * event_entropy) + np.sum(self.P_ei * end_entropy) + np.sum(self.P_ei * end_end_entropy)

    def _event_entropy(self, o_t, pi_t):
        """
        Compute entropy of P^{event}_{e_i}(o(t+1)'|o(t), pi(t)) for every event e_i
        :param o_t: observation o(t)
        :param pi_t: policy pi(t)
        :return: Array of computed entropy with an entry for every event e_i
        """
        entropy = np.zeros(self.num_models, dtype=np.float64)
        for i in range(self.num_models):
            input = np.append(o_t, pi_t)
            x = torch.from_numpy(input)
            output = self.P_ei_networks[i].forward(x)
            mu = output[0].detach().numpy()
            sigma = torch.diag(output[1]).detach().numpy()

            entropy_i = self._compute_gauss_entropy(mu, sigma)
            entropy[i] = entropy_i
        return entropy

    def _event_boundary_entropy(self, o_t, pi_t):
        """
        Compute entropy of P^{end}_{e_i}(o(t+tau)'|o(t), pi(t)) * P^{start}_{e_j}(o(t+tau)'|pi(t))
        for all combinations of e_i and e_j
        :param o_t: observation o(t)
        :param pi_t: policy pi(t)
        :return: Array of sums of computed entropy with an entry for every event e_i
        """
        entropy = np.zeros(self.num_models, dtype=np.float64)
        for i in range(self.num_models):
            # Compute the end distribution for event e_i
            input = np.append(o_t, pi_t)
            x_end = torch.from_numpy(input)
            output_end = self.P_end_networks[i].forward(x_end)
            mu_end = output_end[0].detach().numpy()
            sigma_end = torch.diag(output_end[1]).detach().numpy()
            for j in range(self.num_models):
                # Compute the start distributions for event e_j
                x_start = torch.from_numpy(pi_t)
                output_start = self.P_start_networks[j].forward(x_start)
                mu_start = output_start[0].detach().numpy()
                sigma_start = torch.diag(output_start[1]).detach().numpy()
                # Compute the product of the Gaussian distributions
                mu_3, sigma_3 = self._product_of_gaussians(mu_start, sigma_start, mu_end, sigma_end, self.pinv_matrix)
                # Compute the entropy of the resulting distribution
                entropy_ij = self._compute_gauss_entropy(mu_3, sigma_3)
                entropy[i] += entropy_ij
        return entropy

    def _event_boundary_boundary_entropy(self, o_t, pi_t):
        """
        Compute entropy of P^{end}_{e_i}(o(t+tau_1)'|o(t), pi(t)) * P^{start}_{e_j}(o(t+tau_1)'|pi(t))
         * P^{end}_{e_j}(o(t+tau_2)'| o(t+tau_1)', pi(t)) * P^{start}_{e_k}(o(t+tau_2)'| pi(t))
        for all combinations of e_i, e_j, and e_k
        :param o_t: observation o(t)
        :param pi_t: policy pi(t)
        :return: Array of sums of computed entropy with an entry for every event e_i
        """
        entropy = np.zeros(self.num_models, dtype=np.float64)
        for i in range(self.num_models):
            # Compute the end distribution for event e_i
            input = np.append(o_t, pi_t)
            x_end = torch.from_numpy(input)
            output_end = self.P_end_networks[i].forward(x_end)
            mu_end = output_end[0].detach().numpy()
            sigma_end = torch.diag(output_end[1]).detach().numpy()
            for j in range(self.num_models):
                # Compute the start distributions for event e_j
                x_start = torch.from_numpy(pi_t)
                output_start = self.P_start_networks[j].forward(x_start)
                mu_start = output_start[0].detach().numpy()
                sigma_start = torch.diag(output_start[1]).detach().numpy()

                # Compute the end distribution for event e_j based on the expected
                # start observation distribution of e_j
                input_start_end = np.append(mu_start, pi_t)
                x_start_end = torch.from_numpy(input_start_end)
                output_start_end = self.P_end_networks[j].forward(x_start_end)
                mu_start_end = output_start_end[0].detach().numpy()
                sigma_start_end = torch.diag(output_start_end[1]).detach().numpy()

                for k in range(self.num_models):
                    # Compute the start distribution for e_k
                    output_start_end_start = self.P_start_networks[k].forward(x_start)
                    mu_start_end_start = output_start_end_start[0].detach().numpy()
                    sigma_start_end_start = torch.diag(output_start_end_start[1]).detach().numpy()

                    # Compute products of all 4 Gaussian distributions:
                    # 1. Boundary from ending e_i to starting e_j
                    mu_3, sigma_3 = self._product_of_gaussians(mu_start, sigma_start, mu_end, sigma_end, self.pinv_matrix)
                    # 2. Boundary from starting e_j to ending e_j
                    mu_4, sigma_4 = self._product_of_gaussians(mu_start_end, sigma_start_end, mu_3, sigma_3, self.pinv_matrix)
                    # 3. Boundary from ending e_j to starting e_k
                    mu_5, sigma_5 = self._product_of_gaussians(mu_start_end_start, sigma_start_end_start, mu_4, sigma_4, self.pinv_matrix)
                    entropy_ijk = self._compute_gauss_entropy(mu_5, sigma_5)
                    entropy[i] += entropy_ijk

        return entropy
    
    @staticmethod
    def _compute_gauss_entropy(mu, sigma):
        """
        Compute entropy of a Gaussian distribution
        :param mu: mean of Gaussian
        :param sigma: covariance matrix of Gaussian
        :return: entropy of N(x)(mu, sigma)
        """

        e = 0.000001 # to increase numerical stability
        sigma = sigma + np.eye(len(sigma)) * e        
        
        var = multivariate_normal(mean=mu, cov=sigma)
        return var.entropy()

    @staticmethod
    def _product_of_gaussians(mu_1, sigma_1, mu_2, sigma_2, pinv_matrix):
        """
        Computes the product of two Gaussian distributions
        :param mu_1: mean of first Gaussian
        :param sigma_1: vector of variances of first Gaussian
        :param mu_2: mean of second Gaussian
        :param sigma_2: vector of variances of second Gaussian
        :param pinv_matrix: compute pseudo inverse or simple inverse of diagonal cov-matrix
        :return: mean and covariance matrix of resulting Gaussian
        """

        if pinv_matrix:
            # General case for full covariance matrix
            sum_sigma_inv = np.linalg.pinv(sigma_1 + sigma_2)
        else:
            # Sufficient for diagonal matrices and faster to compute:
            sum_sigma_inv = np.diag(1.0/(sigma_1 + sigma_2))
        sigma_3 = np.matmul(np.matmul(sigma_1, sum_sigma_inv), sigma_2)
        mu_1_factor = np.matmul(sigma_2, np.matmul(sum_sigma_inv, mu_1))
        mu_2_factor = np.matmul(sigma_1, np.matmul(sum_sigma_inv, mu_2))
        mu_3 = mu_1_factor + mu_2_factor
        return mu_3, sigma_3
    
    @staticmethod
    def _product_of_gaussians_gradients(mu_1, sigma_1, mu_2, sigma_2, pinv_matrix):
        """
        Computes the product of two Gaussian distributions
        :param mu_1: mean of first Gaussian
        :param sigma_1: vector of variances of first Gaussian
        :param mu_2: mean of second Gaussian
        :param sigma_2: vector of variances of second Gaussian
        :param pinv_matrix: compute pseudo inverse or simple inverse of diagonal cov-matrix
        :return: mean and covariance matrix of resulting Gaussian
        """

        if pinv_matrix:
            # General case for full covariance matrix
            sum_sigma_inv = torch.linalg.pinv(torch.diag(sigma_1) + torch.diag(sigma_2))
        else:
            # Sufficient for diagonal matrices and faster to compute:
            sum_sigma_inv = torch.diag(1.0/(sigma_1 + sigma_2))
        sigma_3 = torch.matmul(torch.matmul(torch.diag(sigma_1), sum_sigma_inv), torch.diag(sigma_2))
        mu_1_factor = torch.matmul(torch.diag(sigma_2), torch.matmul(sum_sigma_inv, mu_1))

        mu_2_factor = torch.matmul(torch.diag(sigma_1), torch.matmul(sum_sigma_inv, mu_2))
        mu_3 = mu_1_factor + mu_2_factor
        return mu_3, sigma_3

    # ------------- RESETTING OR SAVING THE SYSTEM -------------
    def reset(self):
        self.P_ei = np.ones(self.num_models) * (1.0/self.num_models)
        self.current_event = -1
        self.o_t_minus_1 = None

    def save(self, directory, epoch=-1):
        """
        Save the whole system
        :param directory: target directory to save
        :param epoch: number of current epoch
        """
        dir_name = directory + '/checkpoint_' + str(epoch) + '/'
        buffer_dir = dir_name + 'buffers/'
        os.makedirs(dir_name, exist_ok=True)
        os.makedirs(buffer_dir, exist_ok=True)
        for i in range(self.num_models):
            dir_name_i = dir_name + 'net_' + str(i)
            torch.save({
                'start_net': self.P_start_networks[i].state_dict(),
                'start_opt': self.P_start_optimizers[i].state_dict(),
                'start_buffer_index': self.P_start_buffer[i].get_index(),
                'event_net': self.P_ei_networks[i].state_dict(),
                'event_opt': self.P_ei_optimizers[i].state_dict(),
                'event_buffer_index': self.P_ei_buffer[i].get_index(),
                'end_net': self.P_end_networks[i].state_dict(),
                'end_opt': self.P_end_optimizers[i].state_dict(),
                'end_buffer_index': self.P_end_buffer[i].get_index(),
            }, dir_name_i)

            self.P_start_buffer[i].save(buffer_dir, 'start_' + str(i))
            self.P_ei_buffer[i].save(buffer_dir, 'event_' + str(i))
            self.P_end_buffer[i].save(buffer_dir, 'end_' + str(i))

    def load(self, directory, epoch=-1):
        """
        Load the whole system
        :param directory: target directory to load from
        :param epoch: number of epoch to be loaded
        """
        dir_name = directory + '/checkpoint_' + str(epoch) + '/'
        buffer_dir = dir_name + 'buffers/'
        for i in range(self.num_models):
            dir_name_i = dir_name + 'net_' + str(i)
            checkpoint = torch.load(dir_name_i)
            self.P_start_networks[i].load_state_dict(checkpoint['start_net'])
            self.P_start_optimizers[i].load_state_dict(checkpoint['start_opt'])
            self.P_ei_networks[i].load_state_dict(checkpoint['event_net'])
            self.P_ei_optimizers[i].load_state_dict(checkpoint['event_opt'])
            self.P_end_networks[i].load_state_dict(checkpoint['end_net'])
            self.P_end_optimizers[i].load_state_dict(checkpoint['end_opt'])

            self.P_start_buffer[i].load(buffer_dir, 'start_' + str(i), index=checkpoint['start_buffer_index'])
            self.P_ei_buffer[i].load(buffer_dir, 'event_' + str(i), index=checkpoint['event_buffer_index'])
            self.P_end_buffer[i].load(buffer_dir, 'end_' + str(i), index=checkpoint['end_buffer_index'])


    # ------------- OFFLINE DATA COLLECTION  -------------
    def get_offline_data(self, o_t, pi_t, e_i, done):
        """
        Simulates update step without supervised training.
        A helper to create data for offline training
        :param o_t: current observation o(t)
        :param pi_t: last policy pi(t-1)
        :param e_i: supervised label for current event e(t)
        :param done: flag if event sequence ends after this sample
        :return: 4-tuple with
            - Which component does the current observation belong to: 'start', 'dynamics', or 'end'
            - Event number
            - List of inputs
            - List of target outputs
        """

        if self.o_t_minus_1 is None:
            assert self.current_event == -1

            # store observations and policies in trajectory of the current event
            self.o_t_minus_1 = o_t
            self.current_event = e_i
            self.observation_trajectory.append(np.copy(o_t))
            self.policy_trajectory.append(np.copy(pi_t))
            return "start", e_i, [pi_t], [o_t]

        elif self.current_event != e_i or done:

            inp_traj, target_traj = self._get_end_trajectory(o_t)
            last_event = self.current_event

            # Clear the trajectory memory
            self.policy_trajectory.clear()
            self.observation_trajectory.clear()

            # Reset observation and event knowledge
            self.o_t_minus_1 = None
            self.current_event = -1

            return "end", last_event, inp_traj, target_traj
        else:
            assert self.current_event == e_i

            last_obs = self.o_t_minus_1
            # store observations and policies in trajectory of the current event
            self.observation_trajectory.append(np.copy(o_t))
            self.policy_trajectory.append(np.copy(pi_t))
            self.o_t_minus_1 = o_t
            return "dynamics", e_i, [np.append(last_obs, pi_t)], [o_t]


    def _get_end_trajectory(self, o_t):
        """
        Helper for offline data creation. Collects training data for event end
        :param o_t: current observation o(t)
        :return: list of inputs and list of target observation
        """
        length = len(self.observation_trajectory)
        assert length == len(self.policy_trajectory)
        input_list = []
        target_list = []
        for i in range(length):
            o_i = self.observation_trajectory[i]
            pi_i = self.policy_trajectory[i]
            input_i = np.append(o_i, pi_i)
            input_list.append(input_i)
            target_list.append(np.copy(o_t))
        return input_list, target_list
     
    def get_tensor(self, o_t, o_t_minus_1, pi_t):
        """
        Creates tensors for o_t, pi and o_t appended to pi_t
        :param o_t: current observation o(t)
        :param o_t_minus_1: previous observation o(t - 1)
        :param pi_t: last policy pi(t-1)
        :return: list of  o_t as tensor, pi each for all time steps and x_pi as tensor
        """
        # split tensor in different parts to replace the shape of the agent and the patient
        o_t_first_part, o_t_sa, o_t_middle_part, o_t_sp, o_t_end_part = torch.split(torch.Tensor(o_t),
                                [3, 1, self.dim_observation - 2 * (3+1), 1, 3])
        # concatenate the different parts
        o_t_tensor = torch.cat((o_t_first_part, self.x_pi_sa.unsqueeze(0), o_t_middle_part, self.x_pi_sp.unsqueeze(0),
                                o_t_end_part))
        pi_tensor = torch.Tensor(pi_t)
        # check whether there was an observation at time t-1, if not, there is no x_pi_tensor
        if type(o_t_minus_1) == type(None):
            x_pi_tensor = None
        else:
            x_pi_tensor = torch.Tensor(np.append(o_t_minus_1, pi_t))
            x_pi_s_sa, x_pi_sa, x_pi_g_sa, x_pi_sp, x_pi_g_sp = torch.split(x_pi_tensor,
                                [3, 1, self.dim_observation - 2 * (3+1), 1, 6])
            x_pi_tensor = torch.cat((x_pi_s_sa, self.x_pi_sa.unsqueeze(0), x_pi_g_sa, self.x_pi_sp.unsqueeze(0),
                                     x_pi_g_sp))
        return o_t_tensor, pi_tensor, x_pi_tensor

    def get_sum_neg_log_LH(self):
        """
        Sums the negative log LH
        :return: negative log LH
        """
        return torch.sum(torch.mul(self.log_LH, -1))  