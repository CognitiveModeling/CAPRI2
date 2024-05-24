"""
Simple agent-patient interaction scenario

The scenario is implemented in the OpenAi Gym interface (gym.openai.com)
While currently not all functions of the gym class are properly implemented,
future versions might enable the usage of this scenario for RL-based algorithms.

"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import interaction_gym_rendering as rendering
import numpy as np
import pyglet
import random
from pyglet.gl import *
from pyglet.image.codecs.png import PNGImageDecoder


class InteractionEventGym(gym.Env):
    """
    Gym implementation of a simple agent-patient interaction scenario with different
    interaction events and event sequences

    18-dimensional observation:
    observation space =
    - x_agent y_agent z_agent (positions of agent)
    - s_agent (shape/appearance of agent)
    - v_x_agent v_y_agent v_z_agent (velocities of agent)
    - rel_x rel_y rel_z (relative position of agent towards patient)
    - distance (distance between agent and patient)
    - x_patient y_patient z_patient (positions of patient)
    - s_patient (shape/appearance of patient)
    - v_x_patient v_y_patient v_z_patient (velocities of patient)

    3-dimensional action space is a one-hot encoding where the agent is looking
    (0 = agent, 1 = patient, 2 = no entity)

    There are four possible events:
    - e(t) == 0 => e_still: Agent and patient remain still
    - e(t) == 1 => e_random: Agent moves to randomly selected goal position (decreasing velocity)
    - e(t) == 2 ==> e_reach: Agent moves towards patient
    - e(t) == 3 => e_transport: Agent moves together with patient to randomly selected goal position

    Possible event sequences during training:
    - E(t) = 0 => E_still = e_still (for all agents)
    - E(t) = 1 => E_random = e_random -> e_still (for all agents)
    - E(t) = 2 => E_grasp = e_reach -> e_transport -> e_random (only hand agents)

    Event sequences during testing:
    E_test = e_reach -> e_transport -> e_random (hand or claw agents)
    E_random = e_random -> e_still
    #E_reach = e_reach
    """

    # Defining the event types
    E_UNDEFINED = -1
    E_STILL = 0
    E_RANDOM = 1
    E_REACH = 2
    E_TRANSPORT = 3

    # Defining the event sequence types
    E_SEQ_UNDEFINED = -1
    E_SEQ_STILL = 0
    E_SEQ_RANDOM = 1
    E_SEQ_GRASP = 2

    # ------------- INITIALIZATION -------------
    def __init__(self, sensory_noise_base=1.0, sensory_noise_focus=0.01, r_seed=0, randomize_colors=False,
                 percentage_reaching=1.0 / 3.0):
        """
        Initialization of the simulation
        :param sensory_noise_base: standard deviation of Gaussian distributed noise added on
                                    on all dimensions of observations that are not focused on
        :param sensory_noise_focus: standard deviation of Gaussian distributed noise of
                                    dimensions of the observation currently in focus
        :param r_seed: random seed
        :param randomize_colors: should hand and claw color be determined randomly when initializing?
        :param percentage_reaching: percentage of E_grasp sequences during training
        """

        self.seed = r_seed

        # OBSERVATIONS:
        # bounds of the observation space:
        obs_upper_limits = np.array([10, 10, 10, 10, 10, 10, 10, 20, 20, 10, 300, 10, 10, 10, 10, 10, 10, 10])
        obs_lower_limits = np.array([-10, -10, 0, 0, -10, -10, -10, -20, -20, -10, 0, -10, -10, 0, 0, 10, 10, 10])
        self.observation_space = spaces.Box(obs_lower_limits, obs_upper_limits, dtype=np.float64)
        # starting position of agent and patient after initialization
        self.agent_pos = np.array([0., -8., 5.], dtype=np.float64)
        self.patient_pos = np.array([0., 0., 0.], dtype=np.float64)
        # limits for agent/patient position
        self.origin = np.array([0., 0., 0.], dtype=np.float64)
        self.pos_upper_limits = np.array([10, 10, 10], dtype=np.float64)
        self.pos_lower_limits = np.array([-10, -10, 0], dtype=np.float64)
        # sensory noise
        self.sensory_noise_base = sensory_noise_base  # noise on observation if not focused by the agent
        self.sensory_noise_focus = sensory_noise_focus  # noise on observation if focused by the agent
        self.sensory_noise_other = np.zeros(18, dtype=np.float64)  # sd of other noise added to observation
        # appearance of entities
        self.agent_color = 0.0
        self.patient_color = 10.0
        # velocity definitions
        self.velocity_constant = 0.1  # Scaling factor for determining the next positions
        self.velocity_limits = np.array([0.1, 0.5])
        # previous positions used for velocity computation
        self.old_agent_pos = self.agent_pos.copy()
        self.old_patient_pos = self.patient_pos.copy()
        # o(t):
        self.obs_state = self._get_observation()

        # ACTIONS
        # actions are different gaze positions
        self.gaze_pos = np.zeros(2, dtype=np.float64)
        # position of no fixation
        self.no_fixation_gaze_pos = np.random.uniform(-10, 10, (3, 1))
        action_limits = np.array([1, 1, 1])
        self.action_space = spaces.Box(action_limits * 0, action_limits, dtype=np.float64)

        # VISUALIZATION
        self.viewer = None
        # all entities are composed of a Geom and a Transform (determining positon)
        self.agent_geom = None
        self.agent_trans = None
        self.patient_geom = None
        self.patient_trans = None
        # on top of the Geoms used for the physics we use sprites for visualization
        # that also have an assigned Image, Geom, and Transform and orientation
        self.agent_sprite = None
        self.agent_sprite_geom = None
        self.agent_sprite_trans = None
        self.agent_sprite_orientation = 0
        self.patient_sprite = None
        self.patient_sprite_geom = None
        self.patient_sprite_trans = None
        self.patient_sprite_orientation = 0
        # background image is treated the same way
        self.background_sprite = None
        self.background_geom = None
        self.background_trans = None
        # event names are also sprites
        self.current_event_sprite = None
        # hand and claw have extra sprites (for nice looking grasps)
        self.agent_sprite_extra = None
        self.agent_sprite_extra_geom = None
        self.agent_sprite_extra_trans = None
        # gaze is also visualized through a sprite
        self.gaze_geom = None
        self.gaze_trans = None
        # rendering  its own random generator used to generate randomized colors for
        # different runs. This is done to avoid rendering having an effect on the
        # pseudo-randomization of all other methods for reproducibility
        self.color_randomizer = random.Random(r_seed)

        # Event-related information
        # how often is a reaching event seen during training
        self.percentage_reaching = percentage_reaching
        self.current_event = self.E_UNDEFINED
        self.current_event_sequence = self.E_SEQ_UNDEFINED
        # goal position and maximal distance to determine event ending
        self.agent_goal_pos = None
        self.goal_threshold = 0.5
        self.random_motion_v = np.array([0.0, 0.0, 0.0])
        # during a "stand still" event the event ends after a predefined time
        self.t_still = 0  # counter for time spent in still event
        self.t_end_still = 100  # still event ends after 100 time steps
        self.control_active = False  # TODO implement possibility to control hand as well (not just gaze)
        self.attached = False  # Patient attached to hand?
        self.testing_phase = False
        self.event_sequence_over = False

        self.event_end = False # defines whether in the current time step the event is over

        # appearance of agents
        self.hand_color = 0.0
        self.claw_color = 10.0
        if randomize_colors:
            self.hand_color = random.gauss(2, 1)
            self.claw_color = random.gauss(8, 1)
        self.randomize_colors = randomize_colors


    @staticmethod
    def seed(seed):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    # ------------- STEP -------------
    def step(self, action):
        """
        Core function of this class - is called every time step.
        Receives the action policy pi(t) as an input. The policy must be a one-hot-encoding
        with length 3, stating which entity the system fixates.
        During step the current event is determined, agent and patent are moved, and the next
        observation is determined
        :param action: one-hot-encoding of gaze policy pi(t-1)
        :return: next observation o(t), reward r(t) (currently not used),
                    bool if current event sequence is over, current event e(t)
        """

        # determine exact gaze position based on policy
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.gaze_pos = self._get_gaze_position(action)

        # default: the event is not over
        self.event_end = False #

        # determine ongoing event
        self._determine_event()

        # determine motor action of agent based on event
        motor_action = None
        if not self.control_active:
            motor_action = self._determine_agent_motor_action()
        else:  # infer motor action of agent
            print("In current version is motor control not implemented")
            assert False

        # memorize old positions for velocity computations
        self.old_agent_pos = self.agent_pos.copy()
        self.old_patient_pos = self.patient_pos.copy()

        # move agent according to motor command
        self.agent_pos = self.agent_pos + motor_action
        if self.attached:
            # patient is only moved if attached to the agent
            self.patient_pos = self.patient_pos + motor_action

        # check boundaries of agent and patient position
        self._clip_positions()
        self.obs_state = self._get_observation()

        # determine if event sequence is over (done)
        done = 0
        if self.event_sequence_over:
            done = 1

        # determine reward
        # TODO: reward is currently always 0
        reward = 0

        # normalize observation o(t) and add noise before returning it
        return self._add_gaze_based_noise(self.obs_state, action), reward, done, self.current_event, self.event_end

    def _get_gaze_position(self, policy):
        """
        Determine the gaze position based on the one-hot encoding of looking (policy)
        :param policy: gaze policy pi(t-1)
        :return: fixation position in 2D-coordinates
        """
        assert 1 in policy and np.sum(policy) == 1
        if policy[0] == 1:  # The system looks at the agent.
            return self.agent_pos[0:2]
        if policy[1] == 1:  # The system looks at the patient.
            return self.patient_pos[0:2]
        return self.no_fixation_gaze_pos  # No fixation, looking at center of the screen.

    def _get_observation(self):
        """
        :return: observation o(t), not normalized!
        """
        v_agent = (self.agent_pos - self.old_agent_pos) * (10 / self.velocity_limits[1])
        v_patient = (self.patient_pos - self.old_patient_pos) * (10 / self.velocity_limits[1])
        obs = np.array([self.agent_pos[0], self.agent_pos[1], self.agent_pos[2], self.agent_color,
                        v_agent[0], v_agent[1], v_agent[2], (self.agent_pos[0] - self.patient_pos[0]) / 2.0,
                        (self.agent_pos[1] - self.patient_pos[1]) / 2.0,
                        (self.agent_pos[2] - self.patient_pos[2]) / 2.0,
                        np.linalg.norm(self.patient_pos - self.agent_pos) / 30.0, self.patient_pos[0],
                        self.patient_pos[1], self.patient_pos[2], self.patient_color,
                        v_patient[0], v_patient[1], v_patient[2]], dtype=np.float64)
        return obs

    def _add_gaze_based_noise(self, obs, policy):
        """
        Add noise on observation and normalize it
        :param obs: not normalized observation o(t)
        :param policy: gaze policy pi(t-1)
        :return: normalized observation o(t) with added noise
        """

        # noise for non-fixated entities:
        sensory_noise = np.random.normal(0.0, self.sensory_noise_base, 18)
        # noise for fixated entities:
        sensory_noise_focus = np.random.normal(0.0, self.sensory_noise_focus, 7)

        if policy[0] == 1:  # looking at agent --> less noise on agent-related information
            sensory_noise[0:7] = sensory_noise_focus[0:7]
        elif policy[1] == 1:  # looking at patient --> less noise on patient-related information
            sensory_noise[11:18] = sensory_noise_focus[0:7]
        # Add other noise not stemming from fixation on o_t (if it exists)
        sensory_noise += self.sensory_noise_other
        return (obs + sensory_noise) * 1.0 / 10.0  # add noise and normalize such that output is within [-1, 1]

    def set_other_noise(self, sd_obs):
        """
        Add other noise on sensory observation, not coming from fixation
        Can for example be used to systematically add noise on specific
        dimension of observation such as the agent's appearance
        :param sd_obs: standard deviation of normal distribution noise added on o_t
        """
        assert sd_obs.size == 18
        self.sensory_noise_other = sd_obs

    def _determine_event(self):
        """
        :return: current event e(t)
        """

        # Memorize last event
        last_event = self.current_event

        # Stay in still event for self.t_end_still times
        if self.current_event == self.E_STILL:
            self.t_still = self.t_still + 1
            if self.t_still > self.t_end_still:
                self.event_sequence_over = True

        if (not self.attached) and (self.current_event == self.E_REACH):
            # if e_reach is active check if agent reached patient and transition to transport
            distance_agent_patient = np.linalg.norm(self.patient_pos - self.agent_pos)
            if distance_agent_patient <= self.goal_threshold:
                # transition from e_reach to e_transport
                self.attached = True
                self.current_event = self.E_TRANSPORT
        elif (self.current_event == self.E_RANDOM) or (self.current_event == self.E_TRANSPORT):
            # if e_random or e_transport is active check if goal position is reached
            distance_to_goal = np.linalg.norm(self.agent_goal_pos - self.agent_pos)
            if distance_to_goal <= self.goal_threshold:
                if self.current_event == self.E_TRANSPORT:
                    # after e_transport the hand moves away randomly
                    self.current_event = self.E_RANDOM
                    self._determine_goal_pos()
                    self.attached = False
                else:
                    # after e_random is over e_still starts for E_random
                    # or the sequence is over for E_grasp or E_test
                    if self.current_event_sequence == self.E_SEQ_GRASP:
                        self.event_sequence_over = True
                    else:
                        assert self.E_SEQ_RANDOM
                        self.current_event = self.E_STILL

        if last_event != self.current_event:
            # an event transition happened
            # the new goal position needs to be determined
            # the event is over (- New!)
            self.event_end = True
            self._determine_goal_pos()
            # if we are rendering the event sprite needs to be replaced
            if self.viewer is not None:
                self._load_event_sprite()

    def _determine_goal_pos(self):
        """
        When a new event starts, we need to determine the goal of this event
        """
        if self.current_event == self.E_STILL:
            # still event means no motion of agent
            # hence, the goal position of the agent stays the same
            self.agent_goal_pos = self.agent_pos
        elif self.current_event == self.E_RANDOM:
            # random motion means we sample a starting velocity
            # then we determine the goal based on v * 30 on a radius around the current agent position
            goal_found = False
            goal_search_counter = 0
            # we do this sampling iteratively because we might end up with non-reachable goal positions
            while not goal_found:
                v_candidate = np.random.uniform(-1, 1, 3)
                v_candidate = self.velocity_limits[1] / np.sqrt(np.sum(v_candidate ** 2)) * v_candidate
                goal_candidate = self.agent_pos + 30 * v_candidate
                goal_search_counter = goal_search_counter + 1
                if np.amax(goal_candidate) < 9 and np.amin(goal_candidate) > -9 and goal_candidate[2] > 0:
                    # If the goal position is within the predefined range we can accept it
                    self.random_motion_v = 1.0 / (1.0 - 1.0 / 30.0) * v_candidate
                    self.agent_goal_pos = goal_candidate
                    goal_found = True
                if goal_found is False and goal_search_counter > 100 \
                        and self.current_event_sequence == self.E_SEQ_RANDOM:
                    # if after 100 trials we do not find a suitable goal and we just started a new sequence
                    # we re-sample the starting position of the agent to speed up the process
                    self.agent_pos = np.random.uniform(self.pos_lower_limits, self.pos_upper_limits, 3)
                    goal_search_counter = 0
        elif self.current_event == self.E_REACH:
            # the agent's goal during reaching is the position of the patient
            self.agent_goal_pos = self.patient_pos
        elif self.current_event == self.E_TRANSPORT:
            # during transport we sample a random goal position on the floor
            self.agent_goal_pos = np.random.uniform(self.pos_lower_limits, self.pos_upper_limits, 3)
            # during testing the goal positions for transportation are always the same
            if self.testing_phase:
                # not depending on the position, but set goal (corners)
                if self.transport_goal_sign is not None:
                    self.agent_goal_pos[0] = 7.7 * np.sign(self.transport_goal_sign[0])
                    self.agent_goal_pos[1] = 7.7 * np.sign(self.transport_goal_sign[1])
                else:
                    self.agent_goal_pos[0] = 7.7 * np.sign(self.agent_pos[0])
                    self.agent_goal_pos[1] = 7.7 * np.sign(self.agent_pos[1])
            self.agent_goal_pos[2] = 0.0

    def _determine_agent_motor_action(self):
        """
        Determine the agent's motor action given the current event
        :return: motor action that determines the agent's change in position
        """

        if self.current_event == self.E_RANDOM:
            # During random motion the agent moves with decreasing velocity
            next_motor_action = self.random_motion_v
            self.random_motion_v = (1.0 - 1.0 / 30.0) * self.random_motion_v
            if np.sqrt(np.sum(self.random_motion_v ** 2)) < self.velocity_limits[0]:
                # Decrease velocity of the motion
                self.random_motion_v = self.velocity_limits[0] / np.sqrt(np.sum(self.random_motion_v ** 2)) \
                                       * self.random_motion_v
            return next_motor_action

        # for reaching and transportation the agent moves constantly towards its goal
        next_motor_action = self.agent_goal_pos - self.agent_pos
        # normalize action and multiply with velocity constant
        next_motor_action = next_motor_action * (1.0 / (np.linalg.norm(next_motor_action) + 0.00001))
        next_motor_action = next_motor_action * self.velocity_constant
        return next_motor_action

    def _clip_positions(self):
        """
        Positions are clipped if they exceed the boundaries of the simulation
        """
        np.clip(self.agent_pos, self.pos_lower_limits, self.pos_upper_limits, self.agent_pos)
        np.clip(self.patient_pos, self.pos_lower_limits, self.pos_upper_limits, self.patient_pos)

    # ------------- RESET -------------
    def reset(self, condition, agent_color=None, patient_color=None):
        """
        Reset the simulation to a new agent-patient pair and event sequence
        :return: First observation o(0)
        """

        # reset is only called in training
        self.testing_phase = False

        # reset all internal flags
        self._reset_internal_stuff()

        claw_grasping = False
        if condition is not None:
            # conditions for hands in training
            if isinstance(condition, float):
                # allow claws performing grasping sequences with given probability
                claw_grasping = True
            elif isinstance(condition, tuple):
                # sample hand appearance from a given distribution
                range_hand = condition
                func = range_hand[0]
                self.hand_color = func(range_hand[1], range_hand[2])

        # sample event
        r_event = random.random()
        if r_event < self.percentage_reaching:
            # reaching event
            self.current_event = self.E_REACH
            self.current_event_sequence = self.E_SEQ_GRASP
            # reaching is generally only done by hands
            self.agent_color = self.hand_color
            # ... or also by claws (variant training)
            if claw_grasping:
                r_claw = random.random()
                if r_claw < condition:
                    self.agent_color = self.claw_color
                else:
                    self.agent_color = self.hand_color
        else:
            # standing still or random?
            r_event2 = random.random()
            if r_event2 < 0.5:
                # staying still event
                self.current_event = self.E_STILL
                self.current_event_sequence = self.E_SEQ_STILL
            else:
                # random motion
                self.current_event = self.E_RANDOM
                self.current_event_sequence = self.E_SEQ_RANDOM
            # sample the agent's color
            self.agent_color = np.random.uniform(0, 10, 1)[0]

        # set agent color directly
        if agent_color != None:
            self.agent_color = agent_color

        # sample the patient's color:
        if patient_color == None:
            self.patient_color = np.random.uniform(1, 10, 1)[0]
            if self.randomize_colors:
                self.patient_color = np.random.uniform(0, 10, 1)[0]
        else:
            self.patient_color = patient_color

        # sample positions of agent and patient
        self._sample_starting_positions()

        # determine goal position
        self._determine_goal_pos()

        # determine velocity
        self.velocity_constant = random.random() * (self.velocity_limits[1] - self.velocity_limits[0]) + \
                                 self.velocity_limits[0]

        # Get new observation
        self.obs_state = self._get_observation()

        # If the event sequence starts with 'reaching' the agent needs to be oriented towards the patient
        # (otherwise it looks weird)
        if self.current_event == self.E_REACH:
            sprite_origin = np.array([0, 1], np.float64)
            sprite_goal = np.array([(self.patient_pos[0] - self.agent_pos[0]) *
                                    (self.patient_pos[0] - self.agent_pos[0]),
                                    (self.patient_pos[1] - self.agent_pos[1]) *
                                    (self.patient_pos[1] - self.agent_pos[1])])
            dot_product_orientation = np.dot(sprite_origin, sprite_goal)
            self.agent_sprite_orientation = np.arccos(dot_product_orientation / (np.linalg.norm(sprite_origin) *
                                                                                 np.linalg.norm(sprite_goal)))

            if self.patient_pos[1] > self.agent_pos[1] and self.patient_pos[0] > self.agent_pos[0]:
                self.agent_sprite_orientation = self.agent_sprite_orientation - 1.57
            else:
                if self.patient_pos[1] < self.agent_pos[1]:
                    self.agent_sprite_orientation = self.agent_sprite_orientation + 1.57
                if self.patient_pos[0] > self.agent_pos[0]:
                    self.agent_sprite_orientation = self.agent_sprite_orientation + 1.57

        return np.array(self.obs_state, dtype=np.float64) * 1.0 / 10.0  # normalize

    def reset_to_grasping(self, claw, agent_color=None, patient_color=None,
                          start_pos_signs=None,
                          transport_goal_sign=None):
        """
        Reset simulation to E_test event sequence
        :param claw: bool if claw or hand is used
        :param agent_color: appearance of agent (=agency estimate)
        :param patient_color: appearance of patient
        :param start_pos_signs: the corner where the agent starts
        :param transport_goal_sign: the corner the agent should transport the patient to
        :return: first observation o(0)
        """

        # reset_to_grasping is only called during testing phase
        self.testing_phase = True

        # resetting everything required
        self._reset_internal_stuff()

        # determine the agent's appearance
        if agent_color == None:
            if claw:
                # agent is a claw
                self.agent_color = self.claw_color
            else:
                # agent is a hand
                self.agent_color = self.hand_color
        else:
            # set agent
            self.agent_color = agent_color

        # determine the patient's appearance
        if patient_color == None:
            self.patient_color = np.random.uniform(1, 10, 1)[0]
            if self.randomize_colors:
                self.patient_color = np.random.uniform(0, 10, 1)[0]
        else:
            self.patient_color = patient_color

        # sample positions of the two entities
        self._sample_starting_positions_testing(start_pos_signs=start_pos_signs)

        self.current_event = self.E_REACH  # during testing we always start with a reaching event
        self.current_event_sequence = self.E_SEQ_GRASP

        # goal position is determined
        self._determine_goal_pos()

        # Goal position for the transport event
        self.transport_goal_sign = transport_goal_sign

        # determine the velocity of the movement
        self.velocity_constant = 0.1  # Velocity is always the same during testing

        # determine the observation
        self.obs_state = self._get_observation()

        # determine the orientation of the agent such that it is oriented towards the patient
        # (otherwise reaching would look weird)
        sprite_origin = np.array([0, 1], np.float64)
        sprite_goal = np.array([(self.patient_pos[0] - self.agent_pos[0]) *
                                (self.patient_pos[0] - self.agent_pos[0]),
                                (self.patient_pos[1] - self.agent_pos[1]) *
                                (self.patient_pos[1] - self.agent_pos[1])])
        dot_product_orientation = np.dot(sprite_origin, sprite_goal)
        self.agent_sprite_orientation = np.arccos(dot_product_orientation /
                                                  (np.linalg.norm(sprite_origin) * np.linalg.norm(sprite_goal)))
        if self.patient_pos[1] > self.agent_pos[1] and self.patient_pos[0] > self.agent_pos[0]:
            self.agent_sprite_orientation = self.agent_sprite_orientation - 1.57
        else:
            if self.patient_pos[1] < self.agent_pos[1]:
                self.agent_sprite_orientation = self.agent_sprite_orientation + 1.57
            if self.patient_pos[0] > self.agent_pos[0]:
                self.agent_sprite_orientation = self.agent_sprite_orientation + 1.57

        return np.array(self.obs_state, dtype=np.float64) * 1.0 / 10.0  # normalize observation

    def _reset_internal_stuff(self):
        """
        Helper for reset() and reset_to_grasping() that
        resets internal flags and counters
        """
        self.event_sequence_over = False
        self.t_still = 0
        self.agent_sprite_orientation = 0
        self.attached = False  # patient is never attached when resetting the simulation
        # resetting rendering as well
        if self.viewer:
            self.viewer.deactivate_video_mode()
            self.viewer.close()
            self.viewer = None

        # randomly reset where no fixation is visualized
        self.no_fixation_gaze_pos = np.random.uniform(-10, 10, 2)

        # gaze is reset to be outside of frame
        self.gaze_pos = self.no_fixation_gaze_pos
        self.gaze_geom = None
        self.current_event = self.E_UNDEFINED
        self.current_event_sequence = self.E_SEQ_UNDEFINED

    def _sample_starting_positions(self):
        """
        Sample starting positions of agent and patient when resetting
        during training
        """
        self.agent_pos = np.random.uniform(self.pos_lower_limits, self.pos_upper_limits, 3)
        self.patient_pos = np.random.uniform(self.pos_lower_limits, self.pos_upper_limits, 3)
        self.patient_pos[2] = 0.0  # the patient always lies on the floor
        # if the agent and patient are too close to each other continue sampling until a correct position is found
        while np.linalg.norm(self.patient_pos - self.agent_pos) < 1:
            # entities should be at least 1 unit apart from each other
            self.agent_pos = np.random.uniform(self.pos_lower_limits, self.pos_upper_limits, 3)
            self.patient_pos = np.random.uniform(self.pos_lower_limits, self.pos_upper_limits, 3)
            self.patient_pos[2] = 0.0
        self.old_agent_pos = self.agent_pos.copy()
        self.old_patient_pos = self.patient_pos.copy()

    def _sample_starting_positions_testing(self,start_pos_signs=None):
        """
        Sample starting position of agent and patient when
        resetting to testing sequence E_test
        """
        self.agent_pos = np.array([5.8, 5.8, 6.5])
        if start_pos_signs is None:
            # sample randomly
            signs = np.random.uniform(-1, 1, 2)
        else:
            # set which corner the agent starts in
            signs = np.array(start_pos_signs)
        self.agent_pos[0:2] = self.agent_pos[0:2] * np.sign(signs)
        self.patient_pos = np.array([0.0, 0.0, 0.0])
        self.old_agent_pos = self.agent_pos.copy()
        self.old_patient_pos = self.patient_pos.copy()

    # ------------- RENDERING -------------
    def render(self, store_video=False, video_identifier=1, mode='human'):
        """
        Renders the simulation
        :param store_video: bool to save screenshots or not
        :param video_identifier: number to label video of this simulation
        :param mode: inherited from gym, currently not used
        """

        # Constant values of window size, sprite sizes, etc...
        screen_width = 600
        screen_height = 600
        scale = 30.0
        object_width = 50
        object_height = 50
        sprite_scale = 0.4
        sprite_width = 350 * sprite_scale
        sprite_height = 350 * sprite_scale
        agent_scale = 1.0
        patient_scale = 1.0

        if self.viewer is None:
            # we create a new viewer (window)
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # visualizing the agent
            l, r, t, b = -object_width / 2, object_width / 2, object_height / 2, -object_height / 2
            agent_polygon = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.agent_geom = agent_polygon
            self.agent_trans = rendering.Transform()
            rand_color = self._create_color(for_agent=True)
            agent_polygon.set_color(rand_color[0], rand_color[1], rand_color[2])
            agent_polygon.add_attr(self.agent_trans)
            agent_sprite_file = self._determine_sprites(self.agent_color)
            agent_image = pyglet.image.load(agent_sprite_file, decoder=PNGImageDecoder())
            agent_sprite = pyglet.sprite.Sprite(img=agent_image)
            glEnable(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            agent_sprite.scale = sprite_scale
            sprite_geom = rendering.SpriteGeom(agent_sprite)
            self.viewer.add_geom(sprite_geom)
            self.agent_sprite_geom = sprite_geom
            self.agent_sprite_trans = rendering.Transform()
            sprite_geom.add_attr(self.agent_sprite_trans)

            # hand and claw receives additional sprite for grasping (thumb above object)
            if self.agent_color == self.hand_color or self.agent_color == self.claw_color:
                thumb_image = pyglet.image.load(self._determine_extra_sprites(self.agent_color == self.hand_color),
                                                decoder=PNGImageDecoder())
                thumb_sprite = pyglet.sprite.Sprite(img=thumb_image)
                thumb_sprite.scale = sprite_scale
                self.agent_sprite_extra_geom = rendering.SpriteGeom(thumb_sprite)
                self.viewer.add_geom(self.agent_sprite_extra_geom)
                self.agent_sprite_extra_trans = rendering.Transform()
                self.agent_sprite_extra_geom.add_attr(self.agent_sprite_extra_trans)

            # visualize the patient
            patient_polygon = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.patient_geom = patient_polygon
            self.patient_trans = rendering.Transform()
            patient_polygon.add_attr(self.patient_trans)
            rand_color = self._create_color(for_agent=False)
            patient_polygon.set_color(rand_color[0], rand_color[1], rand_color[2])
            patient_sprite_file = self._determine_sprites(self.patient_color)
            patient_image = pyglet.image.load(patient_sprite_file, decoder=PNGImageDecoder())
            patient_sprite = pyglet.sprite.Sprite(img=patient_image)
            glEnable(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            patient_sprite.scale = sprite_scale
            sprite_geom2 = rendering.SpriteGeom(patient_sprite)
            self.viewer.add_geom(sprite_geom2)
            self.patient_sprite_geom = sprite_geom2
            self.patient_sprite_trans = rendering.Transform()
            sprite_geom2.add_attr(self.patient_sprite_trans)

            # the ongoing event is displayed through a sprite in the top left corner
            event_img = pyglet.image.load('Sprites/still.png', decoder=PNGImageDecoder())
            event_sprite = pyglet.sprite.Sprite(img=event_img)
            self.current_event_sprite = rendering.SpriteGeom(event_sprite)
            self.current_event_sprite.set_z(-1)
            self._load_event_sprite()
            current_event_sprite_transform = rendering.Transform()
            current_event_sprite_transform.set_translation(0, screen_height - 50)
            self.current_event_sprite.add_attr(current_event_sprite_transform)
            self.viewer.add_geom(self.current_event_sprite)

            # The current gaze position is visualized by a small red dot
            fixation_file = pyglet.image.load('Sprites/redDot.png', decoder=PNGImageDecoder())
            fixation_sprite = pyglet.sprite.Sprite(img=fixation_file)
            fixation_sprite.scale = 0.1
            self.gaze_geom = rendering.SpriteGeom(fixation_sprite)
            self.gaze_trans = rendering.Transform()
            self.gaze_geom.add_attr(self.gaze_trans)
            self.viewer.add_geom(self.gaze_geom)

            # background image (only needed for having nice videos)
            if store_video:
                background_img = pyglet.image.load('Sprites/background.png', decoder=PNGImageDecoder())
                background_sprite = pyglet.sprite.Sprite(img=background_img)
                background_sprite.scale = 1
                self.background_geom = rendering.SpriteGeom(background_sprite)
                self.background_geom.set_z(-10)
                self.background_trans = rendering.Transform()
                self.background_geom.add_attr(self.background_trans)
                self.background_trans.set_translation(0, 0)
                self.viewer.add_geom(self.background_geom)

        # during video recording images of the simulation are saved
        if store_video:
            self.viewer.activate_video_mode("Video" + str(video_identifier) + "/")

        # determine the sprite positions, sizes and orientations:
        agent_x = self.agent_pos[0] * scale + (screen_width / 2.0)
        agent_y = self.agent_pos[1] * scale + (screen_height / 2.0)
        patient_x = self.patient_pos[0] * scale + (screen_width / 2.0)
        patient_y = self.patient_pos[1] * scale + (screen_height / 2.0)
        gaze_x = self.gaze_pos[0] * scale + (screen_width / 2.0)
        gaze_y = self.gaze_pos[1] * scale + (screen_height / 2.0)
        self.gaze_trans.set_translation(gaze_x, gaze_y)
        self.gaze_geom.set_z(11)
        agent_scale += self.agent_pos[2] / 10 * 2
        patient_scale += self.patient_pos[2] / 10 * 2
        self.agent_geom.set_z(self.agent_pos[2] - 2)
        self.patient_geom.set_z(self.patient_pos[2])
        self.agent_sprite_geom.set_z(self.agent_pos[2] - 2)
        self.patient_sprite_geom.set_z(self.patient_pos[2] + 0.25)
        self.agent_sprite_trans.set_rotation(self.agent_sprite_orientation)
        if self.agent_sprite_extra_geom:
            self.agent_sprite_extra_geom.set_z(self.agent_pos[2] + 2)
            self.agent_sprite_extra_trans.set_rotation(self.agent_sprite_orientation)

        # compute the rotation matrix for correction
        theta = self.agent_sprite_orientation
        r = np.array(((np.cos(theta), -np.sin(theta)),
                      (np.sin(theta), np.cos(theta))))
        sprite_center = np.array([agent_scale * sprite_width / 2.0, agent_scale * sprite_height / 2.0])
        new_sprite_center = np.dot(r, sprite_center)
        self.agent_trans.set_translation(agent_x, agent_y)
        self.agent_sprite_trans.set_translation(agent_x - new_sprite_center[0], agent_y - new_sprite_center[1])
        self.patient_sprite_trans.set_translation(patient_x - patient_scale * sprite_width / 2.0,
                                                  patient_y - patient_scale * sprite_height / 2.0)
        if self.agent_sprite_extra_geom:
            self.agent_sprite_extra_trans.set_translation(agent_x - new_sprite_center[0],
                                                          agent_y - new_sprite_center[1])
            self.agent_sprite_extra_trans.set_scale(agent_scale, agent_scale)
        self.patient_trans.set_translation(patient_x, patient_y)
        self.agent_trans.set_scale(agent_scale, agent_scale)
        self.agent_sprite_trans.set_scale(agent_scale, agent_scale)
        self.patient_sprite_trans.set_scale(patient_scale, patient_scale)
        self.patient_trans.set_scale(patient_scale, patient_scale)

        return self.viewer.render(mode == 'rgb_array')

    def _load_event_sprite(self):
        """
        When rendering, the top left corner displays the current event through a sprite
        """
        #
        if self.current_event == self.E_STILL:
            image = pyglet.image.load('Sprites/still.png', decoder=PNGImageDecoder())
            event_sprite = pyglet.sprite.Sprite(img=image)
            self.current_event_sprite.replaceSprite(event_sprite)
        elif self.current_event == self.E_RANDOM:
            image = pyglet.image.load('Sprites/random.png', decoder=PNGImageDecoder())
            event_sprite = pyglet.sprite.Sprite(img=image)
            self.current_event_sprite.replaceSprite(event_sprite)
        elif self.current_event == self.E_REACH:
            image = pyglet.image.load('Sprites/grasp.png', decoder=PNGImageDecoder())
            event_sprite = pyglet.sprite.Sprite(img=image)
            self.current_event_sprite.replaceSprite(event_sprite)
        else:
            image = pyglet.image.load('Sprites/push.png', decoder=PNGImageDecoder())
            event_sprite = pyglet.sprite.Sprite(img=image)
            self.current_event_sprite.replaceSprite(event_sprite)

    def _create_color(self, for_agent):
        """
        When rendering does not use the sprites, the entities are displayed as differently
        colored circles. This function determines a RGB triplet and alpha value.
        We simply choose a random color and let the appearance value determine its brightness
        :param for_agent: bool stating if color is used for agent or patient
        :return: color to display the entity
        """
        color = np.ones(4)
        color[0] = self.color_randomizer.random()
        color[1] = self.color_randomizer.random()
        color[2] = self.color_randomizer.random()
        normed_color = color * 1.0 / (np.linalg.norm(color) + 0.000001)
        normed_color[3] = 1  # alpha value is always 1
        if for_agent:
            return normed_color * self.agent_color / 10.0
        return normed_color * self.patient_color / 10.0

    def _determine_sprites(self, color):
        """
        Determine sprites for entity
        :param color: appearance of entity
        :return: relative path to sprite
        """
        # return self._determine_sprites_pencil
        return self._determine_sprite_silhouettes(color)

    def _determine_sprites_pencil(self, color):
        """
        Render the entities through hand drawn pencil sprites
        (Not available online due to copyright)
        :param color: appearance of entity
        :return: relative path to sprite as string
        """

        # Special case hand:
        if color == self.hand_color:
            return "Sprites/Pencil_Sprites/hand_pencil_big.png"

        # Special case claw:
        if color == self.hand_color:
            return "Sprites/Pencil_Sprites/claw_pencil.png"

        if color < 1:
            return "Sprites/Pencil_Sprites/pineapple_pencil.png"
        elif color < 2:
            return "Sprites/Pencil_Sprites/apple_pencil.png"
        elif color < 3:
            return "Sprites/Pencil_Sprites/pear_pencil.png"
        elif color < 4:
            return "Sprites/Pencil_Sprites/pineapple_pencil.png"
        elif color < 5:
            return "Sprites/Pencil_Sprites/ball_pencil.png"
        elif color < 6:
            return "Sprites/Pencil_Sprites/teddy_pencil.png"
        elif color < 7:
            return "Sprites/Pencil_Sprites/pear_pencil.png"
        elif color < 8:
            return "Sprites/Pencil_Sprites/ball_pencil.png"
        elif color < 9:
            return "Sprites/Pencil_Sprites/apple_pencil.png"
        return "Sprites/Pencil_Sprites/teddy_pencil.png"

    def _determine_sprite_silhouettes(self, color):
        """
        Render the entities through silhouette sprites
        :param color: appearance of entity
        :return: relative path to sprite as string
        """
        if color == self.hand_color:
            return "Sprites/Silhouette_Sprites/hand_silhouette_sprite.png"
        if color == self.claw_color:
            return "Sprites/Silhouette_Sprites/claw_silhouette_sprite.png"
        if color < 1:
            return "Sprites/Silhouette_Sprites/ball_silhouette_sprite.png"
        elif color < 2:
            return "Sprites/Silhouette_Sprites/apple_silhouette_sprite.png"
        elif color < 3:
            return "Sprites/Silhouette_Sprites/horse_silhouette_sprite.png"
        elif color < 4:
            return "Sprites/Silhouette_Sprites/pen_silhouette_sprite.png"
        elif color < 5:
            return "Sprites/Silhouette_Sprites/fork_silhouette_sprite.png"
        elif color < 6:
            return "Sprites/Silhouette_Sprites/cherry_silhouette_sprite.png"
        elif color < 7:
            return "Sprites/Silhouette_Sprites/teddy_silhouette_sprite.png"
        elif color < 8:
            return "Sprites/Silhouette_Sprites/mug_silhouette_sprite.png"
        elif color < 9:
            return "Sprites/Silhouette_Sprites/phone_silhouette_sprite.png"
        return "Sprites/Silhouette_Sprites/teddy_silhouette_sprite.png"

    def _determine_extra_sprites(self, hand):
        """
        Hands and claws have one extra sprite for visualizing transportation
        :param hand: bool if hand or claw
        :return: relative path to sprite
        """
        # return self._determine_extra_sprites_pencil(hand)
        return self._determine_extra_sprites_silhouette(hand)

    @staticmethod
    def _determine_extra_sprites_pencil(hand):
        """
        Extra sprite in hand-drawn pencil style
        (online not available due to copyright)
        :param hand: bool if hand or claw
        :return: relative path to extra sprite
        """
        if hand:
            return 'Sprites/Pencil_Sprites/hand_pencil_big_thumb.png'
        return 'Sprites/Pencil_Sprites/claw_pencil_thumb.png'

    @staticmethod
    def _determine_extra_sprites_silhouette(hand):
        """
            Extra sprite in silhouette style
            :param hand: bool if hand or claw
            :return: relative path to extra sprite
        """
        if hand:
            return 'Sprites/Silhouette_Sprites/hand_silhouette_sprite.png'
        return 'Sprites/Silhouette_Sprites/claw_silhouette_sprite_up.png'

    # ------------- CLOSE -------------
    def close(self):
        """
        Shut down the gym
        """
        if self.viewer:
            self.viewer.deactivate_video_mode()
            self.viewer.close()
            self.viewer = None