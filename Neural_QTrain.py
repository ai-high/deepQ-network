import gym
import tensorflow as tf
import numpy as np
import random


# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.99 # discount factor
INITIAL_EPSILON = 0.8  # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON_DECAY_STEPS = 2000  # decay period
BATCH_SIZE = 50
UPDATE_TARGET_NETWORK = 50
MAX_MEMORY = 1000
LEARNING_RATE = 0.01
# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph
with tf.variable_scope("Main"):
    fc1 = tf.layers.dense(inputs = state_in,units=64,activation=tf.nn.relu)
    #fc2 = tf.layers.dense(inputs = fc1,units=12,activation=tf.nn.relu)

    # TODO: Network outputs
    q_values = tf.layers.dense(inputs = fc1,units = ACTION_DIM)
    q_action = tf.reduce_sum(tf.multiply(action_in,q_values),axis=1)

    # TODO: Loss/Optimizer Definition
    loss = tf.losses.mean_squared_error(q_action, target_in)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)


################
# Target Network
################
with tf.variable_scope("Target"):
    target_fc1 = tf.layers.dense(inputs = state_in, units=64 ,activation=tf.nn.relu)
    #target_fc2 = tf.layers.dense(inputs = target_fc1,units=12,activation=tf.nn.relu)

    # TODO: Network outputs
    target_q_values = tf.layers.dense(inputs = target_fc1,units = ACTION_DIM)
    target_q_action = tf.reduce_sum(tf.multiply(action_in,target_q_values),axis=1)

    # TODO: Loss/Optimizer Definition
    target_loss = tf.losses.mean_squared_error(target_q_action, target_in)
    target_optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(target_loss)



# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
memory = []
ave_rewards = []
ave_reward = 0
for episode in range(EPISODE):
    if(ave_reward):
        ave_rewards.append(ave_reward)
    if(len(ave_rewards) >= 3 and ave_rewards[len(ave_rewards)-1] >= 190 and ave_rewards[len(ave_rewards)-2] >= 190 and ave_rewards[len(ave_rewards)-3] >= 190):
        LEARNING_RATE = 0.0
    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    if(epsilon > FINAL_EPSILON):
        epsilon -= (epsilon - FINAL_EPSILON) / EPSILON_DECAY_STEPS
    # Move through env according to e-greedy policy
    for step in range(STEP):

        if step % UPDATE_TARGET_NETWORK == 0:
            main = [t for t in tf.trainable_variables() if t.name.startswith("Main")]
            main = sorted(main, key=lambda v: v.name)
            target_n = [t for t in tf.trainable_variables() if t.name.startswith("Target")]
            target_n = sorted(target_n, key=lambda v: v.name)

            update = []
            for main_x, target_x in zip(main, target_n):
                op = target_x.assign(main_x)
                update.append(op)
            for u in update:
                session.run(u)

        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))
        memory.append((state,action,(reward,done),next_state))
        if len(memory) >= MAX_MEMORY:
            del memory[0]

        if(len(memory) > BATCH_SIZE):
            batch = random.sample(memory, BATCH_SIZE)
            states = [val[0] for val in batch]
            actions = [val[1] for val in batch]
            rewards = [val[2] for val in batch]
            next_states = [val[3] for val in batch]
            targets = []
            nextstate_q_values = target_q_values.eval(feed_dict={
                state_in: next_states
            })

            # TODO: Calculate the target q-value.
            # hint1: Bellman
            # hint2: consider if the episode has terminated
            for i in range(len(nextstate_q_values)):
                #print(rewards)
                if(rewards[i][1]):
                    targets.append(rewards[i][0])
                else:
                    targets.append(rewards[i][0] + GAMMA * np.max(nextstate_q_values[i]))
                # Do one training step

            session.run([optimizer], feed_dict={
                target_in: targets,
                action_in: actions,
                state_in: states
            })

        # Update
        state = next_state
        if done:
            break


    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                #env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()


