#!/usr/bin/env python3
from pickle import FALSE, TRUE
import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve, manage_memory, save_loss, save_score
import random

from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import rospy
import rospkg
import tensorflow as tf

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    ENV_NAME = 'ZebratWall-v0'
    rospy.init_node('zebrat_ppo', anonymous=True, log_level=rospy.WARN)      
    env = StartOpenAI_ROS_Environment(ENV_NAME)

    load_checkpoint = False
    score_list, reward_list, step_list, episode_list = [], [], [], []
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    n_games = 1000

    figure_file = '/home/tian/simulation_ws/src/zebrat/zebrat_training/scripts/PPO-discrete/plots/ppo.png'

    best_score = env.reward_range[0]


    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        episode_list.append(i+1)
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            reward_list.append(reward)
            print("Action: " +str(action))
            print("1. old obs, in shape" + str(observation))
            print("2. action chosen is: "+str(action))
            print("3. obs got, in shape: " +str(observation_))
            print("4. The reward for this step is :  " + str(reward))
            print("5. done flag: " + str(done))
            print(info)
            print("Is crash?: "+ env.crash)
            n_steps += 1
            print('This is EPISODE: ' + str(i+1))
            print ('This is STEP: ' + str(n_steps))
            score += reward
            agent.store_transition(observation, action,
                                   prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation
            step_list.append(n_steps)
        score_list.append(score)
        
        avg_score = np.mean(score_list[-100:])

        if (score > best_score) and (n_steps > 100):
            if not load_checkpoint:
                agent.save_models()
            best_score = score
        save_score(episode_list, score_list, '/home/tian/simulation_ws/src/zebrat/zebrat_training/scripts/PPO-discrete/plots/score.txt')     
        save_loss(step_list,reward_list,'/home/tian/simulation_ws/src/zebrat/zebrat_training/scripts/PPO-discrete/plots/loss.txt')

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_list))]
    plot_learning_curve(x, score_list, figure_file)