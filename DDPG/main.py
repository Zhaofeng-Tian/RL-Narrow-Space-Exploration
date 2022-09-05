#!/usr/bin/env python3
from pickle import FALSE, TRUE
import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve, manage_memory, save_loss, save_reward, save_score
import random

from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import rospy
import rospkg
import tensorflow as tf



if __name__ == '__main__':

    manage_memory()
    # gym-ros env initialization
    ENV_NAME = 'ZebratWall-v1'
    #ENV_NAME = 'ZebratReal-v0'
    rospy.init_node('zebrat_ddpg', anonymous=True, log_level=rospy.WARN)          
    env = StartOpenAI_ROS_Environment(ENV_NAME)
    # env = gym.make('LunarLanderContinuous-v2')

    agent = Agent(obs_shape=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0],
                  alpha=0.0001, beta=0.0002, epsilon = 0.5)
    n_games = 503

    print("**********************************************")
    print ("obs_shape is: " + str(env.observation_space.shape))
    print("**********************************************")
    figure_file = '/home/tian/simulation_ws/src/zebrat/zebrat_training/scripts/DDPG-V1/plots/ddpg.png'

    best_score = env.reward_range[0]
    rospy.logerr('env.reward_range[0]: ' + str(best_score))
    
    #best_score = -10000
    # six lists to save the info of training
    score_list = []
    step_list = []
    episode_list = []
    actor_loss = []
    critic_loss = []
    reward_list = []




    
    # **********  Modify Here!!!  ***************************   
    load_checkpoint = False
    
    load_train = True
    # ********************************************************





    if load_checkpoint:
        agent.load_models()
        evaluate = False
    elif load_train:
        agent.load_models()
        evaluate = True

    else:
        evaluate = False
    
    for i in range(n_games):
        success = 0
        step_ctr = 0
        episode_list.append(i+1)
        observation = env.reset()
        done = False
        score = 0
        #step_episode = 0
        while not done:
            step_ctr += 1
            #step_episode +=1
            step_list.append(step_ctr)
            print("---> this is total step: "+ str(step_ctr))
            check_policy =(step_ctr < agent.batch_size) or (random.random() > (agent.epsilon + (1-agent.epsilon)*i/n_games))
            rospy.logerr('IS_RANDOM:   ' + str(check_policy))
            #action = agent.choose_action(observation, evaluate,is_random=(step_ctr < agent.batch_size) or (random.random() > (agent.epsilon + (1-agent.epsilon)*i/n_games)) )
            action = agent.choose_action(observation, evaluate,is_random=(random.random() < agent.epsilon-agent.epsilon*i/n_games ))
            
            #rospy.logerr('Epsilon = ' + str(agent.epsilon))
            #rospy.logerr('Step counter: ' + str(step_ctr) + '  Batch size:' + str (agent.batch_size))
            #rospy.logerr('Check batch switch: '+str(step_ctr < agent.batch_size))
            #rospy.logerr('Check epsilon switch: ' + str((random.random() > (agent.epsilon + (1-agent.epsilon)*i/n_games))))
            #rospy.logerr('Check whole swich: '+ str((step_ctr < agent.batch_size) or (random.random() > (agent.epsilon + (1-agent.epsilon)*i/n_games))))
            rospy.logerr('Action Policy using: ' + agent.policy)
            print("Action: " +str(action))
            observation_, reward, done, info = env.step(action)
            reward_list.append(reward)
            if reward == 50:
                success+= 1
            print("1. old obs, in shape" + str(observation))
            print("2. action chosen is: "+str(action))
            print("3. obs got, in shape: " +str(observation_))
            print("4. The reward for this step is :  " + str(reward))
            print("5. done flag: " + str(done))
            print(info)
            print("Is crash?: "+ env.crash)
            score += reward

            agent.store_transition(observation, action, reward,
                                   observation_, done)
            if not load_checkpoint:
                agent.learn()
                actor_loss.append(agent.get_actor_loss())
                critic_loss.append(agent.get_critic_loss())
            observation = observation_
        # end while

        score_list.append(score)
        avg_score = np.mean(score_list[-100:])

        if score > best_score:
            best_score = score
            if not load_checkpoint:
                if step_ctr >= 65:
                    agent.save_models()
                    rospy.logerr('GAME Round ' + str(i+1) +" Model Saved !!!")

        save_reward(step_list,reward_list,'/home/tian/simulation_ws/src/zebrat/zebrat_training/scripts/DDPG-V1/plots/loss.txt')
        save_score(episode_list, score_list, '/home/tian/simulation_ws/src/zebrat/zebrat_training/scripts/DDPG-V1/plots/score.txt')
        print('Check the length of five lists: ')
        print('step {:} actor_loss {:} critic_loss {:} reward {:}'.format(len(step_list), len(actor_loss), len(critic_loss), len(reward_list)))
        print ('episode {:} score {:}'.format(len(episode_list), len(score_list)))

        print('episode {} score {:.1f} avg score {:.1f}'.format(i, score, avg_score))
        print('Success Rate: ' + str(success/n_games*100)+'%')
    '''
    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_list, figure_file)
        save_loss(step_list,actor_loss,critic_loss,reward_list,'/home/tian/simulation_ws/src/zebrat/zebrat_training/scripts/DDPG-V1/plots/loss.txt')
        save_score(episode_list, score_list, '/home/tian/simulation_ws/src/zebrat/zebrat_training/scripts/DDPG-V1/plots/score.txt')
    print('Check the length of five lists: ')
    print('step {:} actor_loss {:} critic_loss {:} reward {:}'.format(len(step_list), len(actor_loss), len(critic_loss), len(reward_list)))
    print ('episode {:} score {:}'.format(len(episode_list), len(score_list)))'''