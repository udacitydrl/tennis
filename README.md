# Deep Reinforcement Learning Project 3: Tennis

## Introduction
This project solves the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. Two agents are created and trained to control rackets to bounce a ball over a net.. 

![tennis](tennis.gif)

A reward of +0.1 is provided for each step if an agent hits the ball over the net. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play as long as possible. 


The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, The agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the average over both agents). Specifically,
* After each episode, the rewards that each agent received  is summed up to get a score for each agent. This yields 2 (potentially different) scores.  The average of these 2 scores is taken.
* This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started

Download the environment from one of the links below for your operating system:
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(_For AWS_) [click here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md) if you like to enable a virtual screen, then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

## Instructions
* To download the project [click here](https://github.com/udacitydrl/tennis) or run git clone https://github.com/udacitydrl/tennis from a terminal.
* To train the agent, start jupyter notebook, open Twennis.ipynb and execute! For more information, please check instructions inside the notebook.
