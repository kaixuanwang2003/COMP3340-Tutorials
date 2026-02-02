# Introduction to RoboTwin & Your First MLP Policy to Control a Robot

RoboTwin is a simulation platform developed by Meta AI that allows users to train and evaluate embodied AI agents in a variety of robotic tasks. It provides a realistic environment for simulating robot interactions, making it an excellent tool for research and development in robotics and AI.

In this tutorial, we will guide you through the process of setting up RoboTwin and creating your first Multi-Layer Perceptron (MLP) policy to control a robot within the simulation. By the end of this tutorial, you will have a basic understanding of how to use RoboTwin and implement an MLP policy for robotic control.

## RoboTwin

[RoboTwin Documentations](https://robotwin-platform.github.io/doc/)

RoboTwin provides a comprehensive set of tools and features for simulating robotic environments. It supports various robot models and tasks, allowing users to experiment with different control strategies and algorithms.

## Setting Up RoboTwin

Please make sure that you have a compatible system and hardware. RoboTwin requires a powerful GPU on Linux system, to simulate complex robotic environments efficiently.

If you don't have a GPU-enabled Linux system, you can consider using [CS GPU farm](https://ai.hku.hk/research/major-facilities?view=article&id=131&catid=24).

It is possible, if you are using CPUs for RoboTwin, but it will be much, much slower.

Clone the [Course Project Repo](https://github.com/kaixuanwang2003/COMP3340-2026-Project/tree/main), other than the official RoboTwin repo for the MLP policy support.

```bash
git clone https://github.com/kaixuanwang2003/COMP3340-2026-Project.git
```

Follow the [installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) to set up RoboTwin on your machine, you can omit cloning the official RoboTwin repo, as we have already done that in the course project repo.

## Different Types of Action and Observation

In terms of robot control policies, there are generally two types of observations and actions:

1. **State-based Observations and Actions**: These policies use compact state representations, such as joint angles, velocities, and object positions, to make decisions. The actions are typically low-dimensional commands, such as joint torques or end-effector poses.
2. **Vision-based Observations and Actions**: These policies rely on visual inputs, such as camera images, to perceive the environment. The actions may involve higher-level commands, such as pixel-level manipulations or end-effector movements based on visual feedback.

And for the action type, there are generally two types:

1. End Effector Pose-based Actions: These actions specify the desired position and orientation of the robot's end-effector in 3D space. The policy outputs target poses, which are then converted into joint commands using inverse kinematics or motion planning algorithms.
2. Joint Torque/Position-based Actions: These actions directly control the torques applied to the robot's joints. The policy outputs torque values for each joint, allowing for more fine-grained control of the robot's movements.

## Walking through MLP Implementation in PyTorch

We will use the PyTorch framework to implement our MLP policy for controlling the robot in RoboTwin.

## Try your Policy Out

Check the [documentations](./documentations.md) for the MLP state-based policy for RoboTwin to try it out!

Try different super parameters, like dropout rate, learning rate, batch size, observation horizon, action horizon... and see how they affect the performance!

We don't expect you to get a policy to work - that is very challenging in just a few hours, especially with such a simple policy network, but we hope you can get a taste of how to implement a simple MLP policy for robot control using RoboTwin and PyTorch. Enjoy experimenting!

## Optional Reading Materials

[Tutorial on Quaternions](https://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/index.htm) and a [more in-depth one](https://lisyarus.github.io/blog/posts/introduction-to-quaternions.html)

[Tutorial on Inverse Kinematics](https://oscarliang.com/inverse-kinematics-and-trigonometry-basics/)

[The RoboTwin Paper](https://arxiv.org/pdf/2506.18088)
