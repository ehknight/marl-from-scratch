# Emergent Complexity in an Adversarial Tag Game through Multi-Agent Reinforcement Learning... From Scratch! (in C++)

In this project, **two agents compete in an adversarial game of tag**,
where the “Tagger" attempts to touch the “Taggee". **Both agents  
autonomously learn, without human feedback,** to better accomplish their
respective goals: the tagger receives a small reward for getting closer
to the taggee and receives a large reward for tagging it, and vice versa
for the tagger.

At the beginning, both agents’ behaviors are completely random. However,
**as each agent improves in achieving their goal, complex behaviors
emerge, leading to increasingly complex games over the course of
training**. For instance, agents will begin to **juke** and **feint**,
attempting to deceive their opponent.

One might expect that each agent would overfit to its opponent,
resulting in brittle, overly specific strategies, since each agent has
only seen the other single agent throughout the course of training.
Surprisingly, the policies seem, qualitatively, to be fairly general and
able to reliably outplay human opponents.


## Implementation summary

This project required building the following components:

  - **Tag.** Each ball has momentum and slides around, and can move in
    one of four directions (including one no-op). The game board is a
    torus, so the screen wraps around.

  - **Neural networks.** A limited nonlinear neural network framework,
    including backpropogation and adaptive gradient optimization.

  - **Deep Q-Learning (DQL) agent** Implemented a standard DQN, with
    target networks and an epsilon-greedy policy. Also implemented an
    experience buffer, which the DQNs sample from.

The code was written in C++. It uses the Eigen library for fast matrix
multiplications.


##  Technical details

  - The taggee has slightly lower speed and slightly higher acceleration
    than the taggee. This makes for more interesting games since each
    ball has a certain “niche."

  - Each agent is equipped with the same architecture ( 10k parameters),
    but each agent learns separately.

  - The game runs at around 33 frames per second while a human is
    playing it, and much faster while not rendering. The full system can
    be fully trained in about 30 minutes.
