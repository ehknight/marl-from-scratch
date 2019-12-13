#pragma once
#include "net.h"

struct transitions {
    MatrixXd states;
    VectorXi actions;
    VectorXd rewards;
    VectorXi dones;
    MatrixXd nextStates;
};

// Q-learning agent with similar structure
// to a DQN I implemeneted a couple years
// ago, located here: https://github.com/ehknight/dqn.
// Standard features like target nets,
// eps-greedy policy.
class Agent {
public:
    Agent(Net* qNet,
          Net* qTargNet,
          int nActions,
          double epsDecay,
          double gamma,
          double tau);

    double fit(Optimizer* opt, transitions batch);
    int act(VectorXd state, bool print=false, double overrideEps=-1);
    void updateTargetNet(bool copy=false);
    void decayEpsilon();
    double eps;
    Net* qNet;
    Net* qTarg;

protected:
    int nActions;
    double epsDecay;
    double gamma;
    double tau;
};

Agent buildAgent(int stateSize,
                 int nActions,
                 double epsDecay,
                 double gamma,
                 double tau,
                 vector<int> hiddens);
