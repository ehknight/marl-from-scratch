#pragma once
#include <Eigen/Dense>
#include "agent.h"

class Memory {
public:
    Memory(int stateSize, int capacity);
    transitions sampleBatch(int nSamples);
    void store(VectorXd state, int action, double reward, bool done, VectorXd nextState);

protected:
    int t;
    int size;
    int capacity;
    MatrixXd states;
    VectorXi actions;
    VectorXd rewards;
    VectorXi dones;
    MatrixXd nextStates;
};
