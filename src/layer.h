#pragma once

#include <vector>
#include <Eigen/Dense>
#include "optimizer.h"

using namespace std;
using namespace Eigen;

class Layer{
public:
    virtual MatrixXd forwards(MatrixXd X) = 0;
    virtual void backwards(MatrixXd upstreamGrad) = 0;
    virtual void applyGrads(Optimizer* opt) = 0;
    virtual vector<MatrixXd*> getWeights() = 0;
    MatrixXd grad;
    bool trainable;
    string name;

protected:
    int inFeatures;
    int outFeatures;

    // cache input and output
    MatrixXd input;
    MatrixXd output;
};
