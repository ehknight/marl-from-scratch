#pragma once

#include <Eigen/Dense>
#include "vector.h"
#include "layer.h"
#include "optimizer.h"

using namespace std;
using namespace Eigen;

/*
 * A simple PyTorch-like dense layer framework
 * that supports forwards and backwards propogation.
 * Loosely based off of this Python/Numpy tutorial:'
 * https://medium.com/towards-artificial-intelligence/nothing-but-numpy-understanding
 * -creating-neural-networks-with-computational-graphs-from-scratch-6299901091b0
 */

VectorXd matToVec(MatrixXd A);

MatrixXd vecToMat(VectorXd v);

class Linear : public Layer {
public:
    Linear(int inFeatures, int outFeatures, string name);
    MatrixXd forwards(MatrixXd X);
    void backwards(MatrixXd upstreamGrad);
    void applyGrads(Optimizer* opt);
    vector<MatrixXd *> getWeights();

private:
    MatrixXd W;
    MatrixXd b;
    MatrixXd dW;
    MatrixXd db;
};

class ReLU : public Layer {
public:
    ReLU(int inFeatures, int outFeatures, string name);
    MatrixXd forwards(MatrixXd X);
    void backwards(MatrixXd upstreamGrad);
    void applyGrads(Optimizer* opt);
    vector<MatrixXd *> getWeights();
};

/////////////////////////////////////////////

class LossLayer {
public:
    virtual double forwards(MatrixXd yTrue, MatrixXd yPred) = 0;
    virtual void backwards() = 0;
    MatrixXd grad;

protected:
    MatrixXd yTrue;
    MatrixXd yPred;
};

class MSELayer : public LossLayer {
public:
    MSELayer();
    double forwards(MatrixXd yTrue, MatrixXd yPred);
    void backwards();
};

/////////////////////////////////////////////

class Net {
public:
    Net(vector<Layer *> layers, LossLayer* lossLayer);
    MatrixXd forwards(MatrixXd X);
    double backwards(MatrixXd yTrue, MatrixXd yPred);
    void applyGrads(Optimizer* opt);
    vector<Layer *> layers;
    void read(string dirname);
    void write(string dirname);

protected:
    LossLayer* lossLayer;
};

/////////////////////////////////////////////

Net* MLP(int inSize, int outSize, vector<int> hidSizes);
