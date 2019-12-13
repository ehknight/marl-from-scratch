#include "agent.h"
#include "random.h"

VectorXd rowSlice(MatrixXd X, VectorXi idx) {
    VectorXd v(X.rows());
    for (int i = 0; i < X.rows(); i++) {
        v.row(i) << X.row(i)[idx[i]];
    }
    return v;
}

MatrixXd maskCombine(MatrixXd base, VectorXd target, VectorXi idxs) {
    assert(base.rows() == target.rows());
    for (int i = 0; i < base.rows(); i++) {
        base.row(i)[idxs[i]] = target[i];
    }
    return base;
}

Agent::Agent(Net* qNet,
             Net* qTarg,
             int nActions,
             double epsDecay,
             double gamma,
             double tau) {
    this->qNet = qNet;
    this->qTarg = qTarg;
    this->nActions = nActions;
    this->epsDecay = epsDecay;
    this->gamma = gamma;
    this->tau = tau;
    this->eps = 1;
}

void Agent::updateTargetNet(bool copy) {
    for (unsigned int i = 0; i < qNet->layers.size(); i++) {
        if (qNet->layers[i]->trainable) {
            vector<MatrixXd*> qWeights = qNet->layers[i]->getWeights();
            vector<MatrixXd*> qTargWeights = qTarg->layers[i]->getWeights();
            assert(qWeights.size() == qTargWeights.size());
            for (unsigned int j = 0; j < qWeights.size(); j++) {
                MatrixXd update;
                MatrixXd theta = *qWeights[j];
                MatrixXd targ = *qTargWeights[j];
                if (copy)
                    update = theta;
                else
                    update = theta * (1 - tau) + targ * tau;
                *qTargWeights[j] = update;
                assert(*(qTarg->layers[i]->getWeights()[j]) == update);
            }
        }
    }
}

void Agent::decayEpsilon() {
    this->eps *= epsDecay;
}

double Agent::fit(Optimizer* opt, transitions batch) {
    MatrixXd qPredAll = qNet->forwards(batch.states);
    VectorXd qPred = rowSlice(qPredAll, batch.actions);

    MatrixXd qPredNext = qTarg->forwards(batch.nextStates);
    VectorXd qPredNextMax = qPredNext.rowwise().maxCoeff();
    VectorXd qTarg = batch.rewards + gamma * qPredNextMax;

    qPred = (batch.dones).select(batch.rewards, qPred);
    MatrixXd qTargAll = maskCombine(qPredAll, qTarg, batch.actions);

    qNet->backwards(qTargAll, qPredAll);
    qNet->applyGrads(opt);
    return (qTarg - qPred).mean();
}

int Agent::act(VectorXd X, bool print, double overrideEps) {
    double _eps = (overrideEps >= 0 ? overrideEps : this->eps);
    if (_eps > randomReal(0, 1)) {
        return randomInteger(0, nActions-1);
    } else {
        int maxCol;
        MatrixXd Xmat = vecToMat(X);
        MatrixXd Qvals = qNet->forwards(Xmat);
        if (print)
            cout << "QVals: " << Qvals << endl;
        matToVec(Qvals).maxCoeff(&maxCol);
        return maxCol;
    }
}

Agent buildAgent(int stateSize,
                 int nActions,
                 double epsDecay,
                 double gamma,
                 double tau,
                 vector<int> hiddens) {

    Net* Q = MLP(stateSize, nActions, hiddens);
    Net* Qtarg = MLP(stateSize, nActions, hiddens);
    Agent agent(Q, Qtarg, nActions, epsDecay, gamma, tau);
    agent.updateTargetNet(true);
    return agent;
}
