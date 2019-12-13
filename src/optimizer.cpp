#include <iostream>
#include "optimizer.h"

SGD::SGD(double lr) {
    this->lr = lr;
}

void SGD::applyGrads(Eigen::MatrixXd& theta, Eigen::MatrixXd dTheta, std::string /* name */) {
    theta -= lr * dTheta;
}

Adam::Adam(double lr, double beta1, double beta2) {
    this->lr = lr;
    this->beta1 = beta1;
    this->beta2 = beta2;
}

// math ref: https://ruder.io/optimizing-gradient-descent/index.html#momentum
void Adam::applyGrads(Eigen::MatrixXd& theta, Eigen::MatrixXd dTheta, std::string name) {
    if (!vLast.count(name)) {
        std::cout << "initting " << name << std::endl;
        mLast[name] = Eigen::ArrayXXd::Zero(theta.rows(), theta.cols());
        vLast[name] = Eigen::ArrayXXd::Zero(theta.rows(), theta.cols());
    }
    // Decaying averages
    Eigen::ArrayXXd dThetaArray = dTheta.array();
    mLast[name] = (beta1 * (mLast[name]) + (1 - beta1) * dThetaArray).matrix();
    vLast[name] = (beta2 * (vLast[name]) + (1 - beta2) * dThetaArray.square()).matrix();
    // Bias correction
    Eigen::ArrayXXd m = (1 / (1 - beta1)) * (mLast[name]);
    Eigen::ArrayXXd v = (1 / (1 - beta2)) * (vLast[name]);
    // Param update
    theta -= (lr * (((v.cwiseSqrt() + 1e-8)).cwiseInverse()) * m).matrix();
}
