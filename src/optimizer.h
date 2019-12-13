#pragma once
#include <map>
#include <Eigen/Dense>

class Optimizer {
public:
    virtual void applyGrads(Eigen::MatrixXd& theta, Eigen::MatrixXd dTheta, std::string name) = 0;
};

class SGD : public Optimizer {
public:
    SGD(double lr);
    void applyGrads(Eigen::MatrixXd& theta, Eigen::MatrixXd dTheta, std::string name);

protected:
    double lr;
};

class Adam : public Optimizer {
public:
    Adam(double lr, double beta1=0.9, double beta2=0.999);
    void applyGrads(Eigen::MatrixXd& theta, Eigen::MatrixXd dTheta, std::string name);

protected:
    double lr;
    double beta1;
    double beta2;
    // stanford c++ does _not_ play well with eigen
    std::map<std::string, Eigen::ArrayXXd> mLast;
    std::map<std::string, Eigen::ArrayXXd> vLast;
};
