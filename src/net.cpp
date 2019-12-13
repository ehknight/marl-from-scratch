#include "strlib.h"
#include "net.h"
#include "optimizer.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <io.h>

const bool DEBUG = false;

VectorXd matToVec(MatrixXd A) {
    // from https://stackoverflow.com/a/22896750
    return Eigen::Map<VectorXd>(A.data(), A.cols()*A.rows());
}

MatrixXd vecToMat(VectorXd v) {
    return Eigen::Map<MatrixXd>(v.data(), 1, v.cols()*v.rows());
}

Linear::Linear(int _inFeatures, int _outFeatures, string name) {
    this->inFeatures = _inFeatures;
    this->outFeatures = _outFeatures;
    this->name = name;
    this->trainable = true;

    this->W = MatrixXd::Random(inFeatures, outFeatures) * 0.1;
    this->b = MatrixXd::Ones(outFeatures, 1) * 0.01;

    this->dW = MatrixXd::Zero(inFeatures, outFeatures);
    this->db = MatrixXd::Zero(outFeatures, 1);
}

MatrixXd Linear::forwards(MatrixXd X) {
    this->input = X;
    this->output = X * W;
    this->output.rowwise() += matToVec(b).transpose();
    return this->output;
}

void Linear::backwards(MatrixXd upstreamGrad) {
    if (DEBUG) {
        cout << "input: " << endl << this->input << endl;
        cout << "upstream grad: " << endl << upstreamGrad << endl;
    }
    this->dW = this->input.transpose() * upstreamGrad;
    this->db = upstreamGrad.colwise().sum();
    this->grad = upstreamGrad * W.transpose();
}

void Linear::applyGrads(Optimizer* opt) {
    if (DEBUG) {
        cout << "dW: " << endl << dW << endl;
        cout << "db: " << endl << db << endl;
    }
    opt->applyGrads(W, dW, name+"_W");
    opt->applyGrads(b, db.transpose(), name+"_b");
}

vector<MatrixXd*> Linear::getWeights() {
    MatrixXd* Wptr = &(this->W);
    MatrixXd* bPtr = &(this->b);
    return {Wptr, bPtr};
}

////////////////////////////////////////////////////

ReLU::ReLU(int _inFeatures, int _outFeatures, string name) {
    assert(_inFeatures == _outFeatures);
    this->inFeatures = _inFeatures;
    this->outFeatures = _outFeatures;
    this->name = name;
    this->trainable = false;
}

MatrixXd ReLU::forwards(MatrixXd X) {
    this->input = X;
    this->output = X.cwiseMax(0);
    return this->output;
}

void ReLU::backwards(MatrixXd upstreamGrad) {
    MatrixXd zeros = MatrixXd::Zero(input.rows(), inFeatures);
    MatrixXd ones = MatrixXd::Ones(input.rows(), inFeatures);
    MatrixXd reluGrad = (input.array() > 0).select(ones, zeros);
    this->grad = reluGrad.cwiseProduct(upstreamGrad);
}

void ReLU::applyGrads(Optimizer* /* opt */) {}

vector<MatrixXd*> ReLU::getWeights() {return {};}

////////////////////////////////////////////////////

MSELayer::MSELayer() {}

double MSELayer::forwards(MatrixXd _yTrue, MatrixXd _yPred) {
    this->yTrue = _yTrue;
    this->yPred = _yPred;
    return (_yTrue - yPred).array().square().mean();
}

void MSELayer::backwards() {
    this->grad = yPred - yTrue;
}


////////////////////////////////////////////////////

Net::Net(vector<Layer *> layers, LossLayer* lossLayer) {
    this->layers = layers;
    this->lossLayer = lossLayer;
}

MatrixXd Net::forwards(MatrixXd X) {
    MatrixXd out = X;
    for (Layer* layer : layers) {
        out = layer->forwards(out);
    }
    return out;
}

double Net::backwards(MatrixXd yTrue, MatrixXd yPred) {
    assert(yTrue.rows() == yPred.rows());
    assert(yTrue.cols() == yPred.cols());

    double cost = lossLayer->forwards(yTrue, yPred);
    lossLayer->backwards();
    MatrixXd grad = lossLayer->grad;

    if (DEBUG) {
        cout << "Cost grad: " << endl << grad << endl;
    }

    vector<Layer *> reversedLayers = this->layers;
    reverse(reversedLayers.begin(), reversedLayers.end());

    for (Layer* l : reversedLayers) {
        l->backwards(grad);
        grad = l->grad;
    }
    return cost;
}

void Net::applyGrads(Optimizer* opt) {
    for (Layer* l : layers) {
        l->applyGrads(opt);
    }
}

namespace Eigen {
    // THIS UTILIY write_binary IS COPIED FROM USER "ANDREA" ON STACK OVERFLOW
    // AND CAN BE FOUND AT LINK https://stackoverflow.com/a/25389481
    // It is used for writing matrix files
    template<class Matrix>
    void write_binary(const string filename, const Matrix& matrix){
        std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
        typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(typename Matrix::Index));
        out.write((char*) (&cols), sizeof(typename Matrix::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
        out.close();
    }

    // THIS UTILIY read_binary IS COPIED FROM USER "ANDREA" ON STACK OVERFLOW
    // AND CAN BE FOUND AT LINK https://stackoverflow.com/a/25389481
    // It is used for reading matrix files
    template<class Matrix>
    void read_binary(const string filename, Matrix& matrix){
        std::ifstream in(filename, std::ios::in | std::ios::binary);
        typename Matrix::Index rows=0, cols=0;
        in.read((char*) (&rows),sizeof(typename Matrix::Index));
        in.read((char*) (&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
        in.close();
    }
}

void Net::write(string dirName) {
    mkdir(dirName.c_str());
    for (Layer* l : layers) {
        for (unsigned int i = 0; i < l->getWeights().size(); i++) {
            string name = dirName + "/" + l->name
                        + "_" + integerToString(int(i)) + ".mtx";
            Eigen::write_binary(name, *(l->getWeights()[i]));
        }
    }
}

void Net::read(string dirName) {
    for (Layer* l : layers) {
        for (unsigned int i = 0; i < l->getWeights().size(); i++) {
            string name = dirName + "/" + l->name
                        + "_" + integerToString(int(i)) + ".mtx";
            Eigen::read_binary(name, *(l->getWeights()[i]));
        }
    }
}

Net* MLP(int inSize, int outSize, vector<int> hidSizes) {
    vector<Layer *> layers; hidSizes.insert(hidSizes.begin(), inSize);
    for (unsigned int i=0; i < hidSizes.size() - 1; i++) {
        Linear* l = new Linear(hidSizes[i], hidSizes[i+1], "Hidden"+to_string(i));
        ReLU* r = new ReLU(hidSizes[i+1], hidSizes[i+1], "ReLU"+to_string(i));
        layers.push_back(l);
        layers.push_back(r);
    }
    Linear* outLayer = new Linear(hidSizes[hidSizes.size()-1], outSize, "Out");
    layers.push_back(outLayer);

    MSELayer* lossLayer = new MSELayer();
    return new Net(layers, lossLayer);
}
