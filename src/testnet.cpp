#include "net.h"
#include "optimizer.h"

// XOR test for NN library. Change
// main -> main to verify.
int main2() {
    MatrixXd X = MatrixXd::Zero(4, 2);
    X << 0, 0,
         0, 1,
         1, 0,
         1, 1;

    MatrixXd y = MatrixXd::Zero(4, 1);
    y << 0,
         1,
         1,
         0;

    Net net = *MLP(2, 1, {3});
    // Optimizer* opt = new SGD(0.1);
    Optimizer* opt = new Adam(0.01);

    for (int i = 0; i < 250; i++)  {
        MatrixXd yPred = net.forwards(X);
        double loss = net.backwards(y, yPred);
        net.applyGrads(opt);
        if (i % 1 == 0) {
            cout << "=====================" << endl;
            cout << "y:" << endl << y << endl;
            cout << "yPred:" << endl << yPred << endl;
            cout << "loss:" << endl << loss << endl;
            cout << "=====================" << endl;
        }
    }
    return 0;
}
