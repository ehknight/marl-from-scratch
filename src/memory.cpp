#include <algorithm>
#include "memory.h"
#include "random.h"
using namespace std;

Memory::Memory(int stateSize, int capacity) {
    this->t = 0;
    this->size = 0;
    this->capacity = capacity;

    this->states = MatrixXd::Zero(capacity, stateSize);
    this->actions = VectorXi::Zero(capacity);
    this->rewards = VectorXd::Zero(capacity);
    this->dones = VectorXi::Zero(capacity);
    this->nextStates = MatrixXd::Zero(capacity, stateSize);

}

transitions Memory::sampleBatch(int nSamples) {
    vector<int> idxs;
    for (int i = 0; i < nSamples; i++) {
        int idx = randomInteger(0, this->size-2);
        idxs.push_back(idx);
    }
    transitions batch = {
        this->states(idxs, all),
        this->actions(idxs),
        this->rewards(idxs),
        this->dones(idxs),
        this->nextStates(idxs, all),
    };
    return batch;
}

void Memory::store(VectorXd state, int action, double reward, bool done, VectorXd nextState) {
    this->states.row(t) = state;
    this->actions.row(t) << action;
    this->rewards.row(t) << reward;
    this->dones.row(t) << int(done);
    this->nextStates.row(t) = nextState;

    this->t = (this->t + 1) % capacity;
    this->size = max(this->t, this->size);
}
