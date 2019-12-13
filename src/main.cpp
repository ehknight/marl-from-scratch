/*
 * CS 106B/X Sample Project
 * last updated: 2018/09/19 by Marty Stepp
 *
 * This project helps test that your Qt Creator system is installed correctly.
 * Compile and run this program to see a console and a graphical window.
 * If you see these windows, your Qt Creator is installed correctly.
 */

#include <iostream>
#include "game.h"
#include "agent.h"
#include "memory.h"
#include "optimizer.h"

// from https://stackoverflow.com/a/8666442.
// needed because linux / mac don't have
// a windows.h (shocker).
#ifdef _WIN32
#include "windows.h"
#endif

using namespace std;

const int BATCH_SIZE = 64;
const double LEARNING_RATE = 1e-3;
const double EPS_DECAY = 0.995;
const double GAMMA = 0.995;
const vector<int> HIDDENS = {128, 64};

const int EPISODES = 1500;
const int EPISODE_LENGTH = 400;
const int TRAIN_EVERY = 4;
const int TRAIN_WAIT = 7500;
const int UPDATE_TARGET_EVERY = 25;
const int RENDER_EVERY = 100;

const int MAX_X = 1000;
const int MAX_Y = 500;

const int SAVE_EVERY = 10;
const string TAGGER_SAVE_TO = "savedTagger";
const string TAGGEE_SAVE_TO = "savedTaggee";
const string TAGGER_LOAD_FROM = "savedTagger";
const string TAGGEE_LOAD_FROM = "savedTaggee";

const bool LOAD = false; // load previously saved network
const int EPS_UNTIL_HUMAN_CONTROL = LOAD ? 0 : 750;
const Controllable HUMAN_CONTROL = Controllable::Taggee;

// NOTE: KEYBOARD INPUT ONLY WORKS ON WINDOWS MACHINES.
// Unfortunately, getting the key state on Linux or Mac is extremely hard...
int getInput() {
    // WASD input
    // ref: https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getkeystate
    #ifdef _WIN32
    if (GetKeyState(0x57) < 0) return 0; // up
    if (GetKeyState(0x53) < 0) return 1; // down
    if (GetKeyState(0x41) < 0) return 2; // left
    if (GetKeyState(0x44) < 0) return 3; // right
    #endif
    return 4; // no op
}

int main() {
    player tagger = {30, 20, 2.5, {0, 0}, "red"};
    player taggee = {30, 12, 7, {0, 0}, "blue"};
    GWindow gw(MAX_X, MAX_Y);
    Game game(&gw, MAX_X, MAX_Y, tagger, taggee);

    int stateSize = 10;
    int nActions = 5;
    Agent taggerA = buildAgent(stateSize, nActions, EPS_DECAY, GAMMA, 0, HIDDENS);
    Agent taggeeA = buildAgent(stateSize, nActions, EPS_DECAY, GAMMA, 0, HIDDENS);
    Memory taggerMemory(stateSize, 100000);
    Memory taggeeMemory(stateSize, 100000);
    Optimizer* taggerOpt = new Adam(LEARNING_RATE);
    Optimizer* taggeeOpt = new Adam(LEARNING_RATE);

    // Load if necessary
    if (LOAD) {
        // Read Q-networks
        taggerA.qNet->read(TAGGER_SAVE_TO);
        taggeeA.qNet->read(TAGGEE_SAVE_TO);
        taggerA.qTarg->read(TAGGER_SAVE_TO);
        taggeeA.qTarg->read(TAGGEE_SAVE_TO);
        // Set randomization to zero
        taggerA.eps = 0.05;
        taggeeA.eps = 0.05;
        cout << "Loaded succesfully." << endl;
    }

    int step = 0;
    VectorXd state = stateToVec(game.reset());
    for (int ep = 0; ep < EPISODES; ep++) {
        int epLen = 0;
        double taggerCost = 0;
        double taggeeCost = 0;
        double taggerEpRet = 0;
        double taggeeEpRet = 0;
        cout << "Episode: " << ep << " | Epsilon: " << taggerA.eps << " | ";
        state = stateToVec(game.reset());

        for (int i = 0; i < EPISODE_LENGTH; i++) {
            int taggerAction = taggerA.act(state);
            int taggeeAction = taggeeA.act(state);
            transition t = game.step(taggerAction, taggeeAction);

            VectorXd nextState = stateToVec(t.s);
            taggerMemory.store(state, taggerAction, t.taggerR, t.d, nextState);
            taggeeMemory.store(state, taggeeAction, t.taggeeR, t.d, nextState);

            taggerEpRet += t.taggerR;
            taggeeEpRet += t.taggeeR;
            state = nextState;

            // Train!
            if ((step % TRAIN_EVERY == 0 && step > TRAIN_WAIT)
                 && !(ep >= EPS_UNTIL_HUMAN_CONTROL)) {
                transitions taggerBatch = taggerMemory.sampleBatch(BATCH_SIZE);
                transitions taggeeBatch = taggeeMemory.sampleBatch(BATCH_SIZE);
                taggerCost = taggerA.fit(taggerOpt, taggerBatch);
                taggeeCost = taggeeA.fit(taggeeOpt, taggeeBatch);
             }

            if (t.d) break;
            step++;
            epLen++;
        }
        // Visualize every so often
        if (ep % RENDER_EVERY == 0 || ep >= EPS_UNTIL_HUMAN_CONTROL) {
            for (int testEp = 0; testEp < 1; testEp++) {
                state = stateToVec(game.reset());
                for (int i = 0; i < EPISODE_LENGTH; i++) {
                    int taggerAction = taggerA.act(state, true);
                    int taggeeAction = taggeeA.act(state, true);

                    // Human input
                    if (ep >= EPS_UNTIL_HUMAN_CONTROL) {
                        int input = getInput();
                        if (input == 4); // no op
                        else if (HUMAN_CONTROL == Controllable::Tagger)
                            taggerAction = input;
                        else if (HUMAN_CONTROL == Controllable::Taggee)
                            taggeeAction = input;
                        gw.pause(20);
                    }

                    transition t = game.step(taggerAction, taggeeAction);
                    state = stateToVec(t.s);

                    cout << "Tagger Reward: " << t.taggerR << " | Taggee Reward: " << t.taggeeR << " | Move: ";
                    if (taggerAction == GameMove::Up) cout << "^";
                    if (taggerAction == GameMove::Down) cout << "v";
                    if (taggerAction == GameMove::Left) cout << "<";
                    if (taggerAction == GameMove::Right) cout << ">";
                    cout << endl;

                    game.render();
                    gw.pause(10);
                    if (t.d) break;
                }
            }
        }

        // Update target net
        if (ep % UPDATE_TARGET_EVERY == 0) {
            taggerA.updateTargetNet(true);
            taggeeA.updateTargetNet(true);
        }

        // Save models
        if (ep % SAVE_EVERY == 0) {
            taggerA.qNet->write(TAGGER_SAVE_TO);
            taggeeA.qNet->write(TAGGEE_SAVE_TO);
            cout << "Saved models" << endl;
        }

        taggerA.decayEpsilon();
        taggeeA.decayEpsilon();
        cout << "EpLen: " << epLen << " | ";
        cout << "Tagger Loss: " << taggerCost << " | ";
        cout << "Taggee Loss: " << taggeeCost << " | ";
        cout << "Tagger Return: " << taggerEpRet << " | ";
        cout << "Taggee Return: " << taggeeEpRet << endl;
    }

    return 0;
}
