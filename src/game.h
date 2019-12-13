#pragma once
#include <Eigen/Dense>
#include "gwindow.h"
#include "glabel.h"

using namespace Eigen;

enum GameMove {Up, Down, Left, Right, none};
enum Controllable {Tagger, Taggee, Nobody};

struct coord { double x; double y; };

struct state {
    coord taggerLocation;
    coord taggeeLocation;
    coord taggerVelocity;
    coord taggeeVelocity;
    double maxX;
    double maxY;
};

struct transition {
    state s;
    double taggerR;
    double taggeeR;
    bool d;
};

struct player {
    double size;
    double maxSpeed;
    double accel;
    coord vel;
    std::string color;
};

VectorXd stateToVec(state s);

/*
 * This tag game was inspired by TagPro,
 * which can be played here:
 * https://tagpro.koalabeast.com/
 */
class Game {
public:
    Game(GWindow* gw, int maxX, int maxY, player tagger, player taggee);
    ~Game();
    void render();
    state reset();
    void taggerStep(GameMove move);
    void taggeeStep(GameMove move);
    transition step(GameMove taggerMove, GameMove taggeeMove);
    transition step(int taggerMove, int taggeeMove);

private:
    int maxX;
    int maxY;
    coord taggerPos;
    coord taggeePos;
    player tagger;
    player taggee;
    GameMove taggerAction;
    GameMove taggeeAction;
    double lastDist;

    GOval* taggerOval;
    GOval* taggeeOval;
    GLabel* taggerLabel;
    GLabel* taggeeLabel;
};
