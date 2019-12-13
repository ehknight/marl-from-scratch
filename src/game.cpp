#include "game.h"
#include "math.h"
#include "random.h"
using namespace std;

const bool TORUS = true;

double clamp(double x, double low, double high) {
    return std::max(low, std::min(x, high));
}

double vecDistance(double x1, double x2, double maxX,
                   double y1, double y2, double maxY) {
    if (TORUS)
        // distance along a torus
        // modified from https://stackoverflow.com/a/2123977
        return sqrt(pow(min(abs(x1 - x2), maxX - abs(x1 - x2)), 2)
                  + pow(min(abs(y1 - y2), maxY - abs(y1 - y2)), 2));
    else
        return sqrt(pow(x2-x1, 2) + pow(y2-y1, 2));
}

VectorXd stateToVec(state s) {
    VectorXd v(10);
    v << s.taggerLocation.x,
         s.taggerLocation.y,
         s.taggeeLocation.x,
         s.taggeeLocation.y,

         s.taggerVelocity.x,
         s.taggerVelocity.y,
         s.taggeeVelocity.x,
         s.taggeeVelocity.y,

         s.maxX,
         s.maxY;
    return v;
}

string acStr(GameMove move) {
    if (move == GameMove::Up) return "^";
    else if (move == GameMove::Down) return "v";
    else if (move == GameMove::Left) return "<";
    else if (move == GameMove::Right) return ">";
    return "x";
}

coord playerStep(player& p, GameMove move, coord l, int maxX, int maxY) {
    coord accel = {0, 0};
    if (move == GameMove::Up) accel = {0, -p.accel};
    else if (move == GameMove::Down) accel = {0, p.accel};
    else if (move == GameMove::Left) accel = {-p.accel, 0};
    else if (move == GameMove::Right) accel = {p.accel, 0};
    else if (move == GameMove::none) accel = {0, 0};
    else {
        std::cout << "Unknown move: " << move << std::endl;
        assert(0);
    }

    coord vel = {
       clamp(accel.x + p.vel.x, -p.maxSpeed, p.maxSpeed) * 0.9,
       clamp(accel.y + p.vel.y, -p.maxSpeed, p.maxSpeed) * 0.9 //drag
    };
    p.vel = vel;

    if (TORUS) {
        double newX = l.x + vel.x;
        double newY = l.y + vel.y;
        return {
            newX > maxX ? newX - maxX : (newX < 0 ? maxX - newX : newX),
            newY > maxY ? newY - maxY : (newY < 0 ? maxY - newY : newY),
        };
    } else {
        double playerMaxX = double(maxX) - p.size;
        double playerMaxY = double(maxY) - p.size;
        return {
            clamp(l.x + vel.x, p.size, playerMaxX),
            clamp(l.y + vel.y, p.size, playerMaxY)
        };
    }
}

Game::Game(GWindow* gw, int maxX, int maxY, player tagger, player taggee) {
    this->maxX = maxX;
    this->maxY = maxY;
    this->tagger = tagger;
    this->taggee = taggee;
    reset();

    taggerLabel = new GLabel("");
    taggeeLabel = new GLabel("");
    taggerLabel->setFont("SansSerif-16");
    taggeeLabel->setFont("SansSerif-16");
    gw->add(taggerLabel, taggerPos.x, taggerPos.y);
    gw->add(taggeeLabel, taggeePos.x, taggeePos.y);

    taggerOval = new GOval(taggerPos.x, taggerPos.y, tagger.size, tagger.size);
    taggeeOval = new GOval(taggeePos.x, taggeePos.y, taggee.size, taggee.size);
    taggerOval->setFillColor(tagger.color);
    taggeeOval->setFillColor(taggee.color);
    gw->add(taggerOval);
    gw->add(taggeeOval);
}

Game::~Game() {
    delete taggerLabel;
    delete taggeeLabel;
    delete taggerOval;
    delete taggeeOval;
}

state Game::reset() {
    double gutterSpace = (tagger.size + taggee.size);
    this->taggerPos = {randomReal(gutterSpace, double(maxX) - gutterSpace),
                 randomReal(tagger.size, maxY - tagger.size)};
    this->taggeePos = {randomReal(gutterSpace, double(maxX) - gutterSpace),
                 randomReal(taggee.size, maxY - taggee.size)};
    this->lastDist = vecDistance(taggeePos.x, taggerPos.x, maxX,
                                 taggeePos.y, taggerPos.y, maxY);

    state s = {this->taggerPos, this->taggeePos,
               this->tagger.vel, this->taggee.vel,
               double(maxX), double(maxY)};
    return s;
}

void Game::render() {
    this->taggerOval->setLocation(taggerPos.x, taggerPos.y);
    this->taggeeOval->setLocation(taggeePos.x, taggeePos.y);

    this->taggerLabel->setLocation(taggerPos.x, taggerPos.y);
    this->taggeeLabel->setLocation(taggeePos.x, taggeePos.y);
    this->taggerLabel->setLabel(" " + acStr(taggerAction));
    this->taggeeLabel->setLabel(" " + acStr(taggeeAction));
}

void Game::taggerStep(GameMove move) {
    this->taggerPos = playerStep(tagger, move, taggerPos, maxX, maxY);
}

void Game::taggeeStep(GameMove move) {
    this->taggeePos = playerStep(taggee, move, taggeePos, maxX, maxY);
}

transition Game::step(GameMove taggerMove, GameMove taggeeMove) {
    this->taggerAction = taggerMove;
    this->taggeeAction = taggeeMove;
    taggerStep(taggerMove);
    taggeeStep(taggeeMove);

    // Calculate change in distance
    double dist = vecDistance(taggeePos.x, taggerPos.x, maxX,
                              taggeePos.y, taggerPos.y, maxY);
    double dDist = dist - this->lastDist;
    this->lastDist = dist;

    // Calculate if the balls touch
    bool touching = (dist <= (tagger.size + taggee.size) / (2.));

    // Define state
    state s = {this->taggerPos, this->taggeePos,
               this->tagger.vel, this->taggee.vel,
               double(maxX), double(maxY)};

    if (touching) return {s, 1000, -1000, true};
    else return {s, -dDist, dDist, false};
}

transition Game::step(int taggerMove, int taggeeMove) {
    // enum cast syntax: https://stackoverflow.com/a/11453291
    return Game::step(static_cast<GameMove>(taggerMove),
                      static_cast<GameMove>(taggeeMove));
}
