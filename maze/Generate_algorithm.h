#ifndef _GLIBCXX_CTIME
#include <ctime>
#endif
#ifndef minired_maze
#include "Maze.h"
#endif
class MazeByRandom : public Maze {
 public:
  MazeByRandom(int row, int col, int percent) : Maze(row, col) {
    srand(time(NULL));
    for(int i = 0; i < row*col; i++){
      int t = rand()%100;
      if(t < percent) maze[i] = WALL;
    }
  }
};

class MazeByKruskal : public Maze {
 public:
  MazeByKruskal(int row, int col) : Maze(row, col) {}
  void kruskal();
  int invalid(int, int);
};

class MazeByPrim : public Maze {
 public:
  MazeByPrim(int row, int col) : Maze(row, col) {}
  void prim(int r, int c);

 private:
  void connect(int index);
  int invalid(int, int, vector<int> &);
};

class MazeByDivision: public Maze {
 public:
  MazeByDivision(int row, int col):Maze(row, col){}
  void Division();
  void subDivision(int, int, int, int);
  int walljudge(int, int);
};