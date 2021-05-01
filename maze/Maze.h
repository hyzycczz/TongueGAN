#ifndef _GLIBCXX_VECTOR
#include <vector>
#endif

#ifndef _GLIBCXX_IOSTREAM
#include<iostream>
#endif

#ifndef _GLIBCXX_QUEUE
#include <queue>
#endif

#ifndef minired_maze
#define minired_maze

#define WALL -1
#define ROUTE 0
#define PATH 1

using namespace std;

class Maze {
 protected:
  vector<int> maze;
  int col, row;

 public:
  Maze(int row, int col);
  void init();
  void printMaze();
  int findbyrecusive(int, int, int, int);
  int findbyDijkstra(int, int, int, int);
  int findbyAstar(int, int, int, int);
  ~Maze();
 private:
  int subfindbyrecusive(int, int, int, int);
  int checkandadd(queue<int>&, int, int);
  void goback_goback(int);
  int lookaroundandfindmin(int);
};

#endif

cout << "hi 韜雨";