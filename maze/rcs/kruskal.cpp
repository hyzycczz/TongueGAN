#include <ctime>
#include <iostream>
#include <vector>

#include "Generate_algorithm.h"
#include "Maze.h"
using namespace std;

void MazeByKruskal::kruskal() {
  srand(time(NULL));
  vector<int> vec;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      if (i & 1)
        maze[i * col + j] = WALL;
      else if (j & 1)
        maze[i * col + j] = WALL;
      else
        vec.push_back(i * col + j);
    }
  }
  while (1) {
    int dex = rand() % vec.size();
    int dir = rand() % 4;
    int index = vec[dex];
    if (invalid(index, dir)) continue;

    int set = 0;
    for (int i = 0; i < vec.size(); i++) {
      if (!maze[vec[i]]) set++;
    }
    if (set == 1) break;
    // printMaze();
    // system("pause");
  }
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      if (maze[i * col + j] > 0) maze[i * col + j] = ROUTE;
}

int MazeByKruskal::invalid(int index, int dir) {
  int Aroot, Broot;
  switch (dir) {
    case 0:  // left
      if (index % col == 0) return 1;
      Aroot = index, Broot = index - 2;
      while (maze[Aroot]) Aroot = maze[Aroot];
      while (maze[Broot]) Broot = maze[Broot];
      if (Broot == Aroot)
        return 1;
      else {
        maze[Broot] = Aroot;
        maze[index - 1] = ROUTE;
        return 0;
      }
      break;
    case 1:  // up
      if (index < col) return 1;
      Aroot = index, Broot = index - 2 * col;
      while (maze[Aroot]) Aroot = maze[Aroot];
      while (maze[Broot]) Broot = maze[Broot];
      if (Broot == Aroot)
        return 1;
      else {
        maze[Broot] = Aroot;
        maze[index - col] = ROUTE;
        return 0;
      }
      break;
    case 2:  // right
      if (index % col == col - 1) return 1;
      Aroot = index, Broot = index + 2;
      while (maze[Aroot]) Aroot = maze[Aroot];
      while (maze[Broot]) Broot = maze[Broot];
      if (Broot == Aroot)
        return 1;
      else {
        maze[Broot] = Aroot;
        maze[index + 1] = ROUTE;
        return 0;
      }
      break;
    case 3:  // down
      if (index >= (row - 1) * col) return 1;
      Aroot = index, Broot = index + 2 * col;
      while (maze[Aroot]) Aroot = maze[Aroot];
      while (maze[Broot]) Broot = maze[Broot];
      if (Broot == Aroot)
        return 1;
      else {
        maze[Broot] = Aroot;
        maze[index + col] = ROUTE;
        return 0;
      }
      break;
    default:
      break;
  }
}


/*
只能為基數*基數迷宮
kruskal 的終止條件

*/