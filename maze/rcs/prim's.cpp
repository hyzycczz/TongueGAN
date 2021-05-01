#include "Maze.h"
#include "Generate_algorithm.h"
using namespace std;



void MazeByPrim::prim(int r, int c) {
  if (maze[r * col + c] == WALL) {
    cout << "you can't start here";
    return;
  }
  srand(time(NULL));
  vector<int> cache;
  cache.push_back(r * col + c);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      if (i & 1)
        maze[i * col + j] = WALL;
      else if (j & 1)
        maze[i * col + j] = WALL;
      else
        maze[i * col + j] = 2;
    }
  }
  while (cache.size()) {
    int dex = rand() % cache.size();
    int index = cache[dex];
    cache.erase(cache.begin() + dex);
    connect(index);
    for (int i = 0; i < 4; i++) {
      invalid(index, i, cache);
    }
  }
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      if (!(i & 1) && !(j & 1)) maze[i * col + j] = ROUTE;
    }
  }
}

void MazeByPrim::connect(int index) {
  if (index % col != 0 && maze[index - 2] == ROUTE)
    maze[index - 1] = ROUTE;
  else if (index >= col && maze[index - 2 * col] == ROUTE)
    maze[index - col] = ROUTE;
  else if (index % col != col - 1 && maze[index + 2] == ROUTE)
    maze[index + 1] = ROUTE;
  else if (index < (row - 1) * col && maze[index + 2 * col] == ROUTE)
    maze[index + col] = ROUTE;
  //else cout<<"i am in worst case";
  maze[index] = ROUTE;
  //cout<<endl;
}

int MazeByPrim::invalid(int index, int dir, vector<int> &vec) {
  int find = 0;
  switch (dir) {
    case 0:  // left
      if(index % col == 0 || maze[index-2] == ROUTE) break;
      for (int i = 0; i < vec.size(); i++) {
        if (vec[i] == index - 2) find = 1;
      }
      if(!find) vec.push_back(index - 2);
      break;   
    case 1:  // up
      if(index < col || maze[index-2*col] == ROUTE) break;
      for (int i = 0; i < vec.size(); i++) {
        if (vec[i] == index - 2 * col) find = 1;
      }
      if(!find) vec.push_back(index - 2 * col);
      break;
    case 2:  // right
      if(index % col == col -1 || maze[index+2] == ROUTE) break;
      for (int i = 0; i < vec.size(); i++) {
        if (vec[i] == index + 2) find = 1;
      }
      if(!find) vec.push_back(index + 2);
      break;
    case 3:  // down
      if(index >= (row - 1) * col || maze[index+2*col] == ROUTE) break;
      for (int i = 0; i < vec.size() && index < (row - 1) * col; i++) {
        if (vec[i] == index + 2 * col) find = 1;
      }
      if(!find) vec.push_back(index + 2 * col);
      break;
    default:
      break;
  }
}
