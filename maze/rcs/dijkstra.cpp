#include "Maze.h"

#define START 2
#define END 2

int Maze::findbyDijkstra(int r0, int c0, int r1, int c1) {
  maze[r0 * col + c0] = START;
  maze[r1 * col + c1] = ROUTE;
  queue<int> cache;
  cache.push(r0 * col * c0);
  while (cache.size()) {
    if(checkandadd(cache, r1, c1)){
      break;
    }
    cache.pop();
  }
  maze[r1 * col + c1] = END;
  /*-----------------if somethings wrong--------------------
  for (int i = 0; i < row * col; i++) {
    if (maze[i] > 2) {
      for (int i = 0; i < col + 2; i++) cout << "XXXX";
      cout << '\n';
      for (int i = 0; i < row; i++) {
        cout << "XXXX";
        for (int j = 0; j < col; j++) {
          printf("%4d",maze[i * col + j]);
        }
        cout << "XXXX\n";
      }
      for (int i = 0; i < col + 2; i++) cout << "XXXX";
      break;
    }
  }
  -------------------------------------------------------*/
}

int Maze::checkandadd(queue<int> &cache, int r1, int c1) {
  int destination = r1 * col + c1;
  int index = cache.front();

  if (index % col != col - 1 && maze[index + 1] == ROUTE) {
    maze[index + 1] = maze[cache.front()] + 1;
    cache.push(index + 1);
    if (index + 1 == destination) {
      goback_goback(destination);
      return 1;
    }
  }
  if (index < col * (row - 1) && maze[index + col] == ROUTE) {
    maze[index + col] = maze[cache.front()] + 1;
    cache.push(index + col);
    if (index + col == destination){
      goback_goback(destination);
      return 1;
    }
  }
  if (index % col != 0 && maze[index - 1] == ROUTE) {
    maze[index - 1] = maze[cache.front()] + 1;
    cache.push(index - 1);
    if (index - 1 == destination) {
      goback_goback(destination);
      return 1;
    }
  }
  if (index >= col && maze[index - col] == ROUTE) {
    maze[index - col] = maze[cache.front()] + 1;
    cache.push(index - col);
    if (index - col == destination) {
      goback_goback(destination);
      return 1;
    }
  }
  return 0;
}


#undef START
#undef END