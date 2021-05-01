#include "Maze.h"
#define START 2
#define END 2

Maze::Maze(int row, int col) {
  for (int i = 0; i < row * col; i++) maze.push_back(0);
  Maze::col = col;
  Maze::row = row;
}
void Maze::init() {
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++) maze[i * col + j] = 0;
}

void Maze::clear() {
  for (int i = 0; i < row * col; i++) {
    if (maze[i] != WALL) maze[i] = ROUTE;
  }
}

void Maze::printMaze() {
  for (int i = 0; i < col + 2; i++) cout << "XX";
  cout << '\n';
  for (int i = 0; i < row; i++) {
    cout << "XX";
    for (int j = 0; j < col; j++) {
      if (maze[i * col + j] == WALL)
        cout << "XX";
      else if (maze[i * col + j] == ROUTE)
        cout << "  ";
      else if (maze[i * col + j] == PATH)
        cout << "* ";
      else {
        if (maze[i * col + j] >= 10)
          cout << maze[i * col + j];
        else
          cout << maze[i * col + j] << " ";
      }  // if(maze[i * col + j] == 2)
    }
    cout << "XX\n";
  }
  for (int i = 0; i < col + 2; i++) cout << "XX";
}

Maze::~Maze() {}
/*--------------------------path algorithm-----------------------------*/

int Maze::findbyrecusive(int r0, int c0, int r1, int c1) {
  maze[r0 * col + c0] = ROUTE;
  maze[r1 * col + c1] = ROUTE;
  if (subfindbyrecusive(r0, c0, r1, c1)) {
    maze[r0 * col + c0] = START;
    maze[r1 * col + c1] = END;
    return 1;
  } else
    return 0;
}

/*----------------------private function---------------------------*/
int Maze::subfindbyrecusive(int r0, int c0, int r1, int c1) {
  if (r0 >= row || r0 < 0 || c0 >= col || c0 < 0) return 0;
  if (r0 == r1 && c0 == c1) {
    return 1;
  }
  if (!maze[r0 * col + c0])
    maze[r0 * col + c0] = PATH;
  else
    return 0;
  if (subfindbyrecusive(r0, c0 + 1, r1, c1) ||
      subfindbyrecusive(r0 + 1, c0, r1, c1) ||
      subfindbyrecusive(r0, c0 - 1, r1, c1) ||
      subfindbyrecusive(r0 - 1, c0, r1, c1))
    return 1;
  else
    return 0;
}

void Maze::goback_goback(int destination) {
  int index = destination;
  while (maze[index] != START) {
    int min = lookaroundandfindmin(index);
    if (min == 0) break;
    if (index % col != col - 1 && maze[index + 1] == min) {
      maze[index] = PATH;
      index += 1;
    } else if (index < col * (row - 1) && maze[index + col] == min) {
      maze[index] = PATH;
      index += col;
    } else if (index % col != 0 && maze[index - 1] == min) {
      maze[index] = PATH;
      index -= 1;
    } else if (index >= col && maze[index - col] == min) {
      maze[index] = PATH;
      index -= col;
    }
  }
  for (int i = 0; i < row * col; i++) {
    if (maze[i] > 2) {
      maze[i] = ROUTE;
    }
  }
}

int Maze::lookaroundandfindmin(int index) {
  int min = 0;
  if (index % col != col - 1 && maze[index + 1] > PATH) {
    if (!min) min = maze[index + 1];
  }
  if (index % col != 0 && maze[index - 1] > PATH) {
    if (min > maze[index - 1] || !min) min = maze[index - 1];
  }
  if (index >= col && maze[index - col] > PATH) {
    if (min > maze[index - col] || !min) min = maze[index - col];
  }
  if (index < (row - 1) * col && maze[index + col] > PATH) {
    if (min > maze[index + col] || !min) min = maze[index + col];
  }
  return min;
}

#undef START
#undef END