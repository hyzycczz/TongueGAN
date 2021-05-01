#include "Maze.h"
#include "Generate_algorithm.h"

using namespace std;

void MazeByDivision::Division() {
  srand(time(NULL));
  subDivision(0, row, 0, col);
}

void MazeByDivision::subDivision(int R0, int R1, int C0, int C1) {
  //  12
  //  34
  if ((R1 - R0) <= 2 || (C1 - C0) <= 2) return;
  int r = R0 + (rand() % (R1 - 1 - R0));
  int c = C0 + (rand() % (C1 - 1 - C0));
  if (r == R0) r += 1;
  if (c == C0) c += 1;
  //cout << r << c << '\n';
  for (int i = R0; i < R1; i++) maze[i * col + c] = -1;
  for (int i = C0; i < C1; i++) maze[r * col + i] = -1;

  subDivision(R0, r, C0, c);          // 1
  subDivision(r + 1, R1, C0, c);      // 3
  subDivision(R0, r, c + 1, C1);      // 2
  subDivision(r + 1, R1, c + 1, C1);  // 4

  int dir = rand() % 4;  // l = 0 , r = 1, u = 2, d = 3
  for (int i = 0; i < 4; i++) {
    if (i == dir) i++;
    if (i == 4) break;
    int hole_r, hole_c;
    switch (i) {
      case 0:
        hole_c = rand() % (c - C0) + 1;
        while (walljudge(r, c - hole_c) < -2) {
          hole_c = rand() % (c - C0) + 1;
        }
        maze[r * col + c - hole_c] = 0;
        break;
      case 1:
        hole_c = rand() % (C1 - c - 1) + 1;
        while (walljudge(r, c + hole_c) < -2) {
          hole_c = rand() % (C1 - c - 1) + 1;
        }
        maze[r * col + c + hole_c] = 0;
        break;
      case 2:
        hole_r = rand() % (r - R0) + 1;
        while (walljudge(r - hole_r, c) < -2) {
          hole_r = rand() % (r - R0) + 1;
        }
        maze[(r - hole_r) * col + c] = 0;
        break;
      case 3:
        hole_r = rand() % (R1 - r - 1) + 1;
        while (walljudge(r + hole_r, c) < -2) {
          hole_r = rand() % (R1 - r - 1) + 1;
        }
        maze[(r + hole_r) * col + c] = 0;
        break;
      default:
        break;
    }
  }
}

int MazeByDivision::walljudge(int r, int c) {
  if (r + 1 == row)
    return maze[r * col + c + 1] + maze[r * col + c - 1] +
           maze[(r - 1) * col + c];
  if (r - 1 == -1)
    return maze[r * col + c + 1] + maze[r * col + c - 1] +
           maze[(r + 1) * col + c];
  if (c + 1 == col)
    return maze[r * col + c - 1] + maze[(r + 1) * col + c] +
           maze[(r - 1) * col + c];
  if (c - 1 == -1)
    return maze[r * col + c + 1] + maze[(r + 1) * col + c] +
           maze[(r - 1) * col + c];
  return maze[r * col + c + 1] + maze[r * col + c - 1] +
         maze[(r + 1) * col + c] + maze[(r - 1) * col + c];
}


/*
可能會遇到的問題:
無解:
  路有兩格寬
有解:
  打得洞剛好跟下一個division的牆壁重疊
*/