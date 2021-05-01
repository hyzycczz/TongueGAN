#include "Maze.h"

#define START 2
#define END 2
#define abs(x) ((x > 0) ? x : -(x))

class mintomax {
 public:
  mintomax *next = NULL;
  mintomax(){};
  mintomax(int G, int H, int R, int C);
  int F = 0, G = 0, H = 0, R, C;
  void insert(int g, int h, int r, int c);
  int compare(mintomax *, mintomax *);
  void pop();
  void print();
  ~mintomax() {}
};

mintomax::mintomax(int G, int H, int R, int C) {
  this->G = G;  // distance from strat point
  this->H = H;  // distance to end point
  this->R = R;
  this->C = C;
  this->F = G + H;
}
int mintomax::compare(mintomax *cur, mintomax *newdata) {
  if(!cur) return 0;
  if (cur->F > newdata->F) {
    return 0;
  } else
    return 1;
}
void mintomax::insert(int g, int h, int r, int c) {
  mintomax *newdata = new mintomax(g, h, r, c);
  if (next == NULL) {
    next = newdata;
  } else {
    mintomax **tmp = &next;
    while (compare(*tmp, newdata)) {
      tmp = &((*tmp)->next);
    }
    newdata->next = (*tmp);
    (*tmp) = newdata;
  }
}

void mintomax::pop() {
  mintomax *tmp = next;
  next = next->next;
  delete tmp;
}
void mintomax::print() {
  mintomax *tmp = next;
  while (tmp) {
    cout << tmp->F<<' ';
    tmp = tmp->next;
  }
  cout << endl;
}

int Maze::findbyAstar(int r0, int c0, int r1, int c1) {
  maze[r0 * col + c0] = START;
  maze[r1 * col + c1] = ROUTE;
  mintomax close, open;
  int G = START, H = abs(r1 - r0) + abs(c1 - c0), R = r0, C = c0;
  //cout<<col;
  while (H != 0) {
    close.insert(G, H, R, C);
    maze[R * col + C] = G;
    
    int index = R * col + C;
    if (index % col != col - 1 && maze[R * col + C + 1] == ROUTE) {
      G = lookaroundandfindmin(R * col + C + 1) + 1;
      H = abs(R - r1) + abs(C + 1 - c1);
      open.insert(G, H, R, C + 1);
    }
    if (index % col != 0 && maze[R * col + C - 1] == ROUTE) {
      G = lookaroundandfindmin(R * col + C - 1) + 1;
      H = abs(R - r1) + abs(C - 1 - c1);
      open.insert(G, H, R, C - 1);
    }
    if (index >= col && maze[(R - 1) * col + C] == ROUTE) {
      G = lookaroundandfindmin((R - 1) * col + C) + 1;
      H = abs(R - 1 - r1) + abs(C - c1);
      open.insert(G, H, R - 1, C);
    }
    if (index < (row - 1) * col && maze[(R + 1) * col + C] == ROUTE) {
      G = lookaroundandfindmin((R + 1) * col + C) + 1;
      H = abs(R + 1 - r1) + abs(C - c1);
      open.insert(G, H, R + 1, C);
    }
    if(!open.next) break;
    mintomax *tmp = open.next;
    R = tmp->R;
    C = tmp->C;
    G = tmp->G;
    H = tmp->H;
    open.pop();
  }
  goback_goback(r1 * col + c1);
  maze[r1 * col + c1] = START;
}

#undef START
#undef END
#undef abs