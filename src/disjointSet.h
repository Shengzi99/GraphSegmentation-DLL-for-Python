#pragma once
//NOTE: this disjoint set implementation only suit for GraphSeg usage, there's something different from original disjoint seg
typedef struct {
    long rank;
    long parent;
    long size;
} element;

class forest {
public:
    forest(long elementNum);
    ~forest();
    long find(long x);
    void join(long x, long y);
    long size(long x) const { return elts[x].size; }
    long num_sets() const { return regionNum; }

private:
    element* elts;
    long regionNum;
};

forest::forest(long elementNum) {
    elts = new element[elementNum];
    regionNum = elementNum;
    for (long i = 0; i < elementNum; i++) {
        elts[i].rank = 0;
        elts[i].size = 1;
        elts[i].parent = i;
    }
}

forest::~forest() {
    delete[] elts;
}

long forest::find(long x) {
    long y = x;
    while (y != elts[y].parent)
        y = elts[y].parent;
    elts[x].parent = y;
    return y;
}

void forest::join(long x, long y) {
    // it's worth noitice that, "join" here can only accept root element as input
    // , due to the special needs of graphseg
    if (elts[x].rank > elts[y].rank) {
        elts[y].parent = x;
        elts[x].size += elts[y].size;
    }
    else {
        elts[x].parent = y;
        elts[y].size += elts[x].size;
        if (elts[x].rank == elts[y].rank)
            elts[y].rank++;
    }
    regionNum--;
}
