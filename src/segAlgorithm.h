#pragma once
#include <algorithm>
#include <cmath>
#include "disjointSet.h"
using namespace std;
/*
	implementation of Felzenszwalb et.al(2004). "Efficient graph-based image segmentation." IJCV
*/

typedef struct {
	double w;
	int a, b;
} edge;

bool operator<(const edge& a, const edge& b) {
	return a.w < b.w;
}

double diff(const double* img, long channel, long height, long width, long h1, long w1, long h2, long w2) {
	long hTw = height * width;
	double squareSum = 0;
	for (int ch = 0; ch < channel; ch++) {
		int indx1 = ch * hTw +h1 * width + w1;
		int indx2 = ch * hTw +h2 * width + w2;
		squareSum += (pow(img[indx1] - img[indx2], 2));
	}
	return sqrt(squareSum);
}

// GraphSeg core algorithm
forest* segGraph(long vNum, long eNum, edge* edges, double k) {
	sort(edges, edges + eNum);
	forest* F = new forest(vNum);

	double* threshold = new double[vNum];
	for (int i = 0; i < vNum; i++)
		threshold[i] = k;

	for (int i = 0; i < eNum; i++) {
		edge* edge_ptr = &edges[i];

		int a = F->find(edge_ptr->a);
		int b = F->find(edge_ptr->b);
		if (a != b) {
			if ((edge_ptr->w <= threshold[a]) && (edge_ptr->w <= threshold[b])) {
				F->join(a, b);
				a = F->find(a);
				threshold[a] = edge_ptr->w + (k / F->size(a));
			}
		}
	}
	delete[] threshold;
	return F;
}