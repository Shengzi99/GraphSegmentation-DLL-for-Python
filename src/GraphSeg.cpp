#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <set>
#include <string>
#include <assert.h>
#include "segAlgorithm.h"
#include "gaussianFilter.h"

namespace py = pybind11;

class GraphSeg {
public:
	GraphSeg(py::array_t<double, py::array::c_style | py::array::forcecast> inImg);
	~GraphSeg();

	void segment(double k, double sigma, long min_size);
	py::array_t<long> getSeg();
	py::array_t<double> getFilteredImage();
	py::array_t<long> getSupPixelGraph(long edge_method);
	long getRegionNum() { return region_num; };
	py::array_t<long> getRegionSize();
private:
	const double* img;
	double* img_filtered;
	long* seg;
	long* supEdges;

	long ch;
	long w;
	long h;

	long region_num;
	long* region_size;
	long* segImage(const double* img, long ch, long h, long w, double k, long min_size);
	void gaussianFilter(double sigma);
};

GraphSeg::GraphSeg(py::array_t<double, py::array::c_style | py::array::forcecast> inImg) {
	const long ndim = (long)inImg.ndim();
	if (ndim != 3) {
		throw std::invalid_argument("expected input shape [channel, height, width]. for gray scale image input shape should be [1, height, width]");
	}
	else {
		py::buffer_info buff_inf = inImg.request();
		std::vector<pybind11::ssize_t> shape = buff_inf.shape;
		this->img = (const double*)buff_inf.ptr;
		this->ch = shape[0];
		this->h = shape[1];
		this->w = shape[2];
		this->img_filtered = new double[ch * h * w];
		this->seg = new long[this->w * this->h];
		this->region_num = this->w * this->h;
		this->region_size = NULL;
		this->supEdges = NULL;
	}
}

GraphSeg::~GraphSeg() {
	delete[] seg;
	delete[] img_filtered;
	delete[] supEdges;
	delete[] region_size;
}

void GraphSeg::segment(double k, double sigma, long min_size) {
	if (sigma != -1) {
		gaussianFilter(sigma);
		this->seg = segImage(this->img_filtered, ch, h, w, k, min_size);
	}
	else {
		this->seg = segImage(this->img, ch, h, w, k, min_size);
	}
}

py::array_t<long> GraphSeg::getSeg() {
	return py::array_t<long>(
		{ h, w },
		{ w * sizeof(long), sizeof(long) },
		seg);
}

py::array_t<double> GraphSeg::getFilteredImage() {
	return py::array_t<double>(
		{ch, h, w}, 
		{ h * w * sizeof(double), w * sizeof(double), sizeof(double) }, 
		img_filtered);
}

py::array_t<long> GraphSeg::getRegionSize() {

	return py::array_t<long>(
		{ region_num },
		{ sizeof(long) },
		region_size);
}

py::array_t<long> GraphSeg::getSupPixelGraph(long edge_method) {
	delete[] supEdges;
	long edgeNum = 0;

	if (edge_method == 0) {
		set<long>* nb = new set<long>[region_num];
		for (long i = 0; i < h; i++) {
			for (long j = 0; j < w; j++) {
				if ((i<h-1)&&(seg[(i + 1)*w + j] > seg[i * w + j])) {
					nb[seg[i * w + j]].insert(seg[(i + 1) * w + j]);
				}
				if ((i>0)&&(seg[(i - 1) * w + j] > seg[i * w + j])) {
					nb[seg[i * w + j]].insert(seg[(i - 1) * w + j]);
				}
				if ((j<w-1)&&(seg[i * w + j + 1] > seg[i * w + j])) {
					nb[seg[i * w + j]].insert(seg[i * w + j + 1]);
				}
				if ((j>0)&&(seg[i * w + j - 1] > seg[i * w + j])) {
					nb[seg[i * w + j]].insert(seg[i * w + j - 1]);
				}
			}
		}
		for (long i = 0; i < region_num; i++) {
			edgeNum += nb[i].size();
		}
		supEdges = new long[2 * edgeNum];

		long curIndex = 0;
		for (long i = 0; i < region_num; i++) {
			for (auto it = nb[i].begin(); it != nb[i].end();it++) {
				supEdges[curIndex] = i;
				curIndex++;
				supEdges[curIndex] = *it;
				curIndex++;
			}
		}

		delete[] nb;
	}
	else if (edge_method == 1){
		edgeNum = region_num * (region_num - 1) / 2;
		supEdges = new long[2 * edgeNum];
		long curIndex = 0;
		for (long i = 0; i < region_num; i++){
			for (long j = i + 1; j < region_num; j++){
				supEdges[curIndex] = i;
				curIndex++;
				supEdges[curIndex] = j;
				curIndex++;
			}
		}
		// assert((2 * edgeNum) == curIndex);
	}
	else {assert(0);}

	return py::array_t<long>(
		{ edgeNum,  2L},
		{ 2 * sizeof(long), sizeof(long) },
		supEdges);
}

void GraphSeg::gaussianFilter(double sigma) {
	gaussian_filter(img, img_filtered, ch, h, w, sigma);
}

// segent an [channel, height, width] shaped image
long* GraphSeg::segImage(const double* img, long ch, long h, long w, double k, long min_size) {
	delete[] region_size;
	region_size = NULL;

	long pixNum = h * w;
	edge* edges = new edge[pixNum * 4];
	long eNum = 0;
	for (long y = 0; y < h; y++) {
		for (long x = 0; x < w; x++) {
			if (x < w - 1) {
				edges[eNum].a = y * w + x;
				edges[eNum].b = y * w + (x + 1);
				edges[eNum].w = diff(img, ch, h, w, y, x, y, x + 1);
				eNum++;
			}
			if (y < h - 1) {
				edges[eNum].a = y * w + x;
				edges[eNum].b = (y + 1) * w + x;
				edges[eNum].w = diff(img, ch, h, w, y, x, y + 1, x);
				eNum++;
			}

			if ((x < w - 1) && (y < h - 1)) {
				edges[eNum].a = y * w + x;
				edges[eNum].b = (y + 1) * w + (x + 1);
				edges[eNum].w = diff(img, ch, h, w, y, x, y + 1, x + 1);
				eNum++;
			}

			if ((x < w - 1) && (y > 0)) {
				edges[eNum].a = y * w + x;
				edges[eNum].b = (y - 1) * w + (x + 1);
				edges[eNum].w = diff(img, ch, h, w, y, x, y - 1, x + 1);
				eNum++;
			}
		}
	}

	forest* F = segGraph(pixNum, eNum, edges, k);

	// merge region smaller than min_size
	for (long i = 0; i < eNum; i++) {
		long a = F->find(edges[i].a);
		long b = F->find(edges[i].b);
		if ((a != b) && ((F->size(a) < min_size) || (F->size(b) < min_size))) {
			F->join(a, b);
		}
	}

	delete[] edges;

	// get root index of every node
	long* result = new long[pixNum];
	for (long i = 0; i < pixNum; i++) {
		result[i] = F->find(i);
	}

	long* match = new long[pixNum];
	long* res_copy = new long[pixNum];

	// get a copy of result, and sort the copy
	for (long i = 0; i < pixNum; i++) {
		res_copy[i] = result[i];
	}
	sort(res_copy, res_copy + pixNum);

	// count the number of unique value in res_copy, and get the initial root number to ordered number map
	long count = 0;
	for (long i = 0; i < pixNum; i++) {
		match[res_copy[i]] = count;
		if ((i < pixNum -1 ) && (res_copy[i + 1] > res_copy[i]))
			count++;
	}

	// map result value to ordered number 
	for (long i = 0; i < pixNum; i++) {
		result[i] = match[result[i]];
	}

	region_num = count + 1;
	region_size = new long[region_num];
	for (long i = 0; i < pixNum; i++) {		
		long root = F->find(i);
		region_size[match[root]] = F->size(root);
	}

	delete F;
	delete[] match;
	delete[] res_copy;
	return result;
}

PYBIND11_MODULE(GraphSeg, m) {
	py::class_<GraphSeg>(m, ("GraphSeg"))
		.def(py::init< py::array_t<double, py::array::c_style | py::array::forcecast>>())
		.def("segment", &GraphSeg::segment)
		.def("getSeg", &GraphSeg::getSeg, py::return_value_policy::copy)
		.def("getFilteredImage", &GraphSeg::getFilteredImage, py::return_value_policy::copy)
		.def("getSupPixelGraph", &GraphSeg::getSupPixelGraph, py::return_value_policy::copy)
		.def("getRegionNum", &GraphSeg::getRegionNum)
		.def("getRegionSize", &GraphSeg::getRegionSize, py::return_value_policy::copy);

}