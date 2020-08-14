#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <set>
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
	py::array_t<long> getSupPixelGraph();
	long getRegionNum() { return region_num; };
private:
	const double* img;
	double* img_filtered;
	long* seg;
	long* supEdges;

	long ch;
	long w;
	long h;

	long region_num;
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
	}
}

GraphSeg::~GraphSeg() {
	delete[] seg;
	delete[] img_filtered;
	delete[] supEdges;
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

py::array_t<long> GraphSeg::getSupPixelGraph() {
	delete[] supEdges;
	set<long>* nb = new set<long>[region_num];

	long edgeNum = 0;
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
	edge* edges = new edge[h * w * 4];
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

	forest* F = segGraph(h * w, eNum, edges, k);

	for (long i = 0; i < eNum; i++) {
		long a = F->find(edges[i].a);
		long b = F->find(edges[i].b);
		if ((a != b) && ((F->size(a) < min_size) || (F->size(b) < min_size))) {
			F->join(a, b);
		}
	}

	delete[] edges;

	long* result = new long[h * w];
	for (long y = 0; y < h; y++) {
		for (long x = 0; x < w; x++) {
			result[y * w + x] = F->find(y * w + x);
		}
	}
	long* match = new long[h * w];
	long* res_copy = new long[h * w];
	for (long i = 0; i < h * w; i++) {
		res_copy[i] = result[i];
	}
	sort(res_copy, res_copy + (h * w));

	long count = 0;
	for (long i = 0; i < h * w; i++) {
		match[res_copy[i]] = count;
		if ((i < h * w -1 ) && (res_copy[i + 1] > res_copy[i]))
			count++;
	}
	for (long i = 0; i < h * w; i++) {
		result[i] = match[result[i]];
	}

	region_num = count + 1;

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
		.def("getRegionNum", &GraphSeg::getRegionNum);

}