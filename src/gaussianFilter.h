#pragma once
#include<iostream>
using namespace std;

void gaussian_filter(const double* src, double* dst, long channel, long height, long width, double sigma) {
	// generate 1-dimensional kernel
	sigma = max(sigma, 0.01);
	double kernel[2] = { exp(-0.5 * pow((0 / sigma), 2)), exp(-0.5 * pow((1 / sigma), 2))};

	//normalize kernel
	double sum = kernel[0] + 2 * kernel[1];
	kernel[0] /= sum;
	kernel[1] /= sum;

	long hTw = height * width;
	//do row filter for every channel
	double* temp = new double[channel * hTw];
	for (long ch = 0; ch < channel; ch++) {
		const double* curP_s = src + ch * hTw;
		double* curP_t = temp + ch * hTw;
		for (long h = 0; h < height; h++) {
			for (long w = 0; w < width; w++) {
				curP_t[h * width + w] = 
					kernel[0] * curP_s[h * width + w] +
					kernel[1] * (curP_s[h * width + max(w - 1, 0L)] + curP_s[h * width + min(w + 1, width-1)]);
			}
		}
	}

	//do column filter for every channel
	for (long ch = 0; ch < channel; ch++) {
		double* curP_t = temp + ch * hTw;
		double* curP_d = dst + ch * hTw;
		for (long h = 0; h < height; h++) {
			for (long w = 0; w < width; w++) {
				curP_d[h * width + w] = 
					kernel[0] * curP_t[h * width + w] +
					kernel[1] * (curP_t[max(h - 1, 0L) * width + w] + curP_t[min(h + 1, height - 1) * width + w]);
			}
		}
	}
	delete[] temp;
}