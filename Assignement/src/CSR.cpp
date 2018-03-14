/*
 * CSR.cpp
 *
 *  Created on: 14 mars 2018
 *      Author: yoann
 */

#include "CSR.h"
#include <string>
#include <fstream>
#include <algorithm>
#include <iostream>

CSR::CSR(const std::string filePath) {

	// Open the file:
	std::ifstream fin(filePath.c_str());

	// Declare variables:
	int M, N, L;

	// Ignore headers and comments:
	while (fin.peek() == '%') fin.ignore(2048, '\n');

	// Read defining parameters:
	fin >> M >> N >> L;

	(*this).M = M;
	(*this).N = N;
	std::cout << "size:" << L << std::endl;
	// Read the data
	for (int l = 0; l < L; l++)
	{
		int m, n;
		double data;
		fin >> m >> n >> data;
		(*this).a.push_back(data);
		std::cout << "m[" << m << "]["<< n << "]: " << data << std::endl;

	}

	fin.close();
}

CSR::~CSR() {
	// TODO Auto-generated destructor stub
}

