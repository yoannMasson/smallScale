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
	int M, N, L;//NUmber of row, number of column, number of NN entries
	int* column;
	int* row;
	double* entry;
	// Ignore headers and comments:
	while (fin.peek() == '%') fin.ignore(2048, '\n');

	// Read defining parameters:
	fin >> M >> N >> L;

	(*this).M = M;
	(*this).N = N;
	column = new int[L];
	row = new int[L];
	entry = new double[L];

	// Read the data
	for (int l = 0; l < L; l++)
	{
		int m,n;
		double data;
		fin >> m >> n >> data;
		row[l] = m;
		column[l] = n;
		entry[l] = data;
	}
	fin.close();

	//Parse into CSR
	for(int currentRow = 1; currentRow <= M ; currentRow++ ){
		for(int i = 0 ; i < L ; i++){
			if(row[i] == currentRow ){
				(*this).as.push_back(entry[i]);
				(*this).ja.push_back(column[i]);
			}
		}
	}

	(*this).irp.push_back(0);
	int oldValue(0);

	for(int i = 1 ; i <= M ; i++){//The row containing the next NN entry
		int nbValuePerRow = 0;
		for(int j = 0 ; j < N ; j++ ){//iterating through the row array to find the number of NN entry
			if(row[j] == i){
				nbValuePerRow++;
			}
		}
		(*this).irp.push_back(oldValue+nbValuePerRow);
		oldValue += nbValuePerRow;
	}


	for(int i = 0; i < (*this).as.size(); i ++){
		std::cout << (*this).as[i] << " ";
	}
	std::cout << std::endl;

	for(int i = 0; i < (*this).irp.size(); i ++){
			std::cout << (*this).irp[i] << " ";
		}
	std::cout << std::endl;

	for(int i = 0; i < (*this).ja.size(); i ++){
		std::cout << (*this).ja[i] << " ";
	}
	std::cout << std::endl;



}

CSR::~CSR() {
	// TODO Auto-generated destructor stub
}

