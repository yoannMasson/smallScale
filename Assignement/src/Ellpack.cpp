/*
 * Ellpack.cpp
 *
 *  Created on: 29 mars 2018
 *      Author: yoann
 */

#include "Ellpack.h"
#include <string>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <omp.h>

Ellpack::Ellpack(const std::string filePath) {
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
	for (int l = 0; l < L; l++){
		int m,n;
		double data;
		fin >> m >> n >> data;
		row[l] = m;
		column[l] = n;
		entry[l] = data;
	}
	fin.close();

	//Parse into Ellpack

	//Finding MAXNZ
	this->MAXNZ=0;
	int nbValuePerRow[M];
	for(int i = 0 ; i <= M ; i++ ) nbValuePerRow[i] = 0;
	for(int i = 0; i < L ; i++){
		nbValuePerRow[row[i]] ++ ;
	}
	for(int i = 1 ; i <= M ; i++ ){
		if(nbValuePerRow[i] > this->MAXNZ){
			this->MAXNZ = nbValuePerRow[i];
		}
	}

	//2D Array of coefficient
	for(int currentRow = 1; currentRow <= M ; currentRow++ ){
		for(int i = 0; i < L ; i++ ){
			if(row[i] == currentRow){

			}
		}
	}


}


Ellpack::~Ellpack() {
	// TODO Auto-generated destructor stub
}

