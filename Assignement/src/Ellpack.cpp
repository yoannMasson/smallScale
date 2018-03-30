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

	this->M = M;
	this->N = N;
	this->L = L;


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
	//Initialization
	this->ja = new int*[this->MAXNZ];
	for(int i = 0; i < this->MAXNZ; ++i) {
		this->ja[i] = new int[this->M];
		for(int j = 0 ; j < this->M ; j++ ){
			this->ja[i][j] = -1;
		}
	}
	this->as = new double*[this->MAXNZ];
	for(int i = 0; i < this->MAXNZ; ++i) {
			this->as[i] = new double[this->M];
			for(int j = 0 ; j < this->M ; j++ ){
				this->as[i][j] = 0;
			}
		}


	//2D Array of coefficient
	for(int currentRow = 1; currentRow <= M ; currentRow++ ){
		int positionX = 0;
		for(int i = 0; i < L ; i++ ){
			if(row[i] == currentRow){
				this->as[positionX][currentRow-1] = entry[i];
				positionX++;
			}
		}
	}

	//2D Array of column indices
	for(int currentRow = 1; currentRow <= M ; currentRow++ ){
		int positionX = 0;
		for(int i = 0; i < L ; i++ ){
			if(row[i] == currentRow){
				this->ja[positionX][currentRow-1] = column[i]-1;
				positionX++;
			}
		}
	}
}

/**
 * Redefinition of the << operator.
 */
std::ostream& operator<<(std::ostream& os, Ellpack& obj)
{
	os << "M: " << obj.M << std::endl;
	os << "N: " << obj.N << std::endl;
	os << "L: " << obj.L << std::endl;
	os << "MAXNZ: " << obj.MAXNZ << std::endl;

	os << "JA: " << std::endl;
	for(int i = 0; i < obj.MAXNZ; i ++){
		for(int j = 0; j < obj.M ; j++ ){
			os << obj.ja[i][j] << " ";
		}
		os << std::endl;
	}
	os << std::endl;

	os << "AS: " << std::endl;
	for(int i = 0; i < obj.MAXNZ; i ++){
		for(int j = 0; j < obj.M ; j++ ){
			os << obj.as[i][j] << " ";
		}
		os << std::endl;
	}
	os << std::endl;

	os << std::endl;
	return os;
}


Ellpack::~Ellpack() {
	// TODO Auto-generated destructor stub
}

