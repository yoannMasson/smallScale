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
#include <ctime>
#include <omp.h>

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

	this->M = M;
	this->N = N;
	this->L = L;
	this->irp = new int[M+1];
	this->as = new double[L];
	this->ja = new int[L];

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
	int position = 0;
	for(int currentRow = 1; currentRow <= M ; currentRow++ ){
		for(int i = 0 ; i < L ; i++){
			if(row[i] == currentRow ){
				this->as[position] = entry[i];
				this->ja[position] = column[i]-1;
				position++;
			}
		}
	}

	this->irp[0] = 0;
	int oldValue(0);
	position = 1;
	for(int i = 1 ; i <= M ; i++){//The row containing the next NN entry
		int nbValuePerRow = 0;
		for(int j = 0 ; j < L ; j++ ){//iterating through the row array to find the number of NN entry
			if(row[j] == i){
				nbValuePerRow++;
			}
		}
		this->irp[position] = oldValue+nbValuePerRow;
		position++;
		oldValue += nbValuePerRow;
	}

}

/**
 * Compute the product
 */
double CSR::serialVectorProduct(double* vector, double* solution){
	clock_t begin = clock();
	double t;
	for(int i = 0; i < (*this).M; i++ ){
		t = 0;
		for(int j = (*this).irp[i]; j <= (*this).irp[i+1]-1;j++){
			t += as[j]*vector[ja[j]];
		}
		solution[i]=t;
	}
	clock_t end = clock();
	return  double(end - begin) / CLOCKS_PER_SEC;
}


/**
 * Compute the vector product using OPEN MP
 */
double CSR::openMPVectorProduct(double* vector, double* solution){

	double t;
	int tid,i,j;
	clock_t begin = clock();
#pragma omp parallel num_threads(8) private(tid,i,j,t) shared(vector,solution)
	{
		tid = omp_get_thread_num();
#pragma omp for schedule(static,2)
		for(i = 0; i < (*this).M; i++ ){
			t = 0;
			for(j = (*this).irp[i]; j <= (*this).irp[i+1]-1;j++){
				t += as[j]*vector[ja[j]];
			}
			solution[i]=t;
		}
	}
	clock_t end = clock();
	return  double(end - begin) / CLOCKS_PER_SEC;
}



/**
 * Redefinition of the << operator.
 */
std::ostream& operator<<(std::ostream& os, CSR& obj)
{
	os << "M: " << obj.M << std::endl;
	os << "N: " << obj.N << std::endl;

	os << "AS: " << std::endl;
	for(int i = 0; i < obj.L; i ++){
		os << obj.as[i] << " ";
	}
	os << std::endl;

	os << "IRP: " << std::endl;
	for(int i = 0; i < obj.M+1; i ++){
		os << obj.irp[i] << " ";
	}
	os << std::endl;

	os << "JA: " << std::endl;
	for(int i = 0; i < obj.L ; i ++){
		os << obj.ja[i] << " ";
	}
	os << std::endl;
	return os;
}

int CSR::getM(){
	return this->M;
}

int CSR::getN(){
	return this->N;
}

CSR::~CSR() {


}

