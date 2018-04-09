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
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>  // For CUDA runtime API

__global__ void gpuVectorProduct(double* as,int* ja,int* irp, int M, int L,double* vector, double* solution){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < M){
		double t = 0;
		for(int j = irp[idx]; j <= irp[idx+1]-1;j++){
			t += as[j]*vector[ja[j]];
		}
		solution[idx]=t;
	}
}


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
	return  double(end - begin) ;
}


/**
 * Compute the vector product using OPEN MP
 */
double CSR::openMPVectorProduct(double* vector, double* solution,int nCore){

	
	double begin = omp_get_wtime();
   #pragma omp parallel num_threads(nCore) shared(vector,solution)
    {
	double t;
	int chunk = this->M/omp_get_num_threads();
#pragma omp for  schedule(static,chunk)
		for(int i = 0; i < (*this).M; i++ ){
			t = 0;
			for(int j = (*this).irp[i]; j <= (*this).irp[i+1]-1;j++){
				t += as[j]*vector[ja[j]];
			}
			solution[i]=t;
		}
	}
	double end = omp_get_wtime();
	return  double(end - begin) ;
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


double CSR::cudaVectorProduct(double* vector, double* solution){
	// Allocate global memory on the GPU.
	double* d_vector = 0 ;
	double* d_solution = 0;
	int* d_irp  = 0;
	int* d_ja = 0;
	double* d_as = 0;


	cudaMalloc((void**) &d_vector, this->getN() * sizeof(double));
	cudaMalloc((void**) &d_solution, this->getM() * sizeof(double));
	cudaMalloc((void**) &d_ja, this->getL() * sizeof(int));
	cudaMalloc((void**) &d_as, this->getL() * sizeof(double));
	cudaMalloc((void**) &d_irp, (this->getM()+1) * sizeof(int));

	// Copy vectors from the host (CPU) to the device (GPU).
	cudaMemcpy(d_vector, vector, this->getN() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ja, this->getJA(), this->getL() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_as, this->getAS(), this->getL() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_irp, this->getIRP(), (this->getM()+1) * sizeof(int), cudaMemcpyHostToDevice);

	//Calling method and mesuring time
	int nbBlock = 1;
	int nbThread = this->getM();
	if(nbThread >= 1024){
		nbBlock = (this->getM()/1024)+1;
		nbThread = 1024;
	}
	double begin = omp_get_wtime();
	gpuVectorProduct<<<nbBlock,nbThread>>>(d_as,d_ja,d_irp,this->getM(),this->getL(),d_vector,d_solution);
	cudaDeviceSynchronize();
	double end = omp_get_wtime();
	//get back the result from the GPU
	cudaMemcpy(solution, d_solution, this->getM() * sizeof(double), cudaMemcpyDeviceToHost);

	//Clean Memory
	cudaFree(d_vector);
	cudaFree(d_ja);
	cudaFree(d_as);
	cudaFree(d_irp);
	cudaFree(d_solution);

	return  double(end - begin) ;

}



int CSR::getM(){
	return this->M;
}

int CSR::getN(){
	return this->N;
}

int CSR::getL(){
	return this->L;
}

int* CSR::getJA(){
	return this->ja;
}

int* CSR::getIRP(){
	return this->irp;
}

double* CSR::getAS(){
	return this->as;
}

CSR::~CSR() {


}
