/*
 * Main.cpp
 *
 *  Created on: 9 mars 2018
 *      Author: yoann
 */
#include <iostream>
#include <fstream>
#include "CSR.h"
#include "Ellpack.h"
using namespace std;

int main(){

	std::string matrixPath = "../matricesFile/cage4.mtx";
	//Compute CSR
//	try{
//		CSR csr(matrixPath);
//		std::cout << csr;
//		int SIZE = csr.getM();
//		double vector[csr.getN()];
//		double solutionSerial[SIZE];
//		double solutionOpenMP[SIZE];
//		double timeOpenMP,timeSerial;
//
//		for(int i = 0 ; i < csr.getN(); i++){
//			vector[i] = i+1;
//		}
//
//		timeOpenMP = csr.openMPVectorProduct(vector,solutionSerial);
//		timeSerial = csr.serialVectorProduct(vector,solutionOpenMP);
//
//		for(int i = 0; i < SIZE ; i++ ){
//			std::cout << solutionOpenMP[i] << " ";
//			if(solutionSerial[i] != solutionOpenMP[i]){
//				std::cout << "i:"<<i<< " is not the same openMP: " << solutionOpenMP[i] << ", serial: " << solutionSerial[i] << std::endl;
//			}
//		}
//		std::cout << "It took " << timeOpenMP <<" secondes to compute openMP-y"<< std::endl ;
//		std::cout << "It took " << timeSerial <<" secondes to compute with serial"<< std::endl ;
//
//	}catch(const std::ifstream::failure & e){
//		std::cout << "Error openning/reading/closing file";
//	}

	//Compute ELlpack
	try{

		Ellpack ep(matrixPath);
		std::cout << ep;
		/*int SIZE = ep.getM();
		double vector[ep.getN()];
		double solutionSerial[SIZE];
		double solutionOpenMP[SIZE];
		double timeOpenMP,timeSerial;

		for(int i = 0 ; i < ep.getN(); i++){
			vector[i] = i+1;
		}

		timeOpenMP = ep.openMPVectorProduct(vector,solutionSerial);
		timeSerial = ep.serialVectorProduct(vector,solutionOpenMP);

		for(int i = 0; i < SIZE ; i++ ){
			if(solutionSerial[i] != solutionOpenMP[i]){
				std::cout << "i:"<<i<< " is not the same openMP: " << solutionOpenMP[i] << ", serial: " << solutionSerial[i] << std::endl;
			}
		}
		std::cout << "It took " << timeOpenMP <<" secondes to compute openMP-y"<< std::endl ;
		std::cout << "It took " << timeSerial <<" secondes to compute with serial"<< std::endl ;
		 */
	}catch(const std::ifstream::failure & e){
		std::cout << "Error openning/reading/closing file";
	}

}


