/*
 * Main.cpp
 *
 *  Created on: 9 mars 2018
 *      Author: yoann
 */
#include <iostream>
#include <fstream>
#include "CSR.h"
using namespace std;

int main(){

	try{
		CSR matrix1("../matricesFile/bcsstk17.mtx");
		int SIZE = matrix1.getM();
		double vector[matrix1.getN()];
		double solutionSerial[SIZE];
		double solutionOpenMP[SIZE];
		double timeOpenMP,timeSerial;

		for(int i = 0 ; i < matrix1.getN(); i++){
			vector[i] = i+1;
		}

		timeOpenMP = matrix1.openMPVectorProduct(vector,solutionSerial);
		timeSerial = matrix1.serialVectorProduct(vector,solutionOpenMP);

		for(int i = 0; i < SIZE ; i++ ){
			if(solutionSerial[i] != solutionOpenMP[i]){
				std::cout << "i:"<<i<< " is not the same openMP: " << solutionOpenMP[i] << ", serial: " << solutionSerial[i] << std::endl;
			}
		}
		std::cout << "It took " << timeOpenMP <<" secondes to compute openMP-y"<< std::endl ;
		std::cout << "It took " << timeSerial <<" secondes to compute with serial"<< std::endl ;

	}catch(const std::ifstream::failure & e){
		std::cout << "Error openning/reading/closing file";
	}
}


