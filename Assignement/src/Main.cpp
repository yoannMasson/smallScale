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

	std::string matrixPath = "../matricesFile/bcsstk17.mtx";

	try{
		//------------------------------------Initialization - Preprocessing
		CSR csr(matrixPath);
		Ellpack ep(matrixPath);
	//	std::cout << ep <<std::endl;
		int SIZE = csr.getM();
		double vector[csr.getN()];
		double solutionSerialCSR[SIZE];
		double solutionOpenMPCSR[SIZE];
		double solutionSerialEllpack[SIZE];
		double solutionOpenMPEllpack[SIZE];
		double timeOpenMPCSR,timeSerialCSR,timeOpenMPEllpack,timeSerialEllpack;

		for(int i = 0 ; i < csr.getN(); i++){
			vector[i] = i+1;
		}

		//-----------------------------------Compute CSR
		timeOpenMPCSR = csr.openMPVectorProduct(vector,solutionSerialCSR);
		timeSerialCSR = csr.serialVectorProduct(vector,solutionOpenMPCSR);

		std::cout <<"CSR: "<< std::endl;
		std::cout << "It took " << timeOpenMPCSR <<" secondes to compute openMP-y"<< std::endl ;
		std::cout << "It took " << timeSerialCSR <<" secondes to compute with serial"<< std::endl ;
		std::cout << endl;

		//--------------------------Compute Ellpack
		//timeOpenMP = ep.openMPVectorProduct(vector,solutionOpenMP);
		timeSerialEllpack = ep.serialVectorProduct(vector,solutionSerialEllpack);
		std::cout <<"Ellpack: "<< std::endl;
		//std::cout << "It took " << timeOpenMP <<" secondes to compute openMP-y"<< std::endl ;
		std::cout << "It took " << timeSerialEllpack <<" secondes to compute with serial"<< std::endl ;

		//--------------------------Compare Result
		for(int i = 0; i < SIZE ; i++ ){
			if(solutionSerialCSR[i] != solutionOpenMPCSR[i] || solutionSerialEllpack[i] != solutionSerialCSR[i]){
				std::cout << "i:"<<i<< " is not the same for all solution" << std::endl;
			}
		}
		std::cout << endl;

//		//---------------------------Print Result vectors
//		std::cout <<"CSR - Serial: "<< std::endl;
//		for(int i = 0; i < ep.getM(); i++ ){
//			std::cout << solutionSerialCSR[i] << " ";
//		}
//		std::cout << std::endl;
//
//		std::cout <<"CSR - OpenMP: "<< std::endl;
//		for(int i = 0; i < ep.getM(); i++ ){
//			std::cout << solutionOpenMPCSR[i] << " ";
//		}
//		std::cout << std::endl;
//
//		std::cout <<"ELlpack - Serial: "<< std::endl;
//		for(int i = 0; i < ep.getM(); i++ ){
//			std::cout << solutionSerialEllpack[i] << " ";
//		}
//		std::cout << std::endl;

//		for(int i = 0; i < ep.getM(); i++ ){
//			std::cout << solutionOpenMPEllpack[i] << " ";
//		}
//		std::cout << std::endl;

	}catch(const std::ifstream::failure & e){
		std::cout << "Error openning/reading/closing file";
	}


}


