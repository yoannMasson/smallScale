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
#include <cuda.h>
#include <cuda_runtime.h> 

using namespace std;


int main(){

	std::string matrixPath = "../matricesFile/cage4.mtx";

	try{
		//------------------------------------Initialization - Preprocessing
		CSR csr(matrixPath);
		Ellpack ep(matrixPath);
		//	std::cout << ep <<std::endl;
		int SIZE = csr.getM();
		double vector[csr.getN()];

		double solutionSerialCSR[SIZE];
		double solutionOpenMPCSR[SIZE];
		double solutionCudaCSR[SIZE];

		double solutionSerialEllpack[SIZE];
		double solutionOpenMPEllpack[SIZE];
		double solutionCudaEllpack[SIZE];

		double timeOpenMPCSR, timeSerialCSR, timeOpenMPEllpack, timeSerialEllpack, timeCudaCSR, timeCudaEllpack;

		for(int i = 0 ; i < csr.getN(); i++){
			vector[i] = i+1;
		}

		//-----------------------------------Compute CSR
		timeOpenMPCSR = csr.openMPVectorProduct(vector,solutionSerialCSR);
		timeSerialCSR = csr.serialVectorProduct(vector,solutionOpenMPCSR);
		timeCudaCSR = csr.cudaVectorProduct(vector,solutionCudaCSR);
		std::cout <<"CSR: "<< std::endl;
		std::cout << "It took " << timeOpenMPCSR <<" secondes to compute openMP-y"<< std::endl ;
		std::cout << "It took " << timeSerialCSR <<" secondes to compute with serial"<< std::endl ;
		std::cout << "It took " << timeCudaCSR <<" secondes to compute with cuda"<< std::endl ;
		std::cout << endl;

		//--------------------------Compute Ellpack
		timeOpenMPEllpack = ep.openMPVectorProduct(vector,solutionOpenMPEllpack);
		timeSerialEllpack = ep.serialVectorProduct(vector,solutionSerialEllpack);
		std::cout <<"Ellpack: "<< std::endl;
		std::cout << "It took " << timeOpenMPEllpack <<" secondes to compute openMP-y"<< std::endl ;
		std::cout << "It took " << timeSerialEllpack <<" secondes to compute with serial"<< std::endl ;

		//--------------------------Compare Result
		for(int i = 0; i < SIZE ; i++ ){
			std::cout << solutionCudaCSR[i] << std::endl;
			if(solutionSerialCSR[i] - solutionOpenMPCSR[i] > 0.0001
			|| solutionSerialEllpack[i] - solutionSerialCSR[i] > 0.0001
			|| solutionOpenMPEllpack[i] - solutionSerialEllpack[i] > 0.0001
			
			){

				std::cout << solutionCudaCSR[i] << "gfgf " << solutionSerialCSR[i] << std::endl;
				
			}
		}
		std::cout << endl;

		//---------------------------Print Result vectors
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
//		std::cout <<"CSR - Cuda: "<< std::endl;
//		for(int i = 0; i < ep.getM(); i++ ){
//			std::cout << solutionCudaCSR[i] << " ";
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


