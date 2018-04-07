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

	string matrixPath = "../matricesFile/bcsstk17.mtx";

	try{
		//------------------------------------Initialization - Preprocessing
		CSR csr(matrixPath);
		Ellpack ep(matrixPath);
		cout << csr.getL() <<endl;
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
		cout <<"CSR: "<< endl;
		timeSerialCSR = 0;
		for(int j=0;j<20;j++){//To get an average over 20 runs
			timeSerialCSR += csr.serialVectorProduct(vector,solutionSerialCSR);
		}
		cout << "It took " << timeSerialCSR/20 <<" secondes to compute with serial, Mflops:" << ((2*csr.getL())/timeSerialCSR)/1000<< endl ;
		
		for(int i = 1 ; i <= 8 ; i++ ){
			timeOpenMPCSR = 0;
			for(int j=0;j<20;j++){//To get an average over 20 runs
				timeOpenMPCSR += csr.openMPVectorProduct(vector,solutionOpenMPCSR,i);
			}
			cout << i << "cores: It took " << timeOpenMPCSR/20 <<" secondes to compute openMP-y, Mflops:" << ((2*csr.getL())/timeOpenMPCSR)/1000 << endl ;
		}
		
		//--------------------------Compute Ellpack
		cout <<"Ellpack: "<< endl;
		
		timeSerialEllpack = 0;
		for(int j=0;j<20;j++){//To get an average over 20 runs
			timeSerialEllpack += ep.serialVectorProduct(vector,solutionSerialEllpack);
		}
		cout << "It took " << timeSerialEllpack/20 <<" secondes to compute with serial, Mflops:" << ((2*csr.getL())/timeSerialEllpack)/1000<< endl ;
		
		
		for(int i = 1 ; i <= 8 ; i++ ){
			timeOpenMPEllpack = 0;
			for(int j=0;j<20;j++){//To get an average over 20 runs
				timeOpenMPEllpack +=ep.openMPVectorProduct(vector,solutionOpenMPEllpack,i);
			}
			cout << i << "cores: It took " << timeOpenMPEllpack/20 <<" secondes to compute openMP-y, Mflops:" << ((2*csr.getL())/timeOpenMPEllpack)/1000 << endl ;
		}
		
		
		
		//timeCudaCSR = csr.cudaVectorProduct(vector,solutionCudaCSR);
		//cout << "It took " << timeCudaCSR <<" secondes to compute with cuda, Mflops:" << ((2*csr.getL())/timeCudaCSR)/1000 << endl ;

		
	
		
		//timeCudaEllpack = ep.cudaVectorProduct(vector,solutionCudaEllpack);
		//cout << "It took " << timeCudaEllpack <<" secondes to compute with cuda, Mflops:" << ((2*csr.getL())/timeCudaEllpack)/1000 << endl ;
		cout << endl;
		//--------------------------Compare Result
		for(int i = 0; i < SIZE ; i++ ){
		//	cout << solutionCudaCSR[i] << endl;
			if(abs(solutionSerialCSR[i] - solutionOpenMPCSR[i]) > 0.0001
			|| abs(solutionSerialEllpack[i] - solutionSerialCSR[i]) > 0.0001
			|| abs(solutionOpenMPEllpack[i] - solutionSerialEllpack[i]) > 0.0001
			//|| abs(solutionCudaCSR[i] - solutionSerialCSR[i]) > 0.0001
			//|| abs(solutionCudaEllpack[i] - solutionSerialEllpack[i]) > 0.0001
			){

				cout << i <<": "<< solutionCudaCSR[i] << " - " << solutionSerialCSR[i] << " - " << solutionCudaEllpack[i] << endl;
				
			}
		}
		cout << endl;

	/*	//---------------------------Print Result vectors
		cout <<"CSR: "<< endl;
		for(int i = 0; i < ep.getM(); i++ ){
			cout << solutionSerialCSR[i] << " ";
		}
		cout << endl;
		for(int i = 0; i < ep.getM(); i++ ){
			cout << solutionOpenMPCSR[i] << " ";
		}
		cout << endl;

		for(int i = 0; i < ep.getM(); i++ ){
			cout << solutionCudaCSR[i] << " ";
		}
		cout << endl;
		
		cout <<"ELlpack: "<< endl;
		for(int i = 0; i < ep.getM(); i++ ){
			cout << solutionSerialEllpack[i] << " ";
		}
		cout << endl;
		for(int i = 0; i < ep.getM(); i++ ){
			cout << solutionOpenMPEllpack[i] << " ";
		}
		cout << endl;
		for(int i = 0; i < ep.getM(); i++ ){
			cout << solutionCudaEllpack[i] << " ";
		}
		cout << endl;*/

	}catch(const ifstream::failure & e){
		cout << "Error openning/reading/closing file";
	}


}


