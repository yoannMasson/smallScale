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

	string matrixPath = "../matricesFile/thermomech_TK.mtx";
	
	try{
		clock_t begin = clock();
		//------------------------------------Initialization - Preprocessing
		CSR csr(matrixPath);
		Ellpack ep(matrixPath);
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
		int nbRun = 100;
		cout << "Average time over " << nbRun << " runs " <<endl;
		cout <<"CSR: "<< endl;
		timeSerialCSR = 0;
		for(int j=0;j<nbRun;j++){//To get an average
			timeSerialCSR += csr.serialVectorProduct(vector,solutionSerialCSR);
		}
		timeSerialCSR = timeSerialCSR/nbRun;
		cout << "It took " << timeSerialCSR <<" secondes to compute with serial, Mflops:" << ((2*csr.getL())/timeSerialCSR)/1000000<< endl ;

		for(int i = 1 ; i <= 16 ; i++ ){
			timeOpenMPCSR = 0;
			for(int j=0;j<nbRun;j++){//To get an average
				timeOpenMPCSR += csr.openMPVectorProduct(vector,solutionOpenMPCSR,i);
			}
			timeOpenMPCSR = timeOpenMPCSR/nbRun;
			cout << i << "cores: It took " << timeOpenMPCSR <<" secondes to compute openMP-y, Mflops:" << ((2*csr.getL())/timeOpenMPCSR)/1000000 << endl ;
			
		}


		for(int j=0;j<nbRun;j++){//To get an average
			timeCudaCSR += csr.cudaVectorProduct(vector,solutionCudaCSR);
		}
		timeCudaCSR = timeCudaCSR/nbRun;
		cout << "It took " << timeCudaCSR <<" secondes to compute with cuda, Mflops:" << ((2*csr.getL())/timeCudaCSR)/1000000 << endl ;

		//--------------------------Compute Ellpack
		cout <<"Ellpack: "<< endl;

		timeSerialEllpack = 0;
		for(int j=0;j<nbRun;j++){//To get an average
			timeSerialEllpack += ep.serialVectorProduct(vector,solutionSerialEllpack);
		}
		timeSerialEllpack = timeSerialEllpack/nbRun;
		cout << "It took " << timeSerialEllpack <<" secondes to compute with serial, Mflops:" << ((2*csr.getL())/timeSerialEllpack)/1000000<< endl ;


		for(int i = 1 ; i <= 16 ; i++ ){
			timeOpenMPEllpack = 0;
			for(int j=0;j<nbRun;j++){//To get an average
				timeOpenMPEllpack +=ep.openMPVectorProduct(vector,solutionOpenMPEllpack,i);
			}
			timeOpenMPEllpack = timeOpenMPEllpack/nbRun;
			cout << i << "cores: It took " << timeOpenMPEllpack <<" secondes to compute openMP-y, Mflops:" << ((2*csr.getL())/timeOpenMPEllpack)/1000000 << endl ;
		}

		timeCudaEllpack = 0;
		for(int j=0;j<nbRun;j++){//To get an average
			timeCudaEllpack += ep.cudaVectorProduct(vector,solutionCudaEllpack);
		}

		timeCudaEllpack = timeCudaEllpack/nbRun;
		cout << "It took " << timeCudaEllpack <<" secondes to compute with cuda, Mflops:" << ((2*csr.getL())/timeCudaEllpack)/1000000<< endl ;


		//--------------------------Compare Result
		bool correct = true;
		for(int i = 0; i < SIZE ; i++ ){
			if(abs(solutionSerialCSR[i] - solutionOpenMPCSR[i]) > 0.01
			|| abs(solutionSerialEllpack[i] - solutionSerialCSR[i]) > 0.01
			|| abs(solutionOpenMPEllpack[i] - solutionSerialEllpack[i]) > 0.01
			|| abs(solutionCudaCSR[i] - solutionSerialCSR[i]) > 0.01
			|| abs(solutionCudaEllpack[i] - solutionSerialEllpack[i]) > 0.01
			){
				correct = false;
			}
		}

		if(correct){
			cout << "All values are the same";
		}else{
			cout << "Values are different";
		}
		cout << endl;

		//---------------------------Print Result vectors
		// cout <<"CSR: "<< endl;
		// for(int i = 0; i < ep.getM(); i++ ){
		// 	cout << solutionSerialCSR[i] << " ";
		// }
		// cout << endl;
		// for(int i = 0; i < ep.getM(); i++ ){
		// 	cout << solutionOpenMPCSR[i] << " ";
		// }
		// cout << endl;
		//
		//  for(int i = 0; i < ep.getM(); i++ ){
		//  	cout << solutionCudaCSR[i] << " ";
		//  }
		//  cout << endl;
		//
		// cout <<"ELlpack: "<< endl;
		// for(int i = 0; i < ep.getM(); i++ ){
		//		cout << solutionSerialEllpack[i] << " ";
		// }
		// cout << endl;
		// for(int i = 0; i < ep.getM(); i++ ){
		// 	cout << solutionOpenMPEllpack[i] << " ";
		// }
		// cout << endl;
		// for(int i = 0; i < ep.getM(); i++ ){
		// 	cout << solutionCudaEllpack[i] << " ";
		// }
		// cout << endl;
	clock_t end = clock();
	cout << "Overall programm took " << (end-begin)/CLOCKS_PER_SEC << "secondes" << endl;
	}catch(const ifstream::failure & e){
		cout << "Error openning/reading/closing file";
	}


}
