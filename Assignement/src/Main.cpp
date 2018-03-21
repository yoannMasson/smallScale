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
		CSR matrix1("../matricesFile/cage4.mtx");
		std::cout << matrix1;
		int tab[50];
		for(int i = 0 ; i < 50; i++){
			tab[i] = 1;
		}

		matrix1.serialVectorProduct(tab);
	}catch(std::ifstream::failure e){
		std::cout << "Error openning/reading/closing file";
	}




}


