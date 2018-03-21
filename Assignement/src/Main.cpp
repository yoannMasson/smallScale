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
		CSR matrix1("../matricesFile/testWiki.mtx");
		std::cout << matrix1;
		int tab[50];
		tab[0] = 0;
		for(int i = 1 ; i < 50; i++){
			tab[i] = 2;
		}

		matrix1.serialVectorProduct(tab);
	}catch(const std::ifstream::failure & e){
		std::cout << "Error openning/reading/closing file";
	}




}


