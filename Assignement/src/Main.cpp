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
	}catch(std::ifstream::failure e){
		std::cout << "Error openning/reading/closing file";
	}



}


