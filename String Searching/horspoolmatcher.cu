/*
 Copyright 2015. All rights reserved
 Author: Teh Rong Jiunn
 version: 1.0
*/

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>
#include <fstream>

#define ASIZE 255	//ASCII code size
#define DATA_SIZE 1024000
#define YES 1

using namespace std;

typedef struct{
	char* text;
	char* pattern;
	int* shifts;
	int* results;
}stringSearchData;

__device__ int shifts[ASIZE];
__device__ int results[DATA_SIZE];


__global__ void stringSearchKernel(char *pattern, int m, char* text, int n, int shifts[], int textPerThread, int textPerBlock) {
	extern __shared__ char blockText[];

	//
	// Copy text from global memory to shared memory
	//
	int blockStart = blockIdx.x * (textPerBlock - m + 1);
	for (int i = 0; i < textPerThread - m + 1; i++)
	{	
		int sLocation = blockDim.x * i + threadIdx.x; //shared memory location
		if(sLocation >= textPerBlock) break;
		int gLocation = blockStart + blockDim.x * i + threadIdx.x; //global memory location
		if(gLocation >= n) break;
		blockText[sLocation] = text[gLocation];
	}
	__syncthreads();

	//
	// Matching pattern with block text
	//
	int threadStart = threadIdx.x * (textPerThread - m + 1);
	int tLocation = threadStart; //text location
	int pLocation = m - 1; //pattern location

	if(tLocation + pLocation < textPerBlock){ //skip if exceeds block size
		while(tLocation + pLocation < threadStart + textPerThread){ //matching within thread size
			int k = pLocation;
			while (pattern[k] == blockText[tLocation + k]){
				k -= 1;
				if (k < 0){
					results[blockStart + tLocation] = YES;
					break;
				}
			}
			tLocation += shifts[blockText[tLocation + k]];		
		}
	}	
	__syncthreads();

	return;
}

float stringSearch(int n, int m, stringSearchData host){
	stringSearchData device;
	cudaError_t cudaStatus;
	cudaEvent_t eventStart, eventStop;
	float time = 0;

	//
	// Initializing the GPU timers
	//
	cudaStatus = cudaEventCreate(&eventStart);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaEventCreate of eventStart failed!\n");
		goto Error;
	}

	cudaStatus = cudaEventCreate(&eventStop);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaEventCreate of eventStop failed!\n");
		goto Error;
	}

	//
	// Allocate memory for data and copy to device
	//
	cudaStatus = cudaGetSymbolAddress((void**)&device.shifts, shifts);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaGetSymbolAddress of shifts failed!\n");
		goto Error;
	}

	cudaStatus = cudaGetSymbolAddress((void**)&device.results, results);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaGetSymbolAddress of shifts failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&device.pattern, (m + 1)*sizeof(char));
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc of device.pattern failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&device.text, (n + 1)*sizeof(char));
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc of device.text failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(device.text, host.text, sizeof(char)*(n + 1), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMemcpy to device.text failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(device.pattern, host.pattern, sizeof(char)*(m + 1), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMemcpy to device.pattern failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(device.shifts, host.shifts, sizeof(int)*ASIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMemcpy to device.shifts failed!\n");
		goto Error;
	}
	
	//
	// Call the string search kernel
	//
	int numOfBlock = 4;
	int threadsPerBlock = 512;

	int textPerBlock = n / numOfBlock + (m - 1);
	if (n % textPerBlock != 0) textPerBlock += 1;
	int textPerThread = textPerBlock / threadsPerBlock + (m - 1);
	if (n % threadsPerBlock != 0) textPerThread += 1;
	
	cudaStatus = cudaEventRecord(eventStart, 0);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaEventRecord eventStart failed!\n");
		goto Error;
	}
	//49152 bytes shared memory
	stringSearchKernel<<<numOfBlock,threadsPerBlock,textPerBlock*sizeof(char)>>>(device.pattern, m, device.text, n, device.shifts, textPerThread, textPerBlock);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaDeviceSynchronize stringSearchKernel failed!\n");
		goto Error;
	}

	cudaStatus = cudaEventRecord(eventStop, 0);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaEventRecord eventStop failed!\n");
		goto Error;
	}

	cudaStatus = cudaEventSynchronize(eventStop);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaEventSynchronize eventStop failed!\n");
		goto Error;
	}

	cudaStatus = cudaEventElapsedTime(&time, eventStart, eventStop);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaEventElapsedTime failed!\n");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(host.results, device.results, n * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMemcpy to host.results failed!\n");
		goto Error;
	}

	//
	// Clean up used device memory
	//
	Error:
	cudaEventDestroy(eventStart);
	cudaEventDestroy(eventStop);
	cudaFree(device.pattern);
	cudaFree(device.shifts);
	cudaFree(device.text);
	cudaFree(device.results);

	return time;
}

char* readfile(const char* filename) {

	string line;
	unsigned int i = 0;
	char* data = (char*)malloc(DATA_SIZE * sizeof(char));
	ifstream myfile(filename);
	if (myfile.is_open())
	{
		while (getline(myfile,line))
		{
			if (i < 1) strcpy(data, line.c_str());
			else strcat(data, line.c_str());
			i++;
		}
		myfile.close();
	}
	else printf("Unable to open file");
	
	return data;
}

void displayResults(int n, int  results[]) {
	int noMatch = 0;
	int total = 0;
	for (int i = 0; i < n; i++){
		if (results[i] == YES){
			printf("Found match at %d.\n", i);
			total += 1;
		}
		else noMatch++;
	}
	if (noMatch == n) printf("No match found.\n");
	else printf("Total occurrence: %d\n",total);
}

int main() {
	stringSearchData host;
	int n = 0; // length of host text
	int m = 0; // length of host pattern
	float time;
	
    //
    // Read in the 'pattern' to be matched against the string in the data
    //
    host.text = readfile("Alice.txt");
    host.pattern = readfile("Pattern.txt");
    n = strlen(host.text);
    m = strlen(host.pattern);
	
    //
    // Initialize the shift index
    //
    host.shifts = (int*) malloc( ASIZE * sizeof(int) );
    for( int i = 0; i < ASIZE; i++ )
        host.shifts[i] = m;
	
	//
	// Compute shift table
	//
	for (int i = 0; i <= m - 2; i++)
		host.shifts[host.pattern[i]] = m - 1 - i;

	//
	// Perform the string search
	//
	host.results = (int*)malloc(n * sizeof(int));
	time = stringSearch(n, m, host);	
	printf("Time taken: %f ms\n", time);
	displayResults(n, host.results);

	//
	// Clean up used host memory
	//
	free(host.text);
	free(host.pattern);
	free(host.shifts);
	free(host.results);
 
	cudaDeviceReset();
	
	return 0;
}
