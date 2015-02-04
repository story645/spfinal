#include "cuda_proximity.h"
#include "cuda_timer.h"
#include <cudpp.h>
#include <cutil.h>
#include "reduce_kernel.h"
#include <time.h>
#include "cuda_test.h"


void testall(float *data, unsigned int nSamples, unsigned int dim, 
             unsigned int dmin, unsigned int dmax, 
	     unsigned int BS, unsigned int LSH, unsigned int RESULT)
{
	unsigned int nQueries = nSamples*.20;
	unsigned int K = 250;
	
	printf("nSamples=%d, nQueries=%d, dims=%d, K=%d\n",
	nSamples, nQueries, dim, K);
	
	float* query = NULL;
	unsigned int* KNNResult = NULL;
	unsigned int* KNNResult_query = NULL;
	
	
	CPUMALLOC((void**)&query, sizeof(float) * nQueries * dim);
	CPUMALLOC((void**)&KNNResult, sizeof(unsigned int) * nSamples * K);
	CPUMALLOC((void**)&KNNResult_query, sizeof(unsigned int) * nQueries * K);
	
	
	
	
	//pick random samples for the queries
	uint init;
	uint qi = 0;
	for(uint q = 0; q < nQueries; q++){
	     init = (rand() % nSamples)*dim;
	     for(uint di=init; di<(init+dim); di++){
	     query[qi] = data[di];
	     qi++;
	     }
	}
	
	
	float* d_data = NULL;
	float* d_query = NULL;
	unsigned int* d_KNNResult = NULL;
	unsigned int* d_KNNResult_query = NULL;
	GPUMALLOC((void**)&d_data, sizeof(float) * nSamples * dim);
	GPUMALLOC((void**)&d_query, sizeof(float) * nQueries * dim);
	GPUMALLOC((void**)&d_KNNResult, sizeof(unsigned int) * nSamples * K);
	GPUMALLOC((void**)&d_KNNResult_query, sizeof(unsigned int) * nQueries * K);
	
	TOGPU(d_data, data, sizeof(float) * nSamples * dim);
	TOGPU(d_query, query, sizeof(float) * nQueries * dim);


    	if(BS){
	bruteforce(K, dim, nSamples, nQueries, d_data, d_query,
	           KNNResult, d_KNNResult, KNNResult_query,
		   d_KNNResult_query, RESULT);
	}

	if(LSH ){
	locsenhash(K, dim, nSamples, nQueries, d_data, d_query,
                   KNNResult, d_KNNResult, KNNResult_query,
                   d_KNNResult_query, RESULT);
	}
	
	GPUFREE(d_data);
	GPUFREE(d_KNNResult);
	CPUFREE(data);
	CPUFREE(KNNResult);
	
	GPUFREE(d_query);
	GPUFREE(d_KNNResult_query);
	CPUFREE(query);
	CPUFREE(KNNResult_query);
	
}


void bruteforce(unsigned int K, unsigned int dim, 
                unsigned int nSamples, unsigned int nQueries, 
		float* d_data, float* d_query, 
		unsigned int* KNNResult, unsigned int* d_KNNResult, 
		unsigned int* KNNResult_query, unsigned int* d_KNNResult_query, 
		unsigned int RESULT){
      //data points self query using radixsort
      unsigned int timer3 = 0;
      startTimer(&timer3);
      proximityComputation_bruteforce2(d_data, nSamples, d_data, nSamples,
      dim, K, 0.0f, d_KNNResult);
      FROMGPU(KNNResult, d_KNNResult, sizeof(unsigned int) * nSamples * K);
      endTimer("brute-force KNN - data points self query- using radixsort", &timer3);
      if(RESULT){printResults("knn_bf2.txt", KNNResult, nSamples, K);}
	
	
      //separate data/query points using radixsort	 
      unsigned int timer4 = 0;
      startTimer(&timer4);
      proximityComputation_bruteforce2(d_data, nSamples, d_query, nQueries,
      dim, K, 0.0f, d_KNNResult_query);
      FROMGPU(KNNResult_query, d_KNNResult_query, sizeof(unsigned int) * nQueries * K);
      endTimer("brute-force KNN - separate data/query points - using radixsort", &timer4);
      if(RESULT){printResults("knn_query_bf2.txt", KNNResult_query, nQueries, K);}
}

void locsenhash(unsigned int K, unsigned int dim,
            unsigned int nSamples, unsigned int nQueries,
	    float* d_data, float* d_query,
	    unsigned int* KNNResult, unsigned int* d_KNNResult,
            unsigned int* KNNResult_query, unsigned int* d_KNNResult_query, 
	    unsigned int RESULT){
	    
	    float* h_lower = NULL;
	    float* h_upper = NULL;
	    CPUMALLOC((void**)&h_lower, sizeof(float) * dim);
	    CPUMALLOC((void**)&h_upper, sizeof(float) * dim);
	    for(unsigned int i = 0; i < dim; ++i){
	    		 h_upper[i] = 1;
	    		 h_lower[i] = 0;
	    }
	    int LSH_L = 5;
	    
	    //data points self query
	    unsigned int timer1 = 0;
	    startTimer(&timer1);
	    proximityComputation_LSH(d_data, nSamples, d_data,
	    				     nSamples, dim, K, LSH_L, 0.0f, h_upper, h_lower, d_KNNResult);
	    FROMGPU(KNNResult, d_KNNResult, sizeof(unsigned int) * nSamples * K);
	    endTimer("LSH KNN -data point self query", &timer1);
	    if(RESULT){printResults("knn_lsh.txt", KNNResult, nSamples, K);}

	    unsigned int timer2 = 0;
	    startTimer(&timer2);
	    proximityComputation_LSH(d_data, nSamples, d_query, nQueries, dim, K,
	                              LSH_L, 0.0f, h_upper, h_lower, d_KNNResult_query);
	    FROMGPU(KNNResult_query, d_KNNResult_query, sizeof(unsigned int) * nQueries * K);
	    endTimer("LSH KNN - separte data/query points", &timer2);
	    if(RESULT){printResults("knn_query_lsh.txt", KNNResult_query, nQueries, K);}

	   CPUFREE(h_lower);
	   CPUFREE(h_upper); 
}
	    
void printResults(char *filename, unsigned int* output, 
                  unsigned int nLines, unsigned int K){
     FILE * fileout = fopen(filename, "w");
     for(uint i =0; i<nLines; ++i){
     	      for(uint j=0; j < K; ++j){
     	      	       fprintf(fileout, "%d", output[j * nLines + i]);
	      }	       
     fprintf(fileout, "\n");
     }
     fclose(fileout);
}

