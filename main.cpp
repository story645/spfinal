#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
#include <cudpp.h>
#include <cutil.h>

#include "cuda_test.h"
#include "cpu_test.h"
#include "cuda_defs.h"

#define DEBUG 1
#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

#define TIME 1
#define LAT 190
#define LON 384
#define VARNAME "TMP_2maboveground"
#define DMIN 205
#define DMAX 320
#define MONTHS 12

void unpack_data(char *filename, float *data_in, char *varname);
int main(int argc, char *argv[]){
   
   int dmin, dmax, nSamples, dim;
   float *dataT=NULL;
   
   if (!DEBUG){
      int start = 1985;
      int end = 1986;
      //int end = 2009;
      
      int tsamples = ((end-start) + 1)*MONTHS; 
      dim = 9; //num forecasts
      dmin = DMIN;
      dmax = DMAX;
      nSamples = tsamples * LAT * LON;
      int foffset, yoffset, moffset;
      
      char filename[23];
      float *data = NULL;
      CPUMALLOC((void**)&data, sizeof(float)*nSamples*dim);	
      CPUMALLOC((void**)&dataT, sizeof(float)*nSamples*dim);
      printf("%d\n", nSamples*dim);
      
      char ffilename[25];
      int zcount=0;
      for (int f = 1; f<=9; f++){
	 foffset = nSamples*(f-1);	
	 for (int y =start; y<=end; y++){
	    yoffset = (LAT*LON*12) * (y-start);
	    for (int m = 1; m<=MONTHS; m++){
	       sprintf(ffilename, "../data/TMP_%d%02d_f%02d.nc", y, m, f);
	       moffset = LAT*LON*(m-1);
	       unpack_data(ffilename, (data+(foffset+yoffset+moffset)), VARNAME);
	       
	    }
	    
	 }	
      }	
      assert((foffset+yoffset+moffset+LAT*LON)==(dim*nSamples));
      //Transpose
      //http://stackoverflow.com/a/16743203/1267531	
      for(unsigned int n=0; n<(nSamples*dim); n++){
	 dataT[n] = data[nSamples*(n%dim) +(n/dim)];   
      }
   
      CPUFREE(data);
   }else{
      //Fake data, rows of all 0s...4s
      char *sfilename = "../data/simple.nc";
      char *varname = "data";
      dmin=0;
      dmax=4;
      nSamples = 100;
      dim = 5;
      CPUMALLOC((void**)&dataT, sizeof(float)*nSamples*dim);
      unpack_data(sfilename, dataT, varname);
      /**
      for (int i=0; i<nSamples; i++){ 
	 for(int j=0; j<dim; j++){
	    printf("%1.0f ",dataT[(i*dim)+j]);
	 }
	 printf("\n");
      }**/
   }
   
   
   /*  Allocate enough space.. */
   unsigned int LSH = 0;
   unsigned int BS = 0;
   unsigned int RESULT = 0;
   unsigned int KNN = 0;
   if (argc>1){LSH = atoi(argv[1]);}
   if (argc>2){BS = atoi(argv[2]);}
   if (argc>3){RESULT = atoi(argv[3]);}
   if (argc>4){KNN = atoi(argv[4]);}
	
   
   testall(dataT, nSamples, dim, dmin, dmax, LSH, BS, RESULT);
   if(KNN){
      test_kdtree(dataT, nSamples, dim);
   }
   
   CPUFREE(dataT);
   return 0;
}

	
void unpack_data(char *filename, float *data_in, char *varname){
	//netCDF fileit and data var
	int ncid, varid;

	//loop inds and error_handling
	int ld, la, ln, retval;
	
	/** Open the file. NC_NOWRITE tells netCDF 
  	we want read-only access  to the file.*/
  	if((retval = nc_open(filename, NC_NOWRITE, &ncid)))		
		ERR(retval);		
	
	/** Get the varid of the data variable, 
 	    based on its name. */
   	if ((retval = nc_inq_varid(ncid, varname, &varid)))
		ERR(retval);
	
	/* Read the data. */	
   	if ((retval = nc_get_var_float(ncid, varid, data_in)))
      		ERR(retval);	

   	/* Close the file, freeing all resources. */
   	if ((retval = nc_close(ncid)))
     		 ERR(retval);

   	//printf("*** SUCCESS reading %s!\n", filename);
}

