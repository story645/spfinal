#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
#include <cudpp.h>
#include <cutil.h>

#include "cuda_test.h"
#include "cpu_test.h"
#include "cuda_defs.h"

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

#define TIME 1
#define LAT 190
#define LON 384
#define VARNAME "TMP_2maboveground"

void unpack_data(char *filename, float *data_in);
int main()
{
	
	unsigned int start = 1985;
	unsigned int end = 2009;
	unsigned int tsamples = ((end-start) + 1)*12; 
	unsigned int nf = 9;
	
	unsigned int dim = 9;
        unsigned int nSamples = tsamples * LAT * LON;

	char filename[23];
	float *data = NULL;
	float *dataT = NULL;
	CPUMALLOC((void**)&data, sizeof(float)*nSamples*dim);	
	CPUMALLOC((void**)&dataT, sizeof(float)*nSamples*dim);

	printf("%d\n", nSamples*dim);

	
	int moffset, yoffset, foffset;
	/**
 	char filename[23];
	for (int y =start; y<=end; y++){
		yoffset = (LAT*LON*12) * (y-start);
		for (int m = 1; m<=12; m++){
			sprintf(filename, "../data/TMP_%d%02d_a.nc", y, m);
			moffset = LAT*LON*(m-1);
			unpack_data(filename, (data+yoffset+moffset)); 
		}
	}
	**/

	char ffilename[25];
	for (int f = 1; f<=9; f++){
		foffset = nSamples*(f-1);	
		for (int y =start; y<=end; y++){
			yoffset = (LAT*LON*12) * (y-start);
        	 	for (int m = 1; m<=12; m++){
                		sprintf(ffilename, "../data/TMP_%d%02d_f%02d.nc", y, m, f);
				moffset = LAT*LON*(m-1);
				unpack_data(ffilename, (data+(foffset+yoffset+moffset)));
    	            }	
        	}	
	}	
	assert((foffset+yoffset+moffset+LAT*LON)==(dim*nSamples));
	//Transpose
	//http://stackoverflow.com/a/16743203/1267531	
	for(int n=0; n<(nSamples*dim); n++){
		dataT[n] = data[nSamples*(n%dim) +(n/dim)];
	}
	/*  Allocate enough space.. */
	testall(dataT, nSamples, dim);
       // test_kdtree(data, nSamples, dim);
	return 0;
}

	
void unpack_data(char *filename, float *data_in){
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
   	if ((retval = nc_inq_varid(ncid, VARNAME, &varid)))
		ERR(retval);
	
	/* Read the data. */	
   	if ((retval = nc_get_var_float(ncid, varid, data_in)))
      		ERR(retval);	

   	/* Close the file, freeing all resources. */
   	if ((retval = nc_close(ncid)))
     		 ERR(retval);

   	printf("*** SUCCESS reading %s!\n", filename);
}

