#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>
#include <string>
#include "gmm_scoring.h"
#include "../../utils/timer.h"

#define MAX_SOURCE_SIZE (0x100000)

size_t globalWorkSize[3];
size_t localWorkSize[3];


float feature_vect[] = {2.240018,    2.2570236,    0.11304555,   -0.21307051,
	0.8988138,   0.039065503,  0.023874786,  0.13153112,
	0.15324382,  0.16986738,   -0.020297153, -0.26773554,
	0.40202165,  0.35923952,   0.060746543,  0.35402644,
	0.086052455, -0.10499257,  0.04395058,   0.026407119,
	-0.48301497, 0.120889395,  0.67980754,   -0.19875681,
	-0.5443737,  -0.039534688, 0.20888293,   0.054865785,
	-0.4846478,  0.1,          0.1,          0.1};

float *means_vect;
float *precs_vect;
float *weight_vect;
float *factor_vect;
float *score_vect;

/*
cl_mem logZero = -3.4028235E38;
cl_mem maxLogValue = 7097004.5;
cl_mem minLogValue = -7443538.0;
cl_mem naturalLogBase = (float)1.00011595E-4;
cl_mem inverseNaturalLogBase = 9998.841;
// fixed for a given accoustic model
cl_mem comp_size = 32;
cl_mem feat_size = 29;
cl_mem senone_size = 5120;
*/



extern "C"

int main(int argc, char *argv[]) {


	if (argc < 2) {
		fprintf(stderr, "[ERROR] Invalid arguments provided.\n\n");
		fprintf(stderr, "Usage: %s [INPUT FILE]\n\n", argv[0]);
		exit(0);
	}
	STATS_INIT("kernel", "gpu_gaussian_mixture_model");
	PRINT_STAT_STRING("abrv", "gpu_gmm");

	// Load the kernel source code into the array source_str

	int comp_size = 32;
	int senone_size = 5120;

	int means_array_size = senone_size * comp_size * comp_size;
	int comp_array_size = senone_size * comp_size;

	means_vect = (float *)malloc(means_array_size * sizeof(float));
	precs_vect = (float *)malloc(means_array_size * sizeof(float));
	weight_vect = (float *)malloc(comp_array_size * sizeof(float));
	factor_vect = (float *)malloc(comp_array_size * sizeof(float));

	float *means_vect2 = (float *)malloc(means_array_size * sizeof(float));
	float *precs_vect2 = (float *)malloc(means_array_size * sizeof(float));
	float *weight_vect2 = (float *)malloc(comp_array_size * sizeof(float));
	float *factor_vect2 = (float *)malloc(comp_array_size * sizeof(float));

	score_vect = (float *)malloc(senone_size * sizeof(float));


	int blockSizeX = 256;
	int gridSizeX = (int)ceil(senone_size / blockSizeX);

	int div_grid = ((int)(gridSizeX / 32));
	gridSizeX = (div_grid + 1) * 32;

	// load model from file
	FILE *fp = fopen(argv[1], "r");
	if (fp == NULL) {  // checks for the file
		printf("\n Can’t open file");
		exit(-1);
	}

	int idx = 0;
	for (int i = 0; i < senone_size; i++) {
		for (int j = 0; j < comp_size; j++) {
			for (int k = 0; k < comp_size; k++) {
				float elem;
				fscanf(fp, "%f", &elem);
				means_vect[idx] = elem;
				idx = idx + 1;
			}
		}
	}

	idx = 0;
	for (int i = 0; i < senone_size; i++) {
		for (int j = 0; j < comp_size; j++) {
			for (int k = 0; k < comp_size; k++) {
				float elem;
				fscanf(fp, "%f", &elem);
				precs_vect[idx] = elem;
				idx = idx + 1;
			}
		}
	}

	idx = 0;
	for (int i = 0; i < senone_size; i++) {
		for (int j = 0; j < comp_size; j++) {
			float elem;
			fscanf(fp, "%f", &elem);
			weight_vect[idx] = elem;
			idx = idx + 1;
		}
	}

	idx = 0;
	for (int i = 0; i < senone_size; i++) {
		for (int j = 0; j < comp_size; j++) {
			float elem;
			fscanf(fp, "%f", &elem);
			factor_vect[idx] = elem;
			idx = idx + 1;
		}
	}

	fclose(fp);

	int idx3 = 0;
	for (int j = 0; j < comp_size; j++) {
		for (int i = 0; i < senone_size; i++) {
			int ij = j + i * comp_size;
			weight_vect2[idx3] = weight_vect[ij];
			factor_vect2[idx3] = factor_vect[ij];
			idx3 += 1;
		}
	}

	int idx4 = 0;
	for (int k = 0; k < comp_size; k++) {
		for (int j = 0; j < comp_size; j++) {
			for (int i = 0; i < senone_size; i++) {
				int ijk = k + comp_size * j + i * comp_size * comp_size;
				means_vect2[idx4] = means_vect[ijk];
				precs_vect2[idx4] = precs_vect[ijk];
				idx4 += 1;
			}
		}
	}

	for (int i = 0; i < senone_size; i++) {
		for (int j = 0; j < comp_size; j++) {
			for (int k = 0; k < 29; k++) {
				int ijk = k + comp_size * j + i * comp_size * comp_size;
				int kji = i + senone_size * j + k * comp_size * senone_size;
				if (means_vect2[kji] != means_vect[ijk]) {
					printf("%f != %f\n", means_vect2[kji], means_vect[ijk]);
				}
			}
		}
	}



	PRINT_STAT_INT("blockSizeX", blockSizeX);
	PRINT_STAT_INT("gridSizeX", gridSizeX);
	printf("\n");
	size_t block[3] = {128, 1, 1};
	size_t grid[3] = {1, 1, 1};
	grid[0] = (senone_size + block[0] - 1) / block[0];

	if (grid[0] < 32) grid[0] = 32;
	
	// each time needed for computing score of a given feature vect

	// Create a program from the kernel source
	localWorkSize[0] = block[0];
	localWorkSize[1] = block[1];
	localWorkSize[2] = block[2];
	globalWorkSize[0] = grid[0]*localWorkSize[0];
	globalWorkSize[1] = grid[1]*localWorkSize[1];
	globalWorkSize[2] = grid[2]*localWorkSize[2];
	
        Launch_params_t lparm={.ndim=3, .gdims={globalWorkSize[0], globalWorkSize[1], globalWorkSize[2]}, .ldims={localWorkSize[0], localWorkSize[1],localWorkSize[2]}};
	computeScore(feature_vect, means_vect2, precs_vect2, weight_vect2, factor_vect2, score_vect, lparm);

//	__cu2cl_EventElapsedTime(&cuda_elapsedTime, ePerfEvent);
//	PRINT_STAT_DOUBLE("gpu_gmm", cuda_elapsedTime);

//	__cu2cl_EventElapsedTime(&cuda_elapsedTime, ePerfEvent);
//	PRINT_STAT_DOUBLE("device_to_host", cuda_elapsedTime);
	STATS_END();

#ifdef TESTING
	FILE *f = fopen("../input/gmm_scoring.gpu", "w");
	printf("Writing ouptut to ../input/gmm_scoring.gpu\n");
	for (int i = 0; i < senone_size; ++i) fprintf(f, "%.0f\n", score_vect[i]);

	fclose(f);
#endif
//	clEnqueueMarker(__cu2cl_CommandQueue, &ePerfEvent);
//	clWaitForEvents(1, &PerfEvent);

	free(means_vect);
	free(precs_vect);
	free(weight_vect);
	free(factor_vect);
	free(score_vect);

}
