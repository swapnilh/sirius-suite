#include <CL/opencl.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>
#include <string>

#include "/home/swapnilh/workloads/sirius-suite/utils/timer.h"

#define MAX_SOURCE_SIZE (0x100000)

cl_platform_id __cu2cl_Platform;
cl_device_id __cu2cl_Device;
cl_context __cu2cl_Context;
cl_command_queue __cu2cl_CommandQueue;

size_t globalWorkSize[3];
size_t localWorkSize[3];


size_t __cu2cl_LoadProgramSource(const char *filename, const char **progSrc) {
	FILE *f = fopen(filename, "r");
	fseek(f, 0, SEEK_END);
	size_t len = (size_t) ftell(f);
	*progSrc = (const char *) malloc(sizeof(char)*len);
	rewind(f);
	fread((void *) *progSrc, len, 1, f);
	fclose(f);
	return len;
}

cl_int __cu2cl_EventElapsedTime(float *ms, cl_event start, cl_event end ) {
	cl_int ret;
	cl_ulong s, e;
	//float fs, fe;
	ret |= clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &s, NULL);
	ret |= clGetEventProfilingInfo(end, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL);
	s = e - s;
	*ms = ((float) s)/1000000.0;
	return ret;
}



void __cu2cl_EventElapsedTime(cl_double *g_NDRangePureExecTimeMs, cl_event perf_event ) {
	cl_ulong start = 0, end = 0;

	clGetEventProfilingInfo(perf_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(perf_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

	//END-START gives you hints on kind of “pure HW execution time”
	//the resolution of the events is 1e-09 sec
	*g_NDRangePureExecTimeMs = (cl_double)(end - start)*(cl_double)(1e-06);
}


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
//	const char *progSrc;
//	size_t progLen;

	clGetPlatformIDs(1, &__cu2cl_Platform, NULL);
	clGetDeviceIDs(__cu2cl_Platform, CL_DEVICE_TYPE_GPU, 1, &__cu2cl_Device, NULL);
	__cu2cl_Context = clCreateContext(NULL, 1, &__cu2cl_Device, NULL, NULL, NULL);
	__cu2cl_CommandQueue = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, NULL);

	if (argc < 2) {
		fprintf(stderr, "[ERROR] Invalid arguments provided.\n\n");
		fprintf(stderr, "Usage: %s [INPUT FILE]\n\n", argv[0]);
		exit(0);
	}
	STATS_INIT("kernel", "gpu_gaussian_mixture_model");
	PRINT_STAT_STRING("abrv", "gpu_gmm");

	// Load the kernel source code into the array source_str
	FILE *fp_kernel;
	char *source_str;
	size_t source_size;

	fp_kernel = fopen("gmm_scoring.cl", "r");
	if (!fp_kernel) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp_kernel);
	fclose( fp_kernel );

	cl_mem dev_feat_vect;

	cl_double cuda_elapsedTimedouble;
	float cuda_elapsedTime;
	cl_event eStart, eStop;
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

	cl_mem dev_means_vect;
	cl_mem dev_precs_vect;
	cl_mem dev_weight_vect;
	cl_mem dev_factor_vect;

	score_vect = (float *)malloc(senone_size * sizeof(float));

	cl_mem dev_score_vect;

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

	//  cudaEventCreate(&eStart);
	//  cudaEventCreate(&eStop);



	// just one time to load acoustic model
	*(void **)&dev_means_vect = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(float) * means_array_size, NULL, NULL);
	*(void **)&dev_precs_vect = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(float) * means_array_size, NULL, NULL);
	*(void **)&dev_weight_vect = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(float) * comp_array_size, NULL, NULL);
	*(void **)&dev_factor_vect = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(float) * comp_array_size, NULL, NULL);

	clEnqueueWriteBuffer(__cu2cl_CommandQueue, dev_means_vect, CL_TRUE, 0, sizeof(float) * means_array_size, means_vect2, 0, NULL, NULL);
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, dev_precs_vect, CL_TRUE, 0, sizeof(float) * means_array_size, precs_vect2, 0, NULL, NULL);
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, dev_weight_vect, CL_TRUE, 0, sizeof(float) * comp_array_size, weight_vect2, 0, NULL, NULL);
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, dev_factor_vect, CL_TRUE, 0, sizeof(float) * comp_array_size, factor_vect2, 0, NULL, NULL);

	*(void **)&dev_feat_vect = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(float) * comp_size, NULL, NULL);
	*(void **)&dev_score_vect = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(float) * senone_size, NULL, NULL);

	PRINT_STAT_INT("blockSizeX", blockSizeX);
	PRINT_STAT_INT("gridSizeX", gridSizeX);
	printf("\n");
	size_t block[3] = {128, 1, 1};
	size_t grid[3] = {1, 1, 1};
	grid[0] = (senone_size + block[0] - 1) / block[0];

	if (grid[0] < 32) grid[0] = 32;
	
	// SWAPNIL- why twice, same in the original cuda code too.
	clEnqueueMarker(__cu2cl_CommandQueue, &eStart);

	// each time needed for computing score of a given feature vect
	clEnqueueMarker(__cu2cl_CommandQueue, &eStart);
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, dev_feat_vect, CL_TRUE, 0, comp_size * sizeof(float), feature_vect, 0, NULL, NULL);
	clEnqueueMarker(__cu2cl_CommandQueue, &eStop);
	clWaitForEvents(1, &eStop);
	__cu2cl_EventElapsedTime(&cuda_elapsedTime, eStart, eStop);
	PRINT_STAT_DOUBLE("host_to_device", cuda_elapsedTime);
//        printf(",\n\thost_to_device : % 20.16lf\n", cuda_elapsedTime);



	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(__cu2cl_Context, 1, 
			(const char **)&source_str, (const size_t *)&source_size, NULL);

	// Build the program
	clBuildProgram(program, 1, &__cu2cl_Device, NULL, NULL, NULL);

	// Create the OpenCL kernel
	 cl_kernel __cu2cl_Kernel_computeScore = clCreateKernel(program, "computeScore", NULL);

	clEnqueueMarker(__cu2cl_CommandQueue, &eStart);
	clSetKernelArg(__cu2cl_Kernel_computeScore, 0, sizeof(cl_mem), &dev_feat_vect);
	clSetKernelArg(__cu2cl_Kernel_computeScore, 1, sizeof(cl_mem), &dev_means_vect);
	clSetKernelArg(__cu2cl_Kernel_computeScore, 2, sizeof(cl_mem), &dev_precs_vect);
	clSetKernelArg(__cu2cl_Kernel_computeScore, 3, sizeof(cl_mem), &dev_weight_vect);
	clSetKernelArg(__cu2cl_Kernel_computeScore, 4, sizeof(cl_mem), &dev_factor_vect);
	clSetKernelArg(__cu2cl_Kernel_computeScore, 5, sizeof(cl_mem), &dev_score_vect);
	localWorkSize[0] = block[0];
	localWorkSize[1] = block[1];
	localWorkSize[2] = block[2];
	globalWorkSize[0] = grid[0]*localWorkSize[0];
	globalWorkSize[1] = grid[1]*localWorkSize[1];
	globalWorkSize[2] = grid[2]*localWorkSize[2];
	clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_computeScore, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	clEnqueueMarker(__cu2cl_CommandQueue, &eStop);
	clWaitForEvents(1, &eStop);

//	__cu2cl_EventElapsedTime(&cuda_elapsedTimedouble, eStop);
//        printf(",\n\tgpu_gmm : % 20.16lf\n", cuda_elapsedTimedouble);
	__cu2cl_EventElapsedTime(&cuda_elapsedTime, eStart, eStop);
	PRINT_STAT_DOUBLE("gpu_gmm", cuda_elapsedTime);

	clEnqueueMarker(__cu2cl_CommandQueue, &eStart);
	clEnqueueReadBuffer(__cu2cl_CommandQueue, dev_score_vect, CL_TRUE, 0, senone_size * sizeof(float), score_vect, 0, NULL, NULL);
	clEnqueueMarker(__cu2cl_CommandQueue, &eStop);
	clWaitForEvents(1, &eStop);
	__cu2cl_EventElapsedTime(&cuda_elapsedTime, eStart, eStop);
	PRINT_STAT_DOUBLE("device_to_host", cuda_elapsedTime);

	STATS_END();

#if TESTING
	FILE *f = fopen("../input/gmm_scoring.gpu", "w");

	for (int i = 0; i < senone_size; ++i) fprintf(f, "%.0f\n", score_vect[i]);

	fclose(f);
#endif

	clEnqueueMarker(__cu2cl_CommandQueue, &eStop);
	clWaitForEvents(1, &eStop);
	__cu2cl_EventElapsedTime(&cuda_elapsedTime, eStart, eStop);

	free(means_vect);
	free(precs_vect);

	free(weight_vect);
	free(factor_vect);

	free(score_vect);

	clReleaseMemObject(dev_means_vect);
	clReleaseMemObject(dev_precs_vect);
	clReleaseMemObject(dev_weight_vect);
	clReleaseMemObject(dev_factor_vect);

	clReleaseMemObject(dev_feat_vect);
	clReleaseMemObject(dev_score_vect);

	clReleaseCommandQueue(__cu2cl_CommandQueue);
	clReleaseContext(__cu2cl_Context);
}
