/* This is the Porter stemming algorithm, coded up as thread-safe ANSI C
  by the author.

  It may be be regarded as cononical, in that it follows the algorithm
  presented in

  Porter, 1980, An algorithm for suffix stripping, Program, Vol. 14,
  no. 3, pp 130-137,

  only differing from it at the points maked --DEPARTURE-- below.

  See also http://www.tartarus.org/~martin/PorterStemmer

  The algorithm as described in the paper could be exactly replicated
  by adjusting the points of DEPARTURE, but this is barely necessary,
  because (a) the points of DEPARTURE are definitely improvements, and
  (b) no encoding of the Porter stemmer I have seen is anything like
  as exact as this version, even with the points of DEPARTURE!

  You can compile it on Unix with 'gcc -O3 -o stem stem.c' after which
  'stem' takes a list of inputs and sends the stemmed equivalent to
  stdout.

  The algorithm as encoded here is particularly fast.

  Release 2 (the more old-fashioned, non-thread-safe version may be
  regarded as release 1.)
*/

#include <CL/opencl.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>  /* for isupper, islower, tolower */
#include <string.h> /* for memcmp, memmove */
#include <pthread.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>

#include "../../utils/timer.h"

/* You will probably want to move the following declarations to a central
  header file.
*/

#define MAX_SOURCE_SIZE (0x100000)

cl_platform_id __cu2cl_Platform;
cl_device_id __cu2cl_Device;
cl_context __cu2cl_Context;
cl_command_queue __cu2cl_CommandQueue;

size_t globalWorkSize[3];
size_t localWorkSize[3];
cl_mem __cu2cl_Mem_stem_list;
cl_program __cu2cl_Program_porter_cu;
cl_kernel __cu2cl_Kernel_stem_gpu;

const char *getErrorString(cl_int error);

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

cl_int __cu2cl_MallocHost(void **ptr, size_t size, cl_mem *clMem) {
    cl_int ret;
    *clMem = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, size, NULL, NULL);
    *ptr = clEnqueueMapBuffer(__cu2cl_CommandQueue, *clMem, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &ret);
    return ret;
}

cl_int __cu2cl_EventElapsedTime(float *ms, cl_event start, cl_event end) {
    cl_int ret;
    cl_ulong s, e;
//    float fs, fe;
    ret = clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &s, NULL);
    ret |= clGetEventProfilingInfo(end, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL);
    s = e - s;
    *ms = ((float) s)/1000000.0;
    return ret;
}

cl_int __cu2cl_FreeHost(void *ptr, cl_mem clMem) {
    cl_int ret;
    ret = clEnqueueUnmapMemObject(__cu2cl_CommandQueue, clMem, ptr, 0, NULL, NULL);
    ret |= clReleaseMemObject(clMem);
    return ret;
}

struct stemmer;

extern struct stemmer *create_stemmer(void);
extern void free_stemmer(struct stemmer *z);

extern int stem(struct stemmer *z, char *b, int k);

/* The main part of the stemming algorithm starts here.
*/

#define TRUE 1
#define FALSE 0

#define INC 32 /* size units in which s is increased */

/* stemmer is a structure for a few local bits of data,
*/

struct stemmer {
  char b[INC + 1]; /* buffer for word to be stemmed */
  int k;           /* offset to the end of the string */
  int j;           /* a general offset into the string */
};

/* Member b is a buffer holding a word to be stemmed. The letters are in
  b[0], b[1] ... ending at b[z->k]. Member k is readjusted downwards as
  the stemming progresses. Zero termination is not in fact used in the
  algorithm.

  Note that only lower case sequences are stemmed. Forcing to lower case
  should be done before stem(...) is called.


  Typical usage is:

      struct stemmer * z = create_stemmer();
      char b[] = "pencils";
      int res = stem(z, b, 6);
          /- stem the 7 characters of b[0] to b[6]. The result, res,
             will be 5 (the 's' is removed). -/
      free_stemmer(z);
*/

extern struct stemmer *create_stemmer(void) {
  return (struct stemmer *)malloc(sizeof(struct stemmer));
  /* assume malloc succeeds */
}

extern void free_stemmer(struct stemmer *z) { free(z); }

/* cons(z, i) is TRUE <=> b[i] is a consonant. ('b' means 'z->b', but here
  and below we drop 'z->' in comments.
*/

/* step1ab(z) gets rid of plurals and -ed or -ing. e.g.

      caresses  ->  caress
      ponies    ->  poni
      ties      ->  ti
      caress    ->  caress
      cats      ->  cat

      feed      ->  feed
      agreed    ->  agree
      disabled  ->  disable

      matting   ->  mat
      mating    ->  mate
      meeting   ->  meet
      milling   ->  mill
      messing   ->  mess

      meetings  ->  meet

*/

/* In stem(z, b, k), b is a char pointer, and the string to be stemmed is
  from b[0] to b[k] inclusive.  Possibly b[k+1] == '\0', but it is not
  important. The stemmer adjusts the characters b[0] ... b[k] and returns
  the new end-point of the string, k'. Stemming never increases word
  length, so 0 <= k' <= k.
*/


/* In stem(z, b, k), b is a char pointer, and the string to be stemmed is
  from b[0] to b[k] inclusive.  Possibly b[k+1] == '\0', but it is not
  important. The stemmer adjusts the characters b[0] ... b[k] and returns
  the new end-point of the string, k'. Stemming never increases word
  length, so 0 <= k' <= k.
*/



/*--------------------stemmer definition ends here------------------------*/

#define ARRAYSIZE 1000000
#define A_INC 10000

static int a_max = ARRAYSIZE;
static int i_max = INC; /* maximum offset in s */
struct stemmer *stem_list;
cl_mem gpu_stem_list;

#define LETTER(ch) (isupper(ch) || islower(ch))

int load_data(struct stemmer *stem_list, FILE *f) {
  int a_size = 0;
  while (TRUE) {
    int ch = getc(f);
    if (ch == EOF) return a_size;
    char *s = (char *)malloc(i_max + 1);
    if (LETTER(ch)) {
      int i = 0;

      while (TRUE) {
        if (i == i_max) {
          i_max += INC;
          s = (char *)realloc(s, i_max + 1);
        }
        ch = tolower(ch); /* forces lower case */

        stem_list[a_size].b[i] = ch;
        s[i] = ch;
        i++;
        ch = getc(f);
        if (!LETTER(ch)) {
          ungetc(ch, f);
          break;
        }
      }

      stem_list[a_size].k = i - 1;
      if (a_size == a_max) {
        a_max += A_INC;
        stem_list = (struct stemmer *)realloc(stem_list,
                                              a_max * sizeof(struct stemmer));
      }
      a_size += 1;
    }
  }
}

int main(int argc, char *argv[]) {
	cl_int ret;
	clGetPlatformIDs(1, &__cu2cl_Platform, NULL);
	clGetDeviceIDs(__cu2cl_Platform, CL_DEVICE_TYPE_GPU, 1, &__cu2cl_Device, NULL);
	__cu2cl_Context = clCreateContext(NULL, 1, &__cu2cl_Device, NULL, NULL, NULL);
	__cu2cl_CommandQueue = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, NULL);

	// Load the kernel source code into the array source_str
	FILE *fp_kernel;
	char *source_str;
	size_t source_size;

	fp_kernel = fopen("porter.cl", "r");
	if (!fp_kernel) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp_kernel);
	fclose( fp_kernel );
	
	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(__cu2cl_Context, 1, 
			(const char **)&source_str, (const size_t *)&source_size, &ret);
	if (ret != CL_SUCCESS) {
		printf("Error creating program:%s",getErrorString(ret));
		return 0;
	}
	// Build the program
	ret = clBuildProgram(program, 1, &__cu2cl_Device, NULL, NULL, NULL);
	printf("BUILT CORRECTLY!\n");
	if (ret != CL_SUCCESS) {
		printf("Error building program:%s",getErrorString(ret));
		char buffer[10240];
		clGetProgramBuildInfo(program, __cu2cl_Device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
		fprintf(stderr, "CL Compilation failed:\n%s", buffer);	
		return 0;
	}
	// Create the OpenCL kernel
	cl_kernel __cu2cl_Kernel_stem_gpu = clCreateKernel(program, "stem_gpu", &ret);
	if (ret != CL_SUCCESS) {
		printf("Error creating kernel:%s",getErrorString(ret));
		return 0;
	}

	if (argc < 2) {
		fprintf(stderr, "[ERROR] Invalid arguments provided.\n\n");
		fprintf(stderr, "Usage: %s [INPUT FILE]\n\n", argv[0]);
		exit(0);
	}
	/* Timing */
	STATS_INIT("kernel", "gpu_porter_stemming");
	PRINT_STAT_STRING("abrv", "gpu_stemmer");

	cl_event eStart, eStop;
	float cuda_elapsedTime;

	// allocate data
	FILE *f;
	f = fopen(argv[1], "r");
	if (f == 0) {
		fprintf(stderr, "File %s not found\n", argv[1]);
		exit(1);
	}

	__cu2cl_MallocHost((void **)&stem_list, ARRAYSIZE* sizeof(struct stemmer), &__cu2cl_Mem_stem_list);
//	stem_list = (struct stemmer * ) malloc (ARRAYSIZE * sizeof(struct stemmer));

	int words = load_data(stem_list, f);
	PRINT_STAT_INT("words", words);
	fclose(f);
	f = fopen("../input/porter.gpu", "w");

	for (int i = 0; i < words; ++i) fprintf(f, "b=%s j=%d k=%d\n", stem_list[i].b, stem_list[i].j, stem_list[i].k);

	fclose(f);


	*(void **)&gpu_stem_list = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, words * sizeof(struct stemmer), NULL, NULL);

	clEnqueueMarker(__cu2cl_CommandQueue, &eStart);
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, gpu_stem_list, CL_TRUE, 0, words * sizeof(struct stemmer), stem_list, 0, NULL, NULL);
	clEnqueueMarker(__cu2cl_CommandQueue, &eStop);
	clWaitForEvents(1, &eStop);
	__cu2cl_EventElapsedTime(&cuda_elapsedTime, eStart, eStop);
	PRINT_STAT_DOUBLE("host_to_device", cuda_elapsedTime);

	clEnqueueMarker(__cu2cl_CommandQueue, &eStart);
	size_t block[3] = {256, 1, 1};
	size_t grid[3] = {1, 1, 1};
	grid[0] = ceil(words * 1.0 / block[0]);

	clEnqueueMarker(__cu2cl_CommandQueue, &eStart);
	int err=0;
	err = clSetKernelArg(__cu2cl_Kernel_stem_gpu, 0, sizeof(cl_mem), &gpu_stem_list);
        if (err != CL_SUCCESS) { 
		printf("Unable to set arguments 1\n"); 
		return 0;
	}
	err |= clSetKernelArg(__cu2cl_Kernel_stem_gpu, 1, sizeof(int), &words);
        if (err != CL_SUCCESS) { 
		printf("Unable to set arguments 2\n"); 
		return 0;
	}

	localWorkSize[0] = block[0];
	localWorkSize[1] = block[1];
	localWorkSize[2] = block[2];
	globalWorkSize[0] = grid[0]*localWorkSize[0];
	globalWorkSize[1] = grid[1]*localWorkSize[1];
	globalWorkSize[2] = grid[2]*localWorkSize[2];
	ret = clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel_stem_gpu, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("\n\nError launching kernel:%s\n",getErrorString(ret));
		return 0;
	}

	clEnqueueMarker(__cu2cl_CommandQueue, &eStop);
	clWaitForEvents(1, &eStop);
	__cu2cl_EventElapsedTime(&cuda_elapsedTime, eStart, eStop);
	PRINT_STAT_DOUBLE("gpu_stemmer", cuda_elapsedTime);
	clEnqueueMarker(__cu2cl_CommandQueue, &eStart);
	clEnqueueReadBuffer(__cu2cl_CommandQueue, gpu_stem_list, CL_TRUE, 0, words * sizeof(struct stemmer), stem_list, 0, NULL, NULL);
	clEnqueueMarker(__cu2cl_CommandQueue, &eStop);
	clWaitForEvents(1, &eStop);
	__cu2cl_EventElapsedTime(&cuda_elapsedTime, eStart, eStop);
	PRINT_STAT_DOUBLE("device_to_host", cuda_elapsedTime);

	clReleaseEvent(eStart);
	clReleaseEvent(eStop);

	STATS_END();
#ifdef TESTING
	printf("TESTING MODE ON! Writing to output\n");
	f = fopen("../input/stem_porter.gpu", "w");

//	for (int i = 0; i < words; ++i) fprintf(f, "b=%s j=%d k=%d\n", stem_list[i].b, stem_list[i].j, stem_list[i].k);
	for (int i = 0; i < words; ++i) fprintf(f, "%s\n", stem_list[i].b);

	fclose(f);
#endif
	__cu2cl_FreeHost(stem_list, __cu2cl_Mem_stem_list);
	clReleaseMemObject(gpu_stem_list);


	clReleaseKernel(__cu2cl_Kernel_stem_gpu);
	clReleaseProgram(__cu2cl_Program_porter_cu);
	clReleaseCommandQueue(__cu2cl_CommandQueue);
	clReleaseContext(__cu2cl_Context);
	return 0;
}
const char *getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}
