/* HEADER FILE GENERATED BY cloc VERSION 0.7.4 */
/* THIS FILE:  /home/swapnilh/workloads/modified_sirius_suite/sirius-suite/gmm/gpu_hsa/gmm_scoring.h  */
/* INPUT FILE: /home/swapnilh/workloads/modified_sirius_suite/sirius-suite/gmm/gpu_hsa/gmm_scoring.cl  */
#ifdef __cplusplus
#define _CPPSTRING_ "C" 
#endif
#ifndef __cplusplus
#define _CPPSTRING_ 
#endif
#ifndef __SNACK_DEFS
typedef struct transfer_t Transfer_t;
struct transfer_t { int nargs ; size_t* rsrvd1; size_t* rsrvd2 ; size_t* rsrvd3; } ;
typedef struct lparm_t Launch_params_t;
struct lparm_t { int ndim; size_t gdims[3]; size_t ldims[3]; Transfer_t transfer  ;} ;
#define __SNACK_DEFS
#endif
extern _CPPSTRING_ void computeScore(const float* feature_vect,float* means_vect,float* precs_vect,float* weight_vect,float* factor_vect,float* score_vect, const Launch_params_t lparm);
