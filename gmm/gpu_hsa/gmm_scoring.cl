__constant float logZero = -3.4028235E38;
__constant float maxLogValue = 7097004.5;
__constant float minLogValue = -7443538.0;
__constant float naturalLogBase = (float)1.00011595E-4;
__constant float inverseNaturalLogBase = 9998.841;
// fixed for a given accoustic model
__constant int comp_size = 32;
__constant int feat_size = 29;
__constant int senone_size = 5120;
 
__kernel void    computeScore(const __global float *feature_vect, __global float *means_vect,                 __global float *precs_vect, __global float *weight_vect, __global float *factor_vect,                 __global float *score_vect) {
  int i = get_group_id(0) * get_local_size(0) + get_local_id(0);

  if (i < senone_size) {
    float local_score_vect = logZero;

#pragma unroll 32
    for (int j = 0; j < comp_size; j++) {
      // getScore
      float logDval = 0.0f;
#pragma unroll 29
      for (int k = 0; k < feat_size; k++) {
        int idx = i + senone_size * j + k * comp_size * senone_size;
        float logDiff = feature_vect[k] - means_vect[idx];
        logDval += logDiff * logDiff * precs_vect[idx];
      }

      // Convert to the appropriate base.
      if (logDval != logZero) {
        logDval = logDval * inverseNaturalLogBase;
      }

      int idx2 = i + j * senone_size;

      // Add the precomputed factor, with the appropriate sign.
      logDval -= factor_vect[idx2];

      if (logDval < logZero) {
        logDval = logZero;
      }
      // end of getScore

      float logVal2 = logDval + weight_vect[idx2];

      float logHighestValue = local_score_vect;
      float logDifference = local_score_vect - logVal2;

      // difference is always a positive number
      if (logDifference < 0) {
        logHighestValue = logVal2;
        logDifference = -logDifference;
      }

      float logValue = -logDifference;
      float logInnerSummation;
      if (logValue < minLogValue) {
        logInnerSummation = 0.0;
      } else if (logValue > maxLogValue) {
        logInnerSummation = FLT_MAX;
      } else {
        if (logValue == logZero) {
          logValue = logZero;
        } else {
          logValue = logValue * naturalLogBase;
        }
        logInnerSummation = native_exp(logValue);
      }

      logInnerSummation += 1.0;

      float returnLogValue;
      if (logInnerSummation <= 0.0) {
        returnLogValue = logZero;
      } else {
        returnLogValue = native_log(logInnerSummation) * inverseNaturalLogBase;
        if (returnLogValue > FLT_MAX) {
          returnLogValue = FLT_MAX;
        } else if (returnLogValue < -FLT_MAX) {
          returnLogValue = -FLT_MAX;
        }
      }
      // sum log
      local_score_vect = logHighestValue + returnLogValue;
    }
    score_vect[i] = local_score_vect;
  }
}

