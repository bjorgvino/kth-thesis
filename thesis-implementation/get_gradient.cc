#include <math.h>
#include <string.h>
#include "mex.h"

int number_of_actions(const mxArray *v) {
  const int *dims = mxGetDimensions(v);
  return dims[0];
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  // Input validation
  if (nrhs != 6)
    mexErrMsgTxt("Wrong number of inputs. The functions takes 6 parameters."); 
  if (nlhs > 1)
    mexErrMsgTxt("Wrong number of outputs. The function returns a matrix.");
  
  // Extract and name the inputs
  const mxArray *belief_inp = prhs[0];
  const mxArray *p_out = prhs[1];
  int action = (int)mxGetScalar(prhs[2]);
  double prob = mxGetScalar(prhs[3]);
  int N = (int)mxGetScalar(prhs[4]);
  int M = (int)mxGetScalar(prhs[5]);

  // More input validation
  if (mxGetNumberOfDimensions(belief_inp) != 2 || mxGetClassID(belief_inp) != mxDOUBLE_CLASS)
    mexErrMsgTxt("Invalid input: belief should be a Nx1 double vector.");

  if (mxGetNumberOfDimensions(p_out) != 2 || mxGetClassID(p_out) != mxDOUBLE_CLASS)
    mexErrMsgTxt("Invalid input: policy_output should be a Nx1 double vector.");

  // Set the size of the output matrix and initialize array
  int gradient_dim[2] = {M, N};
  mxArray *gradient = mxCreateNumericArray(2, gradient_dim, mxDOUBLE_CLASS, mxREAL);
  
  // Set variables
  double *belief = (double *)mxGetPr(belief_inp);
  double *policy_out = (double *)mxGetPr(p_out); // double pointer to p_out
  int a = action - 1; // Convert from MATLAB to C indexing
  double *gradient_val = (double *)mxGetPr(gradient); // double pointer to output matrix
  
  double cj = 1 / policy_out[a];

  int row, col;
  double policy_out_sum, cij;

  // Sum over policy_out
  for (row = 0; row < M; row++) {
    policy_out_sum += policy_out[row];
  }

  // Calculate gradient
  for (row = 0; row < M; row++) {
    if (row == a)
      cij = 0;
    else
      cij = cj * (policy_out_sum - policy_out[a] - policy_out[row]);

    for (col = 0; col < N; col++) {
      if (row == a)
        gradient_val[row + col * N] = belief[col] * prob * (1 - (1 + cij) * prob);
      else
        gradient_val[row + col * N] = -belief[col] * prob * (1 - (1 + cij) * prob);
    }
  }

  // Return the gradient
  plhs[0] = gradient;
}
