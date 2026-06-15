#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nlhs > 0) {
        plhs[0] = mxCreateDoubleScalar(1.0);
    }
}
