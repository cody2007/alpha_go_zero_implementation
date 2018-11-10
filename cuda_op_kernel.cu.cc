#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <curand.h>
#include <curand_kernel.h>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "includes.h"
#include "cuda_includes.h"

#include "kernels/return_state.cu"
#include "kernels/verify_integrity.cu"
#include "kernels/init_op.cu" // allocates memory

#include "kernels/vars.cu.cc"
#include "kernels/init_state.cu" // inits new set of games

#include "kernels/move_unit.cu"
#include "kernels/move_random_ai.cu"
#include "kernels/create_batch.cu"
#include "kernels/return_winner.cu"

#include "kernels/session_backup.cu.cc"
#include "kernels/prob_to_coord.cu"
#include "kernels/prob_to_coord_valid_mvs.cu"
#include "kernels/max_prob_to_coord_valid_mvs.cu"

#endif

