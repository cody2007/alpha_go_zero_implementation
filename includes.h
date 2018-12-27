//#define CUDA_DEBUG 1

#define PANIC(A) { printf(A " %s:%i\n", __FILE__,__LINE__); exit(1); }
#define ASSERT(S, A) { if(!(S)) PANIC(A) }

#define N_TURNS 40//20
#define BATCH_SZ 128
#define N_PLAYERS 2

#define MAP_SZ_X 7

#define MAP_SZ_Y MAP_SZ_X

#define MAP_SZ (MAP_SZ_X*MAP_SZ_Y)
#define BATCH_MAP_SZ (BATCH_SZ*MAP_SZ_X*MAP_SZ_Y)

#define MAP_LOOP for(int loc = 0; loc < (MAP_SZ+1); loc++)

#define N_INPUT_CHANNELS 3

// return var indices
#define BOARD_IDX 0
#define VALID_MV_MAP_INTERNAL_IDX 2

#define RETURN_VARS 1
#define SET_VARS 0

