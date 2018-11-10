__global__ void init_rand_states(int32_t RAND_SEED, int32_t map_sz, curandState_t * rand_states){
	int32_t offset = blockIdx.x*map_sz + threadIdx.x;
	curand_init(RAND_SEED + offset, 0, 1, &rand_states[offset]);
}

#define CMALLOC(VAR, SZ) {err = cudaMalloc((void**) &VAR, SZ*sizeof(VAR[0])); MALLOC_ERR_CHECK_R}
#define MALLOC_CHAR(VAR, SZ) {VAR = (char*) malloc(SZ*sizeof(VAR[0])); ASSERT(VAR != 0, "malloc failed"); } 
#define MALLOC_INT32(VAR, SZ) {VAR = (int32_t*) malloc(SZ*sizeof(VAR[0])); ASSERT(VAR != 0, "malloc failed"); } 
#define MALLOC_UINT32(VAR, SZ) {VAR = (uint32_t*) malloc(SZ*sizeof(VAR[0])); ASSERT(VAR != 0, "malloc failed"); } 

void init_op_launcher(){
	cudaError_t err;
	op_initialized = 1;

	///////////////////////////////// gpu buffers
	// game state
	CMALLOC(board, BATCH_MAP_SZ); 
	CMALLOC(board2, BATCH_MAP_SZ); 

	CMALLOC(board_prev, BATCH_MAP_SZ);
	CMALLOC(board_prev2, BATCH_MAP_SZ);

	CMALLOC(board_pprev, BATCH_MAP_SZ);
	CMALLOC(board_pprev2, BATCH_MAP_SZ);

	CMALLOC(n_captures, N_PLAYERS*BATCH_SZ);
	CMALLOC(n_captures2, N_PLAYERS*BATCH_SZ);

	CMALLOC(ai_to_coord, BATCH_SZ); // input to move_unit, output from move_random_ai

	CMALLOC(valid_mv_map_internal, BATCH_MAP_SZ) // input to move_unit, output from create_batch
	
	CMALLOC(moved_internal, BATCH_SZ) // [BATCH_SZ] used in move_random_ai, req. input to move_unit_launcher, results not used

	////// random seed
	int32_t RAND_SEED = time(NULL);
	err = cudaMalloc((void**) &rand_states, BATCH_MAP_SZ*sizeof(curandState_t));
	init_rand_states <<< BATCH_SZ, MAP_SZ >>> (RAND_SEED, MAP_SZ, rand_states);

}
