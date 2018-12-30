__global__ void move_random_ai_kernel(int * to_coord, char * board, curandState_t* rand_states, char * valid_mv_map_internal){
	
	int gm = blockIdx.x;
	int gm_offset = gm*MAP_SZ;

	COUNT_VALID

	// select random move
	int rand_ind = (curand(&rand_states[gm]) % (n_valid_mvs-1)) + 1;
	
	to_coord[gm] = valid_mv_inds[rand_ind];

	DASSERT(to_coord[gm] >= 0 && to_coord[gm] < MAP_SZ && board[gm_offset + to_coord[gm]] == 0)

}

void move_random_ai_launcher(int * moving_player){
	cudaError_t err;
	REQ_INIT

	move_random_ai_kernel <<< BATCH_SZ, 1 >>> (ai_to_coord, board, rand_states, valid_mv_map_internal); CHECK_CUDA_ERR

	move_unit_launcher(ai_to_coord, moving_player, moved_internal); 
	VERIFY_BUFFER_INTEGRITY
}


