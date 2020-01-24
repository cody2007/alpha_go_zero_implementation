__global__ void max_prob_to_coord_valid_mvs_kernel(half * prob_map, int16_t * to_coord, 
		char * board, char * valid_mv_map_internal){
	int gm = blockIdx.x;
	int gm_offset = gm*MAP_SZ;
	half * prob_map_cur = &prob_map[gm_offset];

	COUNT_VALID

	// determine max prob
	float max_prob = -999;
	int16_t max_map_loc = -1;
	for(int mv_ind = 1; mv_ind < n_valid_mvs; mv_ind++){ // skip pass move
		int map_loc = valid_mv_inds[mv_ind];
		CHK_VALID_MAP_COORD(map_loc)
		DASSERT(board[gm*MAP_SZ + map_loc] == 0)
		if((float)prob_map_cur[map_loc] <= max_prob)
			continue;
		max_map_loc = map_loc;
		max_prob = prob_map_cur[map_loc];
	}

	to_coord[gm] = max_map_loc;
}

void max_prob_to_coord_valid_mvs_launcher(float * prob_map, int16_t * to_coord){
	cudaError_t err;
	REQ_INIT

	max_prob_to_coord_valid_mvs_kernel <<< BATCH_SZ, 1 >>> ((half*)prob_map, to_coord, board, 
		valid_mv_map_internal); CHECK_CUDA_ERR

	VERIFY_BUFFER_INTEGRITY
}


