__global__ void prob_to_coord_valid_mvs_kernel(half * prob_map, int16_t * to_coord, 
		char * board, curandState_t* rand_states, char * valid_mv_map_internal){
	int gm = blockIdx.x;
	int gm_offset = gm*MAP_SZ;
	half * prob_map_cur = &prob_map[gm_offset];

	COUNT_VALID
	
	float rand_val = (float)(curand(&rand_states[gm]) % RAND_RES);
	rand_val /= (float)RAND_RES;

	// compute probs sum over valid mvs
	float probs_sum_orig = 0;
	for(int mv_ind = 1; mv_ind < n_valid_mvs; mv_ind++){ // skip pass move
		int map_loc = valid_mv_inds[mv_ind];
		CHK_VALID_MAP_COORD(map_loc)
		DASSERT(board[gm*MAP_SZ + map_loc] == 0)
		probs_sum_orig += (float)prob_map_cur[map_loc];
	}
	if(probs_sum_orig == 0) probs_sum_orig = 1;
	//assert(probs_sum_orig >= 0);
	
	float probs_sum = 0;
	for(int mv_ind = 1; mv_ind < n_valid_mvs; mv_ind++){ // skip pass move
		int16_t map_loc = valid_mv_inds[mv_ind];
		float p = (float)prob_map_cur[map_loc] / probs_sum_orig;
		//if(!(p >= 0 && p <= 1))
		//	printf("prob err %f\n", p);
		//DASSERT(p >= 0 && p <= 1)

		// randomly selected or we're at the last move
		if(((rand_val >= probs_sum) && (rand_val < (probs_sum + p))) || 
				(mv_ind == (n_valid_mvs - 1))){
			to_coord[gm] = map_loc;
			return;
		}
		probs_sum += p;
	}

	to_coord[gm] = -1;
	//assert(0);
}

void prob_to_coord_valid_mvs_launcher(float * prob_map, int16_t * to_coord){
	cudaError_t err;
	REQ_INIT

	prob_to_coord_valid_mvs_kernel <<< BATCH_SZ, 1 >>> ((half*)prob_map, to_coord, board, rand_states, valid_mv_map_internal); CHECK_CUDA_ERR

	VERIFY_BUFFER_INTEGRITY
}


