#define RAND_RES 100000
#define PROB ((float)prob_map[MO + loc] / probs_sum_orig)

__global__ void prob_to_coord_kernel(half * prob_map, int16_t * to_coord, curandState_t* rand_states){
	int gm = blockIdx.x;
	int MO = gm*MAP_SZ;
	float rand_val = (float)(curand(&rand_states[gm]) % RAND_RES);
	rand_val /= (float)RAND_RES;

	float probs_sum_orig = 0;
	MAP_LOOP
		probs_sum_orig += (float)prob_map[MO + loc];
	assert(probs_sum_orig >= 0);

	float probs_sum = 0;
	MAP_LOOP{
		if(PROB < 0 || PROB > 1)
			printf("PROB %f\n", PROB);
		//DASSERT(PROB >= 0 && PROB <= 1)

		if((rand_val >= probs_sum) && (rand_val < (probs_sum + PROB))){
			to_coord[gm] = loc;
			return;
		}
		probs_sum += PROB;
	}

	to_coord[gm] = -1;

	DASSERT(probs_sum <= 1.01)
	DASSERT(probs_sum >= .999)
}

void prob_to_coord_launcher(float * prob_map, int16_t * to_coord){
	REQ_INIT
	cudaError_t err; 

	prob_to_coord_kernel <<< BATCH_SZ, 1 >>> ((half*)prob_map, to_coord, rand_states);

	CHECK_CUDA_ERR
	VERIFY_BUFFER_INTEGRITY
}

