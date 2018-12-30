#define DIR_EPS .25
#define RAND_RES 100000
#define PROB (prob_map[MO + loc] / probs_sum_orig)

__global__ void prob_to_coord_kernel(float * prob_map, int * to_coord,
		curandState_t* rand_states, float * dir_pre, float * dir_a){
	int gm = blockIdx.x;
	int MO = gm*MAP_SZ;
	float rand_val = (float)(curand(&rand_states[gm]) % RAND_RES);
	rand_val /= (float)RAND_RES;

	float probs_sum_orig = 0;
	MAP_LOOP
		probs_sum_orig += prob_map[MO + loc];
	assert(probs_sum_orig >= 0);

	////////// add dir noise
	if(*dir_a >= .0001){
		float val = 1;
		// val = np.prod(prob_map ** (dir_a-1))
		MAP_LOOP{
			if(prob_map[MO + loc] == 0)
				continue;
			val *= powf(PROB, (*dir_a)-1);
		}
	
		// add dir_pre * val to prob_map, sum new prob_map
		float probs_sum = 0;
		MAP_LOOP{
			prob_map[MO + loc] = (1-DIR_EPS)*PROB + (*dir_pre) * val * DIR_EPS;
			probs_sum += PROB;
		}
		
		// renormalize to sum to 1
		MAP_LOOP
			prob_map[MO + loc] /= probs_sum;
	}

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

void prob_to_coord_launcher(float * prob_map, int * to_coord, float * dir_pre, float * dir_a){
	REQ_INIT
	cudaError_t err; 

	prob_to_coord_kernel <<< BATCH_SZ, 1 >>> (prob_map, to_coord, rand_states,
			dir_pre, dir_a);

	CHECK_CUDA_ERR
	VERIFY_BUFFER_INTEGRITY
}

