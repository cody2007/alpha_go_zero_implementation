__device__ inline char add_blank_adj_to_stack(int coord, int * coord_stack, int *coord_stack_sz, 
		char * checked, int game_offset, char * board, int * owner){
	DASSERT(coord >= 0 && coord < MAP_SZ);

	ADJ_LOOP(coord)
		if(checked[coord_i]) // already checked
			continue;

		// touching non-owner, therefore blank space is not owned by owner
		if(board[game_offset + coord_i] != 0){
			if(*owner != 0){
				if(*owner != board[game_offset + coord_i])
					return 0;
			}else
				// set owner
				*owner = board[game_offset + coord_i];
			
		}else{ // space is empty; add to stack

			checked[coord_i] = 1;
			coord_stack[*coord_stack_sz] = coord_i;
			*coord_stack_sz = *coord_stack_sz + 1;

			DASSERT(*coord_stack_sz < MAP_SZ)
		}
	} // adj

	return 1; // blank potentially owned by single player
}

#define ADD_BLANK_ADJ_TO_STACK(COORD) add_blank_adj_to_stack(COORD, coord_stack, \
		&coord_stack_sz, checked, game_offset, board, &owner);


#define SCORE_START (MAP_SZ*2)
#define LARGE_VAL 99999
__global__ void return_winner_kernel(float * winner, char * board, int * moving_player, float * score){ 
	int32_t game = blockIdx.x;
	int32_t coord = threadIdx.x;
	int game_offset = game*MAP_SZ;
	int gcoord = game_offset + coord;

	GET_PLAYER_VAL

	__shared__ unsigned score_tmp;
	if(coord == 0) score_tmp = SCORE_START;
	__syncthreads();

	if(board[gcoord] == player_val) // + 1
		atomicInc(&score_tmp, LARGE_VAL);
	else if(board[gcoord] == (-player_val))
		atomicDec(&score_tmp, LARGE_VAL); // -1
	else{
		// determine ownership of blank
		if(board[gcoord] != 0)
			printf("gcoord %i playerval %i board %i\n", gcoord, player_val, board[gcoord]);
		DASSERT(board[gcoord] == 0)
		
		int owner = 0;

		// adj search vars
		char checked[MAP_SZ];
		int coord_stack[MAP_SZ];
		int coord_stack_sz = 0;
		for(int i = 0; i < MAP_SZ; i++) checked[i] = 0; 
		checked[coord] = 1;

		int space_owned = ADD_BLANK_ADJ_TO_STACK(coord);
	
		for(int stack_i = 0; space_owned && (stack_i < coord_stack_sz); stack_i++){
			int coord_j = coord_stack[stack_i];

			DASSERT(coord_j >= 0 && coord_j < MAP_SZ)
			DASSERT(board[game_offset + coord_j] == 0)
			
			space_owned = ADD_BLANK_ADJ_TO_STACK(coord_j);
		}

		// add score to winner
		if(space_owned && owner != 0){
			if(owner == player_val)
				atomicInc(&score_tmp, LARGE_VAL);
			else
				atomicDec(&score_tmp, LARGE_VAL);
		}

	} // empty space

	__syncthreads();
	if(coord != 0)
		return;

	score[game] = (float)(score_tmp) - (float)(SCORE_START);
	if(score_tmp > SCORE_START)
		winner[game] = 1;
	else if(score_tmp < SCORE_START)
		winner[game] = -1;
	else
		winner[game] = 0;

}

void return_winner_launcher(float * winner, int * moving_player, float * score, int * n_captures_out){
	REQ_INIT

	cudaError_t err;
	BMEM(n_captures_out, n_captures, N_PLAYERS*BATCH_SZ)

	return_winner_kernel <<< BATCH_SZ, MAP_SZ >>> (winner, board, moving_player, score);
	VERIFY_BUFFER_INTEGRITY
}
