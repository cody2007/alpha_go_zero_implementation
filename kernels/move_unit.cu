#define N_ADJ 4

#define ADJ_LOOP(COORD)  \
		int coord_x = COORD / MAP_SZ_Y;\
		int coord_y = COORD % MAP_SZ_Y;\
		\
		int X_adj[N_ADJ] = {0, -1, 1, 0};\
		int Y_adj[N_ADJ] = {-1, 0, 0, 1};\
		for(int adj = 0; adj < N_ADJ; adj++){\
			int coord_px = coord_x + X_adj[adj];\
			int coord_py = coord_y + Y_adj[adj];\
			if(coord_py < 0 || coord_py >= MAP_SZ_Y ||\
				coord_px < 0 || coord_px >= MAP_SZ_X)\
					continue;\
			int coord_i = coord_px*MAP_SZ_Y + coord_py;\


__device__ inline int add_adj_to_stack(int coord, int * coord_stack, int coord_stack_sz, 
		char * checked, char op_player_val, int game_offset, char * board){
	DASSERT(coord >= 0 && coord < MAP_SZ);

	ADJ_LOOP(coord)
		if(checked[coord_i]) // already checked
			continue;

		if(board[game_offset + coord_i] == 0) return -1;

		// add to stack
		if(board[game_offset + coord_i] == op_player_val){
			checked[coord_i] = 1;
			coord_stack[coord_stack_sz] = coord_i;
			coord_stack_sz ++;
			DASSERT(coord_stack_sz < MAP_SZ)
		}
	} // adj

	return coord_stack_sz;
}

#define ADD_ADJ_TO_STACK(COORD, PLAYER_VAL) *coord_stack_sz = add_adj_to_stack(COORD, coord_stack, \
		*coord_stack_sz, checked, PLAYER_VAL, game_offset, board);

#define LIBERTY(COORD, PLAYER_VAL) return_liberty(COORD, PLAYER_VAL, game_offset, board, coord_stack, &coord_stack_sz)
__device__ inline char return_liberty(int coord, int player_val, int game_offset, char * board,
		int * coord_stack, int * coord_stack_sz){
	char checked[MAP_SZ];

	//////////// check if there exists a liberty for the placed stone
	*coord_stack_sz = 0;
	for(int i = 0; i < MAP_SZ; i++) checked[i] = 0; checked[coord] = 1;

	ADD_ADJ_TO_STACK(coord, player_val)

	for(int stack_i = 0; stack_i < *coord_stack_sz; stack_i++){
		int coord_j = coord_stack[stack_i];
		
		DASSERT(coord_j >= 0 && coord_j < MAP_SZ)
		DASSERT(board[game_offset + coord_j] == player_val)
		
		ADD_ADJ_TO_STACK(coord_j, player_val)

	} // stack
	
	return *coord_stack_sz == -1;
}

__global__ void move_unit_kernel(int *to_coord, int *moving_player, char * board, int * n_captures, char * moved, char * valid_mv_map_internal){
	int gm = blockIdx.x;
	int game_offset = gm * MAP_SZ;

	moved[gm] = 0;

	DASSERT(to_coord[gm] >= 0 && to_coord[gm] <= MAP_SZ)
	if(to_coord[gm] >= MAP_SZ) return;

	DASSERT(*moving_player == 0 || *moving_player == 1);

	GET_PLAYER_VAL

	int coord = to_coord[gm];

	// position not empty. shouldn't happen? (only when nn is making moves directly frm outputs)
	if(board[game_offset + coord] != 0) return;

	///////////////// check if we have listed this is a valid mv
	if(!valid_mv_map_internal[game_offset + coord]) return; // invalid move
	
	///////////////////////////
	
	board[game_offset + coord] = player_val;

	// adj search vars
	int coord_stack[MAP_SZ];
	int coord_stack_sz;

	///////////// check if we should remove stones
	char removed_stones = 0;
	
	ADJ_LOOP(coord)
		if(board[game_offset + coord_i] == (-player_val) &&
				!LIBERTY(coord_i, -player_val)){

			removed_stones = 1;
			DASSERT(board[game_offset + coord_i] == (-player_val))
			board[game_offset + coord_i] = 0;
			n_captures[*moving_player*BATCH_SZ + gm] ++;

			for(int stack_i = 0; stack_i < coord_stack_sz; stack_i++){
				int coord_j = coord_stack[stack_i];
				DASSERT(board[game_offset + coord_j] == (-player_val))
				board[game_offset + coord_j] = 0;
				n_captures[*moving_player*BATCH_SZ + gm] ++;

			} // stack

		} // opposing player / liberty check
	} // adj

	///////////////// if we've not removed stones, make sure there's a liberty for the placed stone
	if(!removed_stones && !LIBERTY(coord, player_val))
		board[game_offset + coord] = 0;

	// surrounded & could not capture
	if(board[game_offset + coord] == 0) return;
	
	moved[gm] = 1;
}

void move_unit_launcher(int * to_coord, int * moving_player, char * moved){
	REQ_INIT
	cudaError_t err;

	BMEM(board_pprev, board, BATCH_MAP_SZ)
	BMEM(board_prev, board, BATCH_MAP_SZ)
	move_unit_kernel <<< BATCH_SZ, 1 >>> (to_coord, moving_player, board, n_captures, moved, valid_mv_map_internal);

	CHECK_CUDA_ERR
	VERIFY_BUFFER_INTEGRITY
}

