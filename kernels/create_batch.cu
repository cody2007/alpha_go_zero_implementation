#define LIBERTY_TMP(COORD, PLAYER_VAL) return_liberty(COORD, PLAYER_VAL, 0, board_tmp, coord_stack, &coord_stack_sz)

// imgs_shape = [gv.BATCH_SZ, gv.n_rows, gv.n_cols, gv.n_input_channels]
// valid_mv_map = [gv.BATCH_SZ, gv.n_rows, gv.n_cols]

// create batch (for nn) from current game state
__global__ void create_batch_kernel(float * imgs, char * board, char * board_prev, char * board_pprev, int * moving_player, char * valid_mv_map,
		char * valid_mv_map_internal){
	
	int32_t gm = blockIdx.x;
	int32_t map_coord = threadIdx.x;
	int game_offset = gm*MAP_SZ;
	int gcoord = game_offset + map_coord;

	GET_PLAYER_VAL

	//////////// imgs
	int icoord = gm*MAP_SZ*N_INPUT_CHANNELS + map_coord*N_INPUT_CHANNELS;
	if(board[gcoord] == player_val)
		imgs[icoord] = 1;
	else if(board[gcoord] == 0)
		imgs[icoord] = 0;
	else
		imgs[icoord] = -1;

	icoord ++;
	if(board_prev[gcoord] == player_val)
		imgs[icoord] = 1;
	else if(board_prev[gcoord] == 0)
		imgs[icoord] = 0;
	else
		imgs[icoord] = -1;

	icoord ++;
	if(board_pprev[gcoord] == player_val)
		imgs[icoord] = 1;
	else if(board_pprev[gcoord] == 0)
		imgs[icoord] = 0;
	else
		imgs[icoord] = -1;

	//////////// valid moves
	// adj search vars
	int coord_stack[MAP_SZ];
	int coord_stack_sz;

	__syncthreads();
	if(map_coord != 0) return;

	#define ADD_MV(COORD) { valid_mv_map[gcoord] = 1; valid_mv_map_internal[gcoord] = 1; }
	
	for(map_coord = 0; map_coord < MAP_SZ; map_coord++){
		gcoord = game_offset + map_coord;
		
		valid_mv_map[gcoord] = 0;
		valid_mv_map_internal[gcoord] = 0;

		if(board[gcoord] != 0) continue;

		// add move
		if(LIBERTY(map_coord, player_val)){
			ADD_MV(map_coord)
			continue;
		}

		//////////// if no liberty, check if pieces can be captured creating liberty for moving player

		// copy board
		char board_tmp[MAP_SZ]; // just sotre one game, don't waste space for games not eval'd in this worker
		for(int loc = 0; loc < MAP_SZ; loc++)
			board_tmp[loc] = board[game_offset + loc];

		// if we did move here, would we capture?
		char valid_mv = 0;
		board_tmp[map_coord] = player_val; // tmp move here
		ADJ_LOOP(map_coord)
			// remove pieces with no liberty
			if(board_tmp[coord_i] == (-player_val) &&
				!LIBERTY_TMP(coord_i, -player_val)){
					valid_mv = 1;
					board_tmp[coord_i] = 0;

					// remove adj pieces (to then check if final state matches prior state)
					for(int stack_i = 0; stack_i < coord_stack_sz; stack_i++){
						int coord_j = coord_stack[stack_i];
						DASSERT(board_tmp[coord_j] == (-player_val))
						board_tmp[coord_j] = 0;
					} // stack

				} // opposing player / liberty check
		} // adj loop

		if(valid_mv == 0)
			continue;

		////// does this replicate a prior state?
		char matching = 1;
		for(int loc = 0; matching && (loc < MAP_SZ); loc++){
			matching = board_pprev[game_offset + loc] == board_tmp[loc];
		}

		if(matching == 0) ADD_MV(map_coord) 

	} // map loop
}

void create_batch_launcher(float * imgs, int * moving_player, char * valid_mv_map){
	REQ_INIT

	create_batch_kernel <<< BATCH_SZ, MAP_SZ >>> (imgs, board, board_prev, board_pprev, moving_player, valid_mv_map, valid_mv_map_internal);

	VERIFY_BUFFER_INTEGRITY
}
