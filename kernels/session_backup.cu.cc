void session_backup_launcher(){
	REQ_INIT
	cudaError_t err;

	BMEM(board2, board, BATCH_MAP_SZ)
	BMEM(board_prev2, board_prev, BATCH_MAP_SZ)
	BMEM(board_pprev2, board_pprev, BATCH_MAP_SZ)

	BMEM(n_captures2, n_captures, BATCH_SZ)
}

void session_restore_launcher(){
	REQ_INIT
	cudaError_t err;

	RMEM(board2, board, BATCH_MAP_SZ)
	RMEM(board_prev2, board_prev, BATCH_MAP_SZ)
	RMEM(board_pprev2, board_pprev, BATCH_MAP_SZ)

	RMEM(n_captures2, n_captures, BATCH_SZ)
}
