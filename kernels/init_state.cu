void init_state_launcher() {
	CHECK_INIT

	cudaError_t err = cudaMemset(board, 0, sizeof(board[0])*BATCH_MAP_SZ); CHECK_CUDA_ERR
	err = cudaMemset(board_prev, 0, sizeof(board[0])*BATCH_MAP_SZ); CHECK_CUDA_ERR
	err = cudaMemset(board_pprev, 0, sizeof(board[0])*BATCH_MAP_SZ); CHECK_CUDA_ERR
	
	err = cudaMemset(n_captures, 0, sizeof(n_captures[0])*N_PLAYERS*BATCH_SZ); CHECK_CUDA_ERR
}

