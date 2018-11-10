char return_device_buffers(){
	cudaError_t err;

	err = cudaMemcpy(board_cpu, board, BATCH_MAP_SZ*sizeof(board[0]), cudaMemcpyDeviceToHost);  MALLOC_ERR_CHECK

	return 1;
}

