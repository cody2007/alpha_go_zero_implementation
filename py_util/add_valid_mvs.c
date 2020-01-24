static PyObject *add_valid_mvs(PyObject *self, PyObject *args){
	PyArrayObject *valid_mv_map_np;
	int moving_player;
	char * valid_mv_map;

	if(!PyArg_ParseTuple(args, "iO!", &moving_player, &PyArray_Type, &valid_mv_map_np)) return NULL;
	
	/////////////////////// check inputs
	ASSERT(moving_player == 0 || moving_player == 1, "moving player incorrect")
	ASSERT(valid_mv_map_np != NULL, "absent inputs")
	ASSERT(PyArray_TYPE(valid_mv_map_np) == NPY_INT8, "data type incorrect")
	ASSERT(PyArray_NDIM(valid_mv_map_np) == 3, "dims incorrect")
	ASSERT(PyArray_STRIDE(valid_mv_map_np, 2) == sizeof(valid_mv_map[0]), "data not contigious or C-order")

	npy_intp * dims_in = PyArray_DIMS(valid_mv_map_np);

	ASSERT(dims_in[0] == BATCH_SZ, "batch sz incorrect")
	ASSERT(dims_in[1] == MAP_SZ_X, "map sz incorrect")
	ASSERT(dims_in[2] == MAP_SZ_Y, "map sz incorrect")

	valid_mv_map = (char *) PyArray_DATA(valid_mv_map_np);

	////////////////////////////
	for(int gm = 0; gm < BATCH_SZ; gm++){ 
		int TO;
		int game_offset = gm*MAP_SZ;
	
		#ifdef CUDA_DEBUG
			if(tree_sz[gm] >= TREE_BUFFER_SZ)
				printf("tree_sz[%i] = %i tree_start %i\n", gm, tree_sz[gm], tree_start[gm]);
			if(tree_start[gm] < 0 || tree_start[gm] >= tree_sz[gm])
				printf("tree_sz[%i] = %i tree_start %i\n", gm, tree_sz[gm], tree_start[gm]);
		#endif

		int t_ind = tree_start[gm]; TO_FRM_T_IND

		// already created valid moves leaves:
		if(tree_list_start[TO] != -1){
			DASSERT(tree_player[TO] == moving_player);
			#ifdef CUDA_DEBUG
				int n_valid_mvs_chk = 1;
				for(int map_coord = 0; map_coord < MAP_SZ; map_coord++){
					int gcoord = game_offset + map_coord;
					if(valid_mv_map[gcoord]) n_valid_mvs_chk ++;
				}
				if(n_valid_mvs_chk != tree_list_sz[TO]){
					printf("skipping %i moving_player %i n_valid_mvs_chk %i tree_list_sz %i\n", gm, moving_player,
						n_valid_mvs_chk, tree_list_sz[TO]);
					DASSERT(0)
				}
			#endif
			continue;
		}
		
		tree_player[TO] = moving_player;
		tree_list_start[TO] = list_sz[gm];
		tree_list_sz[TO] = 0;

		DASSERT(list_sz[gm] < MV_BUFFER_SZ);
		
		#define LOE (gm*MV_BUFFER_SZ + list_sz[gm])
		#define ADD_MV(COORD) { list_valid_mv_inds[LOE] = COORD;\
				list_valid_tree_inds[LOE] = -1;\
				list_q_total[LOE] = 0;\
				list_visit_count[LOE] = 0;\
				list_prob[LOE] = -1;\
				tree_list_sz[TO] ++;\
				list_sz[gm] ++;\
				assert(list_sz[gm] < MV_BUFFER_SZ); }
		
		ADD_MV(-1) // pass move entry
		
		for(int map_coord = 0; map_coord < MAP_SZ; map_coord++){
			int gcoord = game_offset + map_coord;
			if(!valid_mv_map[gcoord]) continue;

			ADD_MV(map_coord)
		} // map loop
	} // gm

	Py_RETURN_NONE;
}

