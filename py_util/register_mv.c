// register move in tree, initialize node if not already initialized
static PyObject *register_mv(PyObject *self, PyObject *args){
	PyArrayObject *chosen_coord_np;
	int moving_player, * chosen_coord;

	if(!PyArg_ParseTuple(args, "iO!", &moving_player, &PyArray_Type, &chosen_coord_np)) return NULL;
	
	/////////////////////// check inputs
	ASSERT(moving_player == 0 || moving_player == 1, "moving player incorrect")
	ASSERT(chosen_coord_np != NULL, "absent inputs")
	ASSERT(PyArray_TYPE(chosen_coord_np) == NPY_INT32, "data type incorrect")
	ASSERT(PyArray_NDIM(chosen_coord_np) == 1, "dims incorrect")
	ASSERT(PyArray_STRIDE(chosen_coord_np, 0) == sizeof(chosen_coord[0]), "data not contigious or C-order")

	npy_intp * dims_in = PyArray_DIMS(chosen_coord_np);

	ASSERT(dims_in[0] == BATCH_SZ, "batch sz incorrect")

	chosen_coord = (int *) PyArray_DATA(chosen_coord_np);

	///////////////////////////////

	for(int gm = 0; gm < BATCH_SZ; gm++){
		//if(chosen_coord[gm] == -1) continue;

		#ifdef CUDA_DEBUG
			if(tree_sz[gm] >= TREE_BUFFER_SZ){
				printf("tree_sz[%i] %i\n", gm, tree_sz[gm]);
				DASSERT(0);
			}
			if(tree_start[gm] < 0 || tree_start[gm] >= tree_sz[gm]){
				printf("tree_sz[%i] %i\n", gm, tree_sz[gm]);
				printf("tree_start %i\n", tree_start[gm]);
				DASSERT(0);
			}
			if(list_sz[gm] >= MV_BUFFER_SZ){
				printf("list_sz[%i] %i\n", gm, list_sz[gm]);
				DASSERT(0);
			}
			int t_ind2 = tree_start[gm];
			int TO2 = gm*TREE_BUFFER_SZ + t_ind2;
			if(tree_list_start[TO2] < 0 || tree_list_start[TO2] >= list_sz[gm]){
				printf("list_sz[%i] %i\n", gm, list_sz[gm]);
				printf("tree_list_start[%i] %i\n", TO2, tree_list_start[TO2]);
				DASSERT(0);
			}
		#endif

		CUR_TREE_INDS	
		
		// find list index for chosen move
		char found = 0;
		int LOC;
		for(int mv_ind = 0; mv_ind < n_valid_mvs; mv_ind++){
			LOC = LO + mv_ind;
			if(list_valid_mv_inds[LOC] != chosen_coord[gm]) continue;
			
			found = 1;
			break;
		}
	
		#ifdef CUDA_DEBUG
			if(found == 0){
				printf("could not find valid move: gm %i chosen_coord %i n_valid_mvs %i\n", gm, chosen_coord[gm], n_valid_mvs);
				for(int mv_ind = 0; mv_ind < n_valid_mvs; mv_ind++){
					LOC = LO + mv_ind;
					printf("valid: %i\n", list_valid_mv_inds[LOC]);
				}
				for(int gm2 = 0; gm2 < BATCH_SZ; gm2++)
					printf("to_coords[%i] %i\n", gm2, chosen_coord[gm2]);
				//LOC = LO;
			}
		#endif
		ASSERT(found != 0, "could not find move");

		// update pointer to tree_start
		int t_ind_new;
		if(list_valid_tree_inds[LOC] == -1){ 
			
			// create new node, return t_ind_new
			list_valid_tree_inds[LOC] = tree_sz[gm];

			t_ind_new = tree_sz[gm];
			int TO_NEW = gm*TREE_BUFFER_SZ + t_ind_new;
			
			tree_parent[TO_NEW] = t_ind;
			tree_player[TO_NEW] = moving_player == 0;
			tree_list_start[TO_NEW] = -1;
			tree_list_sz[TO_NEW] = 0;

			tree_sz[gm] ++;
			ASSERT(tree_sz[gm] < TREE_BUFFER_SZ, "tree buffer size exceeded");
		}else{ 
			
			// return t_ind_new from list
			t_ind_new = list_valid_tree_inds[LOC];		
			DASSERT(t_ind_new >= 0 && t_ind_new < TREE_BUFFER_SZ);
			
			#ifdef CUDA_DEBUG
				int TO_NEW = gm*TREE_BUFFER_SZ + t_ind_new;
			#endif

			DASSERT(tree_parent[TO_NEW] == t_ind)
			DASSERT(tree_player[TO_NEW] == (!moving_player))
		}
		tree_start[gm] = t_ind_new;
	} // gm

	Py_RETURN_NONE;
}

