static PyObject *backup_visit(PyObject *self, PyObject *args){
	PyArrayObject * q_np;
	float * q;
	int moving_player;

	if(!PyArg_ParseTuple(args, "iO!", &moving_player, &PyArray_Type, &q_np)) return NULL;

	/////////////////// check inputs
	ASSERT(q_np != NULL, "absent inputs")
	ASSERT(PyArray_TYPE(q_np) == NPY_FLOAT32, "data type incorrect")
	ASSERT(PyArray_NDIM(q_np) == 1, "dims must be 1")
	ASSERT(PyArray_STRIDE(q_np, 0) == sizeof(q[0]), "data not contigious or C-order")
	ASSERT(moving_player == 0 || moving_player == 1, "moving_player incorrect")

	npy_intp * dims_in = PyArray_DIMS(q_np);

	ASSERT(dims_in[0] == BATCH_SZ, "batch sz incorrect")

	q = (float *) PyArray_DATA(q_np);

	/////////////////////////

	for(int gm = 0; gm < BATCH_SZ; gm++){
				
		// tree ind
		int TO, LO;
		int t_ind = tree_start[gm]; TO_FRM_T_IND

		while(1){
			int t_ind_prev = t_ind;
			if(tree_parent[TO] == -1) // tree root
				break;

			// inds
			t_ind = tree_parent[TO]; TO_FRM_T_IND
			int l_ind = tree_list_start[TO]; LO_FRM_L_IND
			int n_valid_mvs = tree_list_sz[TO]; CHK_N_VALID_MVS

			// find list index for previous tree ind
			char found = 0;
			int LOC;
			for(int mv_ind = 0; mv_ind < n_valid_mvs; mv_ind++){
				LOC = LO + mv_ind;
				if(list_valid_tree_inds[LOC] != t_ind_prev) continue;
				
				found = 1;
				break;
			}
			assert(found != 0);

			if(tree_player[TO] == moving_player){
				list_visit_count[LOC] ++;
				DASSERT((powf(2, 8*sizeof(list_visit_count[0])) - 3) > (float)list_visit_count[LOC]) // overflow check
				list_q_total[LOC] += q[gm];
			}
		
		}
	} // gm

	Py_RETURN_NONE;
}

