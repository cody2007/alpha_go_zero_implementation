#define ZERO(A, S) memset(A, 0, (S)*sizeof(A[0]));

void init_vecs(){
	
	ZERO(tree_player, BATCH_SZ * TREE_BUFFER_SZ)
	ZERO(list_q_total, BATCH_SZ * MV_BUFFER_SZ)
	ZERO(list_visit_count, BATCH_SZ * MV_BUFFER_SZ)

		

	for(int i = 0; i < (BATCH_SZ*TREE_BUFFER_SZ); i++){
		tree_parent[i] = -1;
		tree_list_start[i] = -1;
		tree_list_sz[i] = -1;
	}

	for(int i = 0; i < (BATCH_SZ*MV_BUFFER_SZ); i++){
		list_valid_mv_inds[i] = -1;
		list_valid_tree_inds[i] = -1;
		list_prob[i] = -1;
	}
}

static PyObject *init_tree(PyObject *self, PyObject *args){
	ZERO(tree_start, BATCH_SZ)
	ZERO(list_sz, BATCH_SZ)

	for(int i = 0; i < BATCH_SZ; i++)
		tree_sz[i] = 1;

	init_vecs();	

	Py_RETURN_NONE;
}

