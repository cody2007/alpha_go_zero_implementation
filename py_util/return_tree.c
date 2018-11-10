/*	tree_sz, tree_start, tree_player, tree_parent, tree_list_sz, tree_list_start, \
		list_sz, list_valid_mv_inds, list_valid_tree_inds, list_q_total, list_prob, \
		list_visit_count = tf_op.return_tree()
*/
static PyObject *return_tree(PyObject *self, PyObject *args){
	
	///// output
	npy_intp dims[4];
	dims[0] = BATCH_SZ;
	dims[1] = TREE_BUFFER_SZ;

	PyObject * tree_sz_np = PyArray_SimpleNew(1, dims, NPY_UINT32);
	PyObject * tree_start_np = PyArray_SimpleNew(1, dims, NPY_UINT32);
	
	PyObject * tree_player_np = PyArray_SimpleNew(2, dims, NPY_INT8);
	PyObject * tree_parent_np = PyArray_SimpleNew(2, dims, NPY_INT32);

	PyObject * tree_list_sz_np = PyArray_SimpleNew(2, dims, NPY_INT32);
	PyObject * tree_list_start_np = PyArray_SimpleNew(2, dims, NPY_INT32);

	PyObject * list_sz_np = PyArray_SimpleNew(1, dims, NPY_UINT32);

	dims[1] = MV_BUFFER_SZ;
	PyObject * list_valid_mv_inds_np = PyArray_SimpleNew(2, dims, NPY_INT16);
	PyObject * list_valid_tree_inds_np = PyArray_SimpleNew(2, dims, NPY_INT32);
	PyObject * list_q_total_np = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
	PyObject * list_prob_np = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
	PyObject * list_visit_count_np = PyArray_SimpleNew(2, dims, NPY_UINT32);

	ASSERT(tree_sz_np && tree_start_np && tree_player_np && tree_parent_np && tree_list_sz_np &&
		tree_list_start_np && list_sz_np && list_valid_mv_inds_np && list_valid_tree_inds_np &&
		list_q_total_np && list_prob_np && list_visit_count_np, "error creating python outputs");

	unsigned * tree_sz_ret = (unsigned *) PyArray_DATA((PyArrayObject*) tree_sz_np);
	unsigned * tree_start_ret = (unsigned *) PyArray_DATA((PyArrayObject*) tree_start_np);
	
	char * tree_player_ret = (char *) PyArray_DATA((PyArrayObject*) tree_player_np);
	int * tree_parent_ret = (int *) PyArray_DATA((PyArrayObject*) tree_parent_np);

	int * tree_list_sz_ret = (int *) PyArray_DATA((PyArrayObject*) tree_list_sz_np);
	int * tree_list_start_ret = (int *) PyArray_DATA((PyArrayObject*) tree_list_start_np);

	unsigned * list_sz_ret = (unsigned *) PyArray_DATA((PyArrayObject*) list_sz_np);

	short * list_valid_mv_inds_ret = (short *) PyArray_DATA((PyArrayObject*) list_valid_mv_inds_np);
	int * list_valid_tree_inds_ret = (int *) PyArray_DATA((PyArrayObject*) list_valid_mv_inds_np);
	float * list_q_total_ret = (float *) PyArray_DATA((PyArrayObject*) list_q_total_np);
	float * list_prob_ret = (float *) PyArray_DATA((PyArrayObject*) list_prob_np);
	unsigned * list_visit_count_ret = (unsigned *) PyArray_DATA((PyArrayObject*) list_visit_count_np);

	////////////////////////////////////// copy
	BMEM(tree_sz_ret, tree_sz, BATCH_SZ)
	BMEM(tree_start_ret, tree_start, BATCH_SZ)

	BMEM(tree_player_ret, tree_player, B_TREE_SZ)
	BMEM(tree_parent_ret, tree_parent, B_TREE_SZ)

	BMEM(tree_list_sz_ret, tree_list_sz, B_TREE_SZ)
	BMEM(tree_list_start_ret, tree_list_start, B_TREE_SZ)

	BMEM(list_sz_ret, list_sz, BATCH_SZ)

	BMEM(list_valid_mv_inds_ret, list_valid_mv_inds, B_MV_SZ)
	BMEM(list_valid_tree_inds_ret, list_valid_tree_inds, B_MV_SZ)
	BMEM(list_q_total_ret, list_q_total, B_MV_SZ)
	BMEM(list_prob_ret, list_prob, B_MV_SZ)
	BMEM(list_visit_count_ret, list_visit_count, B_MV_SZ)
	
	/////////// return
	PyObject * ret = PyList_New(12);
	ASSERT(ret != 0, "err creating output list")

	ASSERT(PyList_SetItem(ret, 0, tree_sz_np) == 0, "failed setting item");
	ASSERT(PyList_SetItem(ret, 1, tree_start_np) == 0, "failed setting item");
	ASSERT(PyList_SetItem(ret, 2, tree_player_np) == 0, "failed setting item");
	ASSERT(PyList_SetItem(ret, 3, tree_parent_np) == 0, "failed setting item");
	ASSERT(PyList_SetItem(ret, 4, tree_list_sz_np) == 0, "failed setting item");
	ASSERT(PyList_SetItem(ret, 5, tree_list_start_np) == 0, "failed setting item");
	ASSERT(PyList_SetItem(ret, 6, list_sz_np) == 0, "failed setting item");
	ASSERT(PyList_SetItem(ret, 7, list_valid_mv_inds_np) == 0, "failed setting item");
	ASSERT(PyList_SetItem(ret, 8, list_valid_tree_inds_np) == 0, "failed setting item");
	ASSERT(PyList_SetItem(ret, 9, list_q_total_np) == 0, "failed setting item");
	ASSERT(PyList_SetItem(ret, 10, list_prob_np) == 0, "failed setting item");
	ASSERT(PyList_SetItem(ret, 11, list_visit_count_np) == 0, "failed setting item");

	return ret;
}

