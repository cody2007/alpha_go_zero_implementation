// inputs: imgs[batch_sz, map_sz_x, map_sz_y, channels]
// randomly rotate/reflect each image
static PyObject *rotate_reflect_imgs(PyObject *self, PyObject *args){
	PyArrayObject *imgs_np, *tree_probs_np;
	PyObject *imgs_r_np, *tree_probs_r_np;
	float * imgs, *imgs_r, *tree_probs, *tree_probs_r;

	if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &imgs_np, &PyArray_Type, &tree_probs_np)) return NULL;
	
	/////////////////////// check inputs
	ASSERT(imgs_np != NULL, "absent inputs")
	ASSERT(PyArray_TYPE(imgs_np) == NPY_FLOAT32 && PyArray_TYPE(tree_probs_np) == NPY_FLOAT32, "data type incorrect")
	ASSERT(PyArray_NDIM(imgs_np) == 4 && PyArray_NDIM(tree_probs_np) == 2, "dims must be 4")
	ASSERT(PyArray_STRIDE(imgs_np, 3) == sizeof(imgs[0]) && PyArray_STRIDE(tree_probs_np, 1) == sizeof(tree_probs[0]), "data not contigious or C-order")

	npy_intp * dims_in = PyArray_DIMS(imgs_np);
	npy_intp * pdims_in = PyArray_DIMS(tree_probs_np);

	ASSERT(dims_in[0] == BATCH_SZ, "batch sz incorrect")

	int map_sz_x = dims_in[1];
	int map_sz_y = dims_in[2];
	int n_chan = dims_in[3];

	ASSERT(map_sz_x == map_sz_y, "board must be sq")
	ASSERT(pdims_in[0] == BATCH_SZ && pdims_in[1] == ((map_sz_x*map_sz_y)+1), "tree_probs incorrect")

	imgs_r_np = PyArray_SimpleNew(4, dims_in, NPY_FLOAT);
	tree_probs_r_np = PyArray_SimpleNew(2, pdims_in, NPY_FLOAT);

	imgs = (float *) PyArray_DATA(imgs_np);
	tree_probs = (float *) PyArray_DATA(tree_probs_np);

	imgs_r = (float *) PyArray_DATA((PyArrayObject*) imgs_r_np);
	tree_probs_r = (float *) PyArray_DATA((PyArrayObject*) tree_probs_r_np);

	float * imgs_r_pre = malloc(BATCH_SZ*map_sz_x*map_sz_y*n_chan*sizeof(imgs[0]));
	float * tree_probs_r_pre = malloc(BATCH_SZ*((map_sz_x*map_sz_y)+ 1)*sizeof(imgs[0]));

	ASSERT(imgs_r_pre && tree_probs_r_pre, "failed allocating");

	#define MAP_LOOP_SEP for(int x = 0; x < map_sz_x; x++){ for(int y = 0; y < map_sz_y; y++){

	#define CP(X, Y) MAP_LOOP_SEP\
				memcpy(&imgs_r_pre[gm_off + x*map_sz_y*n_chan + y*n_chan], \
					&imgs[gm_off + (X)*map_sz_y*n_chan + (Y)*n_chan], n_chan*sizeof(imgs[0]));\
				tree_probs_r_pre[pgm_off + x*map_sz_y + y] = \
					tree_probs[pgm_off + (X)*map_sz_y + Y];\
			}}

	#define CP_F(X, Y) MAP_LOOP_SEP\
				memcpy(&imgs_r[gm_off + x*map_sz_y*n_chan + y*n_chan], \
					&imgs_r_pre[gm_off + (X)*map_sz_y*n_chan + (Y)*n_chan], n_chan*sizeof(imgs[0]));\
				tree_probs_r[pgm_off + x*map_sz_y + y] = \
					tree_probs_r_pre[pgm_off + (X)*map_sz_y + Y];\
			}}
	for(int gm = 0; gm < BATCH_SZ; gm++){
		int op = rand() % 4;
		int trans = rand() % 2;
		int gm_off = gm*map_sz_x*map_sz_y*n_chan;
		int pgm_off = gm*(map_sz_x*map_sz_y + 1);

		//////////////////////////////////
		if(op == 0){ // no transform
			memcpy(&imgs_r_pre[gm_off], &imgs[gm_off], map_sz_x*map_sz_y*n_chan*sizeof(imgs[0]));
			memcpy(&tree_probs_r_pre[pgm_off], &tree_probs[pgm_off], map_sz_x*map_sz_y*sizeof(imgs[0]));
		}else if(op == 1){ // imgs[::-1]
			CP(map_sz_x - 1 - x, y)
		}else if(op == 2){ // imgs[:,::-1]
			CP(x, map_sz_y - 1 - y)
		}else if(op == 3){ // imgs[::-1, ::-1]
			CP(map_sz_x - 1 - x, map_sz_y - 1 - y)
		}

		/////////// transpose
		if(trans == 1){
			CP_F(y, x)
		}else{ // direct cp
			memcpy(&imgs_r[gm_off], &imgs_r_pre[gm_off], map_sz_x*map_sz_y*n_chan*sizeof(imgs[0]));
			memcpy(&tree_probs_r[pgm_off], &tree_probs_r_pre[pgm_off], map_sz_x*map_sz_y*sizeof(imgs[0]));
		}

		// cp no move (last entry of map)
		int ind = pgm_off + (map_sz_x*map_sz_y);
		tree_probs[ind] = tree_probs_r[ind];
	}
	
	PyObject * ret = PyList_New(2);
	ASSERT(ret != 0, "err creating output list")

	ASSERT(PyList_SetItem(ret, 0, imgs_r_np) == 0, "failed setting item");
	ASSERT(PyList_SetItem(ret, 1, tree_probs_r_np) == 0, "failed setting item");

	return ret;
}

