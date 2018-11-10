void vars_launcher(int var_idx, void * outputs, char op);

/// return
#define MAP_COMPUTE(IDX, OP) {tensorflow::TensorShape shape;\
		shape.AddDim(BATCH_SZ);\
		shape.AddDim(MAP_SZ_X);\
		shape.AddDim(MAP_SZ_Y);\
		Tensor* tensor = nullptr;\
		OP_REQUIRES_OK(context, context->allocate_output(0, shape, &tensor));\
		auto outputs = tensor->template flat<int32>();\
		vars_launcher(IDX, outputs.data(), RETURN_VARS);}

#define MAP_COMPUTE_CHAR(IDX, OP) {tensorflow::TensorShape shape;\
		shape.AddDim(BATCH_SZ);\
		shape.AddDim(MAP_SZ_X);\
		shape.AddDim(MAP_SZ_Y);\
		Tensor* tensor = nullptr;\
		OP_REQUIRES_OK(context, context->allocate_output(0, shape, &tensor));\
		auto outputs = tensor->template flat<int8>();\
		vars_launcher(IDX, outputs.data(), RETURN_VARS);}

#define COMPUTE_BATCH_SZ_DT(IDX, DT, OP) {tensorflow::TensorShape shape;\
		shape.AddDim(BATCH_SZ);\
		Tensor* tensor = nullptr;\
		OP_REQUIRES_OK(context, context->allocate_output(0, shape, &tensor));\
		auto outputs = tensor->template flat<DT>();\
		vars_launcher(IDX, outputs.data(), RETURN_VARS);}

//// set
#define SET_MAP_COMPUTE(IDX, OP) {tensorflow::TensorShape shape;\
		shape.AddDim(BATCH_SZ);\
		shape.AddDim(MAP_SZ_X);\
		shape.AddDim(MAP_SZ_Y);\
		const Tensor& inputs_tensor = context->input(0);\
		auto inputs = inputs_tensor.flat<int32>();\
		vars_launcher(IDX, (void*)inputs.data(), SET_VARS);}

#define SET_MAP_COMPUTE_CHAR(IDX, OP) {tensorflow::TensorShape shape;\
		shape.AddDim(BATCH_SZ);\
		shape.AddDim(MAP_SZ_X);\
		shape.AddDim(MAP_SZ_Y);\
		const Tensor& inputs_tensor = context->input(0);\
		auto inputs = inputs_tensor.flat<int8>();\
		vars_launcher(IDX, (void*)inputs.data(), SET_VARS);}

#define SET_COMPUTE_BATCH_SZ_DT(IDX, DT, OP) {tensorflow::TensorShape shape;\
		shape.AddDim(BATCH_SZ);\
		const Tensor& inputs_tensor = context->input(0);\
		auto inputs = inputs_tensor.flat<DT>();\
		vars_launcher(IDX, (void*)inputs.data(), SET_VARS);}	

#include "vars_class_return.cc"
#include "vars_class_set.cc"


