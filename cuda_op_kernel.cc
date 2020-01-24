#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "includes.h"
using namespace tensorflow;

#include "vars.cc" // return / set vars

// return coordinate from probability map, proportionate to probabiltiies
REGISTER_OP("ProbToCoord")
	.Input("prob_map: float16") // [BATCH_SZ, MAP_SZ]
	.Output("to_coord: int16");

// return coordinate from probability map, proportionate to probabiltiies, restricted to only valid mvs
REGISTER_OP("ProbToCoordValidMvs")
	.Input("prob_map: float16")
	.Output("to_coord: int16");

// return max coordinate from probability map, restricted to only valid mvs
REGISTER_OP("MaxProbToCoordValidMvs")
	.Input("prob_map: float16")
	.Output("to_coord: int16");

#define CREATE_BATCH_SHAPES tensorflow::TensorShape imgs_shape, valid_mv_map_shape;\
		imgs_shape.AddDim(BATCH_SZ);\
		imgs_shape.AddDim(MAP_SZ_X);\
		imgs_shape.AddDim(MAP_SZ_Y);\
		imgs_shape.AddDim(N_INPUT_CHANNELS);\
		\
		valid_mv_map_shape.AddDim(BATCH_SZ);\
		valid_mv_map_shape.AddDim(MAP_SZ_X);\
		valid_mv_map_shape.AddDim(MAP_SZ_Y);
	
REGISTER_OP("CreateBatch")
	.Input("moving_player: int8") // [1]
	.Output("imgs: float16")
	.Output("valid_mv_map: int8")

	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		CREATE_BATCH_SHAPES
		tensorflow::shape_inference::ShapeHandle imgs_shape_h, valid_mv_map_shape_h;
		
		c->MakeShapeFromTensorShape(imgs_shape, &imgs_shape_h);
		c->MakeShapeFromTensorShape(valid_mv_map_shape, &valid_mv_map_shape_h);

		c->set_output(0, imgs_shape_h);
		c->set_output(1, valid_mv_map_shape_h);

		return Status::OK();
});

#define RETURN_WINNER_SHAPES tensorflow::TensorShape winner_shape, score_shape, n_captures_shape;\
		winner_shape.AddDim(BATCH_SZ);\
		score_shape.AddDim(BATCH_SZ);\
		n_captures_shape.AddDim(N_PLAYERS);\
		n_captures_shape.AddDim(BATCH_SZ);

REGISTER_OP("ReturnWinner")
	.Input("moving_player: int8") // [1]
	.Output("winner: int8")
	.Output("score: int16")
	.Output("n_captures: int16")

	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		RETURN_WINNER_SHAPES
		tensorflow::shape_inference::ShapeHandle winner_shape_h, score_shape_h, n_captures_shape_h;

		c->MakeShapeFromTensorShape(winner_shape, &winner_shape_h);
		c->set_output(0, winner_shape_h);

		c->MakeShapeFromTensorShape(score_shape, &score_shape_h);
		c->set_output(1, score_shape_h);
		
		c->MakeShapeFromTensorShape(n_captures_shape, &n_captures_shape_h);
		c->set_output(2, n_captures_shape_h);

		return Status::OK();
});


REGISTER_OP("InitState");
REGISTER_OP("EndTurn");
REGISTER_OP("SessionBackup");
REGISTER_OP("SessionRestore");
REGISTER_OP("MoveRandomAi")
	.Input("moving_player: int8"); // [1]

REGISTER_OP("MoveUnit")
	.Input("to_coord: int16")
	.Input("moving_player: int8") // [1]
	.Output("moved: int8"); // [BATCH_SZ]

void prob_to_coord_launcher(float * prob_map, int16_t * to_coord);
void prob_to_coord_valid_mvs_launcher(float * prob_map, int16_t * to_coord);
void max_prob_to_coord_valid_mvs_launcher(float * prob_map, int16_t * to_coord);

void session_backup_launcher();
void session_restore_launcher();
void return_inputs_launcher(float* out);
void init_state_launcher();
void move_random_ai_launcher(int8_t * moving_player);
void create_batch_launcher(float * imgs, int8_t * moving_player, char * valid_mv_map);
void move_unit_launcher(int16_t * to_coord, int8_t * moving_player, char *moved);
void return_winner_launcher(int8_t * winner, int8_t *moving_player, int16_t * score, int16_t * n_captures_out);

class session_backup : public OpKernel {
	public:
	explicit session_backup(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		session_backup_launcher();
	}
};

class session_restore : public OpKernel {
	public:
	explicit session_restore(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		session_restore_launcher();
	}
};

class prob_to_coord : public OpKernel {
	public:
	explicit prob_to_coord(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		/////////////////////////////////// inputs
    		const Tensor& prob_map_tensor = context->input(0);

		auto prob_map = prob_map_tensor.flat<Eigen::half>();

		// check dims
		TensorShape prob_map_shape = prob_map_tensor.shape();
		ASSERT(prob_map_shape.dims() == 2, "number of dims not correct")
		ASSERT(prob_map_shape.dim_size(0) == BATCH_SZ, "incorrect input size")
		ASSERT(prob_map_shape.dim_size(1) == MAP_SZ, "incorrect input size")

		////////////////////////////////////// outputs
		Tensor* to_coord_tensor = nullptr;

		TensorShape to_coord_shape;
		to_coord_shape.AddDim(BATCH_SZ);

		OP_REQUIRES_OK(context, context->allocate_output(0, to_coord_shape, &to_coord_tensor));

		auto to_coord = to_coord_tensor->template flat<int16>();

		///////////////////
		prob_to_coord_launcher((float*)prob_map.data(), (int16_t*)to_coord.data());
	}
};

class prob_to_coord_valid_mvs : public OpKernel {
	public:
	explicit prob_to_coord_valid_mvs(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		/////////////////////////////////// inputs
    		const Tensor& prob_map_tensor = context->input(0);
		auto prob_map = prob_map_tensor.flat<Eigen::half>();

		// check dims
		TensorShape prob_map_shape = prob_map_tensor.shape();
		ASSERT(prob_map_shape.dims() == 2, "number of dims not correct")
		ASSERT(prob_map_shape.dim_size(0) == BATCH_SZ, "incorrect input size")
		ASSERT(prob_map_shape.dim_size(1) == MAP_SZ, "incorrect input size")

		////////////////////////////////////// outputs
		Tensor* to_coord_tensor = nullptr;

		TensorShape to_coord_shape;
		to_coord_shape.AddDim(BATCH_SZ);

		OP_REQUIRES_OK(context, context->allocate_output(0, to_coord_shape, &to_coord_tensor));

		auto to_coord = to_coord_tensor->template flat<int16_t>();

		///////////////////
		prob_to_coord_valid_mvs_launcher((float*)prob_map.data(), (int16_t*)to_coord.data());
	}
};

class max_prob_to_coord_valid_mvs : public OpKernel {
	public:
	explicit max_prob_to_coord_valid_mvs(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		/////////////////////////////////// inputs
    		const Tensor& prob_map_tensor = context->input(0);
		auto prob_map = prob_map_tensor.flat<Eigen::half>();

		// check dims
		TensorShape prob_map_shape = prob_map_tensor.shape();
		ASSERT(prob_map_shape.dims() == 2, "number of dims not correct")
		ASSERT(prob_map_shape.dim_size(0) == BATCH_SZ, "incorrect input size")
		ASSERT(prob_map_shape.dim_size(1) == MAP_SZ, "incorrect input size")

		////////////////////////////////////// outputs
		Tensor* to_coord_tensor = nullptr;

		TensorShape to_coord_shape;
		to_coord_shape.AddDim(BATCH_SZ);

		OP_REQUIRES_OK(context, context->allocate_output(0, to_coord_shape, &to_coord_tensor));

		auto to_coord = to_coord_tensor->template flat<int16_t>();

		///////////////////
		max_prob_to_coord_valid_mvs_launcher((float*)prob_map.data(), (int16_t*)to_coord.data());
	}
};

class return_winner : public OpKernel {
	public:
	explicit return_winner(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		/////////////////////////////////// inputs
    		const Tensor& moving_player_tensor = context->input(0);
		auto moving_player = moving_player_tensor.flat<int8>();

		// check dims
		TensorShape moving_player_shape = moving_player_tensor.shape();
		ASSERT(moving_player_shape.dims() == 0, "number of dims not correct")

		////////////////////////////////////// outputs
		Tensor* winner_tensor = nullptr, * score_tensor = nullptr, * n_captures_tensor = nullptr;
		RETURN_WINNER_SHAPES

		OP_REQUIRES_OK(context, context->allocate_output(0, winner_shape, &winner_tensor));
		OP_REQUIRES_OK(context, context->allocate_output(1, score_shape, &score_tensor));
		OP_REQUIRES_OK(context, context->allocate_output(2, n_captures_shape, &n_captures_tensor));

		auto winner = winner_tensor->template flat<int8_t>();
		auto score = score_tensor->template flat<int16_t>();
		auto n_captures = n_captures_tensor->template flat<int16>();

		///////////////////
		return_winner_launcher((int8_t*)winner.data(), (int8_t*)moving_player.data(), (int16_t*)score.data(), (int16_t*)n_captures.data());
	}
};

class move_unit : public OpKernel {
	public:
	explicit move_unit(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		///////////////////////////////////// inputs
		const Tensor& to_coord_tensor = context->input(0);
    		const Tensor& moving_player_tensor = context->input(1);

		auto to_coord = to_coord_tensor.flat<int16>();
		auto moving_player = moving_player_tensor.flat<int8>();

		// check dims
		TensorShape to_map_shape = to_coord_tensor.shape();
		ASSERT(to_map_shape.dims() == 1, "number of dims not correct")
		ASSERT(to_map_shape.dim_size(0) == BATCH_SZ, "incorrect input size")

		TensorShape moving_player_shape = moving_player_tensor.shape();
		ASSERT(moving_player_shape.dims() == 0, "number of dims not correct")

		///////////////////////// outputs
		Tensor* moved_tensor = nullptr;

		TensorShape moved_shape;
		moved_shape.AddDim(BATCH_SZ);

		OP_REQUIRES_OK(context, context->allocate_output(0, moved_shape, &moved_tensor));

		auto moved = moved_tensor->template flat<int8>();

		///////////////////
		move_unit_launcher((int16_t*)to_coord.data(), (int8_t*)moving_player.data(), (char*)moved.data());
	}
};

class create_batch : public OpKernel {
	public:
	explicit create_batch(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		///////////////////////////////////// inputs
		const Tensor& moving_player_tensor = context->input(0);
		auto moving_player = moving_player_tensor.flat<int8>();
	
		// check dims
		TensorShape moving_player_shape = moving_player_tensor.shape();
		ASSERT(moving_player_shape.dims() == 0, "number of dims not correct")


		////////////////////////////////////// outputs
		CREATE_BATCH_SHAPES
		Tensor* imgs_tensor = nullptr, *valid_mv_map_tensor = nullptr;
		
		OP_REQUIRES_OK(context, context->allocate_output(0, imgs_shape, &imgs_tensor));
		OP_REQUIRES_OK(context, context->allocate_output(1, valid_mv_map_shape, &valid_mv_map_tensor));

		auto imgs = imgs_tensor->template flat<Eigen::half>();
		auto valid_mv_map = valid_mv_map_tensor->template flat<int8>();

		///////////////////
		create_batch_launcher((float*)imgs.data(), (int8_t*)moving_player.data(), 
			(char *)valid_mv_map.data());
	}
};

class init_state : public OpKernel {
	public:
	explicit init_state(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		init_state_launcher();
	}
};

class move_random_ai : public OpKernel {
	public:
	explicit move_random_ai(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		///////////////////////////////////// inputs
		const Tensor& moving_player_tensor = context->input(0);

    		auto moving_player = moving_player_tensor.flat<int8>();

		// check dims
		TensorShape moving_player_shape = moving_player_tensor.shape();
		ASSERT(moving_player_shape.dims() == 0, "number of dims not correct")

		move_random_ai_launcher((int8_t *)moving_player.data());
	}
};

REGISTER_KERNEL_BUILDER(Name("InitState").Device(DEVICE_GPU), init_state);
REGISTER_KERNEL_BUILDER(Name("MoveRandomAi").Device(DEVICE_GPU), move_random_ai);
REGISTER_KERNEL_BUILDER(Name("CreateBatch").Device(DEVICE_GPU), create_batch);
REGISTER_KERNEL_BUILDER(Name("MoveUnit").Device(DEVICE_GPU), move_unit);
REGISTER_KERNEL_BUILDER(Name("SessionRestore").Device(DEVICE_GPU), session_restore);
REGISTER_KERNEL_BUILDER(Name("SessionBackup").Device(DEVICE_GPU), session_backup);
REGISTER_KERNEL_BUILDER(Name("ReturnWinner").Device(DEVICE_GPU), return_winner);
REGISTER_KERNEL_BUILDER(Name("ProbToCoord").Device(DEVICE_GPU), prob_to_coord);
REGISTER_KERNEL_BUILDER(Name("ProbToCoordValidMvs").Device(DEVICE_GPU), prob_to_coord_valid_mvs);
REGISTER_KERNEL_BUILDER(Name("MaxProbToCoordValidMvs").Device(DEVICE_GPU), max_prob_to_coord_valid_mvs);

