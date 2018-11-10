// maps
REGISTER_OP("SetBoard").Input("inputs: int8");
REGISTER_OP("SetValidMvMapInternal").Input("inputs: int8");

// maps
class SetBoard : public OpKernel {
	public:
	explicit SetBoard(OpKernelConstruction* context) : OpKernel(context) {}
	void Compute(OpKernelContext* context) override {
		SET_MAP_COMPUTE_CHAR(BOARD_IDX, SET_VARS)
	}
};

class SetValidMvMapInternal : public OpKernel {
	public:
	explicit SetValidMvMapInternal(OpKernelConstruction* context) : OpKernel(context) {}
	void Compute(OpKernelContext* context) override {
		SET_MAP_COMPUTE_CHAR(VALID_MV_MAP_INTERNAL_IDX, SET_VARS)
	}
};

// maps
REGISTER_KERNEL_BUILDER(Name("SetBoard").Device(DEVICE_GPU), SetBoard);
REGISTER_KERNEL_BUILDER(Name("SetValidMvMapInternal").Device(DEVICE_GPU), SetValidMvMapInternal);

