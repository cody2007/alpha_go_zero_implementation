// maps
REGISTER_OP("Board").Output("outputs: int8");
REGISTER_OP("ValidMvMapInternal").Output("outputs: int8");

// maps
class Board : public OpKernel {
	public:
	explicit Board(OpKernelConstruction* context) : OpKernel(context) {}
	void Compute(OpKernelContext* context) override {
		MAP_COMPUTE_CHAR(BOARD_IDX, RETURN_VARS)
	}
};

class ValidMvMapInternal : public OpKernel {
	public:
	explicit ValidMvMapInternal(OpKernelConstruction* context) : OpKernel(context) {}
	void Compute(OpKernelContext* context) override {
		MAP_COMPUTE_CHAR(VALID_MV_MAP_INTERNAL_IDX, RETURN_VARS)
	}
};


// maps
REGISTER_KERNEL_BUILDER(Name("Board").Device(DEVICE_GPU), Board);
REGISTER_KERNEL_BUILDER(Name("ValidMvMapInternal").Device(DEVICE_GPU), Board);

