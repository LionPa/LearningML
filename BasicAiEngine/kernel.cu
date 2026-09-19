extern "C" __global__ void swiglu_forward(float* data, int out_features, int batches) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = batches * out_features;
    if (idx < total) {
        int b = idx / out_features;        // номер примера в батче
        int i = idx % out_features;        // номер нейрона
        int row_offset = b * (out_features * 2);

        float a = data[row_offset + i];                  // gate
        float b_val = data[row_offset + out_features + i]; // up
        float silu_a = a / (1.0f + __expf(-a));

        // Пишем результат в упакованном виде [batches x out_features]:
        data[b * out_features + i] = silu_a * b_val;
    }
}

extern "C" __global__ void add_bias_batched(float* data, const float* bias, int b, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = b * n;

    if (idx < total) {
        // feature_idx — это индекс нейрона (от 0 до n-1)
        int feature_idx = idx % n;
        data[idx] += bias[feature_idx];
    }
}

extern "C" __global__ void rmsnorm(const float* input, float* output, float* gamma) {
    b

}