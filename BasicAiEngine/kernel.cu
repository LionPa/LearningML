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

extern "C" __global__ void rmsnorm(const float* input, float* output, float* gamma, int features, float eps, int batches) {
    int current_batch = blockIdx.x; // Почему так? Потому что мы делаем всё в одном блоке. Это можно понять по циклу

    if (current_batch >= batches) return;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = batches * features;

    const float* row_in = input + current_batch * features;
    float* row_out = output + current_batch * features;

    __shared__ float s_sum;

    if (threadIdx.x == 0) {
        s_sum = 0.0f;
    }
    __syncthreads(); // Ждем, пока поток 0 обнулит память

    // Суммируем квадраты
    float local_sum = 0;

    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float val = row_in[i];
        local_sum += val * val;
    }

    atomicAdd(&s_sum, local_sum); // Переписать на Reduction Tree, когда надо будет

    __syncthreads(); // Синкуем всё, что бы убедиться что все закончили

    float s_inv_rms = rsqrtf(s_sum / features + eps); // Каждый поток считает одно и тоже значение. Подумать над возможной оптимизацией

    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        row_out[i] = row_in[i] * s_inv_rms * gamma[i];
    }
}