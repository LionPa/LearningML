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

extern "C" __global__ void swiglu_backward(
    const float* gradient,
    float* grad_linear,
    float* saved_linear,
    int out_features,
    int batches
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = batches * out_features;
    if (idx < total) {
        int b = idx / out_features;        // номер батча
        int i = idx % out_features;        // номер нейрона
        int row_offset = b * (out_features * 2);

        float gate = saved_linear[row_offset + i];
        float up   = saved_linear[row_offset + out_features + i];

        float sig = 1.0f / (1.0f + __expf(-gate));
        float silu_gate = gate * sig;

        float silu_deriv = sig * (1.0f + gate * (1.0f - sig));

        float grad = gradient[b * out_features + i];

        grad_linear[row_offset + i]                = grad * up * silu_deriv;
        grad_linear[row_offset + out_features + i] = grad * silu_gate;
    }
}

extern "C" __global__ void bias_backward(
    const float* grad_linear,
    float* grad_bias,
    int swiglu_features,
    int batches
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < swiglu_features) {
        float sum = 0.0f;

        for (int b = 0; b < batches; ++b) {
            sum += grad_linear[b * swiglu_features + i];
        }

        grad_bias[i] = sum;
    }
}

extern "C" __global__ void rmsnorm_backward(
    const float* grad_output,
    const float* input,
    const float* gamma,
    float* grad_input,
    float* grad_gamma,
    int features,
    float eps,
    int batches
) {
    int b = blockIdx.x;
    if (b >= batches) return;

    const float* row_dy = grad_output + b * features;
    const float* row_x  = input + b * features;
    float* row_dx       = grad_input + b * features;

    __shared__ float s_sq_sum;
    __shared__ float s_dot_sum;

    if (threadIdx.x == 0) {
        s_sq_sum = 0.0f;
        s_dot_sum = 0.0f;
    }
    __syncthreads();

    float local_sq = 0.0f;
    float local_dot = 0.0f;

    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x_val = row_x[i];
        float dy_val = row_dy[i];
        float g_val = gamma[i];

        local_sq += x_val * x_val;
        local_dot += dy_val * g_val * x_val;
    }

    atomicAdd(&s_sq_sum, local_sq);
    atomicAdd(&s_dot_sum, local_dot);

    __syncthreads();

    float rrms = rsqrtf(s_sq_sum / (float)features + eps);
    float c_factor = (s_dot_sum * rrms * rrms) / (float)features;

    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x_val = row_x[i];
        float dy_val = row_dy[i];
        float g_val = gamma[i];

        row_dx[i] = rrms * (dy_val * g_val - x_val * c_factor);

        atomicAdd(&grad_gamma[i], dy_val * x_val * rrms);
    }
}

extern "C" __global__ void mse_loss_backward(
    const float* pred,
    const float* target,
    float* grad_out,
    int total
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < total) {
        grad_out[i] = pred[i] - target[i];
    }
}

extern "C" __global__ void sgd_step(
    float* param,
    const float* grad,
    float lr,
    int total
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < total) {
        param[i] -= lr * grad[i];
    }
}