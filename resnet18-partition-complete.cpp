#include <jni.h>
#include <android/log.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <CL/cl.h>
#include <string>

#define LOG_TAG "ResNet18Partition"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Define ResNet18 layer structure
struct Layer {
    int id;
    float cpu_time;
    float gpu_time;
    float accuracy_loss;
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;
};

// ResNet18 layers (simplified, you should add all 18 layers with real data)
std::vector<Layer> resnet18_layers = {
    {1, 10.0f, 2.0f, 0.01f, 3, 64, 7, 2, 3},
    {2, 15.0f, 3.0f, 0.02f, 64, 64, 3, 1, 1},
    {3, 12.0f, 2.5f, 0.015f, 64, 64, 3, 1, 1},
    {4, 18.0f, 4.0f, 0.025f, 64, 128, 3, 2, 1},
    // ... Add all 18 layers of ResNet18
};

// Structure to represent a partition
struct Partition {
    std::vector<int> config;
    float time;
    float accuracy;

    Partition(const std::vector<int>& c, float t, float a) : config(c), time(t), accuracy(a) {}
};

// Compare partitions based on accuracy (for max heap)
struct ComparePartition {
    bool operator()(const Partition& p1, const Partition& p2) {
        return p1.accuracy < p2.accuracy;
    }
};

class OpenCLDataTransfer {
private:
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem inputBuffer;
    cl_mem outputBuffer;

    bool initOpenCL() {
        cl_int err;
        cl_uint numPlatforms;
        cl_platform_id platform = NULL;

        err = clGetPlatformIDs(1, &platform, &numPlatforms);
        if (err != CL_SUCCESS) {
            LOGE("Failed to get platform ID: %d", err);
            return false;
        }

        cl_device_id device = NULL;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            LOGE("Failed to get device ID: %d", err);
            return false;
        }

        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        if (err != CL_SUCCESS) {
            LOGE("Failed to create context: %d", err);
            return false;
        }

        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) {
            LOGE("Failed to create command queue: %d", err);
            return false;
        }

        return true;
    }

    bool createKernel() {
        const char* kernelSource = R"(
            __kernel void conv2d(__global const float* input, __global float* output,
                                 __global const float* weight, __global const float* bias,
                                 int inChannels, int outChannels, int kernelSize,
                                 int inputWidth, int inputHeight, int stride, int padding) {
                int outX = get_global_id(0);
                int outY = get_global_id(1);
                int outZ = get_global_id(2);
                
                int outWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;
                int outHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
                
                if (outX < outWidth && outY < outHeight && outZ < outChannels) {
                    float sum = bias[outZ];
                    for (int inC = 0; inC < inChannels; inC++) {
                        for (int kY = 0; kY < kernelSize; kY++) {
                            for (int kX = 0; kX < kernelSize; kX++) {
                                int inX = outX * stride + kX - padding;
                                int inY = outY * stride + kY - padding;
                                if (inX >= 0 && inX < inputWidth && inY >= 0 && inY < inputHeight) {
                                    int inIdx = (inC * inputHeight + inY) * inputWidth + inX;
                                    int wIdx = ((outZ * inChannels + inC) * kernelSize + kY) * kernelSize + kX;
                                    sum += input[inIdx] * weight[wIdx];
                                }
                            }
                        }
                    }
                    int outIdx = (outZ * outHeight + outY) * outWidth + outX;
                    output[outIdx] = sum;
                }
            }
        )";

        cl_int err;
        program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
        if (err != CL_SUCCESS) {
            LOGE("Failed to create program: %d", err);
            return false;
        }

        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            LOGE("Failed to build program: %d", err);
            return false;
        }

        kernel = clCreateKernel(program, "conv2d", &err);
        if (err != CL_SUCCESS) {
            LOGE("Failed to create kernel: %d", err);
            return false;
        }

        return true;
    }

public:
    OpenCLDataTransfer() : context(NULL), queue(NULL), program(NULL), kernel(NULL), 
                           inputBuffer(NULL), outputBuffer(NULL) {}

    ~OpenCLDataTransfer() {
        if (inputBuffer) clReleaseMemObject(inputBuffer);
        if (outputBuffer) clReleaseMemObject(outputBuffer);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    bool initialize() {
        if (!initOpenCL()) return false;
        if (!createKernel()) return false;
        return true;
    }

    bool transferAndCompute(const std::vector<float>& input, std::vector<float>& output,
                            const std::vector<float>& weight, const std::vector<float>& bias,
                            int inChannels, int outChannels, int kernelSize,
                            int inputWidth, int inputHeight, int stride, int padding) {
        cl_int err;
        size_t inputSize = input.size() * sizeof(float);
        size_t outputSize = output.size() * sizeof(float);
        size_t weightSize = weight.size() * sizeof(float);
        size_t biasSize = bias.size() * sizeof(float);

        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                            inputSize, (void*)input.data(), &err);
        cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outputSize, NULL, &err);
        cl_mem weightBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                             weightSize, (void*)weight.data(), &err);
        cl_mem biasBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                           biasSize, (void*)bias.data(), &err);

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &weightBuffer);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &biasBuffer);
        err |= clSetKernelArg(kernel, 4, sizeof(int), &inChannels);
        err |= clSetKernelArg(kernel, 5, sizeof(int), &outChannels);
        err |= clSetKernelArg(kernel, 6, sizeof(int), &kernelSize);
        err |= clSetKernelArg(kernel, 7, sizeof(int), &inputWidth);
        err |= clSetKernelArg(kernel, 8, sizeof(int), &inputHeight);
        err |= clSetKernelArg(kernel, 9, sizeof(int), &stride);
        err |= clSetKernelArg(kernel, 10, sizeof(int), &padding);

        if (err != CL_SUCCESS) {
            LOGE("Failed to set kernel arguments: %d", err);
            return false;
        }

        int outWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;
        int outHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
        size_t globalSize[3] = {(size_t)outWidth, (size_t)outHeight, (size_t)outChannels};
        
        err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            LOGE("Failed to enqueue kernel: %d", err);
            return false;
        }

        err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, outputSize, 
                                  output.data(), 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            LOGE("Failed to read output buffer: %d", err);
            return false;
        }

        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(weightBuffer);
        clReleaseMemObject(biasBuffer);

        return true;
    }
};

OpenCLDataTransfer openclTransfer;

// Function to estimate execution time (now including data transfer and computation)
float estimate_execution_time(const std::vector<int>& partition) {
    float total_time = 0.0f;
    bool on_gpu = false;
    std::vector<float> data(224*224*3, 1.0f);  // Simulated input data
    std::vector<float> result(224*224*64);  // Simulated output data
    std::vector<float> weight(64*3*7*7, 1.0f);  // Simulated weights
    std::vector<float> bias(64, 0.0f);  // Simulated bias

    for (size_t i = 0; i < partition.size(); ++i) {
        if (partition[i] == 0) {  // CPU
            if (on_gpu) {
                // Transfer data back to CPU
                total_time += 1.0f;  // Estimated transfer time, adjust as needed
                on_gpu = false;
            }
            total_time += resnet18_layers[i].cpu_time;
        } else {  // GPU
            if (!on_gpu) {
                // Transfer data to GPU
                total_time += 1.0f;  // Estimated transfer time, adjust as needed
                on_gpu = true;
            }
            // Simulate OpenCL computation
            openclTransfer.transferAndCompute(data, result, weight, bias,
                                              resnet18_layers[i].input_channels,
                                              resnet18_layers[i].output_channels,
                                              resnet18_layers[i].kernel_size,
                                              224, 224,  // Assuming input size is 224x224
                                              resnet18_layers[i].stride,
                                              resnet18_layers[i].padding);
            total_time += resnet18_layers[i].gpu_time;
        }
    }

    if (on_gpu) {
        // Transfer final result back to CPU
        total_time += 1.0f;  // Estimated transfer time, adjust as needed
    }

    return total_time;
}

// Function to estimate accuracy
float estimate_accuracy(const std::vector<int>& partition) {
    float total_loss = 0.0f;
    for (size_t i = 0; i < partition.size(); ++i) {
        if (partition[i] == 1) {  // If layer is on GPU
            total_loss += resnet18_layers[i].accuracy_loss;
        }
    }
    return 1.0f - total_loss;
}

// MLMP algorithm implementation
std::vector<int> mlmp_partition(float time_constraint, float accuracy_threshold, int K) {
    std::priority_queue<Partition, std::vector<Partition>, ComparePartition> pq;
    
    // Start with all layers on CPU
    std::vector<int> initial_config(resnet18_layers.size(), 0);
    float initial_time = estimate_execution_time(initial_config);
    float initial_accuracy = estimate_accuracy(initial_config);
    
    pq.push(Partition(initial_config, initial_time, initial_accuracy));
    
    std::vector<int> best_config = initial_config;
    float best_time = initial_time;
    float best_accuracy = initial_accuracy;
    
    while (!pq.empty()) {
        Partition current = pq.top();
        pq.pop();
        
        if (current.time <= time_constraint && current.accuracy > best_accuracy) {
            best_config = current.config;
            best_time = current.time;
            best_accuracy = current.accuracy;
        }
        
        for (size_t i = 0; i < current.config.size(); ++i) {
            if (current.config[i] == 0) {  // If layer is on CPU, try moving to GPU
                std::vector<int> new_config = current.config;
                new_config[i] = 1;
                
                float new_time = estimate_execution_time(new_config);
                float new_accuracy = estimate_accuracy(new_config);
                
                if (new_time <= time_constraint && new_accuracy >= accuracy_threshold) {
                    pq.push(Partition(new_config, new_time, new_accuracy));
                    
                    if (pq.size() > K) {
                        pq.pop();
                    }
                }
            }
        }
    }
    
    return best_config;
}

// JNI function
extern "C" JNIEXPORT j