#include <jni.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <android/log.h>

#define LOG_TAG "ResNet18Partition"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// Define ResNet18 layer structure
struct Layer {
    int id;
    float cpu_time;
    float gpu_time;
    float accuracy_loss;
};

// Sample model with 4 layers
std::vector<Layer> resnet18_layers = {
    {1, 10.0f, 2.0f, 0.01f},
    {2, 15.0f, 3.0f, 0.02f},
    {3, 12.0f, 2.5f, 0.015f},
    {4, 18.0f, 4.0f, 0.025f},

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

// Function to estimate execution time
float estimate_execution_time(const std::vector<int>& partition) {
    float total_time = 0.0f;
    for (size_t i = 0; i < partition.size(); ++i) {
        total_time += (partition[i] == 0) ? resnet18_layers[i].cpu_time : resnet18_layers[i].gpu_time;
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
extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_ResNet18Partition_partitionModel(JNIEnv *env, jobject /* this */, jfloat timeConstraint, jfloat accuracyThreshold) {
    int K = 50;  // Limit for the number of partitions to consider
    std::vector<int> partition = mlmp_partition(timeConstraint, accuracyThreshold, K);
    
    jintArray result = env->NewIntArray(partition.size());
    env->SetIntArrayRegion(result, 0, partition.size(), partition.data());
    
    return result;
}
