package com.example;

public class ResNet18Partition {
    static {
        System.loadLibrary("resnet18-partition");
    }

    public native int[] partitionModel(float timeConstraint, float accuracyThreshold);

    public void runPartition() {
        float timeConstraint = 100.0f;  // Example time constraint in milliseconds
        float accuracyThreshold = 0.95f;  // Example accuracy threshold

        int[] partition = partitionModel(timeConstraint, accuracyThreshold);

        System.out.println("Partition result:");
        for (int i = 0; i < partition.length; i++) {
            System.out.println("Layer " + (i + 1) + ": " + (partition[i] == 0 ? "CPU" : "GPU"));
        }
    }

    public static void main(String[] args) {
        new ResNet18Partition().runPartition();
    }
}
