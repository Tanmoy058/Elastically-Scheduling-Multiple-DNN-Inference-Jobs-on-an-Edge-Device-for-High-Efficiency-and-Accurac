package com.example;

import android.os.Bundle;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TextView resultView = findViewById(R.id.resultView);
        
        ResNet18Partition partitioner = new ResNet18Partition();
        float timeConstraint = 100.0f;  // Example time constraint in milliseconds
        float accuracyThreshold = 0.95f;  // Example accuracy threshold

        int[] partition = partitioner.partitionModel(timeConstraint, accuracyThreshold);

        StringBuilder result = new StringBuilder("Partition result:\n");
        for (int i = 0; i < partition.length; i++) {
            result.append("Layer ").append(i + 1).append(": ")
                  .append(partition[i] == 0 ? "CPU" : "GPU").append("\n");
        }

        resultView.setText(result.toString());
    }
}
