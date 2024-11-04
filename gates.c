#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/*
    This is a very basic neural network with a single neuron.
    The neuron will have two inputs and one output.
    It will model a simple OR/AND gate
*/

// OR GATE
float train[][3] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

// AND GATE
// float train[][3] = {
//     {0, 0, 0},
//     {1, 0, 0},
//     {0, 1, 0},
//     {1, 1, 1},
// };

// NAND GATE
// float train[][3] = {
//     {0, 0, 1},
//     {1, 0, 1},
//     {0, 1, 1},
//     {1, 1, 0},
// };

// XOR GATE
// float train[][3] = {
//     {0, 0, 0},
//     {1, 0, 1},
//     {0, 1, 1},
//     {1, 1, 0},
// };

#define train_count (sizeof(train) / sizeof(train[0])) // Get the number of elements in the array

float rand_float(void) {
    return (float)rand() / (float)RAND_MAX; // Now this gives a value between 0.0 and 1.0
}

// Activation function (Sigmoid)
float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}


float cost(float w1, float w2, float b) {
    // Train the model
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoidf(x1*w1 + x2*w2 + b);
        float error = y - train[i][2]; // calculate how wrong our model is
        result += error * error;  // square the error to make mistakes more meaningful
    }
    result /= train_count;
    return result;
}

int main(void) {
    srand(13);
    
    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();

    float eps = 1e-3; // size of the step
    float rate = 1e-1; // learning rate

    for(size_t i = 0; i < 1000 * 1000; i++) {
        float c = cost(w1, w2, b);

        float dw1 = (cost(w1 + eps, w2, b) - c) / eps; 
        float dw2 = (cost(w1, w2 + eps, b) - c) / eps; 
        float db = (cost(w1, w2, b + eps) - c) / eps;

        // update the model
        w1 -= rate * dw1; 
        w2 -= rate * dw2; 
        b -= rate * db;

        // printf("w1: %f, w2: %f, b: %f, c: %f\n", w1, w2, b, cost(w1, w2, b));
    }
    // printf("w1: %f, w2: %f, b: %f, c: %f\n", w1, w2, b, cost(w1, w2, b));

    // Check how well the model preforms
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            printf("%zu | %zu = %f\n",i, j, sigmoidf(i*w1 + j*w2 + b));
        }
    }

    return 0;
}