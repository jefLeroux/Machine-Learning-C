#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
    This is a very basic neural network with a single neuron.
    The program is meant to learn the basics of neural networks.
    Our first test will be to train a simple linear function
*/

float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};

#define train_count (sizeof(train) / sizeof(train[0])) // Get the number of elements in the array

float rand_float(void) {
    return (float)rand() / (float)RAND_MAX; // This gives a value between 0.0 and 1.0
}


float cost(float w, float b) {
    // Train the model
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x = train[i][0];
        float y = x * w + b;
        float error = y - train[i][1]; // calculate how wrong our model is
        result += error * error;  // square the error to make mistakes more meaningful
    }
    result /= train_count;
    return result;
}

/*
    The only thing we know about our model is that it rougly has the following structure:
    y = x*w
    We don't know the value of w yet so we'll just start with a random value 
*/

int main(void) {
    srand(time(0)); 

    float w = rand_float() * 10.0f; // Random value between 0 and 10
    float b = rand_float() * 5.0f; // Random value between 0 and 5

    float eps = 1e-3; // size of the step
    float rate = 1e-3; // learning rate

    for(int i = 0; i < 10000; i++) {
        float c = cost(w, b);

        float dw = (cost(w + eps, b) - c) / eps; 
        float db = (cost(w, b + eps) - c) / eps; 

        // update the model
        w -= rate * dw; 
        b -= rate * db;

        printf("Cost: %f, w = %f, b = %f\n", cost(w, b), w, b);
    }
    printf("-----------------------------\n");
    printf("w: %f, b: %f\n", w, b);
    return 0;
}