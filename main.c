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
    return 1.0f + (float)rand() / (float)(RAND_MAX / 9.0f); // Now this gives a value between 1.0 and 10.0
}


float cost(float w) {
    // Train the model
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x = train[i][0];
        float y = x * w;
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

int main() {
    srand(time(0)); // for testing purposes

    float w = rand_float(); // Random value between 0 and 10

    float eps = 1e-3; // size of the step
    float rate = 1e-3; // learning rate

    printf("Cost: %f, w = %f\n", cost(w), w);
    for(int i = 0; i < 1000; i++) {
        float costd = (cost(w + eps) - cost(w)) / eps; // the slope of the cost function
        w -= (rate * costd); // update the model
        printf("Cost: %f, w = %f\n", cost(w), w);
    }
    printf("-----------------------------\n");
    printf("w: %f\n", w);
    return 0;
}