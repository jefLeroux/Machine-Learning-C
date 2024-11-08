#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

/*
    This is a very basic neural network with a single neuron.
    The neuron will have two inputs and one output.
    It will model a simple OR/AND gate
*/

typedef float data[3];

// OR GATE
data or_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

// AND GATE
data and_train[] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 1},
};

// NAND GATE
data nand_train[] = {
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};

float rand_float(void) {
    return (float)rand() / (float)RAND_MAX; // this gives a value between 0.0 and 1.0
}

// Activation function (Sigmoid)
float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// allows us to switch our dataset
data *train = or_train;
size_t train_count = 4;  

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

float gcost(float w1, float w2, float b, float *dw1, float *dw2, float *db) {
    *dw1 = 0.0f;
    *dw2 = 0.0f;
    *db = 0.0f;

    float n = train_count;
    for(size_t i = 0; i < n; i++) {
        float xi = train[i][0];
        float yi = train[i][1];
        float zi = train[i][2];
        float ai = sigmoidf(xi*w1 + yi*w2 + b);
        float di = 2*(ai - zi)*ai*(1 - ai);
        *dw1 += di*xi;
        *dw2 += di*yi;
        *db += di;
    }

    *dw1 /= n;
    *dw2 /= n;
    *db /= n;
}

int main(void) {
    srand(13);
    
    bool tracing = true;

    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();

    float rate = 1e-1; // learning rate

    for(size_t i = 0; i < 1 * 1000; i++) {

        float dw1, dw2, db;

#if 0
        float eps = 1e-3; // size of the step
        float c = cost(w1, w2, b);

        dw1 = (cost(w1 + eps, w2, b) - c) / eps; 
        dw2 = (cost(w1, w2 + eps, b) - c) / eps; 
        db = (cost(w1, w2, b + eps) - c) / eps;
#else
    gcost(w1, w2, b, &dw1, &dw2, &db);
#endif
        // update the model
        w1 -= rate * dw1; 
        w2 -= rate * dw2; 
        b -= rate * db;

        tracing && printf("w1: %f, w2: %f, b: %f, c: %f\n", w1, w2, b, cost(w1, w2, b));
    }

    // Check how well the model preforms
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            printf("%zu | %zu = %f\n",i, j, sigmoidf(i*w1 + j*w2 + b));
        }
    }

    return 0;
}