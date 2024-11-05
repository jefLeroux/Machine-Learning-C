#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

/*
    XOR model
*/

// Define our model
typedef struct {
    float or_w1;
    float or_w2;
    float or_b;
    
    float nand_w1;
    float nand_w2;
    float nand_b;

    float and_w1;
    float and_w2;
    float and_b;
} Xor;

float rand_float(void) {
    return (float)rand() / (float)RAND_MAX; // This gives a value between 0.0 and 1.0
}

// Activation function (Sigmoid)
float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Forward functions that essentially feeds the inputs into the model
float forward(Xor m, float x1, float x2) { 
    // First layer
    float a = sigmoidf(m.or_w1*x1 + m.or_w2*x2 + m.or_b); // this is the OR gate output
    float b = sigmoidf(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_b); // this is the NAND gate output

    // Second layer
    return sigmoidf(m.and_w1*a + m.and_w2*b + m.and_b); // this is the AND gate output 
}

Xor rand_xor(void) {
    Xor m;
    m.or_w1 = rand_float();
    m.or_w2 = rand_float();
    m.or_b = rand_float();

    m.nand_w1 = rand_float();
    m.nand_w2 = rand_float();
    m.nand_b = rand_float();

    m.and_w1 = rand_float();
    m.and_w2 = rand_float();
    m.and_b = rand_float();
    return m;
}

void print_xor(Xor m) {
    printf("OR: w1: %f, w2: %f, b: %f\n", m.or_w1, m.or_w2, m.or_b);
    printf("NAND: w1: %f, w2: %f, b: %f\n", m.nand_w1, m.nand_w2, m.nand_b);
    printf("AND: w1: %f, w2: %f, b: %f\n", m.and_w1, m.and_w2, m.and_b);
}

Xor learn(Xor m, Xor g, float rate) {
    m.or_w1 -= rate * g.or_w1;
    m.or_w2 -= rate * g.or_w2;
    m.or_b -= rate * g.or_b;

    m.nand_w1 -= rate * g.nand_w1;
    m.nand_w2 -= rate * g.nand_w2;
    m.nand_b -= rate * g.nand_b;

    m.and_w1 -= rate * g.and_w1;
    m.and_w2 -= rate * g.and_w2;
    m.and_b -= rate * g.and_b;

    return m;
}

typedef float data[3];

// XOR GATE
data xor_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};

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

// NOR GATE
data nor_train[] = {
    {0, 0, 1},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 0},
};

// allows us to switch our dataset
data *train = xor_train;
size_t train_count = 4;  

float cost(Xor m) {
    // Train the model
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(m, x1, x2);
        float error = y - train[i][2]; // calculate how wrong our model is
        result += error * error;  // square the error to make mistakes more meaningful
    }
    result /= train_count;
    return result;
}

// This is obviously inefficient but to make things as explicit as possible we will do it this way
Xor finite_diff(Xor m, float eps) {
    Xor g; // separate variable for gradient descent
    float c = cost(m);
    float saved;

    saved = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (cost(m) - c) / eps;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (cost(m) - c) / eps;
    m.or_w2 = saved;

    saved = m.or_b;
    m.or_b += eps;
    g.or_b = (cost(m) - c) / eps;
    m.or_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (cost(m) - c) / eps;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (cost(m) - c) / eps;
    m.nand_w2 = saved;

    saved = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (cost(m) - c) / eps;
    m.nand_b = saved;

    saved = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (cost(m) - c) / eps;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (cost(m) - c) / eps;
    m.and_w2 = saved;

    saved = m.and_b;
    m.and_b += eps;
    g.and_b = (cost(m) - c) / eps;
    m.and_b = saved;

    return g;
}

int main(void) {
    srand(time(0));

    Xor m = rand_xor();
    Xor g;
    
    float eps = 1e-2;
    float rate = 1e-1;

    bool tracing = false;
    bool blackbox = true;

    for(size_t i = 0; i < 1000*1000; i++) {
        g = finite_diff(m, eps);
        m = learn(m, g, rate);
        tracing && printf("cost: %f\n", cost(m));
    }
    printf("cost: %f\n", cost(m));

    printf("-----------------------------\n");
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            printf("%zu ^ %zu = %f\n",i, j, forward(m, i, j));
        }
    }

    if(!blackbox) {
        printf("-----------------------------\n");
        printf("\"OR\" neuron\n");
        for(size_t i = 0; i < 2; i++) {
            for(size_t j = 0; j < 2; j++) {
                printf("%zu | %zu = %f\n", i, j, sigmoidf(m.or_w1*i + m.or_w2*j + m.or_b));
            }
        }

        printf("-----------------------------\n");
        printf("\"NAND\" neuron\n");
        for(size_t i = 0; i < 2; i++) {
            for(size_t j = 0; j < 2; j++) {
                printf("~(%zu & %zu) = %f\n", i, j, sigmoidf(m.nand_w1*i + m.nand_w2*j + m.nand_b));
            }
        }

        printf("-----------------------------\n");
        printf("\"AND\" neuron\n");
        for(size_t i = 0; i < 2; i++) {
            for(size_t j = 0; j < 2; j++) {
                printf("%zu & %zu = %f\n", i, j, sigmoidf(m.and_w1*i + m.and_w2*j + m.and_b));
            }
        }
    }

    return 0;
}