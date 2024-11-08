#define MLIGHT_IMPLEMENTATION
#include "MLight.h"
#include <stdbool.h>

float td_xor[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0
};

float td_and[] = {
    0,0,0,
    0,1,0,
    1,0,0,
    1,1,1
};

float td_or[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,1
};

float td_nand[] = {
    0,0,1,
    0,1,1,
    1,0,1,
    1,1,0
};

float td_nor[] = {
    0,0,1,
    0,1,0,
    1,0,0,
    1,1,0
};

int main(void) {
    srand(time(0));

    size_t stride = 3;
    float  *td = td_or;
    size_t n = 4;
    Matrix tin = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td
    };

    Matrix tout = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2,
    };

    float rate = 1;
    bool tracing = true;
    bool network = false;

    // this single line represents the entire neural network
    size_t arch[] = {2, 2, 1}; // input, { hidden }, output
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    NEURALNETWORK_RANDOMIZE(nn);

    for (size_t i = 0; i < 5 *1000; i++) {
        nn_backprop(nn, g, tin, tout);
        nn_learn(nn, g, rate);
        tracing && printf("%zu: cost = %f\n", i, nn_cost(nn, tin, tout));
    }

    NEURALNETWORK_PRINT(g);

    // verify
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            MATRIX_AT(NEURALNETWORK_INPUT(nn), 0, 0) = i;
            MATRIX_AT(NEURALNETWORK_INPUT(nn), 0, 1) = j;
            nn_forward(nn);
            printf("%zu ^ %zu = %f\n",i, j, MATRIX_AT(NEURALNETWORK_OUTPUT(nn), 0, 0));
        }
    }

    // print network
    if (network) {
        NEURALNETWORK_PRINT(nn);
    }

    return 0;
}