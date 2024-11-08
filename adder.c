#define MLIGHT_IMPLEMENTATION
#include "MLight.h"
#include <stdbool.h>

#define BITS 4
int main(void)
{
    srand(time(0));

    size_t n = (1<<BITS);
    size_t rows = n*n;
    Matrix ti = matrix_allocate(rows, 2*BITS);
    Matrix to = matrix_allocate(rows, BITS + 1);
    for (size_t i = 0; i < ti.rows; ++i) {
        size_t x = i/n;
        size_t y = i%n;
        size_t z = x + y;
        for (size_t j = 0; j < BITS; ++j) {
            MATRIX_AT(ti, i, j)        = (x>>j)&1;
            MATRIX_AT(ti, i, j + BITS) = (y>>j)&1;
            MATRIX_AT(to, i, j)        = (z>>j)&1;
        }
        MATRIX_AT(to, i, BITS) = z >= n;
    }

    size_t arch[] = {2*BITS, 4*BITS, BITS + 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    NEURALNETWORK_RANDOMIZE(nn);
    
    float rate = 1;
    bool tracing = true;
    bool network = false;

    for (size_t i = 0; i < 10*1000; ++i) {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
        tracing && printf("%zu: c = %f\n", i, nn_cost(nn, ti, to));
    }
    
    size_t fails = 0;
    for (size_t x = 0; x < n; ++x) {
        for (size_t y = 0; y < n; ++y) {
            size_t z = x + y;
            for (size_t j = 0; j < BITS; ++j) {
                MATRIX_AT(NEURALNETWORK_INPUT(nn), 0, j)        = (x>>j)&1;
                MATRIX_AT(NEURALNETWORK_INPUT(nn), 0, j + BITS) = (y>>j)&1;
            }
            nn_forward(nn);
            if (MATRIX_AT(NEURALNETWORK_OUTPUT(nn), 0, BITS) > 0.5f) {
                if (z < n) {
                    printf("%zu + %zu = (OVERFLOW<>%zu)\n", x, y, z);
                    fails += 1;
                }
            } else {
                size_t a = 0;
                for (size_t j = 0; j < BITS; ++j) {
                    size_t bit = MATRIX_AT(NEURALNETWORK_OUTPUT(nn), 0, j) > 0.5f;
                    a |= bit<<j;
                }
                if (z != a) {
                    printf("%zu + %zu = (%zu<>%zu)\n", x, y, z, a);
                    fails += 1;
                }
            }
        }
    }
    if (fails == 0) printf("OK\n");
    return 0;
}