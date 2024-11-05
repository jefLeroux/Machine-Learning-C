# XOR Gate
#machine_learning #c #coding

## Modeling a XOR Gate

To model a XOR Gate using OR, AND and NAND gates we can use the following formula:
`(x|y) & ~(x&y)` this will allow us to create a neural network with 3 neurons and 2 layers that can model a XOR gate

![](https://i.imgur.com/PGFm9DO.png)

### Setup

I'm going to speed up here a little bit because we have seen how to build the basics but there are some things that are very different this time around. Let's start by creating a struct for our model

```C
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
```

This will be how we pass our model along to all the functions that we need. Next we'll create a function that feeds our input to the model we call this a forward function.

```C
float forward(Xor m, float x1, float x2) {
    // First layer
    float a = sigmoidf(m.or_w1*x1 + m.or_w2*x2 + m.or_b); 
    float b = sigmoidf(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_b);

    // Second layer
    return sigmoidf(m.and_w1*a + m.and_w2*b + m.and_b);
}
```

To initiate our model we give each parameter a random value.

```C
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
```

Next we'll create a function that allows us to log the value of each parameter.

```C
void print_xor(Xor m) {
    printf("OR: w1: %f, w2: %f, b: %f\n", m.or_w1, m.or_w2, m.or_b);
    printf("NAND: w1: %f, w2: %f, b: %f\n", m.nand_w1, m.nand_w2, m.nand_b);
    printf("AND: w1: %f, w2: %f, b: %f\n", m.and_w1, m.and_w2, m.and_b);
}
```

To give our self more flexibility in what Gate we might want to model we'll add the following code that makes it super easy to switch gate.

```C
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
```

Our cost function also needs some minor adjustment.

```C
float cost(Xor m) {
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(m, x1, x2);
        float error = y - train[i][2]; 
        result += error * error;  
    }
    result /= train_count;
    return result;
}
```

In order to be able to train the model we need our derivation approximation function but with nine parameters this get's a little lengthy. Note that we're doing here is not at all optimal but for learning and making the code as implicit as possible it works.

```C
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
```

Finally we need a function that can train our model.

```C
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
```

Now for the moment of truth we'll throw all of this together in the main function and see if our model is any good.

```C
int main(void) {
    srand(time(0));

    Xor m = rand_xor();
    Xor g;
    float eps = 1e-2;
    float rate = 1e-1;

    bool tracing = false;

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

	return 0
}
```

Let's run it and see what we get.

```
cost: 0.000025
-----------------------------
0 ^ 0 = 0.005665
0 ^ 1 = 0.995184
1 ^ 0 = 0.995191
1 ^ 1 = 0.004768
```

We have modeled our XOR Gate

### Behind the model

At the start we came up with a formula to create a XOR Gate but does out model actually find this formula? To find out we can add some additional logs at the end of the program. 

```C
bool blackbox = false;

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
```

When we run the program again we get the following output. The model doesn't follow our formula at all but what's really cool is that it still manages to find the solution we want.

```
cost: 0.000025
-----------------------------
0 ^ 0 = 0.005675
0 ^ 1 = 0.995207
1 ^ 0 = 0.995206
1 ^ 1 = 0.004796
-----------------------------
"OR" neuron
0 | 0 = 0.000270
0 | 1 = 0.054045
1 | 0 = 0.054041
1 | 1 = 0.923668
-----------------------------
"NAND" neuron
~(0 & 0) = 0.036071
~(0 & 1) = 0.980890
~(1 & 0) = 0.980882
~(1 & 1) = 0.999986
-----------------------------
"AND" neuron
0 & 0 = 0.003724
0 & 1 = 0.998052
1 & 0 = 0.000000
1 & 1 = 0.001848
```

Because we have a random starting the point the model can even find different solution bellow you can see a part of  the output from two different times I trained the model. In the first case we see that out OR neuron actually acts as or but in the second case it acts as an AND neuron.

```
cost: 0.000300
-----------------------------
0 ^ 0 = 0.018985
0 ^ 1 = 0.983424
1 ^ 0 = 0.983411
1 ^ 1 = 0.017037
-----------------------------
"OR" neuron
0 | 0 = 0.052329
0 | 1 = 0.974248
1 | 0 = 0.973979
1 | 1 = 0.999961
```
```
cost: 0.000298
-----------------------------
0 ^ 0 = 0.018933
0 ^ 1 = 0.983487
1 ^ 0 = 0.983480
1 ^ 1 = 0.016931
-----------------------------
"OR" neuron
0 | 0 = 0.000822
0 | 1 = 0.076452
1 | 0 = 0.076341
1 | 1 = 0.892641
```

This is because we gave the model no idea of our formula so it found it's own solution based on it's starting configuration and that can end up being different.