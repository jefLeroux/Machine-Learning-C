# Perceptron
#machine_learning #c #coding

## Basic 1 neuron network with 1 input

### Setup

The first thing I'm doing to try and learn how to create my own machine learning models is understand what is happening in a very small model using the C coding language.  The first model I'm trying to make tries to predict an output number with only a single number given as input(linear regression approximation). To start this process I created some training data that follows y = 2x. 

```C
float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8}
};
```

To get the amount of data point in the training value I wrote a macro to make sure that even If i change the data the code won't crash

```C
#define train_count (sizeof(train) / sizeof(train[0])) 
```

In order to test my learning model I need a guess as to what my mystery value will be. To make sure the model arrives at the answer alone I created a random number generator that gives me a number between 1 and 10. To make sure the test gives consistent results while trying to develop the model I seeded the random number generator with a set value.

```C
float rand_float(void) {
    return 1.0f + (float)rand() / (float)(RAND_MAX / 9.0f);
}


int main() {
	srand(13);
	float w = rand_float();
}
```

### Training Function

To see how close my model comes to approximating the linear regression I need a way to measure it's accuracy against my training data set. To do so I wrote the following cost function.

```C
float cost(float w) {
    float result = 0.0f;
    for (int i = 0; i < train_count; i++) {
        float x = train[i][0];
        float y = x * w;
        float error = y - train[i][1]; 
        result += error * error;  
    }
    result /= train_count;
    return result;
}
```

This function takes in the value I generated with the rand_float() function and tells me how close it is to my training set. To do this goes over the entire training data set and compares how far of the real value it was. To make sure the program gets punished heavily for mistakes I square this error and then add up all the squares and divide them by the number of datapoints. This gives me a value that quantifies how well my model did. If the value is 0 then the model predicted the number flawlessly, is the number large then well...

`Cost: 1.336699`

I guess I shouldn't be surprised after I chose a simple random value to try and approximate the function. But now this is where the magic happens. What if we introduce a variable that can change our starting random number. 

```C
float eps = 1e-3;

 printf("Result: %f\n", cost(w + eps));
```

Surely this will make my model better right?  Well I guess not. But what if instead of increasing the value we decrease it?

```c
float eps = 1e-3;

printf("Result: %f\n", cost(w - eps));
```

Finally some progress in the right direction.

`Cost: 1.304780`

Now we just have to figure out which direction we need to move our value and then let the network train away.

### Approximating Derivatives 

There is thing in math called a derivative, this is it's definition.

![](https://i.imgur.com/P2Cmk5T.png)

The way to think about this in the context of machine learning is that the h represents our small value that is shifting our parameter to try and find the magic value. Now when we make h approach zero the derivative can tell us in which direction the function is growing (The velocity). Now we are working with computers so let's see what this looks like in code.

```C
float dcost = (cost(w + eps) - cost(w))/eps;
w -= dcost;
```

This is called the finite difference equation, this is not used in modern machine learning but to conceptually understand what is going on while learning I find it quite useful. This function is used to approximate derivatives. There is a problem however, when printing the value that this functions produces we get an enormous value. 

`dcost: -11.727332`

To address this issue we introduce a new variable we call the learning rate. This is a multiplier that allows us to control how precise our step is

```C
srand(13); 
float w = rand_float(); 
float eps = 1e-3; 
float rate = 1e-3; 

float costd = (cost(w + eps) - cost(w)) / eps; 

printf("Cost: %f\n", cost(w));
w -= (rate * costd); // update the model
printf("Cost: %f\n", cost(w));
```

I chose to set the rate to a really small value because we are looking for a relatively small value. The results of this are already quite promising.

```
Cost: 1.336699
Cost: 1.304780
```

There is another benefit to this function that might be a little less obvious but the function works in both directions. Whether our value needs to go up or down the function doesn't change.

### Training The Model

Now let's push this value down we're going to put this process in a loop and see if we can get close to zero.

```C
 printf("Cost: %f\n", cost(w));
    for(int i = 0; i < 2; i++) {
        float costd = (cost(w + eps) - cost(w)) / eps;
        w -= (rate * costd); 
        printf("Cost: %f\n", cost(w));
    }
```

This produces the following output

```
Cost: 1.336699
Cost: 1.304780
Cost: 1.273623
```

In order to really get close to zero we're going to need a "few" more iterations.

```C
int main() {
    srand(13);
    float w = rand_float() * 1000.0f;
    float eps = 1e-3; 
    float rate = 1e-3; 

    printf("Cost: %f\n", cost(w));
    for(int i = 0; i < 1000; i++) {
        float costd = (cost(w + eps) - cost(w)) / eps;
        w -= (rate * costd);
        printf("Cost: %f\n", cost(w));
    }
    return 0;
}
```

With these thousand iterations I got the output of the cost function down to the following value.

`Cost: 0.000001

But it seems like this is the limit even when doubling the amount of iterations in the loop the value will not go any lower. After adding a print statement at the end of the loop to see what the current value of our model is I get this.

`w: 1.999524`

Not a bad approximation. With this we have essentially created a neural network with a single neuron. 

### Bias

We are now going to introduce a bias. This will allow our model to shift our prediction around in case it doesn't go through (0,0). 

![](https://i.imgur.com/Ij3EQIx.png)


For our very simple model this isn't really necessary but it's an important concept in machine learning. This leads to the following changes to our code

```C
float cost(float w, float b) {
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x = train[i][0];
        float y = x * w + b;
        float error = y - train[i][1]; 
        result += error * error; 
    }
    result /= train_count;
    return result;
}

int main() {
    srand(time(0));

    float w = rand_float() * 10.0f; // Random value between 0 and 10
    float b = rand_float() * 5.0f; // Random value between 0 and 5

    float eps = 1e-3;
    float rate = 1e-3; 

	for(int i = 0; i < 1000; i++) {
        float c = cost(w, b);

        float dw = (cost(w + eps, b) - c) / eps;
        float db = (cost(w, b + eps) - c) / eps;

        // update the model
        w -= (rate * dw);
        b -= (rate * db);

        printf("Cost: %f, w = %f, b = %f\n", cost(w, b), w, b);

    }
    printf("-----------------------------\n");
    printf("w: %f, b: %f\n", w. b);
    return 0;
}
```



## Basic 1 neuron network with 2 inputs

Now we will add an additional parameter to the input  of our neural network. In the image below you can see the model we created previously and the bottom model which we will be making now.

![](https://i.imgur.com/E8QyVRd.png)

This will allow us to model things like logic gates(OR/AND).

### Setup

To Save some time I'll only explain what I did differently from our original modal [[#Basic 1 neuron network with 1 input|See here]].

```C
// OR Gate trainning data
float train[][3] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

// New Cost function
float cost(float w1, float w2) {
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = x1*w1 + x2*w2;
        float error = y - train[i][2];
        result += error * error; 
    }
    result /= train_count;
    return result;
}

// New Main fucntion
int main(void) {
    srand(13);
    float w1 = rand_float();
    float w2 = rand_float();

    float eps = 1e-3; 
    float rate = 1e-3;

    for(size_t i = 0; i < 10000; i++) {
        float c = cost(w1, w2);

        float dw1 = (cost(w1 + eps, w2) - c) / eps;
        float dw2 = (cost(w1, w2 + eps) - c) / eps;

        // update the model
        w1 -= rate * dw1;
        w2 -= rate * dw2;

        // printf("Cost: %f, w1 = %f, w2 = %f\n", cost(w1, w2), w1, w2);
    }
    printf("-----------------------------\n");
    printf("Cost: %f, w1: %f, w2: %f\n", cost(w1, w2), w1, w2);
    return 0;
}
```

We now have an additional input parameter so to account for this we adjusted our cost function to have a `x1` and a `x2` that each take 1 input. and adjusted our cost calculation accordingly. Our model now has 2 values it needs to predict so similarly in our main we create `w1` and `w2`. I chose not the print all the training steps in because it's the main thing slowing the model down. 

```
Cost: 0.083338, w1: 0.663432, w2: 0.669222
```

### Activation function

This function determines whether or not a neuron should fire, this can prevent your model from oscillating around non sensical values like ours is currently doing. A common function used for this is called the sigmoid function.

![](https://i.imgur.com/M4ncdI0.png)
This function takes any number and maps it onto the {0, 1} interval. It has the following definition.

![](https://i.imgur.com/rI7BdA6.png)
Now let's implement it into our code.
```C
float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float cost(float w1, float w2) {
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoidf(x1*w1 + x2*w2);
        float error = y - train[i][2];
        result += error * error;
    }
    result /= train_count;
    return result;
}
```
```
w1: 5.537835, w2: 5.537835, c: 0.062508
0 | 0 = 0.500000
0 | 1 = 0.996080
1 | 0 = 0.996080
1 | 1 = 0.999985
```


### Bias

This creates an interesting problem because the sigmoid function always outputs 1/2 for the value zero. To try and combat this we will give the model a bias and see how it responds.

```C
float cost(float w1, float w2, float b) {
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoidf(x1*w1 + x2*w2 + b);
        float error = y - train[i][2];
        result += error * error;
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

        w1 -= rate * dw1;
        w2 -= rate * dw2;
        b -= rate * db;
    }
    printf("w1: %f, w2: %f, b: %f, c: %f\n", w1, w2, b, cost(w1, w2, b));

    // Check how well the model preforms
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            printf("%zu | %zu = %f\n",i, j, sigmoidf(i*w1 + j*w2 + b));
        }
    }

    return 0;
}
```
```
w1: 10.265729, w2: 10.265718, b: -4.903722, c: 0.000024
0 | 0 = 0.007364
0 | 1 = 0.995330
1 | 0 = 0.995330
1 | 1 = 1.000000
```

### OR/AND/NAND

The training we have done so far was modeling an OR gate but what if we wanted to do an AND gate or a NAND gate?

```C
// AND GATE
float train[][3] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 1},
};
```
```
0 | 0 = 0.000000
0 | 1 = 0.007235
1 | 0 = 0.007235
1 | 1 = 0.991256
```

```C
// NAND GATE
float train[][3] = {
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};
```
```
0 | 0 = 1.000000
0 | 1 = 0.992738
1 | 0 = 0.992738
1 | 1 = 0.008706
```

Because the model we trained just tries to fit the data to the best of it's ability it doesn't matter if we try to train it to behave like an OR gate, an AND gate or a NAND gate they will all work.

### XOR 

You might think that if all these other logic gates work than you should be able to also create a XOR gate with a single neuron.

```C
// XOR GATE
float train[][3] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};
```
```
0 | 0 = 0.500214
0 | 1 = 0.500002
1 | 0 = 0.499905
1 | 1 = 0.499693
```

But this doesn't work. XOR is "non-linearly separable" which is to say it's impossible to draw a single straight line that separate the the the input combination that result in a zero or a one output. Single neuron network can only model **linearly separable functions** in the [next document](XOR) we will take a look at how you can model XOR.