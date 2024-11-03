# Machine learning in C

#machine_leanring #c #coding


## Basic 1 neuron network

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

With this we have essentially created a neural network with a single neuron. 