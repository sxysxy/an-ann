# an-ann
It is just an ANN.

A very simple and basic ANN library...

# example:
![](https://github.com/sxysxy/an-ann/raw/master/snapshot/xor_code.png)
![](https://github.com/sxysxy/an-ann/raw/master/snapshot/xor_res.png)

## notice:
Please use std::vector to pass your data to the ANN, and remember the indexes start from 1(for convenience to run the ANN. for example : std::vector({0.0, 1.0, 0.5}) means a vector (1.0, 0.5) (We do not use the first dimension because it's index is zero))

## rough document
Layer:
```
Layer::Layer(int ninputs, const std::function<double(double)> &activation_function, 
            const std::function<double(double)> &diff_activation_function, double biases = 1.0);
 //ninputs : number of nodes in this layer.
 //diff_activation_function : differential function of your activation_function.
 //biases : as its name :)
```

ANN:
```
ANN::ANN(double learning_rate = 0.5);
ANN::push_layer(std::shared_ptr<Layer> layer); //add a layer into your ANN.
ANN::init_state();  //as its name. It should be called after you finish adding all layers into your ANN, 
                    //the weights in your ANN will be given random values.
ANN::advance(const std::vector<double> &input_data);  //run your ANN in the forward direction(from layer 1 to layer n), 
                    //input_data is the data put into input-layer(layer 1)
ANN::back(const std::vector<double> &ans);    //run your ANN in the backward direction(from layer n to layer 1).
                      //ans is your expecting answer for your input_data. After this operation, 
                      //the weights in your ANN will be changed.(According to Gradient Descent method)   
```

Typical usage:
```
auto layer1 = std::shared_ptr<Layer>(new Layer(...)); //fill your arguments to build a layer
...
auto layern = ...;
ANN ann(learnig_rate);
ann.push_layer(layer1);
ann.push_layer(layer2);
...
ann.push_layer(layern);
for(int i = 0; i < limit; i++) {//train it.
    ann.advance(your_input_data);
    ann.back(answer_for_input_data);
}
auto ans = ann.advance(your_input_data);
//then output ans
```