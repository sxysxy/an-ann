//requires c++17
#pragma once
#pragma warning(push)
#pragma warning(disable: 4819)
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <random>
#include <memory>
#include <functional>

constexpr auto sigmoid = [](double x) { //sigmoid函数
    if (x >= 50)return 1.0;
    else if (x <= -50)return 0.0;
    else return 1.0 / (1.0 + exp(-x));
};
constexpr auto dsigmoid = [](double x) { //sigmoid函数的导函数
    double s = sigmoid(x);
    return s * (1 - s);
};
constexpr auto id = [](double x) { //id 函数id(x) = x
    return x;
};
constexpr auto did = [](double x) { // id'(x) = 1
    return 1.0;
};
constexpr auto real_random = []() -> double { //返回一个(0, 1)之间的实数
    static std::default_random_engine e;
    static std::uniform_real_distribution<double> u(0.0, 1.0);
    return u(e);
};

class Layer {
    int p_noutput;
    double p_biases;
public:
    std::vector<double> net, out;
    std::vector<double> diff;
    std::vector<std::vector<double>> iweight, neweight;
    std::function<double(double)> activator;
    std::function<double(double)> dactivator;
    Layer() {}
    Layer(int ninput, const std::function<double(double)> &activation_function, const std::function<double(double)> &diff_activation_function, double biases = 1.0) {
        net.resize(ninput + 1);
        out.resize(ninput + 1);
        iweight.resize(ninput + 1);
        neweight.resize(ninput + 1);
        diff.resize(ninput + 1);
        activator = activation_function;
        dactivator = diff_activation_function;
        p_biases = biases;
    }
    int size() const {
        return net.size() - 1;
    }

    double& operator()(int i) {
        return net[i];
    }

    double biases() const {
        return p_biases;
    }
};
using LayerPtr = std::shared_ptr<Layer>;
class ANN {
    int p_nlayers;
    double p_learning_rate;
public:
    std::vector<LayerPtr> layers;

    ANN(double learning_rate = 0.5) {
        p_nlayers = 0;
        layers.push_back(std::make_shared<Layer>());
        p_learning_rate = learning_rate;
    }
    LayerPtr operator[](int i) {
        return layers[i];
    }

    void push_layer(LayerPtr layer) {
        layers.push_back(layer);
        p_nlayers++;
    }

    void init_state(const std::vector<std::vector<std::vector<double>>> *weights = nullptr) {
        if (p_nlayers < 2) {
            throw std::runtime_error("The number of layers should be more than 1");
        }
        for (int i = 2; i <= p_nlayers; i++) {
            auto l = layers[i];
            for (int j = 1; j <= l->size(); j++) {
                l->iweight[j].resize(layers[i - 1]->size() + 1);
                l->neweight[j].resize(layers[i - 1]->size() + 1);
                for (int k = 1; k <= layers[i - 1]->size(); k++) {
                    if (!weights)
                        l->iweight[j][k] = real_random(); //j,k，j索引当前神经元，k索引上一层神经网络中的神经元
                    else
                        l->iweight[j][k] = (*weights)[i][j][k];
                }
            }
        }
    }

    //前向传播
    std::vector<double> advance(const std::vector<double> &input_data) {
        for (int i = 1; i <= layers[1]->size(); i++) {
            layers[1]->net[i] = layers[1]->out[i] = input_data.begin()[i];
        }
        for (int i = 2; i <= p_nlayers; i++) {
            auto lcur = layers[i];
            auto llast = layers[i - 1];
            for (int j = 1; j <= lcur->size(); j++) {
                lcur->net[j] = 0.0;
                lcur->out[j] = 0.0;
                for (int k = 1; k <= llast->size(); k++) {
                    lcur->net[j] += llast->out[k] * lcur->iweight[j][k];
                }
                lcur->net[j] += llast->biases();
                lcur->out[j] = lcur->activator(lcur->net[j]);
            }
        }
        std::vector<double> res = layers[p_nlayers]->out;
        return res;
    }

    //反向传播
    void back(const std::vector<double> &ans) {
        auto last = layers[p_nlayers];
        for (int i = 1; i <= last->size(); i++) //计算总误差对输出层每个节点的偏导数
            last->diff[i] = (last->out[i] - ans[i]) * last->dactivator(last->net[i]);
        for (int i = p_nlayers - 1; i >= 1; i--) {
            auto lcur = layers[i], llast = layers[i + 1];
            for (int j = 1; j <= lcur->size(); j++) {//更新w
                double diff_layer_j = 0.0;
                for (int k = 1; k <= llast->size(); k++) {
                    double diff = llast->diff[k] * lcur->out[j];
                    llast->neweight[k][j] = llast->iweight[k][j] - p_learning_rate * diff; //计算新的权重，暂时保存到一个新的表中。
                                                                                           //(因原先的权重在之后的计算中还需要用到，这时不能覆盖掉)
                    diff_layer_j += llast->diff[k] * llast->iweight[k][j];
                }
                lcur->diff[j] = diff_layer_j * lcur->dactivator(lcur->net[j]);
            }
        }
        for (int i = 1; i <= p_nlayers; i++) {
            auto x = layers[i];
            for (int j = 1; j <= x->size(); j++) {
                x->iweight[j] = x->neweight[j]; //应用新的权重
            }
        }
    }
};
#pragma warning(pop)