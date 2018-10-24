#include "an_ann.h"

#include <cstdio>
#include <iostream>
#include <ctime>
#include <cstdlib>

//根据一组网上已有的数据进行测试，训练f(0.05, 0.10) -> (0.01, 0.99)
void test1() {
    auto layer1 = std::shared_ptr<Layer>(new Layer(2, af::sigmoid, 0.35));
    auto layer2 = std::shared_ptr<Layer>(new Layer(2, af::sigmoid, 0.60));
    auto layer3 = std::shared_ptr<Layer>(new Layer(2, af::sigmoid));
    ANN ann(0.5);        //学习率
    ann.push_layer(layer1);
    ann.push_layer(layer2);
    ann.push_layer(layer3);
    
    auto weights = std::vector<std::vector<std::vector<double>>>({
        {},  //unused 
        {},  //weights of layer1, unused
        { {},{ 0, 0.15, 0.20 },{ 0, 0.25, 0.30 } },  //layer2
        { {},{ 0, 0.40, 0.45 },{ 0, 0.50, 0.55 } }   //layer3
        });
    ann.init_state(&weights);
    for (int i = 0; i < 50000; i++) { //训练使得f(0,05, 0.10) -> (0.01, 0.99)
        ann.advance({ 0.0, 0.05, 0.10 });
        ann.back({ 0.0, 0.01, 0.99 });
    }
    auto ans = ann.advance({ 0.0, 0.05, 0.10 });
    printf("%.4lf %.4lf\n", ans[1], ans[2]);
    system("pause");
}

//训练学习XOR 即：
// f(0, 0) = {0, 0}; f(0, 1) = f(1, 0) = {1, 0}; f(1, 1) = {0, 0}
void test2() {
    srand(time(0));
    struct data {
        float arg[2], ans[1];
    };
    static const std::initializer_list<data> train_data = {
        { { 0.0, 0.0 },{ 0.0} },
        { { 0.0, 1.0 },{ 1.0} },
        { { 1.0, 0.0 },{ 1.0} },
        { { 1.0, 1.0 },{ 0.0} }
    };

    //建立神经网络
    auto layer1 = std::shared_ptr<Layer>(new Layer(2, af::sigmoid));
    auto layer2 = std::shared_ptr<Layer>(new Layer(2, af::sigmoid));
    auto layer3 = std::shared_ptr<Layer>(new Layer(1, af::sigmoid));
    ANN ann;
    ann.push_layer(layer1);
    ann.push_layer(layer2);
    ann.push_layer(layer3);
    ann.init_state();
    for (int n = 0; n < 4; n++) {
        double a = train_data.begin()[n].arg[0], b = train_data.begin()[n].arg[1];
        for (int i = 0; i < 3000; i++) {
            ann.advance({ 0.0, a, b});
            ann.back({ 0.0, train_data.begin()[n].ans[0] });
        }
        //检查♂训练结果
        printf("%.0lf xor %.0lf = %.0lf\n", a, b, ann.advance({0.0, a, b})[1]);
    }
    system("pause");
}
int main() {
    test2();
    return 0;
}