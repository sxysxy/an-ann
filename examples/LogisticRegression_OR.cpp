#include "../an_ann.h"
#include <iostream>

static auto sigmoid = [](double x) {return 1.0 / (1.0 + exp(-x));};

int main() {
    auto X = matrix({{1.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 0.0}, {1.0, 1.0, 1.0}});
    auto Y = mvector({0.0, 1.0, 1.0, 1.0});
    auto W = mvector::zeros(3);
    auto learning_rate = 0.2;

    for(int i = 0; i < 500; i++) {
        auto predict = (X * W).as_vector().map(sigmoid);
        auto log_predict = predict;
        log_predict.map(log);
        auto loss = (Y * log_predict + (1.0 - Y) * (1-predict).map(log)).sum();
        auto grand = (X.transpose() * (Y - predict)).as_vector();
        W += learning_rate * grand;
    }
    auto predict = (X * W).as_vector().map(sigmoid);
    printf("0 or 0 = %.8lf (%.0lf)\n", predict[0], predict[0]);
    printf("0 or 1 = %.8lf (%.0lf)\n", predict[1], predict[1]);
    printf("1 or 0 = %.8lf (%.0lf)\n", predict[2], predict[2]);
    printf("1 or 1 = %.8lf (%.0lf)\n", predict[3], predict[3]);
    return 0;
}