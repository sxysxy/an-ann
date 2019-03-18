/* using Gradient Descent optimizing method to solve linear equations
 * Consider about Ax = b, constrcut f(x) = 1/2 (Ax-b)^T (Ax-b)
 * So if we can find the x which minimizes f(x) (nearly to zero), the 'x' is just the answer to Ax = b.
 * Here is a solver of https://syzoj.com/problem/34. 
 */

#include "../an_ann.h"

std::default_random_engine random_engine;
std::normal_distribution<double> normal_distribution(0.0, 0.1);
std::uniform_real_distribution<double> uniform_distribution(-1.0, 1.0);
auto normal_rand = std::bind(normal_distribution, random_engine);
auto uniform_rand = std::bind(uniform_distribution, random_engine);

const double NORMAL_BASE = 1e2; //To normalize all input data into [0, 1]

int main() {
    int n; 
    scanf("%d", &n);
    if(n <= 1) {       //too few train data
        int a, b; scanf("%d %d", &a, &b);
        printf("%d\n", b / a);
        return 0;
    }else if(n == 2) { //too few.
        double a, b, y1, c, d, y2; scanf("%lf %lf %lf %lf %lf %lf", &a, &b, &y1, &c, &d, &y2);
        double D = a * d - b * c;
        printf("%.0lf %.0lf\n", (y1 * d - b * y2) / D, (a * y2 - y1 * c) / D);
        return 0;
    }
    auto A = matrix(n, n);
    auto b = mvector(n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            scanf("%lf", &A(i, j));
            A(i, j) /= NORMAL_BASE;   //normalize
        }
        scanf("%lf", &b[i]);
        b[i] /= NORMAL_BASE;          //normalize
    }
    //Gradient Descent method:   
    auto At = A.transpose();
    auto parameters = mvector::zeros(n);
    auto rate = 0.08;
    for(int i = 0; i < 40000; i++) {
        auto predict = (A * parameters).as_vector();
        auto diff = predict - b;
        auto loss = (diff *diff).sum() / 2.0;
        auto gradient = (At * diff).as_vector();
        parameters -= rate * gradient;
    }
    for(auto x : parameters) 
        printf("%.0lf ", x);
    puts("");
    return 0;
}