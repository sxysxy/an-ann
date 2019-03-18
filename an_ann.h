/*   an_ann.h,Header file of an_ann.
 *   This file provides basical mathematical extensions including vector, matrix,
 *     and nerual networks support. (BPNN)
 *                                 license: THE MIT LICENSE
 *                                 author:  HfCloud(https://github.com/sxysxy)
 *                                 date:    created on 2019-03-05, version 0.2
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <random>

/* class mvector, an extension for std::vector<double>
 * mvector looks like a column vector(mathmetically)
 */
class mvector : public std::vector<double> {
public:
    mvector() : std::vector<double>::vector() {}
    mvector(size_t s) : std::vector<double>::vector(s) {resize(s);}
    mvector(const mvector &v) : std::vector<double>::vector(v) {}
    mvector(mvector &&v) : std::vector<double>::vector(std::move(v)) {}
    mvector(const std::initializer_list<double> &list) : std::vector<double>::vector(list) {}

    //foreach element a_i, let a_i = f(a_i, v)
    template<class T, class Fn>
    void each_op_with(const Fn &f, const T &v) {
        for(auto &a : *this) {
            a = f(a, v);
        }
    }

    //foreach i and element a_i, let a_i = f(a_i, b_i)
    template<class Fn> 
    void each_op_with(const Fn &f, const mvector &v) {
        if(size() != v.size()) {
            throw std::invalid_argument("can not operate the two vector!");
        }
        for(size_t i = 0; i < size(); i++) {
            (*this)[i] = f((*this)[i], v[i]);
        }
    }

    //get the sum of all elements.
    double sum() const {
        double v = 0.0;
        for(auto a : *this) v += a;
        return v;
    }

    //returns a s-dimension vector, all elements set to be zero.
    static mvector zeros(size_t s) {
        mvector v(s);
        std::fill(v.begin(), v.end(), 0.0);
        return v;
    }

    //build a mvector from a container.
    template<class Iter>
    static mvector build(Iter _begin, Iter _end) {
        mvector v(_end - _begin);
        std::copy(_begin, _end, v.begin());
        return v;
    }
    template<class T>
    static mvector build(const T &list) {
        return mvector::build(list.begin(), list.end());
    }
    template<class T>
    static mvector build(const std::initializer_list<T> &list) {
        return mvector::build(list.begin(), list.end());
    }

    //add operation
    template<class T>
    mvector &operator+=(const T &v) {
        each_op_with([](double x, double y) {return x + y;}, v);
        return *this;
    }
    template<class T>
    mvector operator+(const T &v) {
        mvector r(*this);
        return r += v;
    }

    //sub
    template<class T>
    mvector &operator-=(const T &v) {
        each_op_with([](double x, double y) {return x - y;}, v);
        return *this;
    }
    template<class T>
    mvector operator-(const T &v) {
        mvector r(*this);
        return r -= v;
    }
   
    //mul
    template<class T>
    mvector &operator*=(const T &v) {
        each_op_with([](double x, double y) {return x * y;}, v);
        return *this;
    }
    template<class T>
    mvector operator*(const T &v) {
        mvector r(*this);
        return r *= v;
    }

    //div
    template<class T>
    mvector &operator/=(const T &v) {
        each_op_with([](double x, double y) {return x / y;}, v);
        return *this;
    }
    template<class T>
    mvector operator/(const T &v) {
        mvector r(*this);
        return r /= v;
    }
    
    //map function, let a_i = f(a_i)
    template<class Fn>
    mvector &map(const Fn &f) {
        each_op_with([&](double x, double y) {return f(x);}, 0);
        return *this;
    }
    // let a_i = (a_i)^2
    mvector &map_square() {
        return map([](double x) {return x * x;});
    }

    //copy...
    mvector &operator=(const mvector &v) {
        std::vector<double>::operator=(v);
    }
};

//mvector into out stream
std::ostream &operator<<(std::ostream &os, const mvector &v) {
    os << "[ ";
    for(size_t i = 0; i < v.size() - 1; i ++) 
        os << v[i] << ", ";
    os << v[v.size()-1] << "]";
    return os;
}

// * operator, when the number is on the left of the operator.
mvector operator*(double a, const mvector &v) {
    mvector r(v);
    return r.map([&](double x){return a * x;}); 
}
mvector operator/(double a, const mvector &v) {
    mvector r(v);
    return r.map([&](double x){return 1/a * x;}); 
}
mvector operator+(double a, const mvector &v) {
    mvector r(v);
    return r.map([&](double x){return a + x;}); 
}
mvector operator-(double a, const mvector &v) {
    mvector r(v);
    return r.map([&](double x){return a - x;});
}

class matrix {
    int _row_size, _col_size;
public:
    std::vector<mvector> data;
    matrix(int m, int n) {
        _row_size = m, _col_size = n;
        data.resize(m);
        for(auto & r : data) {
            r.resize(n);
        }
    }
    matrix(const matrix &m) {
        _row_size = m.row_size();
        _col_size = m.col_size();
        data = m.data;
    }
    matrix(matrix &&m) {
        _row_size = m.row_size();
        _col_size = m.col_size();
        data = std::move(m.data);
    }
    
    double &operator()(int i, int j) {
        return data[i][j];
    }
    double operator()(int i, int j) const {
        return data[i][j];
    }

    int row_size() const{
        return _row_size;
    }
    int col_size() const {
        return _col_size;
    }

    matrix &operator*=(double x) {
        for(auto &row : data) {
            for(auto &element : row) {
                element *= x;
            }
        }
        return *this;
    }
    matrix operator*(double x) {
        matrix r(*this);
        return r *= x;
    }

    //matrix multiply
    matrix operator*(const matrix &m) {
        if(col_size() != m.row_size()) {
            throw std::invalid_argument("Can not multiply the two matrix");
        }
        matrix r(row_size(), m.col_size());
        for(int i = 0; i < row_size(); i++) {
            for(int j = 0; j < m.col_size(); j++) {
                r(i, j) = 0.0;
                for(int k = 0; k < col_size(); k++) {
                    r(i, j) += (*this)(i, k) * m(k, j);
                }
            }
        }
        return r;
    }

    //assume the vector is a column vector.
    matrix operator*(const mvector &v) {
        if(col_size() != v.size()) {
            throw std::invalid_argument("Can not multiply the matrix and the vector");
        }
        matrix r(row_size(), 1);
        for(int i = 0; i < row_size(); i++) {
            for(int j = 0; j < 1; j++) {
                r(i, j) = 0.0;
                for(int k = 0; k < col_size(); k++) {
                    r(i, j) += (*this)(i, k) * v[k];
                }
            }
        }
        return r;
    }

    matrix transpose() {
        matrix r(col_size(), row_size());
        for(int i = 0; i < col_size(); i++)for(int j = 0; j < row_size(); j++) {
            r(i, j) = (*this)(j, i);
        }
        return r;
    }
    
    //get the i-th row, returns a mvector
    mvector row(int i) {
        mvector r(data[i]);
        return r;
    }
    //get the i-th column, returns a mvector
    mvector col(int i) {
        mvector r(row_size());
        for(int j = 0; j < row_size(); j++) {
            r[j] = (*this)(j, i);
        }
        return r;
    }
    //returns a submatrix from i-th row, j-th column, w elements each row, h elements each column. 
    matrix submatrix(int i, int j, int w, int h) {
        if(i + w >= _col_size || j + h >= _row_size || i < 0 || j < 0)
            throw std::invalid_argument("Can not get such a submatrix");
        matrix r(w, h);
        for(int p = i; p < i+w; p++) for(int q = j; q < j+h; q++) 
            r(p-i, q-j) = (*this)(p, q);
        return r;
    }

    //try to convert self into a mvector if possible
    mvector as_vector() {
        if(row_size() == 1) {
            return row(0);
        }else if(col_size() == 1) {
            return col(0);
        }else {
            throw std::invalid_argument("Can not convert to a vector");   
        }
    }

    //construct from 2-d initializer_list
    matrix(const std::initializer_list<std::initializer_list<double>> &d) : matrix(d.size(), (*d.begin()).size()) {
        auto beg = d.begin();
        for(auto & r : data) {
            std::copy(beg->begin(), beg->end(), r.begin());
            beg++;
        }
    }

};

/*
//Nerual network part:
struct ActivationFunction { 
    using func_type = std::function<mvector(mvector)>;
    func_type func, diff_func;
    constexpr ActivationFunction(const func_type &_func, const func_type &_diff_func):func(_func), diff_func(_diff_func) noexcept const{}
};
constexpr static const auto sigmoid = ActivationFunction([](mvector &v) -> mvector{
        return v.map([](double x) {return 1.0 / (1.0 + exp(-x));});
    }, 
    [](mvector &v) -> mvector{
        return v.map([](double x) {
            double s = 1.0 / (1.0 + exp(-x));
            return s * (1-s);});
}); 
*/

class Layer {
    matrix weight;
    
};

class BPNN {

};