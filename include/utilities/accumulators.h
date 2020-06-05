#ifndef ACCUMULATORS_H
#define ACCUMULATORS_H

#include <set>
#include <deque>
#include <cstddef>
#include <array>
#include "data/records.h"
using namespace std;

template <typename T>
class Accumulator
{
    protected:
        const size_t window_size;

        std::deque<T> window;

        T _sum = T(0);

    public:
        Accumulator(size_t window_size);

        virtual void push(T val);

        T sum();

        T at(size_t i);
        T front();
        T back();

        void clear();

        bool full();
        size_t size();
};

template <typename T>
class RollingMean: public Accumulator<T>
{
    private:
        T _mean;
        T _s;

    public:
        RollingMean(size_t window_size);

        void push(T val);

        T mean();
        T var();
        T std();

        T zscore(T val);
        T last_zscore();
};

template <typename T>
class EWMA: public Accumulator<T>
{
    private:
        double _alpha;
        T _mean;

    public:
        EWMA(size_t window_size);

        void push(T val);

        T mean();
};

template <typename T>
class RollingMedian: public Accumulator<T>
{
    private:
        std::multiset<T> min_heap;
        std::multiset<T> max_heap;

    public:
        RollingMedian(size_t window_size);

        void push(T val);

        T median();

        T quartile(double quartile);
        T iqr();

        T min();
        T max();

        T zscore(T val);
        T last_zscore();
};

class DataBuffer{
public:
    typedef array<int,5> arr;
    DataBuffer()=default;

    //是否初始化完毕
    bool isInited();

    //重置
    void clear();

    //打印curr数据
    void printInfo();

    //基于快照record的数据更替
    void push(const data::MarketDepthRecord&);

    //挂单倾斜程度变化率
    int abv_diff();
    double abv_diff_change();

    //挂单量的倾斜程度
    double abv_diff_ratio();

    //一档挂单量变化
    int av_consume();
    int bv_consume();

    //五档挂单量变化
    int av_consume_5d();
    int bv_consume_5d();
private:
    arr AP,BP,AV,BV;
    arr AP_Last,BP_Last,AV_Last,BV_Last;
    int av1_5,bv1_5,last_av1_5,last_bv1_5;
    int count=0;
    bool inited=false;
    int time;

    int priceInLastAp(int);
    int priceInLastBp(int);
    int priceInCurrAp(int);
    int priceInCurrBp(int);
};


#endif
