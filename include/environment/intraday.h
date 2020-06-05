#ifndef ENVIRONMENT_INTRADAY_H
#define ENVIRONMENT_INTRADAY_H

#include "data/basic.h"
#include "market/market.h"
#include "environment/base.h"
#include "utilities/comparison.h"

#include <map>
#include <tuple>
#include <vector>
#include <functional>


namespace environment {

enum class Variable {
    pos, spd, mpm,
    imb, svl, vol,
    rsi, vwap,
    a_dist, a_queue,
    b_dist, b_queue,
    last_action,abv_diff,abv_diff_change,
    abv_diff_ratio,av_consume,bv_consume,
    av_consume_5d,bv_consume_5d
};

template<class T1 = data::basic::MarketDepth,
         class T2 = data::basic::TimeAndSales>
class Intraday: public Base
{
    protected:
        T1 market_depth;
        T2 time_and_sales;

        // Real market parameters
        market::Market* market = nullptr;

        list<Variable> state_vars;
        const static map<string, Variable> v_to_i;


        int last_date = 0;
        int init_date = 0;      // Used for logging...

        long ref_time;

        double ask_level = 0;
        double bid_level = 0;

        void DoAction(int action);
        bool NextState();
        bool UpdateBookProfiles(
            const std::map<double, long, FloatComparator<>>& transactions = {});

        std::function<std::tuple<double, double>(int, int)> l2p_;
        void _place_orders(double sp, double sk, int levelplace,bool replace=true);
        void _place_orders(double,double,bool replace=true);
    public:
        Intraday(Config& c);
        Intraday(Config& c, string symbol, string md_path, string tas_path);
        ~Intraday();

        bool Initialise();
        void LoadData(string symbol, string md_path, string tas_path);

        double getVariable(Variable v);
        void getState(vector<float>& out);

        bool isTerminal();
        string getEpisodeId();

        void printInfo(const int action = -1);

        void LogProfit(int action, double pnl, double bandh);
        void LogMarket();
        //用于记录t0时刻的行情
        void LogTrade(int index, char side, int price, int vol);
        //用于记录t0时刻的open_order在t1时刻的撮合成交情况
        void LogAction(int action, int ask, double ref, int bid, int pos);
        //用于记录t0时刻做出的报单选择
        void LogOpenOrder();
        void LogPnl(double, double, double, double, double);
        //用于记录当前区间内的各种pnl
};

}

#endif
