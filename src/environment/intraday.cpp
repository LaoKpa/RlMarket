#include "environment/intraday.h"

#include "data/records.h"
#include "market/book.h"
#include "market/market.h"
#include "market/measures.h"
#include "utilities/time.h"
#include "utilities/maths.h"
#include "utilities/comparison.h"

#include <map>
#include <cmath>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <stdexcept>

using namespace environment;
using namespace market::measure;

template<class T1, class T2>
const map<string, Variable> Intraday<T1, T2>::v_to_i = {
    {"pos", Variable::pos},
    {"spd", Variable::spd},
    {"mpm", Variable::mpm},
    {"imb", Variable::imb},
    {"svl", Variable::svl},
    {"vol", Variable::vol},
    {"rsi", Variable::rsi},
    {"vwap", Variable::vwap},
    {"a_dist", Variable::a_dist},
    {"a_queue", Variable::a_queue},
    {"b_dist", Variable::b_dist},
    {"b_queue", Variable::b_queue},
    {"last_action", Variable::last_action},
    {"abv_diff",Variable::abv_diff},
    {"abv_diff_change",Variable::abv_diff_change},
    {"abv_diff_ratio",Variable::abv_diff_ratio},
    {"av_consume",Variable::av_consume},
    {"bv_consume",Variable::bv_consume},
    {"av_consume_5d",Variable::av_consume_5d},
    {"bv_consume_5d",Variable::bv_consume_5d}
};

template<class T1, class T2>
Intraday<T1, T2>::Intraday(Config& c):
    Base(c),

    market_depth(),
    time_and_sales(),

    state_vars()
{
    static_assert(is_base_of<data::MarketDepth, T1>::value,
                  "T1 is not a subclass of data::MarketDepth");
    static_assert(is_base_of<data::TimeAndSales, T2>::value,
                  "T2 is not a subclass of data::TimeAndSales");

    auto v = c["state"]["variables"].as<list<string>>();
    for (auto it = v.begin(); it != v.end(); ++it) {
        try {
            state_vars.push_back(v_to_i.at(*it));

        } catch (const out_of_range& oor) {
            cout << "Unknown state variable: " << *it << "." << endl;

            throw;
        }
    }

    if (c["market"]["target_price"]["type"].as<string>() == "midprice") {
        l2p_ = [this](int al, int bl) {
            int midPrice = (ask_book_.price(0)+bid_book_.price(0))/2;
            return std::make_tuple(midPrice+al,midPrice-bl);
        };

    } else {
        l2p_ = [this](int al, int bl) {
            double tp = target_price_->get(),
                   //half_spd = max(0.0, spread_window.mean() / 2.0);
                    half_spd = max(0.0, spread_window.mean() / 2.0);

            return std::make_tuple(
                market->ToPrice(market->ToTicks(tp + al*half_spd)),
                market->ToPrice(market->ToTicks(tp - bl*half_spd))
            );
        };
    }
}

template<class T1, class T2>
Intraday<T1, T2>::Intraday(Config& c,
                           string symbol,
                           string md_path,
                           string tas_path):
    Intraday<T1, T2>(c)
{
    LoadData(symbol, md_path, tas_path);
}

template<class T1, class T2>
Intraday<T1, T2>::~Intraday()
{
    if (market != nullptr)
        delete market;
}

template<class T1, class T2>
bool Intraday<T1, T2>::Initialise()
{
    bool stat = Base::Initialise();

    last_date = 0;
    market->set_date(0);
    market->set_time(0L);
    //market->set_date(0);
    //market->set_time(34200000);

    // Keep loading data until we are ready:
    while (not market->IsOpen())
        if (not UpdateBookProfiles())
            return false;

    time_and_sales.SkipUntil(market->date(), market->time());

    // Update all features with in-market data:
    while (not (f_ask_transactions.full() and
                f_bid_transactions.full() and
                f_vwap_numer.full() and
                f_vwap_denom.full() and
                f_volatility.full() and
                f_midprice.full() and
                target_price_->ready() and
                spread_window.full() and
                dbf.isInited()))
        if (not NextState())
            return false;

    time_and_sales.SkipUntil(market->date(), market->time());

    ref_time = market->time();
    init_date = market->date();

   // _place_orders(1, 1);

    return stat;
}

template<class T1, class T2>
void Intraday<T1, T2>::LoadData(string ticker, string md_path, string tas_path)
{
    market_depth.LoadCSV(md_path);
    time_and_sales.LoadCSV(tas_path);

    std::string symbol = ticker.substr(0, ticker.find_first_of('.'));
    std::string venue  = ticker.substr(ticker.find_first_of('.') + 1);

    market = market::Market::make_market(symbol, venue);
}

template<class T1, class T2>
bool Intraday<T1, T2>::isTerminal()
{
    return (not market->IsOpen()) or
        ((last_date != 0) and (market->date() != last_date));
}

template<class T1, class T2>
string Intraday<T1, T2>::getEpisodeId()
{ return to_string(init_date); }

template<class T1, class T2>
void Intraday<T1, T2>::_place_orders(double al, double bl,int levels,bool replace)
{
    ask_level = al ;
    bid_level = bl ;

    std::tie(ask_quote, bid_quote) = l2p_(ask_level*levels, bid_level * levels);

    risk_manager_.PlaceOrder(market::Side::ask, ask_quote, ORDER_SIZE, replace);
    risk_manager_.PlaceOrder(market::Side::bid, bid_quote, ORDER_SIZE, replace);
}

template<class T1, class T2>
void Intraday<T1,T2>::_place_orders(double per_ask,double per_bid,bool replace){
    //percent以万分之一为单位
    double mp = midprice(ask_book_,bid_book_);
    int al = ceil(per_ask*mp/10000);
    int bl = ceil(per_bid*mp/10000);
    _place_orders(al,bl,1,replace);
}


template<class T1, class T2>
void Intraday<T1, T2>::DoAction(int action)
{
    ref_time = (long) ceil(market->time() + latency_->sample());

    // Do the action
    switch (action) {
        case 0:
            _place_orders(9, 9);
            break;

        case 1:
            ClearInventory();
            _place_orders(9,9);
            break;

        case 2:
            _place_orders(9.5, 9.5);
            break;

        case 3:
            _place_orders(10, 10);
            break;

        case 4:
            _place_orders(7, 12);
            break;

        case 5:
            _place_orders(12, 7);
            break;

        case 6:
            _place_orders(6,13);
            break;

        case 7:
            _place_orders(13,6);
            break;

        case 8:
            break;
           //假设70的胜率，双边手续费千1.3，则一次双向成交最少需要赚千1.8，也就是单边万9
    }
}

// Read a bid or ask row from the data source
// Assumes that such a row is currently in the data source
template<class T1, class T2>
bool Intraday<T1, T2>::NextState()
{
    int target_date = market_depth.NextDate();      //获取下一条record的Date
    long target_time = market_depth.NextTime();     //获取下一条record的Time

    if (not time_and_sales.LoadUntil(target_date, target_time))     //一直导入数据直至next恰好<=target_time
        return false;

    const data::TimeAndSalesRecord& rec_ts = time_and_sales.Record();
    const data::MarketDepthRecord md_rec = market_depth.Record();

    double mp = midprice(ask_book_, bid_book_);
    auto au = ask_book_.ApplyTransactions(rec_ts.transactions, mp),
         bu = bid_book_.ApplyTransactions(rec_ts.transactions, mp);

    if (not UpdateBookProfiles(rec_ts.transactions))
        return false;

    auto adverse_selection =
        market::BookUtils::HandleAdverseSelection(ask_book_, bid_book_);

    tradepnl = get<1>(au) + get<1>(bu) + get<1>(adverse_selection);
    double fee = (abs(get<2>(au))+abs(get<2>(bu))+abs(get<2>(adverse_selection)))
              *TRANSACTION_FEE;
    tradepnl -= fee;

    pnl_step += tradepnl;
    lo_vol_step += get<0>(bu) - get<0>(au) + abs(get<0>(adverse_selection));


    episode_stats.pnl += get<2>(au) + get<2>(bu) + get<2>(adverse_selection)-fee;

    // Update our position:
    risk_manager_.Update(get<0>(bu) + get<0>(au) + get<0>(adverse_selection));

    long mpt = market->ToTicks(midprice(ask_book_, bid_book_));
    double mpm = midprice_move(ask_book_, bid_book_),
            sp = spread(ask_book_, bid_book_);

    pospnl = mpm * risk_manager_.exposure();

    f_midprice.push(mpt);
    f_volatility.push(mpt);
    f_vwap_numer.push(ask_book_.observed_value() + bid_book_.observed_value());
    f_vwap_denom.push(ask_book_.observed_volume() + bid_book_.observed_volume());

    spread_window.push(fmax(0.0, sp));
    target_price_->update(ask_book_, bid_book_);

    return_ups.push(fmax(0.0, mpm));
    return_downs.push(abs(fmin(0.0, mpm)));

    f_ask_transactions.push(ask_book_.observed_volume());
    f_bid_transactions.push(bid_book_.observed_volume());

    dbf.push(md_rec);

    return true;
}

template<class T1, class T2>
bool Intraday<T1, T2>::UpdateBookProfiles(
    const std::map<double, long, FloatComparator<>>& transactions)
{
    ask_book_.StashState();
    bid_book_.StashState();

    while (true) {
        if (not market_depth.LoadNext())
            return false;

        const data::MarketDepthRecord& rec_md = market_depth.Record();

        last_date = market->date();
        market->set_date(rec_md.date);
        market->set_time(rec_md.time);

        ask_book_.ApplyChanges(rec_md.ask_prices,
                               rec_md.ask_volumes,
                               transactions);

        bid_book_.ApplyChanges(rec_md.bid_prices,
                               rec_md.bid_volumes,
                               transactions);

        if (not market_depth.WillTimeChange()) continue;

        // Use a try as a bit of a shortcut. If the books haven't seen
        // enough (2) states to have a midprice move then it will clearly
        // throw an exception when we try to access it. So, instead, we just
        // use a try and start again if we couldn't.
        try {
            if (market::BookUtils::IsValidState(ask_book_, bid_book_))
                break;
        } catch (runtime_error &e) {
            continue;
        }
    }

    return true;
}

template<class T1, class T2>
double Intraday<T1, T2>::getVariable(Variable v)
{
    switch (v) {
        case Variable::pos:
            // Generalise -> ORDER_SIZE (default: 1)
            return double(risk_manager_.exposure()) / ORDER_SIZE;

        case Variable::spd:
            // Generalise -> 1 tick
            return ulb((double)(market->ToTicks(ask_book_.price(0)) -
                                market->ToTicks(bid_book_.price(0))),
                       0.0, 20.0);

        case Variable::mpm:
            // Generalise -> 1 tick
            return ulb(
                (double)(market->ToTicks(f_midprice.front()) -
                         market->ToTicks(f_midprice.back())),
                -10.0, 10.0
                );
        case Variable::abv_diff:
            return (dbf.abv_diff());

        case Variable::abv_diff_ratio:
            return (dbf.abv_diff_ratio());

        case Variable::abv_diff_change:
            return (dbf.abv_diff_change());

        case Variable::av_consume:
            return (dbf.av_consume());

        case Variable::av_consume_5d:
            return (dbf.av_consume_5d());

        case Variable::bv_consume:
            return (dbf.bv_consume());

        case Variable::bv_consume_5d:
            return (dbf.bv_consume_5d());

        case Variable::imb: {
            double v_a = (double) ask_book_.total_volume(),
                   v_b = (double) bid_book_.total_volume();

            // Generalise -> 0.2
            return ((v_a + v_b) > 0 ? 5*(v_b - v_a) / (v_b + v_a) : 0.0);
        }

        case Variable::svl: {
            double q_a = (double) f_ask_transactions.sum(),
                   q_b = (double) f_bid_transactions.sum();

            // Generalise -> 0.2
            return ((q_a + q_b) > 0 ? 5*(q_b - q_a) / (q_a + q_b) : 0.0);
        }

        case Variable::vol:
            return ulb(5.0*f_volatility.std(), 0.0, 10.0);

        case Variable::rsi: {
            double u = return_ups.mean(),
                   d = return_downs.mean();

            // Generalise -> 0.20
            return (u + d) != 0.0 ? 5.0 * (u - d) / (u + d) : 0.0;
        }

        case Variable::vwap: {
            double d = f_vwap_numer.sum() / f_vwap_denom.sum();

            return ulb(
                d / spread_window.mean(), -10.0, 10.0
            );
        }

        case Variable::a_dist:
            // Generalise -> 1 tick
            if (ask_book_.order_count() > 0)
                return ((double) market->ToTicks(ask_book_.best_open_order_price()) -
                        (double) market->ToTicks(ask_book_.price(0)));
            else
                return -100.0;

        case Variable::a_queue:
            // Generalise -> 10%
            if (ask_book_.order_count() > 0)
                return 10.0 * ask_book_.queue_progress();
            else
                return -1.0;

        case Variable::b_dist:
            // Generalise -> 1 tick
            if (bid_book_.order_count() > 0)
                return ((double) market->ToTicks(bid_book_.price(0)) -
                        (double) market->ToTicks(bid_book_.best_open_order_price()));
            else
                return -100.0;

        case Variable::b_queue:
            // Generalise -> 10%
            if (bid_book_.order_count() > 0)
                return 10.0 * bid_book_.queue_progress();
            else
                return -1.0;

        case Variable::last_action:
            return last_action;

        default:
            throw std::invalid_argument("Unknown state-var enum value: " +
                                        to_string((int) v) + ".");
    }
}

template<class T1, class T2>
void Intraday<T1, T2>::getState(std::vector<float>& out)
{
    for (auto it = state_vars.begin(); it != state_vars.end(); ++it)
        out.push_back(getVariable(*it));
}

template<class T1, class T2>
void Intraday<T1, T2>::printInfo(const int action)
{
    cout << "Date: " << market->date() << endl;
    cout << "Time: " << time_to_string(market->time()) << "ms" << endl;

    cout << "Time (TAS): " << time_to_string(time_and_sales.Record().time) << endl;
    cout << "Time (TAS_f): " << time_to_string(time_and_sales.NextTime()) << endl << endl;

    if (market->IsOpen())
        cout << "Market status: Open" << endl;
    else
        cout << "Market status: Closed" << endl;

    cout << endl;

    Base::printInfo(action);
}

template<class T1, class T2>
void Intraday<T1, T2>::LogMarket() {
    if (trade_logger != nullptr)
    {
        trade_logger->info("{}", " ");
        trade_logger->info("{},{},{},{},{},{},{}",
                           market->date(), market->time(), time2sec(market->time()),
                           ask_book_.price(0), ask_book_.volume(ask_book_.price(0)), bid_book_.price(0), bid_book_.volume(bid_book_.price(0)));
        for (int i = 1; i != 5; i++) {
            trade_logger->info(" , , ,{},{},{},{}",
                               ask_book_.price(i), ask_book_.volume(ask_book_.price(i)), bid_book_.price(i), bid_book_.volume(bid_book_.price(i)));
        }
    }
}

template<class T1, class T2>
void Intraday<T1, T2>::LogAction(int action, int ask, double ref, int bid, int pos) {
    if (trade_logger != nullptr)
        trade_logger->info("{},{},{},{},{},{},{}",
                           "","选择报单", action, ask, ref, bid, pos);
}

template<class T1, class T2>
void Intraday<T1, T2>::LogOpenOrder() {
    auto od_iter = ask_book_.open_orders.begin();
    for (; od_iter != ask_book_.open_orders.end(); od_iter++) {
        if (trade_logger != nullptr)
            trade_logger->info("{},{},{},{},{},{}",
                               "","挂单", od_iter->second->id, 'A', od_iter->second->price, od_iter->second->remaining());
    }
    od_iter = bid_book_.open_orders.begin();
    for (; od_iter != bid_book_.open_orders.end(); od_iter++) {
        if (trade_logger != nullptr)
            trade_logger->info("{},{},{},{},{},{}",
                               "","挂单", od_iter->second->id, 'B', od_iter->second->price, od_iter->second->remaining());
    }
}

template<class T1, class T2>
void Intraday<T1, T2>::LogTrade(int index, char side, int price, int vol) {
    if (trade_logger != nullptr) {
        trade_logger->info("{},{},{},{},{},{},{}",
                           "", "", "", index, (int)side, price, vol);
        trade_logger->info("{}"," ");
    }
}

template<class T1, class T2>
void Intraday<T1, T2>::LogProfit(int action, double pnl, double bandh)
{
    if (profit_logger != nullptr)
        profit_logger->info("{},{},{},{},{},{},{},{},{},{},{},{}",
                            market->date(),
                            market->time(),
                            action,
                            risk_manager_.exposure(),
                            midprice(ask_book_, bid_book_),
                            spread(ask_book_, bid_book_),
                            ask_quote, bid_quote,
                            ask_level, bid_level,
                            pnl, bandh);
}

template<class T1, class T2>
void Intraday<T1, T2>::LogPnl(double tradepnl,double pospnl,double totalpnl,
                              double aggpnl,double rew) {
    if (trade_logger != nullptr) {
        trade_logger->info(",TradePnl,{},PosPnl,{},TotalPnl,{}",
                           tradepnl, pospnl, totalpnl);
        trade_logger->info(",AggPnl,{},Reward,{},Episode_pnl,{}",
                           aggpnl, rew, episode_stats.pnl);
    }
}


// Template specialisations
template class environment::Intraday<data::basic::MarketDepth,
        data::basic::TimeAndSales>;
