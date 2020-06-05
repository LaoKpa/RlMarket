#include <ctime>
#include <mutex>
#include <chrono>
#include <vector>
#include <string>
#include <random>
#include <thread>
#include <fstream>
#include <iostream>

#include <iterator>
#include <algorithm>
#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "rl/agent.h"
#include "rl/tiles.h"
#include "experiment/serial.h"
#include "utilities/files.h"
#include "utilities/sampler.h"
#include "environment/intraday.h"

using namespace std;

auto r_eng = default_random_engine{};

typedef tuple<string, string, string> data_sample_t;

vector<data_sample_t> train_set;
vector<data_sample_t> test_set;

// Synchronisation variables for the training threads
int n_threads;
int n_train_episodes;
int n_eval_episodes;

int current_episode = 1;
mutex episode_mutex;

void train(int id, Config &c, rl::Agent* m)
{
    environment::Intraday<> env(c);
    experiment::serial::Learner experiment(c, env);

    data_sample_t ds;
    RandomSampler<data_sample_t> rs(train_set);

    while (true) {
        ds = rs.sample();       //随机从数据（三元组）中抽取一份
        env.LoadData(get<0>(ds), get<1>(ds), get<2>(ds));

        // Run episode:
        if (experiment.RunEpisode(m)) {
            cout << "[" << id << "]";
            cout << " Trained on episode " << current_episode;

            cout << " (" << get<0>(ds) << " - " << env.getEpisodeId() << "):";

            cout << "\n\tRwd = " << env.getEpisodeReward() << endl;
            cout << "\tRho = " << env.getMeanEpisodeReward() << endl;
            cout << "\tPnl = " << env.getEpisodePnL() << endl;
            //Pnl：episode_stats.pnl;
            cout << "\tnTr = " << env.getTotalTransactions() << endl;
            cout << "\tPpt = " << env.getEpisodePnL()/env.getTotalTransactions() << endl;
            cout << endl;

            // Episode completed...
            episode_mutex.lock();
            current_episode++;
            episode_mutex.unlock();

            if (current_episode > n_train_episodes-n_threads+1)
                break;
        }
    }
}

void run(Config &c) {
    // Initiate all random number generators
    unsigned seed = c["debug"]["random_seed"].as<unsigned>(
            chrono::system_clock::now().time_since_epoch().count());

    srand(seed);
    r_eng.seed(seed);

    bool eval_from_train = c["evaluation"]["use_train_sample"].as<bool>(false);
    n_train_episodes = c["training"]["n_episodes"].as<int>();
    n_eval_episodes = c["evaluation"]["n_samples"].as<int>(-1);

    // Partition the data
    auto symbols = c["data"]["symbols"].as<vector<string>>();
    auto train_file_samples = get_file_sample(c["data"]["train"]["md_dir"].as<string>(),
                                        c["data"]["train"]["tas_dir"].as<string>(),
                                            symbols);
    auto test_file_samples = get_file_sample(c["data"]["test"]["md_dir"].as<string>(),
                                              c["data"]["test"]["tas_dir"].as<string>(),
                                              symbols);

    int n_train_samples = c["training"]["n_samples"].as<int>(-1);

    cout << "[-] Training on " << n_train_episodes << " episodes." << endl;
    cout << "[-] Testing on " << n_eval_episodes << " episodes." << endl;
    cout << endl;

    // Set up policy:
    unsigned int n_actions = c["learning"]["n_actions"].as<unsigned int>();

    std::unique_ptr<rl::Policy> p;
    string policy_type = c["policy"]["type"].as<string>("");
    if (policy_type == "greedy")
        p = std::unique_ptr<rl::Policy>(new rl::Greedy(n_actions, seed));

    else if (policy_type == "random")
        p = std::unique_ptr<rl::Policy>(new rl::Random(n_actions, seed));

    else if (policy_type == "epsilon_greedy") {
        float eps = c["policy"]["eps_init"].as<float>(),
                eps_floor = c["policy"]["eps_floor"].as<float>();
        unsigned int eps_T = c["policy"]["eps_T"].as<unsigned int>();

        p = std::unique_ptr<rl::Policy>(
                new rl::EpsilonGreedy(n_actions, eps, eps_floor, eps_T, seed));

    } else if (policy_type == "boltzmann") {
        float tau = c["policy"]["tau_init"].as<float>(),
                tau_floor = c["policy"]["tau_floor"].as<float>();
        unsigned int tau_T = c["policy"]["tau_T"].as<unsigned int>();

        p = std::unique_ptr<rl::Policy>(
                new rl::Boltzmann(n_actions, tau, tau_floor, tau_T, seed));

    } else
        throw runtime_error("Please specify a valid policy!");

    // Set up the agent
    rl::Agent *m;
    string algorithm = c["learning"]["algorithm"].as<string>("");
    if (algorithm == "q_learn")
        m = new rl::QLearn(std::move(p), c);

    else if (algorithm == "double_q_learn")
        m = new rl::DoubleQLearn(std::move(p), c);

    else if (algorithm == "sarsa")
        m = new rl::SARSA(std::move(p), c);

    else if (algorithm == "r_learn")
        m = new rl::RLearn(std::move(p), c);

    else if (algorithm == "online_r_learn")
        m = new rl::OnlineRLearn(std::move(p), c);

    else if (algorithm == "double_r_learn")
        m = new rl::DoubleRLearn(std::move(p), c);

    else
        throw runtime_error("Please specify a valid learning algorithm!");

    // Run training phases:
    if (n_train_episodes > 0) {
        train_set.resize(train_file_samples.size());
        copy(train_file_samples.begin(), train_file_samples.end(), train_set.begin());
        train(0, c, m);
    }

    // Reset counter:
    current_episode = 0;
    cout << endl;

    // Run final testing phase:
    m->GoGreedy();

    test_set.resize(test_file_samples.size());
    copy(test_file_samples.begin(), test_file_samples.end(), test_set.begin());
    data_sample_t ds;
    RandomSampler<data_sample_t> rs(test_set);

    environment::Intraday<> env(c);
    for (int i = 0; i < n_eval_episodes; i++) {
        ds = rs.sample();
        env.LoadData(get<0>(ds), get<1>(ds), get<2>(ds));
        string date_file = get<2>(ds);
        int npos = date_file.find_last_of("_");
        date_file = string(to_string(i)+"_")+(date_file.substr(npos,20));
        experiment::serial::Backtester experiment(c, env, date_file);
        current_episode++;

        if (experiment.RunEpisode(m)) {
            cout << "[0] \33[4mTested on episode " << current_episode
                 << " (" << get<0>(ds) << " - " << env.getEpisodeId() << "):\33[0m";

            cout << "\n\tRwd = " << env.getEpisodeReward() << endl;
            cout << "\tRho = " << env.getMeanEpisodeReward() << endl;
            cout << "\tPnl = " << env.getEpisodePnL() << endl;
            cout << "\tnTr = " << env.getTotalTransactions() << endl;
            cout << "\tPpt = " << env.getEpisodePnL()/env.getTotalTransactions() << endl;
            cout << endl;
        }
    }

    // Output testing stats to a csv:
    env.writeStats(c["output_dir"].as<string>() + "test_stats.csv");

    delete m;
}

int main(int argc, char** argv)
{
    string config_dir("../config/"),
            config_path,
            output_dir,
            symbol,
            algorithm;
    spdlog::set_pattern("%v");

    try {
        namespace po = boost::program_options;
        po::options_description desc("Options");

        desc.add_options()
                ("config,c", po::value<string>(), "Config file path")
                ("output_dir,o", po::value<string>(), "Output directory")
                ("symbol,s", po::value<string>(), "Symbol override")
                ("algorithm,a", po::value<string>(), "Algorithm override")
                ("debug",
                 po::bool_switch()->default_value(false),
                 "Show debug output")
                ("quiet",
                 po::bool_switch()->default_value(false),
                 "Disable episode logging to cout")
                ("help,h", "Display help message");

        po::variables_map vm;

        try {
            po::store(po::parse_command_line(argc, argv, desc), vm);

            if (vm.count("config"))
                config_path = vm["config"].as<string>();
            else
                config_path = "main";

            if (vm.count("output_dir"))
                output_dir = vm["output_dir"].as<string>();
            else {
                output_dir = "/tmp/rl_markets/";

                time_t rawtime;
                struct tm *timeinfo;
                char buffer[80];
                time(&rawtime);
                timeinfo = localtime(&rawtime);
                strftime(buffer, 80, "%d_%m_%Y.%H_%M_%S", timeinfo);

                output_dir += string(buffer) + "/";
            }

            if (vm.count("symbol"))
                symbol = vm["symbol"].as<string>();
            else
                symbol = "";

            if (vm.count("algorithm"))
                algorithm = vm["algorithm"].as<string>();
            else
                algorithm = "";

            if (vm.count("help")) {
                cout << "RL_Markets Help:" << endl
                     << desc << endl;

                return 0;
            }

            po::notify(vm);
        } catch (po::error& e) {
            cerr << "ERROR: " << e.what() << endl << endl;
            cerr << desc << endl;

            return 1;
        }

        if (output_dir.back() != '/')
            output_dir.append("/");

        boost::filesystem::create_directories(output_dir);
        boost::filesystem::create_directories(output_dir+string("profit_log/"));

        // Copy original yaml file for reference:
        if (config_path.find('/') == string::npos)
            config_path = config_dir + config_path + ".yaml";

        if (config_path != output_dir+"config.yaml") {
            ifstream ifs(config_path, fstream::binary);
            ofstream ofs(output_dir + "config.yaml", fstream::binary);

            ofs << ifs.rdbuf() << "" << endl;
        }

        // Start application
        Config config(config_path);

        config["output_dir"] = output_dir;

        if (symbol != "") config["data"]["symbols"] = vector<string> {symbol};
        if (algorithm != "") config["learning"]["algorithm"] = algorithm;

        if (vm["debug"].as<bool>()) config["debug"]["inspect_books"] = true;

        bool quiet = vm["quiet"].as<bool>();

        if (quiet) {
            streambuf *old = cout.rdbuf();
            stringstream ss;

            cout.rdbuf(ss.rdbuf());
            run(config);
            cout.rdbuf(old);

        } else
            run(config);

        cout << output_dir << endl;

    } catch (exception& e) {
        cerr << "Unhandled Exception: " << e.what() << endl;

        return 2;
    }

    return 0;
}
