#include "hsd_dataset.h"

#include <filesystem>
#include <iostream>
#include <vector>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/date_time.hpp>
#include <boost/algorithm/string.hpp>


using namespace std;
namespace fs = boost::filesystem;

static const size_t MIN_NUM_SAMPLES = 1024;

struct Stock
{
    typedef unique_ptr<Stock> Ptr;

    Stock() = default;
    Stock(string symbol, string market) : Symbol(move(symbol)), Market(move(market)) {}

    string Symbol;
    string Market;
    torch::Tensor HistoricalData;
};

struct HSDDataset::HSDDatasetImpl
{
    HSDDatasetImpl(size_t historySize)
        : historySize(historySize)
    {
    }

    vector<Stock::Ptr> stocks;
    size_t historySize;
};

HSDDataset::HSDDataset(string root, size_t historySize)
    : m_impl(new HSDDatasetImpl{historySize})
{
    if (root.empty()) {
        root = "../huge_stock_dataset";
    }

    long baseJulian = boost::gregorian::date(2010, 1, 1).julian_day();

    cout << "Finding all stock price data..." << endl;

    vector<fs::path> files;
    for (auto &p : fs::recursive_directory_iterator(root)) {
        auto ext = fs::extension(p);
        if (ext == ".txt") {
            files.push_back(p.path());
        }
    }

    cout << "Found " << files.size() << " stock symbols. Loading..." << endl;

    long minSamples = numeric_limits<long>::max(),
         maxSamples = numeric_limits<long>::min(),
         avgSamples = 0;

    #pragma omp parallel for
    for (size_t z = 0; z < files.size(); ++z) {
        auto p = files[z];

        auto bn = fs::basename(p);

        vector<string> pts;
        boost::split(pts, bn, boost::is_any_of("."));

        string symbol(move(pts[0])),
               market(move(pts[1]));
        // cout << pts[0] << ", " << pts[1] << endl;

        ifstream histFile(p.string());

        vector<vector<float>> data;

        size_t i = 0;
        string line;
        while (getline(histFile, line)) {
            i += 1;
            // Date,Open,High,Low,Close,Volume,OpenInt
            // 1999-01-22,1.6238,1.8092,1.4379,1.5215,18297633,0

            // Ignore header
            if (i == 1) continue;

            pts.clear();
            boost::split(pts, line, boost::is_any_of(","));

            long recJulian = boost::gregorian::from_simple_string(pts[0]).julian_day();

            // Not necessary that 365 is accurate, but rather that our value range isn't huge.
            float normJul = (recJulian - baseJulian) / 365.0f;

            vector<float> vals(5);
            vals[0] = normJul;
            for (size_t k = 1; k < 5; ++k) {
                vals[k] = stof(pts[k]);
            }

            data.push_back(move(vals));
        }

        if (data.size() < MIN_NUM_SAMPLES) continue;

        auto stock = make_unique<Stock>(move(symbol), move(market));
        auto t = torch::empty({data.size(), 5}, torch::kFloat32);
        auto tAcc = t.accessor<float, 2>();
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t k = 0; k < 5; ++k) {
                tAcc[i][k] = data[i][k];
            }
        }
        stock->HistoricalData = t;

        #pragma omp critical
        {
            minSamples = min(minSamples, t.size(0));
            maxSamples = max(maxSamples, t.size(0));
            avgSamples += t.size(0);

            m_impl->stocks.push_back(move(stock));

            if ((m_impl->stocks.size() % 250) == 0) {
                cout << "\r" << m_impl->stocks.size() << " of " << files.size() << flush;
            }
        }
    }

    cout << "\r" << files.size() << " of " << files.size() << endl;
    cout << "Done!" << endl;
    cout << "Found " << m_impl->stocks.size() << " valid stocks!" << endl;
    cout << "Samples - Min: " << minSamples << ", Max: " << maxSamples << ", Average: " << (avgSamples / double(m_impl->stocks.size())) << endl;
}
HSDDataset::~HSDDataset()
{
    // DO NOT DELETE
}

c10::optional<size_t> HSDDataset::size() const
{
    return {};
}

HSDDataset::ExampleType HSDDataset::get(size_t index)
{
    throw runtime_error("Not implemented yet!");
}