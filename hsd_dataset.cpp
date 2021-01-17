#include "hsd_dataset.h"

#include <filesystem>
#include <iostream>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/date_time.hpp>
#include <boost/algorithm/string.hpp>


using namespace std;
namespace fs = boost::filesystem;


struct Stock
{
    typedef unique_ptr<Stock> Ptr;

    string Symbol;
    string Zone;
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

    for (auto &p : fs::recursive_directory_iterator(root)) {
        auto ext = fs::extension(p);
        if (ext != ".txt") continue;

        auto bn = fs::basename(p);

        vector<string> pts;
        boost::split(pts, bn, boost::is_any_of("."));

        cout << pts[0] << ", " << pts[1] << endl;

        ifstream histFile(p.path().string());

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

            boost::gregorian::date d = boost::gregorian::from_simple_string(pts[0]);

            if (i == 2) {
                cout << d << endl;
            }


        }
    }
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