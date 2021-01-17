#include <torch/torch.h>
#include <iostream>

#include "hsd_dataset.h"

using namespace std;

int main() {
    size_t historySize = 128;
    auto dataset = HSDDataset("", historySize)
                        .map(torch::data::transforms::Stack<>());

    auto loader = torch::data::make_data_loader(
        dataset,
        torch::data::DataLoaderOptions().batch_size(128).enforce_ordering(true).workers(8)
    );

    size_t step = 0;
    for (auto &batch : *loader) {

        step += 1;

        cout << "Step " << step << endl << batch.target << endl;

        if (step == 10) {
            break;
        }
    }

    torch::Tensor tensor = torch::eye(3);
    cout << tensor << endl;
}