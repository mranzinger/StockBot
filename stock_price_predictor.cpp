#include <torch/torch.h>
#include <iostream>

#include "hsd_dataset.h"
#include "hsd_model.h"
#include "hsd_loss.h"

using namespace std;

int main() {
    size_t historySize = 128;
    HSDDataset hsdDataset("", historySize);
    auto dataset = hsdDataset.map(torch::data::transforms::Stack<>());

    auto loader = torch::data::make_data_loader(
        dataset,
        torch::data::DataLoaderOptions().batch_size(128).enforce_ordering(true).workers(8)
    );

    cout << "Constructing Model..." << endl;
    auto model = make_shared<HSDModel>(historySize, 5);
    cout << "\tSending to GPU..." << endl;
    model->to(torch::kCUDA);
    cout << "Done" << endl;

    auto optimizer = torch::optim::Adam(model->parameters(), /*lr=*/1e-3);

    auto criterion = make_shared<HSDLoss>();

    size_t step = 0;
    for (torch::data::Example<> &batch : *loader) {
        torch::Tensor histData = batch.data.to(torch::kCUDA);
        torch::Tensor allTargets = batch.target.to(torch::kCUDA);

        torch::Tensor dateTarget = allTargets.slice(1, 0, 1);

        HSDOutput output = model->forward(histData, dateTarget);

        torch::Tensor priceTarget = allTargets.slice(1, hsdDataset.closingPriceDim(), hsdDataset.closingPriceDim() + 1);

        torch::Tensor loss = criterion->forward(output, priceTarget);

        step += 1;

        break;
    }

    torch::Tensor tensor = torch::eye(3);
    cout << tensor << endl;
}