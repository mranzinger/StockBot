#include <torch/torch.h>
#include <iostream>

#include "hsd_dataset.h"
#include "hsd_model.h"
#include "hsd_loss.h"
#include "meter.h"

using namespace std;

void print_loss_components(const LossInfo &l);

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

    for (auto &p : model->parameters()) {
        if (p.dim() >= 2) {
            torch::nn::init::kaiming_uniform_(p);
        }
    }

    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(1e-3).weight_decay(5e-4));
    // auto optimizer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(1e-4).momentum(0.99).weight_decay(5e-4));

    auto criterion = make_shared<HSDLoss>();

    MeterDict<float> lossMeters;

    size_t step = 0;
    for (torch::data::Example<> &batch : *loader) {
        torch::Tensor histData = batch.data.to(torch::kCUDA);
        torch::Tensor allTargets = batch.target.to(torch::kCUDA);

        torch::Tensor dateTarget = allTargets.slice(1, 0, 1);

        HSDOutput output = model->forward(histData, dateTarget);

        torch::Tensor priceTarget = allTargets.slice(1, hsdDataset.closingPriceDim(), hsdDataset.closingPriceDim() + 1);

        LossInfo loss = criterion->forward(output, priceTarget);

        loss.Loss.backward();
        optimizer.step();

        lossMeters.Add("Global", loss.Loss.item<float>());
        for (auto &kv : loss.Components) {
            lossMeters.Add(kv.first, kv.second.item<float>());
        }

        step += 1;

        if ((step % 100) == 0) {
            cout << "Step " << step << " - Losses " << lossMeters << endl;
        }
    }
}

void print_loss_components(const LossInfo &l)
{
    cout << "Loss: " << l.Loss.item<float>();

    if (l.Components.size() > 0) {
        cout << " - Components - " << l.Components[0].first << ": " << l.Components[0].second.item<float>();
        for (size_t i = 1; i < l.Components.size(); ++i) {
            cout << ", " << l.Components[i].first << ": " << l.Components[i].second.item<float>();
        }
    }
}