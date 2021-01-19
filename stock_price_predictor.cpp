#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#include "hsd_dataset.h"
#include "hsd_model.h"
#include "hsd_loss.h"
#include "meter.h"

using namespace std;

void synchronize();
template<typename TP1, typename TP2>
double time_diff(TP1 t1, TP2 t2);

int main() {
    size_t historySize = 128;
    HSDDataset hsdDataset("", historySize);
    auto dataset = hsdDataset.map(torch::data::transforms::Stack<>());

    auto loader = torch::data::make_data_loader(
        dataset,
        torch::data::DataLoaderOptions().batch_size(256).enforce_ordering(true).workers(8)
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

    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(1e-4).weight_decay(5e-4));
    // auto optimizer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(1e-4).momentum(0.99).weight_decay(5e-4));

    auto criterion = make_shared<HSDLoss>();

    MeterDict<float> lossMeters;
    MeterDict<float> timeMeters;

    auto startTime = std::chrono::high_resolution_clock::now();

    size_t step = 0;
    for (torch::data::Example<> &batch : *loader) {
        auto loopStart = std::chrono::high_resolution_clock::now();

        torch::Tensor histData = batch.data.to(torch::kCUDA, true);
        torch::Tensor allTargets = batch.target.to(torch::kCUDA, true);

        torch::Tensor dateTarget = allTargets.slice(1, 0, 1);

        HSDOutput output = model->forward(histData, dateTarget);

        torch::Tensor priceTarget = allTargets.slice(1, hsdDataset.closingPriceDim(), hsdDataset.closingPriceDim() + 1);

        LossInfo loss = criterion->forward(output, priceTarget);

        // This will also synchronize the device
        lossMeters.Add("Global", loss.Loss.item<float>());
        for (auto &kv : loss.Components) {
            lossMeters.Add(kv.first, kv.second.item<float>());
        }

        auto fpropEnd = std::chrono::high_resolution_clock::now();

        loss.Loss.backward();
        optimizer.step();

        synchronize();

        auto bpropEnd = std::chrono::high_resolution_clock::now();

        timeMeters.Add("Batch", time_diff(startTime, loopStart));
        timeMeters.Add("FProp", time_diff(loopStart, fpropEnd));
        timeMeters.Add("BProp", time_diff(fpropEnd, bpropEnd));
        timeMeters.Add("Step", time_diff(startTime, bpropEnd));

        step += 1;

        if ((step % 100) == 0) {
            cout << "Step " << step << " - Losses " << lossMeters << " - Timings " << timeMeters << endl;
        }

        startTime = std::chrono::high_resolution_clock::now();
    }
}

void synchronize()
{
    cudaDeviceSynchronize();
}

template<typename TP1, typename TP2>
double time_diff(TP1 t1, TP2 t2)
{
    return std::chrono::duration<double>(t2 - t1).count();
}