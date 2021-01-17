#include <torch/torch.h>
#include <iostream>

#include "hsd_dataset.h"

using namespace std;

int main() {
    size_t historySize = 128;
    auto dataset = std::make_unique<HSDDataset>("", historySize);


    torch::Tensor tensor = torch::eye(3);
    cout << tensor << endl;
}