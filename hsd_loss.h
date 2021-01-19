#include <torch/torch.h>

#include <vector>
#include <map>

#include "hsd_model.h"


struct LossInfo
{
    torch::Tensor Loss;

    std::vector<std::pair<std::string, torch::Tensor>> Components;
};


class HSDLoss
    : public torch::nn::Module
{
public:
    LossInfo forward(const HSDOutput &x, torch::Tensor y);

};