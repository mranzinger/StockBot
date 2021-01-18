#include <torch/torch.h>

#include "hsd_model.h"

class HSDLoss
    : public torch::nn::Module
{
public:
    torch::Tensor forward(const HSDOutput &x, torch::Tensor y);

};