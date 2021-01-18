#pragma once

#include <torch/torch.h>


struct HSDOutput
{
    HSDOutput() = default;
    HSDOutput(torch::Tensor closingPrice, torch::Tensor certainty)
        : ClosingPrice(closingPrice), Certainty(certainty)
    {}

    torch::Tensor ClosingPrice;
    torch::Tensor Certainty;
};


class HSDModel
    : public torch::nn::Module
{
public:
    HSDModel(size_t histLen, size_t numFields, size_t embedDim = 512);

    HSDOutput forward(torch::Tensor histData, torch::Tensor dateTarget);

private:
    torch::nn::GRU m_histEncoder{nullptr};
    torch::nn::Linear m_targetEmbedder{nullptr};
    torch::nn::Sequential m_jointEncoder{nullptr};
};