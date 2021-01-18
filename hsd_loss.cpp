#include "hsd_loss.h"

namespace nn = torch::nn;
namespace F = nn::functional;


torch::Tensor HSDLoss::forward(const HSDOutput &x, torch::Tensor y)
{
    auto mse = F::mse_loss(x.ClosingPrice, y, torch::kNone);

    auto expUnc = 2 * x.Certainty.exp();

    auto loss = (mse / expUnc) + 0.5 * x.Certainty;

    return loss.mean();
}