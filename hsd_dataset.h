#include <string>
#include <memory>

#include <torch/torch.h>

class HSDDataset:
    public torch::data::Dataset<HSDDataset>
{
public:
    HSDDataset(std::string root, size_t historySize);
    // Required to enable a unique_ptr on the Impl object
    ~HSDDataset();

    virtual c10::optional<size_t> size() const override;
    virtual ExampleType get(size_t index) override;

private:
    struct HSDDatasetImpl;
    std::shared_ptr<HSDDatasetImpl> m_impl;
};
