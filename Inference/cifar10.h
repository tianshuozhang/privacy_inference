#include <torch/torch.h>
#include <fstream>
#include <vector>
struct Image {
    torch::Tensor tensor;  // 一个3x32x32的张量，代表一个颜色图像
    uint8_t label;
};

std::vector<Image> read_batch(const std::string& filepath);

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
public:
    std::vector<Image> images;
    CustomDataset(const std::string& filepath);

    // 重写 get 函数来获取数据和标签
    torch::data::Example<> get(size_t index) ;

    // 重写 size 函数获取数据集的大小
    torch::optional<size_t> size() const ;
};
