#include <torch/torch.h>
#include <fstream>
#include <vector>
struct Image {
    torch::Tensor tensor;  // 一个3x32x32的张量，代表一个颜色图像
    uint8_t label;
};

std::vector<Image> read_batch(const std::string& filepath);
