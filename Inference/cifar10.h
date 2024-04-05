#include <torch/torch.h>
#include <fstream>
#include <vector>
struct Image {
    torch::Tensor tensor;  // һ��3x32x32������������һ����ɫͼ��
    uint8_t label;
};

std::vector<Image> read_batch(const std::string& filepath);
