#include <torch/torch.h>
#include <fstream>
#include <vector>
struct Image {
    torch::Tensor tensor;  // һ��3x32x32������������һ����ɫͼ��
    uint8_t label;
};

std::vector<Image> read_batch(const std::string& filepath);

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
public:
    std::vector<Image> images;
    CustomDataset(const std::string& filepath);

    // ��д get ��������ȡ���ݺͱ�ǩ
    torch::data::Example<> get(size_t index) ;

    // ��д size ������ȡ���ݼ��Ĵ�С
    torch::optional<size_t> size() const ;
};
