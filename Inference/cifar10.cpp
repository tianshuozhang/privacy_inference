#include "cifar10.h"
std::vector<Image> read_batch(const std::string& folderpath) {
    // CIFAR-10文件中每个样本的大小：1x label + 1024x red_channel + 1024x green_channel + 1024x blue_channel 
    constexpr int NUM_SAMPLES = 10000;  // 每个批次文件有10000个图像

    std::vector<Image> images;
    images.reserve(NUM_SAMPLES*5);
    for (int j = 1; j < 6; ++j) {
        auto filepath = folderpath + "/data_batch_"+ std::to_string(j) + ".bin";
        std::ifstream file(filepath, std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Could not open file " + filepath);
        }

        std::vector<uint8_t> buffer(1024);  // 中间缓冲区

        std::vector<double> mean = { 0.4914, 0.4822, 0.4465 };
        std::vector<double> std = { 0.2023, 0.1994, 0.2010 };
        // Create the normalize transform
        auto normalize = torch::data::transforms::Normalize<>(mean, std);
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            Image img;

            // 读取label
            file.read(reinterpret_cast<char*>(&img.label), 1);

            // 读取并处理红色通道
            file.read(reinterpret_cast<char*>(buffer.data()), 1024);
            torch::Tensor red_channel = torch::tensor(buffer, torch::kFloat32).clone();
            red_channel = red_channel.unsqueeze(0).view({ 1, 32, 32 });

            // 读取并处理绿色通道
            file.read(reinterpret_cast<char*>(buffer.data()), 1024);
            torch::Tensor green_channel = torch::tensor(buffer, torch::kFloat32).clone();
            green_channel = green_channel.unsqueeze(0).view({ 1, 32, 32 });

            // 读取并处理蓝色通道
            file.read(reinterpret_cast<char*>(buffer.data()), 1024);
            torch::Tensor blue_channel = torch::tensor(buffer, torch::kFloat32).clone();
            blue_channel = blue_channel.unsqueeze(0).view({ 1, 32, 32 });

            // 使用torch::cat函数拼接三个通道
            img.tensor = torch::cat({ red_channel, green_channel, blue_channel }, 0);
            // Apply the transform to your tensor
            img.tensor = normalize(img.tensor/255);

            images.push_back(std::move(img));
        }

    }
    

    return images;
}
CustomDataset::CustomDataset(const std::string& filepath) {
    images = read_batch(filepath);
}

// 重写 get 函数来获取数据和标签
torch::data::Example<> CustomDataset:: get(size_t index)   {
    return { images[index].tensor.clone(), torch::tensor((uint8_t)images[index].label) };
}

// 重写 size 函数获取数据集的大小
torch::optional<size_t> CustomDataset:: size() const  {
    return images.size();
}