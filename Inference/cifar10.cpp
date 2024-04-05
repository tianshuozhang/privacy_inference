#include "cifar10.h"
std::vector<Image> read_batch(const std::string& folderpath) {
    // CIFAR-10�ļ���ÿ�������Ĵ�С��1x label + 1024x red_channel + 1024x green_channel + 1024x blue_channel 
    constexpr int NUM_SAMPLES = 10000;  // ÿ�������ļ���10000��ͼ��

    std::vector<Image> images;
    images.reserve(NUM_SAMPLES*5);
    for (int j = 1; j < 6; ++j) {
        auto filepath = folderpath + "/data_batch_"+ std::to_string(j) + ".bin";
        std::ifstream file(filepath, std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Could not open file " + filepath);
        }

        std::vector<uint8_t> buffer(1024);  // �м仺����
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            Image img;

            // ��ȡlabel
            file.read(reinterpret_cast<char*>(&img.label), 1);

            // ��ȡ�������ɫͨ��
            file.read(reinterpret_cast<char*>(buffer.data()), 1024);
            torch::Tensor red_channel = torch::tensor(buffer, torch::kUInt8).clone();
            red_channel = red_channel.unsqueeze(0).view({ 1, 32, 32 });

            // ��ȡ��������ɫͨ��
            file.read(reinterpret_cast<char*>(buffer.data()), 1024);
            torch::Tensor green_channel = torch::tensor(buffer, torch::kUInt8).clone();
            green_channel = green_channel.unsqueeze(0).view({ 1, 32, 32 });

            // ��ȡ��������ɫͨ��
            file.read(reinterpret_cast<char*>(buffer.data()), 1024);
            torch::Tensor blue_channel = torch::tensor(buffer, torch::kUInt8).clone();
            blue_channel = blue_channel.unsqueeze(0).view({ 1, 32, 32 });

            // ʹ��torch::cat����ƴ������ͨ��
            img.tensor = torch::cat({ red_channel, green_channel, blue_channel }, 0);

            images.push_back(std::move(img));
        }

    }
    

    return images;
}
