#include"utils.h"
#pragma comment(lib,"ws2_32.lib")
float test_acc(Base &module,const std::string &datapath ) {
    auto dataset = torch::data::datasets::MNIST(datapath,
        torch::data::datasets::MNIST::Mode::kTest).map(torch::data::transforms::Normalize<>(0.5, 0.5));
    auto dataloader = torch::data::make_data_loader(dataset, torch::data::DataLoaderOptions().batch_size(64));
    int totalCorrect = 0;
    int totalSamples = 0;
    torch::Device device = torch::kCPU;
    
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    }
    module.to(device);
    // �������ݼ������е�����
    for (auto& batch : *dataloader) {
        // ��ȡ����ͱ�ǩ
        std::vector<torch::Tensor> data_vec, target_vec;
        for (const auto& example : batch) {
            data_vec.push_back(example.data);
            target_vec.push_back(example.target);
        }
        auto data = torch::stack(data_vec).to(device);
        auto target = torch::stack(target_vec).to(device);
        

        // ǰ�򴫲�
        auto start = std::chrono::high_resolution_clock::now();
        torch::Tensor output = module.forward({ data });
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "Elapsed time: " << elapsed.count() << " s\n";
        // ����Ԥ��׼ȷ��
        auto predicted = output.argmax(1);
        int64_t correct = predicted.eq(target).sum().item<int64_t>();

        totalCorrect += correct;
        totalSamples += data.size(0);
        break;

    }

    float accuracy = static_cast<float>(totalCorrect) / totalSamples * 100.0;
    return accuracy;

}

void layer_weight_extraction(Base &module, const std::string path) {
    
    auto loadmodule = torch::jit::load(path);
    for (const auto& parameter : loadmodule.named_parameters()) {
        auto& name = parameter.name;
        auto& tensor = parameter.value;
        // ���������������ģ�����ҵ���Ӧ�Ĳ�����Ȼ������ֵ
        // ����Ҫ�����ģ�͵Ĳ��������ģ�͵Ĳ�������ȫƥ��
        
        for (auto& p : module.named_parameters()) {
            if (name == p.key()) {
                p.value().data().copy_(tensor);
                break;
            }
        }
    }
    return ;
}

float test_layer(Base &module, const std::string & datapath) {
    auto dataset = torch::data::datasets::MNIST(datapath,
        torch::data::datasets::MNIST::Mode::kTest).map(torch::data::transforms::Normalize<>(0.5, 0.5));
    auto dataloader = torch::data::make_data_loader(dataset, torch::data::DataLoaderOptions().batch_size(64));
    int totalCorrect = 0;
    int totalSamples = 0;
    
    // �������ݼ������е�����
    for (auto& batch : *dataloader) {
        // ��ȡ����ͱ�ǩ
        std::vector<torch::Tensor> data_vec, target_vec;
        for (const auto& example : batch) {
            data_vec.push_back(example.data);
            target_vec.push_back(example.target);
        }

        auto data = torch::stack(data_vec);
        auto target = torch::stack(target_vec);
        // ��������Դ����������ε����ݺͱ�ǩ...
        
        // ǰ�򴫲�
        
        torch::Tensor output = module.layer_forward(data);
        
        /*
        auto start = std::chrono::high_resolution_clock::now();
        torch::Tensor output = module.layer1->forward({ data });
        std::cout << output.sizes() << std::endl;
        output = module.act(output);
        output = module.pool(output);
        std::cout << output.sizes() << std::endl;
        output = module.layer2->forward({ output });
        std::cout << output.sizes() << std::endl;
        output = module.act(output);
        output = module.pool(output);
        std::cout << output.sizes() << std::endl;
        output = output.view({ -1, 800 });
        output = module.layer3->forward({ output });
        output = module.act(output);
        output = module.layer4->forward({ output });
        output = module.act(output);
        
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "Elapsed time: " << elapsed.count() << " s\n";
        std::ofstream file("./output.txt", std::ios::app);  // ��׷��ģʽ���ļ�
        if (!file) {  // ����ļ��Ƿ�ɹ���
            std::cerr << "Unable to open file.";
            return 1;  // ���ط���ֵ��ʾ�����쳣
        }
        file << "Elapsed time directly compute: " << elapsed.count() << " s\n";
        file.close();  // �ر��ļ�
        */

        // ����Ԥ��׼ȷ��
        auto predicted = output.argmax(1);
        int64_t correct = predicted.eq(target).sum().item<int64_t>();

        totalCorrect += correct;
        totalSamples += data.size(0);
        break;

    }
    float accuracy = static_cast<float>(totalCorrect) / totalSamples * 100.0;
    return accuracy;
}