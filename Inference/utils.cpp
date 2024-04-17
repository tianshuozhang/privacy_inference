#include"utils.h"
#include <fstream>

#pragma comment(lib,"ws2_32.lib")
float test_acc(Base &module,const std::string &datapath ,std::string tag) {
    if (tag == "mnist") {
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
        module.eval();
        // 遍历数据加载器中的批次
        for (auto& batch : *dataloader) {
            // 获取输入和标签
            std::vector<torch::Tensor> data_vec, target_vec;
            for (const auto& example : batch) {
                data_vec.push_back(example.data);
                target_vec.push_back(example.target);
            }
            auto data = torch::stack(data_vec).to(device);
            auto target = torch::stack(target_vec).to(device);


            // 前向传播
            auto start = std::chrono::high_resolution_clock::now();
            torch::Tensor output = module.forward({ data });
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            std::cout << "Elapsed time: " << elapsed.count() << " s\n";
            // 计算预测准确率
            auto predicted = output.argmax(1);
            int64_t correct = predicted.eq(target).sum().item<int64_t>();

            totalCorrect += correct;
            totalSamples += data.size(0);
            break;

        }

        float accuracy = static_cast<float>(totalCorrect) / totalSamples * 100.0;
        return accuracy;
    }
    else if (tag == "cifar10") {
        auto custom_dataset = CustomDataset(datapath).map(torch::data::transforms::Stack<>());
        auto dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(custom_dataset),
            torch::data::DataLoaderOptions().batch_size(64)
        );
        int totalCorrect = 0;
        int totalSamples = 0;
        torch::Device device = torch::kCPU;
        
        if (torch::cuda::is_available()) {
            std::cout << "CUDA available! Training on GPU." << std::endl;
            device = torch::kCUDA;
        }
        module.to(device);
        // 遍历数据加载器中的批次
        for (auto& batch : *dataloader) {
            // 获取输入和标签
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            // 前向传播
            auto start = std::chrono::high_resolution_clock::now();
            torch::Tensor output = module.forward({ data });
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            std::cout << "Elapsed time: " << elapsed.count() << " s\n";
            // 计算预测准确率
            auto predicted = output.argmax(1);
            int64_t correct = predicted.eq(target).sum().item<int64_t>();

            totalCorrect += correct;
            totalSamples += data.size(0);
           

        }

        float accuracy = static_cast<float>(totalCorrect) / totalSamples * 100.0;
        return accuracy;
    }
    

}

void layer_weight_extraction(Base &module, const std::string path) {
    // 创建一个 std::ofstream 对象，你可以在括号中输入你所想要创建的文件的路径和文件名。
    
    auto loadmodule = torch::jit::load(path);
    loadmodule.eval();
    std::cout << "model load" << "\n";
    for (const auto& parameter : loadmodule.named_parameters()) {
        auto& name = parameter.name;
        auto& tensor = parameter.value;
        std::cout << name << "\n";
        // 根据名字在你的新模型中找到相应的参数，然后设置值
        // 这需要你的新模型的参数名与旧模型的参数名完全匹配
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
    
    // 遍历数据加载器中的批次
    for (auto& batch : *dataloader) {
        // 获取输入和标签
        std::vector<torch::Tensor> data_vec, target_vec;
        for (const auto& example : batch) {
            data_vec.push_back(example.data);
            target_vec.push_back(example.target);
        }

        auto data = torch::stack(data_vec);
        auto target = torch::stack(target_vec);
        // 这里你可以处理整个批次的数据和标签...
        
        // 前向传播
        
        torch::Tensor output = module.layer_forward(data);
        

        // 计算预测准确率
        auto predicted = output.argmax(1);
        int64_t correct = predicted.eq(target).sum().item<int64_t>();
        totalCorrect += correct;
        totalSamples += data.size(0);
        break;

    }
    float accuracy = static_cast<float>(totalCorrect) / totalSamples * 100.0;
    return accuracy;
}

void weight(Base& module, const std::string path) {
    // 创建一个 std::ofstream 对象，你可以在括号中输入你所想要创建的文件的路径和文件名。

    auto loadmodule = torch::jit::load(path);
    loadmodule.eval();
    std::cout << "model compare" << "\n";
    for (const auto& parameter : loadmodule.named_parameters()) {
        auto& name = parameter.name;
        auto& tensor = parameter.value;
        std::cout << name << "\n";
        for (auto& p : module.named_parameters()) {
            if (name==p.key())
                std::cout << torch::all(torch::eq(tensor, p.value())).item<bool>() << "\n";
        }
    }
    
    return;
}