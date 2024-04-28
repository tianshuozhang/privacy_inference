#include "vgg11.h"


vgg11::vgg11()
    : layer1(register_module("layer1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1)))),
    layer4(register_module("layer4", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)))),
    layer7(register_module("layer7", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)))),
    layer9(register_module("layer9", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)))),
    layer12(register_module("layer12", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)))),
    layer14(register_module("layer14", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)))),
    layer17(register_module("layer17", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)))),
    layer19(register_module("layer19", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)))),

    layer3(register_module("layer3", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2).padding(0).dilation(1).ceil_mode(false)))),
    layer6(register_module("layer6", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2).padding(0).dilation(1).ceil_mode(false)))),
    layer11(register_module("layer11", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2).padding(0).dilation(1).ceil_mode(false)))),
    layer16(register_module("layer16", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2).padding(0).dilation(1).ceil_mode(false)))),
    layer21(register_module("layer21", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2).padding(0).dilation(1).ceil_mode(false)))),
    
    layer2(register_module("layer2", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)))),
    layer5(register_module("layer5", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)))),
    layer8(register_module("layer8", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)))),
    layer10(register_module("layer10", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)))),
    layer13(register_module("layer13", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)))),
    layer15(register_module("layer15", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)))),
    layer18(register_module("layer18", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)))),
    layer20(register_module("layer20", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)))),
    layer24(register_module("layer24", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)))),
    layer27(register_module("layer27", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)))),

    layer23(register_module("layer23", torch::nn::Linear(torch::nn::LinearOptions(512,512).bias(true)))),
    layer26(register_module("layer26", torch::nn::Linear(torch::nn::LinearOptions(512, 512).bias(true)))),
    layer28(register_module("layer28", torch::nn::Linear(torch::nn::LinearOptions(512, 10).bias(true)))),

    layer22(register_module("layer22", torch::nn::Dropout(torch::nn::DropoutOptions(0.5).inplace(true)))),
    layer25(register_module("layer25", torch::nn::Dropout(torch::nn::DropoutOptions(0.5).inplace(true))))


     {
    filename = "vgg11.txt";
    run_thread = true;

    std::thread serverThread(std::bind(&vgg11::startServer, this));
    std::thread clientThread(std::bind(&vgg11::startClient, this, 5010));
    std::thread clientThread_1(std::bind(&vgg11::startClient, this, 5020));
    clientThread.detach();
    clientThread_1.detach();
    serverThread.detach();
    WSACleanup();
}

vgg11::~vgg11() {
    run_thread = false;
}

ModuleInfo vgg11::GetModuleByName(const std::string& name) {
    ModuleInfo module_info;
    for (auto& pair : this->named_children()) {
        if (pair.key() == name) {
            module_info.ptr = pair.value();
            // Checking the type and setting it
            if (module_info.ptr->as<torch::nn::Conv2d>() != nullptr) {
                module_info.type = "torch::nn::Conv2d";
            }
            else if (module_info.ptr->as<torch::nn::Linear>() != nullptr) {
                module_info.type = "torch::nn::Linear";
            }
            else if (module_info.ptr->as<torch::nn::ReLU>() != nullptr) {
                module_info.type = "torch::nn::ReLU";
            }
            else if (module_info.ptr->as<torch::nn::BatchNorm2d>() != nullptr) {
                module_info.type = "torch::nn::BatchNorm2d";
            }
            else if (module_info.ptr->as<torch::nn::MaxPool2d>() != nullptr) {
                module_info.type = "torch::nn::MaxPool2d";
            }
            else if (module_info.ptr->as<torch::nn::Dropout>() != nullptr) {
                module_info.type = "torch::nn::Dropout";
            }
            else {
                module_info.type = "unknown";
            }
            return module_info;
        }
    }
    // Default return with nullptr and "unknown"
    module_info.ptr = nullptr;
    module_info.type = "unknown";
    return module_info;
}

torch::Tensor vgg11::layer_forward(torch::Tensor x) {

    while (in.size() > 0) Sleep(10);
    mu.lock();
    in.push_back(x);
    mu.unlock();
    
    while (out.size() == 0) Sleep(10);
    mu.lock();
    x = out.front();
    out.erase(out.begin());
    mu.unlock();
    return x;
}

void vgg11::startServer() {
    // Defining length variables

    // Defining send buffer and receive buffer
    initialization();
    // Defining server sockets and receive request sockets
    SOCKET s_server;
    SOCKET s_accept;
    // Server address and client address
    SOCKET s_server_1;
    SOCKET s_accept_1;
    std::thread thread1(connectAndListen, std::ref(s_server), std::ref(s_accept), 5010);
    std::thread thread2(connectAndListen, std::ref(s_server_1), std::ref(s_accept_1), 5020);
    thread1.join();
    thread2.join();

    while (run_thread) {
        if (in.size() > 0) {
            mu.lock();
            auto data = in.front();
            mu.unlock();
            auto start = std::chrono::high_resolution_clock::now();
            auto input = torch::randn_like(data);
            auto input_1 = data - input;

            std::vector<std::string> operations;
            std::string line;
            std::ifstream operationsFile(filename);
            if (operationsFile.is_open())
            {
                while (getline(operationsFile, line))
                {
                    operations.push_back(line);
                }
                operationsFile.close();
            }
            std::cout << "读取完成\n";
            for (auto iter = operations.begin(); iter != operations.end(); ++iter)
            {

                std::thread work(workserver, std::ref(s_accept), std::ref(input));
                std::thread work_1(workserver, std::ref(s_accept_1), std::ref(input_1));
                work.join();
                work_1.join();
                data = input + input_1;
                while ((*(iter)).substr(0, 1) == "1" && iter != operations.end()) ++iter;
                while (iter != operations.end()) {
                    std::string identifier = (*iter).substr(0, 1);
                    if (identifier == "0") {
                        std::string func = (*iter).substr(3);
                    
                       
                        if (func == "out = out;") ;
                        else {
                            std::regex re("layer\\d+.*?\\(");
                            std::smatch match;
                            if (std::regex_search(func, match, re))
                            {
                                std::string numberString = match[0];
                                numberString = numberString.substr(0, numberString.size() - 1);
                                ModuleInfo module_info = this->GetModuleByName(numberString);
                                // 别忘了你需要先设置module_info指针和类型字符串
                                if (module_info.type == "torch::nn::ReLU")
                                {

                                    auto linear_module = module_info.ptr->as<torch::nn::ReLU>();

                                    data = linear_module->forward(data);

                                    // now you can use `linear_module` which is `torch::nn::Linear*`
                                }
                                else if (module_info.type == "torch::nn::MaxPool2d")
                                {
                                    auto conv_module = module_info.ptr->as<torch::nn::MaxPool2d>();
                                    if (conv_module != nullptr)
                                    {

                                        data = (conv_module)->forward(data);
                                    }
                                    // now you can use `conv_module` which is `torch::nn::Conv2d*`
                                }
                                else if (module_info.type == "torch::nn::Dropout")
                                {
                                    auto conv_module = module_info.ptr->as<torch::nn::Dropout>();
                                    if (conv_module != nullptr)
                                    {

                                        data = (conv_module)->forward(data);
                                    }
                                    // now you can use `conv_module` which is `torch::nn::BatchNorm2d*`
                                }
                            }
                        }
                    
                        ++iter;
                    }
                    else {

                        break;
                    };
                }
                if (iter == operations.end()) break;
                input = torch::randn_like(data);
                input_1 = data - input;
            }
            //计时
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            std::cout << "Elapsed time: " << elapsed.count() << " s\n";
            std::ofstream file("./output.txt", std::ios::app);  // 以追加模式打开文件
            if (!file) {  // 检查文件是否成功打开
                std::cerr << "Unable to open file.";
                return;  // 返回非零值表示程序异常
            }
            file << "Elapsed time distribute compute: " << elapsed.count() << " s\n";
            file.close();  // 关闭文件

            mu.lock();
            out.push_back(data);
            in.erase(in.begin());
            mu.unlock();

        }
    }
    // Close sockets
    closesocket(s_server);
    closesocket(s_accept);
}

void vgg11::startClient(int port)
{
    // Defined length variable

    // Defined send buffer and receive buffer
    // Defined server socket, and receiving request socket
    SOCKET s_server;
    // Server address and client address
    SOCKADDR_IN server_addr;


    Sleep(1000); // Sleep for 1 seconds to make sure server is ready
    // Fill in server information
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
    server_addr.sin_port = htons(port);

    // Create Socket
    s_server = socket(AF_INET, SOCK_STREAM, 0);
    if (connect(s_server, (SOCKADDR*)&server_addr, sizeof(SOCKADDR)) == SOCKET_ERROR) {
        std::cout << "Client: Server connection failed!" << std::endl;
    }
    else {
        std::cout << "Client: Server connection succeeded!\n";
    }
    // Send and receive data
    while (run_thread)
    {
        torch::Tensor data;
        std::vector<std::string> operations;
        std::string line;

        std::ifstream operationsFile(filename);
        if (operationsFile.is_open())
        {
            while (getline(operationsFile, line))
            {
                operations.push_back(line);
            }
            operationsFile.close();
        }
        for (auto iter = operations.begin(); iter != operations.end(); ++iter)
        {

            recvData(s_server, data);
            while (1)
            {
                std::string identifier = (*iter).substr(0, 1);
                if (identifier == "1")
                {
                    std::string func = (*iter).substr(3);
                    if (func.find("view") != std::string::npos)
                    {
                        // extract argument values
                        std::smatch match;
                        std::regex argument_regex("\\{([^}]*)\\}");
                        std::regex_search(func, match, argument_regex);
                        std::string argument_string = match[1];
                        std::stringstream ss(argument_string);
                        std::istream_iterator<std::string> begin(ss);
                        std::istream_iterator<std::string> end;
                        std::vector<std::string> arguments(begin, end);
                        std::vector<int64_t> dimensions;
                        for (auto& s : arguments)
                        {
                            dimensions.push_back(stoll(s));
                        }
                        data = data.view(torch::IntList(dimensions));
                    }
                    else
                    {
                        std::regex re("layer\\d+.*?\\(");
                        std::smatch match;
                        if (std::regex_search(func, match, re))
                        {
                            std::string numberString = match[0];
                            numberString = numberString.substr(0, numberString.size() - 1);
                            ModuleInfo module_info = this->GetModuleByName(numberString);
                            // 别忘了你需要先设置module_info指针和类型字符串
                            if (module_info.type == "torch::nn::Linear")
                            {

                                auto linear_module = module_info.ptr->as<torch::nn::Linear>();

                                data = linear_module->forward(data);

                                // now you can use `linear_module` which is `torch::nn::Linear*`
                            }
                            else if (module_info.type == "torch::nn::Conv2d")
                            {
                                auto conv_module = module_info.ptr->as<torch::nn::Conv2d>();
                                if (conv_module != nullptr)
                                {

                                    data = (conv_module)->forward(data);
                                }
                                // now you can use `conv_module` which is `torch::nn::Conv2d*`
                            }
                            else if (module_info.type == "torch::nn::BatchNorm2d")
                            {
                                auto conv_module = module_info.ptr->as<torch::nn::BatchNorm2d>();
                                if (conv_module != nullptr)
                                {

                                    data = (conv_module)->forward(data);
                                }
                                // now you can use `conv_module` which is `torch::nn::BatchNorm2d*`
                            }
                        }

                    }
                    ++iter;

                }
                else {
                    while ((iter + 1) != operations.end() && (*(iter + 1)).substr(0, 1) == "0")
                    {
                        ++iter;
                    }
                    break;

                };
            }
            sendData(s_server, data);
        }
    }

    // Close socket
    closesocket(s_server);
}

torch::Tensor vgg11::forward(torch::Tensor x){

    auto out = layer1(x);
    out = layer2(out);
    out = layer3(out);
    out = layer4(out);
    out = layer5(out);
    out = layer6(out);
    out = layer7(out);
    out = layer8(out);
    out = layer9(out);
    out = layer10(out);
    out = layer11(out);
    out = layer12(out);
    out = layer13(out);
    out = layer14(out);
    out = layer15(out);
    out = layer16(out);
    out = layer17(out);
    out = layer18(out);
    out = layer19(out);
    out = layer20(out);
    out = layer21(out);
    out = out.view({ -1, 512 });
    out = layer22(out);
    out = layer23(out);
    out = layer24(out);
    out = layer25(out);
    out = layer26(out);
    out = layer27(out);
    out = layer28(out);
    return out;
}