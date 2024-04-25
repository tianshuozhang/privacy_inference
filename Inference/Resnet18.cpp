#include"Resnet18.h"
Resnet18::Resnet18()
    : layer1(register_module("layer1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1).bias(false)))),
    layer3(register_module("layer3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)))),
    layer5(register_module("layer5", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)))),
    layer8(register_module("layer8", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)))),
    layer10(register_module("layer10", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)))),
    layer13(register_module("layer13", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1).bias(false)))),
    layer15(register_module("layer15", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(false)))),
    layer17_shortcut(register_module("layer17_shortcut", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 1).stride(2).bias(false)))),
    layer19(register_module("layer19", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(false)))),
    layer21(register_module("layer21", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(false)))),
    layer24(register_module("layer24", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1).bias(false)))),
    layer26(register_module("layer26", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)))),
    layer28_shortcut(register_module("layer28_shortcut", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 1).stride(2).bias(false)))),
    layer30(register_module("layer30", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)))),
    layer32(register_module("layer32", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)))),
    layer35(register_module("layer35", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(2).padding(1).bias(false)))),
    layer37(register_module("layer37", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1).bias(false)))),
    layer39_shortcut(register_module("layer39_shortcut", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 1).stride(2).bias(false)))),
    layer41(register_module("layer41", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1).bias(false)))),
    layer43(register_module("layer43", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1).bias(false)))),

    layer2(register_module("layer2", torch::nn::BatchNorm2d(64))),
    layer4(register_module("layer4", torch::nn::BatchNorm2d(64))),
    layer6(register_module("layer6", torch::nn::BatchNorm2d(64))),
    layer9(register_module("layer9", torch::nn::BatchNorm2d(64))),
    layer11(register_module("layer11", torch::nn::BatchNorm2d(64))),
    layer14(register_module("layer14", torch::nn::BatchNorm2d(128))),
    layer16(register_module("layer16", torch::nn::BatchNorm2d(128))),
    layer18_shortcut(register_module("layer18_shortcut", torch::nn::BatchNorm2d(128))),
    layer20(register_module("layer20", torch::nn::BatchNorm2d(128))),
    layer22(register_module("layer22", torch::nn::BatchNorm2d(128))),
    layer25(register_module("layer25", torch::nn::BatchNorm2d(256))),
    layer27(register_module("layer27", torch::nn::BatchNorm2d(256))),
    layer29_shortcut(register_module("layer29_shortcut", torch::nn::BatchNorm2d(256))),
    layer31(register_module("layer31", torch::nn::BatchNorm2d(256))),
    layer33(register_module("layer33", torch::nn::BatchNorm2d(256))),
    layer36(register_module("layer36", torch::nn::BatchNorm2d(512))),
    layer38(register_module("layer38", torch::nn::BatchNorm2d(512))),
    layer40_shortcut(register_module("layer40_shortcut", torch::nn::BatchNorm2d(512))),
    layer42(register_module("layer42", torch::nn::BatchNorm2d(512))),
    layer44(register_module("layer44", torch::nn::BatchNorm2d(512))),
    layer46(register_module("layer46", torch::nn::Linear(512,10))),
    act(register_module("act", torch::nn::ReLU())){
    filename = "Resnet18.txt";
    run_thread = true;;
    std::thread serverThread(std::bind(&Resnet18::startServer, this));
    std::thread clientThread(std::bind(&Resnet18::startClient, this, 5010));
    std::thread clientThread_1(std::bind(&Resnet18::startClient, this, 5020));
    clientThread.detach();
    clientThread_1.detach();
    serverThread.detach();
    WSACleanup();
}

Resnet18::~Resnet18() {
    run_thread = false;
}

ModuleInfo Resnet18::GetModuleByName(const std::string& name) {
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

torch::Tensor Resnet18::layer_forward(torch::Tensor x) {
    
    while (in.size() > 0) Sleep(10);
    mu.lock();
    in.push_back(x);
    mu.unlock();
    
    while (out.size()==0) Sleep(10);
    mu.lock();
    x=out.front();
    out.erase(out.begin());
    mu.unlock();
    return x;
}

void Resnet18::startServer()
{
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
            torch::Tensor x;
            bool send_x = false;
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
            std::cout << "��ȡ���\n";
            for (auto iter = operations.begin(); iter != operations.end(); ++iter)
            {

                std::thread work(workserver, std::ref(s_accept), std::ref(input));
                std::thread work_1(workserver, std::ref(s_accept_1), std::ref(input_1));
                work.join();
                work_1.join();
                if (send_x) {
                    x = input + input_1;
                    send_x = false;
                }
                else data = input + input_1;
                while ((*(iter)).substr(0, 1) == "1" && iter != operations.end()) ++iter;
                while (iter != operations.end()) {
                    std::string identifier = (*iter).substr(0, 1);
                    if (identifier == "0") {
                        std::string func = (*iter).substr(3);
                        if (func == "out = act(out);") {
                            ModuleInfo module_info = this->GetModuleByName("act");
                            // ����������Ҫ������module_infoָ��������ַ���
                            if (module_info.type == "torch::nn::ReLU")
                            {
                                auto act_module = module_info.ptr->as<torch::nn::ReLU>();

                                data = act_module->forward(data);
                                // now you can use `linear_module` which is `torch::nn::Linear*`
                            }

                        }
                        else if (func == "out = torch::nn::functional::avg_pool2d(out, torch::nn::functional::AvgPool2dFuncOptions(4));") {

                            data = torch::nn::functional::avg_pool2d(data, torch::nn::functional::AvgPool2dFuncOptions(4));
                        }
                        else if (func == "out += x;")
                        {
                            data = data + x;
                        }
                        else if (func == "x = out.clone();")
                        {
                            x = data.clone();
                        }
                        else if (func == "send x;")
                        {
                            send_x = true;
                        }
                        else if (func == "out = out;")
                        {
                            ;
                        }
                        ++iter;
                    }
                    else {

                        break;
                    };
                }
                if (iter == operations.end()) break;
                if (send_x) {
                    input = torch::randn_like(x);
                    input_1 = x - input;
                }
                else {
                    input = torch::randn_like(data);
                    input_1 = data - input;
                }

            }
            //��ʱ
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            std::cout << "Elapsed time: " << elapsed.count() << " s\n";
            std::ofstream file("./output.txt", std::ios::app);  // ��׷��ģʽ���ļ�
            if (!file) {  // ����ļ��Ƿ�ɹ���
                std::cerr << "Unable to open file.";
                return;  // ���ط���ֵ��ʾ�����쳣
            }
            file << "Elapsed time distribute compute: " << elapsed.count() << " s\n";
            file.close();  // �ر��ļ�
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
void Resnet18::startClient(int port) {
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
    while (run_thread) {
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
        for (auto iter = operations.begin(); iter != operations.end(); ++iter) {

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
                            // ����������Ҫ������module_infoָ��������ַ���
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

torch::Tensor Resnet18::forward(torch::Tensor x) {
    
    auto out = layer1->forward(x);
    out = layer2->forward(out);
    out = act(out);
    x = out.clone();
    out = layer3(out);
    out = layer4(out);
    out = act(out);
    out = layer5(out);
    out = layer6(out);
    out += x;
    out = act(out);
    x = out.clone();
    out = layer8(x);
    out = layer9(out);
    out = act(out);
    out = layer10(out);
    out = layer11(out);
    out += x;
    out = act(out);
    x = out.clone();
    out = layer13(x);
    out = layer14(out);
    out = act(out);
    out = layer15(out);
    out = layer16(out);
    x = layer17_shortcut(x);
    out += layer18_shortcut(x);
    out = act(out);
    x = out.clone();
    out = layer19(x);
    out = layer20(out);
    out = act(out);
    out = layer21(out);
    out = layer22(out);
    out +=x;
    out = act(out);
    x = out.clone();
    out = layer24(x);
    out = layer25(out);
    out = act(out);
    out = layer26(out);
    out = layer27(out);
    x = layer28_shortcut(x);
    out += layer29_shortcut(x);
    out = act(out);
    x = out.clone();
    out = layer30(x);
    out = layer31(out);
    out = act(out);
    out = layer32(out);
    out = layer33(out);
    out += x;
    out = act(out);
    x = out.clone();
    out = layer35(x);
    out = layer36(out);
    out = act(out);
    out = layer37(out);
    out = layer38(out);
    x = layer39_shortcut(x);
    out += layer40_shortcut(x);
    out = act(out);
    x = out.clone();
    out = layer41(x);
    out = layer42(out);
    out = act(out);
    out = layer43(out);
    out = layer44(out);
    out += (x);
    out = act(out);
    out = torch::nn::functional::avg_pool2d(out, torch::nn::functional::AvgPool2dFuncOptions(4));
    out = out.view({ -1, 512 });
    out = layer46(out);
    return out;
}