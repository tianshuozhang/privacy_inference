#include"Resnet34.h"
Resnet34::Resnet34()
    : layer1(register_module("layer1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1).bias(false)))),
    layer3(register_module("layer3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)))),
    layer5(register_module("layer5", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)))),
    layer8(register_module("layer8", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)))),
    layer10(register_module("layer10", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)))),
    layer13(register_module("layer13", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)))),
    layer15(register_module("layer15", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)))),
    layer18(register_module("layer18", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1).bias(false)))),
    layer20(register_module("layer20", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(false)))),
    layer22_shortcut(register_module("layer22_shortcut", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 1).stride(2).bias(false)))),
    layer24(register_module("layer24", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(false)))),
    layer26(register_module("layer26", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(false)))),
    layer29(register_module("layer29", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(false)))),
    layer31(register_module("layer31", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(false)))),
    layer34(register_module("layer34", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(false)))),
    layer36(register_module("layer36", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(false)))),
    layer39(register_module("layer39", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1).bias(false)))),
    layer41(register_module("layer41", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)))),
    layer43_shortcut(register_module("layer43_shortcut", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 1).stride(2).bias(false)))),
    layer45(register_module("layer45", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)))),
    layer47(register_module("layer47", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)))),
    layer50(register_module("layer50", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)))),
    layer52(register_module("layer52", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)))),
    layer55(register_module("layer55", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)))),
    layer57(register_module("layer57", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)))),
    layer60(register_module("layer60", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)))),
    layer62(register_module("layer62", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)))),
    layer65(register_module("layer65", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)))),
    layer67(register_module("layer67", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1).bias(false)))),
    layer70(register_module("layer70", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(2).padding(1).bias(false)))),
    layer72(register_module("layer72", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1).bias(false)))),
    layer74_shortcut(register_module("layer74_shortcut", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 1).stride(2).bias(false)))),
    layer76(register_module("layer76", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1).bias(false)))),
    layer78(register_module("layer78", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1).bias(false)))),
    layer81(register_module("layer81", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1).bias(false)))),
    layer83(register_module("layer83", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1).bias(false)))),






    layer2(register_module("layer2", torch::nn::BatchNorm2d(64))),
    layer4(register_module("layer4", torch::nn::BatchNorm2d(64))),
    layer6(register_module("layer6", torch::nn::BatchNorm2d(64))),
    layer9(register_module("layer9", torch::nn::BatchNorm2d(64))),
    layer11(register_module("layer11", torch::nn::BatchNorm2d(64))),
    layer14(register_module("layer14", torch::nn::BatchNorm2d(64))),
    layer16(register_module("layer16", torch::nn::BatchNorm2d(64))),
    layer19(register_module("layer19", torch::nn::BatchNorm2d(128))),
    layer21(register_module("layer21", torch::nn::BatchNorm2d(128))),
    layer23_shortcut(register_module("layer23_shortcut", torch::nn::BatchNorm2d(128))),
    layer25(register_module("layer25", torch::nn::BatchNorm2d(128))),
    layer27(register_module("layer27", torch::nn::BatchNorm2d(128))),

    layer30(register_module("layer30", torch::nn::BatchNorm2d(128))),
    layer32(register_module("layer32", torch::nn::BatchNorm2d(128))),
    layer35(register_module("layer35", torch::nn::BatchNorm2d(128))),
    layer37(register_module("layer37", torch::nn::BatchNorm2d(128))),
    layer40(register_module("layer40", torch::nn::BatchNorm2d(256))),
    layer42(register_module("layer42", torch::nn::BatchNorm2d(256))),
    layer44_shortcut(register_module("layer44_shortcut", torch::nn::BatchNorm2d(256))),
    layer46(register_module("layer46", torch::nn::BatchNorm2d(256))),
    layer48(register_module("layer48", torch::nn::BatchNorm2d(256))),
    layer51(register_module("layer51", torch::nn::BatchNorm2d(256))),
    layer53(register_module("layer53", torch::nn::BatchNorm2d(256))),
    layer56(register_module("layer56", torch::nn::BatchNorm2d(256))),
    layer58(register_module("layer58", torch::nn::BatchNorm2d(256))),
    layer61(register_module("layer61", torch::nn::BatchNorm2d(256))),
    layer63(register_module("layer63", torch::nn::BatchNorm2d(256))),
    layer66(register_module("layer66", torch::nn::BatchNorm2d(256))),
    layer68(register_module("layer68", torch::nn::BatchNorm2d(256))),
    layer71(register_module("layer71", torch::nn::BatchNorm2d(512))),
    layer73(register_module("layer73", torch::nn::BatchNorm2d(512))),
    layer75_shortcut(register_module("layer75_shortcut", torch::nn::BatchNorm2d(512))),
    layer77(register_module("layer77", torch::nn::BatchNorm2d(512))),
    layer79(register_module("layer79", torch::nn::BatchNorm2d(512))),
    layer82(register_module("layer82", torch::nn::BatchNorm2d(512))),
    layer84(register_module("layer84", torch::nn::BatchNorm2d(512))),

    layer86(register_module("layer86", torch::nn::Linear(512, 10))),
    act(register_module("act", torch::nn::ReLU())) {
    filename = "Resnet34.txt";
    run_thread = true;

    std::thread serverThread(std::bind(&Resnet34::startServer, this));
    std::thread clientThread(std::bind(&Resnet34::startClient, this, 5010));
    std::thread clientThread_1(std::bind(&Resnet34::startClient, this, 5020));
    clientThread.detach();
    clientThread_1.detach();
    serverThread.detach();
    WSACleanup();
}

Resnet34::~Resnet34() {
    run_thread = false;
}

ModuleInfo Resnet34::GetModuleByName(const std::string& name) {
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

torch::Tensor Resnet34::layer_forward(torch::Tensor x) {

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
void Resnet34::startServer()
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
                                if (module_info.type == "torch::nn::ReLU")
                                {

                                    auto linear_module = module_info.ptr->as<torch::nn::ReLU>();

                                    data = linear_module->forward(data);

                                    // now you can use `linear_module` which is `torch::nn::Linear*`
                                }
                                else if (module_info.type == "torch::nn::BatchNorm2d")
                                {
                                    auto conv_module = module_info.ptr->as<torch::nn::BatchNorm2d>();
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
void Resnet34::startClient(int port) {
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

torch::Tensor Resnet34::forward(torch::Tensor x) {

    auto data = layer1->forward(x);
    data = layer2->forward(data);
    data = act(data);
    x = data.clone();
    data = layer3(data);
    data = layer4(data);
    data = act(data);
    data = layer5(data);
    data = layer6(data);
    data += x;

    data = act(data);
    x = data.clone();
    data = layer8(x);
    data = layer9(data);
    data = act(data);
    data = layer10(data);
    data = layer11(data);
    data += x;

    data = act(data);
    x = data.clone();
    data = layer13(x);
    data = layer14(data);
    data = act(data);
    data = layer15(data);
    data = layer16(data);
    data += x;

    data = act(data);
    x = data.clone();
    data = layer18(x);
    data = layer19(data);
    data = act(data);
    data = layer20(data);
    data = layer21(data);
    x = layer22_shortcut(x);
    x = layer23_shortcut(x);
    data += x;

    data = act(data);
    x = data.clone();
    data = layer24(x);
    data = layer25(data);
    data = act(data);
    data = layer26(data);
    data = layer27(data);
    data += x;

    data = act(data);
    x = data.clone();
    data = layer29(x);
    data = layer30(data);
    data = act(data);
    data = layer31(data);
    data = layer32(data);
    data += x;
    data = act(data);
    x = data.clone();
    data = layer34(x);
    data = layer35(data);
    data = act(data);
    data = layer36(data);
    data = layer37(data);
    data += x;

    data = act(data);
    x = data.clone();
    data = layer39(x);
    data = layer40(data);
    data = act(data);
    data = layer41(data);
    data = layer42(data);
    x = layer43_shortcut(x);
    x = layer44_shortcut(x);
    data += x;
    data = act(data);
    x = data.clone();
    data = layer45(x);
    data = layer46(data);
    data = act(data);
    data = layer47(data);
    data = layer48(data);
    data += x;
    data = act(data);
    x = data.clone();
    data = layer50(x);
    data = layer51(data);
    data = act(data);
    data = layer52(data);
    data = layer53(data);
    data += x;
    data = act(data);
    x = data.clone();
    data = layer55(x);
    data = layer56(data);
    data = act(data);
    data = layer57(data);
    data = layer58(data);
    data += x;
    data = act(data);

    x = data.clone();
    data = layer60(x);
    data = layer61(data);
    data = act(data);
    data = layer62(data);
    data = layer63(data);
    data += x;
    data = act(data);
    x = data.clone();
    data = layer65(x);
    data = layer66(data);
    data = act(data);
    data = layer67(data);
    data = layer68(data);
    data += x;
    data = act(data);

    x = data.clone();
    data = layer70(x);
    data = layer71(data);
    data = act(data);
    data = layer72(data);
    data = layer73(data);
    x = layer74_shortcut(x);
    x = layer75_shortcut(x);
    data += x;
    data = act(data);
    x = data.clone();
    data = layer76(x);
    data = layer77(data);
    data = act(data);
    data = layer78(data);
    data = layer79(data);
    data += x;
    data = act(data);
    x = data.clone();
    data = layer81(x);
    data = layer82(data);
    data = act(data);
    data = layer83(data);
    data = layer84(data);
    data += x;
    data = act(data);
    data = torch::nn::functional::avg_pool2d(data, torch::nn::functional::AvgPool2dFuncOptions(4));
    data = data.view({ -1, 512 });
    data = layer86(data);
    return data;
}