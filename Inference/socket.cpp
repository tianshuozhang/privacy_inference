#define NOMINMAX
#include"socket.h"
#pragma comment(lib,"ws2_32.lib")
void initialization() {
    WORD w_req = MAKEWORD(2, 2);
    WSADATA wsadata;
    int err;
    err = WSAStartup(w_req, &wsadata);
    if (err != 0) {
        std::cout << "Socket initialization failed!" << std::endl;
    }
    else {
        std::cout << "Socket initialization succeeded!" << std::endl;
    }

    if (LOBYTE(wsadata.wVersion) != 2 || HIBYTE(wsadata.wHighVersion) != 2) {
        std::cout << "Socket version mismatch!" << std::endl;
        WSACleanup();
    }
    else {
        std::cout << "Socket version match!" << std::endl;
    }
}

void recvData(SOCKET& s_server,  torch::Tensor& data) {
    int rec_size=0, recv_len=0;
    std::string temp_str="123";
    // receive data size
    recv_len = recv(s_server, (char*)&rec_size, sizeof(int), 0);
    if (recv_len < 0) {
        std::cout << "Reception failed!" << std::endl;
        return;
    }
    // receive data
    temp_str.resize(rec_size);
    recv_len = 0;
    while (recv_len < rec_size)
    {
        int result = recv(s_server, &temp_str[recv_len], rec_size - recv_len, 0);

        // 检查是否发生错误或连接被关闭
        if (result <= 0)
        {
            std::cout << "Reception failed!" << std::endl;
            return;
        }

        recv_len += result;
    }
    

    // load tensor from stringstream
    
    std::stringstream ss(temp_str);
    //mu.lock();
    torch::load(data, ss);
    //mu.unlock();
}

void sendData(SOCKET& s_server, torch::Tensor& data) {
    std::string temp_str;
    int send_size=0, send_len=0;
    std::stringstream ss;
    // save tensor to stringstream
    
    //mu.lock();
    torch::save(data, ss);
    //mu.unlock();
    
    // send data size
    temp_str = ss.str();
    send_size = temp_str.size();
    
    send_len = send(s_server, (char*)&send_size, sizeof(int), 0);
    if (send_len < 0) {
        std::cout << "Failed to send!" << std::endl;
        return;
    }
    // send data
    send_len = send(s_server, &temp_str[0], send_size, 0);
    if (send_len < 0) {
        std::cout << "Failed to send!" << std::endl;
        return;
    }
    
}

void connectAndListen(SOCKET &s_server, SOCKET &s_accept,int port) {
    int len = 0;
    SOCKADDR_IN server_addr;
    SOCKADDR_IN accept_addr;

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.S_un.S_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(port);

    s_server = socket(AF_INET, SOCK_STREAM, 0);
    if (bind(s_server, (SOCKADDR*)&server_addr, sizeof(SOCKADDR)) == SOCKET_ERROR) {
        std::cout << "Server: Socket binding failed on port " << port << std::endl;
        int iError = WSAGetLastError();
        std::cout << iError << std::endl;
        closesocket(s_server);
        WSACleanup();
        exit(0);
    }
    else {
        std::cout << "Server: Socket binding succeeded on port " << port << std::endl;
    }

    if (listen(s_server, SOMAXCONN) < 0) {
        std::cout << "Server: Failed to set listen status on port " << port << std::endl;
    }
    else {
        std::cout << "Server: Succeeded in setting listen status on port " << port << std::endl;
    }

    std::cout << "Server: Server is listening for connections on port " << port << ", please wait ..." << std::endl;
    len = sizeof(SOCKADDR);

    s_accept = accept(s_server, (SOCKADDR*)&accept_addr, &len);

    if (s_accept == SOCKET_ERROR) {
        std::cout << "Server: Connection failed on port " << port << std::endl;
    }
    else {
        std::cout << "Server: Connection established, ready to receive data on port " << port << std::endl;
    }
}

void workserver(SOCKET& s_accept, torch::Tensor& data) {
    sendData(s_accept, data);
    recvData(s_accept, data);
}