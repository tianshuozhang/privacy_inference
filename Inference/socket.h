#ifndef SOCKET_H
#define SOCKET_H
#include<winsock.h>
#include <thread>
#include <torch/torch.h>
#include <iostream>
#include<sstream>
#include <chrono>
#include <regex>
void initialization();
void connectAndListen(SOCKET& s_server, SOCKET& s_accept, int port);
void workserver(SOCKET& s_accept, torch::Tensor& data);
void sendData(SOCKET& s_server,  torch::Tensor& data);
void recvData(SOCKET& s_server,  torch::Tensor& data);
#endif