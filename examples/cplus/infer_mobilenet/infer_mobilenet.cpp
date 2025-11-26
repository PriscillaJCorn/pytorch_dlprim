#include <iostream>
#include <dlfcn.h>
#include <torch/torch.h>
#include <torch/script.h> 



torch::Device device(torch::kPrivateUse1, 0); 


void load_device() {
    const char* lib_path = "<path-to-pt_ocl.so>";
    /* load dynamic library */
    void* handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        std::cerr << "Failed to load " << lib_path << ": " << dlerror() << std::endl;
        exit(1);
    }
    std::cout << "Dynamic library loaded successfully: " << lib_path << std::endl;
}



/* create a Mnist net */
struct Net : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    Net() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).stride(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1)));
        fc1 = register_module("fc1", torch::nn::Linear(9216, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }

    /* forward */
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = x.view({x.size(0), -1}); // flatten
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return torch::log_softmax(x, /*dim=*/1);
    }
}; 


void infer_net() {
    /* load net */
    Net net;
    net.to(device);
    std::cout << "net : " << net << std::endl;
    std::cout << "Model moved to device: " << device.str() << std::endl;



    /* create dummy input */
    torch::Tensor input = torch::randn({1, 1, 28, 28}).to(device);
    std::cout << "input : " << input << std::endl;

    /* load scripted module */
    std::string scripted_model = "<path-to-mnist_cnn-scripted.pt>";
    std::cout << "Loading scripted module: " << scripted_model << std::endl;
    auto module = torch::jit::load(scripted_model);
    module.to(device);
    module.eval();

    /* inference */
    auto out = module.forward({input}).toTensor();
    auto probs = torch::exp(out);
    auto top = std::get<1>(probs.max(1));
    std::cout << "Predicted class (scripted): " << top.item<int>() << std::endl;
}



int main() {
    load_device();
    infer_net();
    return 0;
} 