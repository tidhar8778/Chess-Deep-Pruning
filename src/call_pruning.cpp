#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

extern "C" int call_pruning(const char* path_to_model_script);

int call_pruning(const char* path_to_model_script) {
    
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(path_to_model_script);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "ok\n";
}