// Week-04 microbenchmark. Sweeps tile sizes for v4_checkpoint and
// reports GFLOP/s alongside cublasSgemm. Median of 20 runs after 5
// warm-up runs, per the §3 Performance target spec.

#include "gemm.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace cg = cudalab::gemm;

namespace {

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

float time_ms(cudaEvent_t s, cudaEvent_t e,
              auto&& fn) {
    cudaEventRecord(s);
    fn();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);
    return ms;
}

}  // namespace

int main() {
    const int M = 4096, N = 4096, K = 4096;
    const double flops = 2.0 * double(M) * double(N) * double(K);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, size_t(M) * K * sizeof(float));
    cudaMalloc(&dB, size_t(K) * N * sizeof(float));
    cudaMalloc(&dC, size_t(M) * N * sizeof(float));
    cudaMemset(dA, 1, size_t(M) * K * sizeof(float));
    cudaMemset(dB, 1, size_t(K) * N * sizeof(float));

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    // cuBLAS baseline.
    cublasHandle_t h; cublasCreate(&h);
    float a = 1.0f, b = 0.0f;
    auto cublas_run = [&]{
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &a, dB, N, dA, K, &b, dC, N);
    };

    for (int i = 0; i < 5; ++i) cublas_run();
    std::vector<float> tcb;
    for (int i = 0; i < 20; ++i) tcb.push_back(time_ms(s, e, cublas_run));
    float ms_cb = median(tcb);
    double gflops_cb = flops / (ms_cb * 1e6);

    std::printf("| Variant | Time (ms) | GFLOP/s | %% of cuBLAS |\n");
    std::printf("|---|---|---|---|\n");
    std::printf("| cublasSgemm | %.3f | %.1f | 100.0 |\n", ms_cb, gflops_cb);

    auto run_variant = [&](const char* name, cg::TileConfig cfg) {
        auto fn = [&]{
            cg::gemm_v4_checkpoint({M, N, K}, 1.0f, dA, dB, 0.0f, dC,
                                   nullptr, cfg);
        };
        for (int i = 0; i < 5; ++i) fn();
        std::vector<float> ts;
        for (int i = 0; i < 20; ++i) ts.push_back(time_ms(s, e, fn));
        float ms = median(ts);
        double g = flops / (ms * 1e6);
        std::printf("| %s | %.3f | %.1f | %.1f |\n",
                    name, ms, g, 100.0 * g / gflops_cb);
    };

    run_variant("v4_checkpoint BM=32",  cg::TileConfig{32, 32, 16, 4, 4});
    run_variant("v4_checkpoint BM=64",  cg::TileConfig{64, 64, 16, 4, 4});
    run_variant("v4_checkpoint BM=128", cg::TileConfig{128, 128, 16, 8, 8});

    cublasDestroy(h);
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
