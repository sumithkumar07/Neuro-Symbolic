#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <string>
#include <map>
#include <cuda_runtime.h>

#define DIM 2000

using HDVector = std::vector<int>;

#define cudaCheckError() {                                          \
 cudaError_t e = cudaGetLastError();                                 \
 if(e != cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0);                                                        \
 }                                                                 \
}

// ==========================================
// SOVEREIGN FUSED FOUNDATION (v4.6)
// THE SOVEREIGN ABSOLUTE | 100% SILICON CORE
// ==========================================

__global__ void kernel_quantize_adaptive(const double* latent, double* quant, int size, double threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double val = latent[idx];
        if (val > threshold) quant[idx] = 1.0; 
        else if (val < -threshold) quant[idx] = -1.0; 
        else quant[idx] = 0.0;
    }
}

// Specialized DFA: Projects error into a hidden projection vector for biases/gates
__global__ void kernel_dfa_projection(const double* B, const double* E, double* PROJ, int vs, int hs) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < hs) {
        double p = 0; for(int k=0; k<vs; k++) p += B[j * vs + k] * E[k];
        PROJ[j] = p;
    }
}

__global__ void kernel_vectorized_dfa_update_dual(double* W1, double* W2, const double* X, const double* PROJ, const double* H, int in_s, int hs, double lr, double clip) {
    int i = blockIdx.y; int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < hs && i < in_s) {
        double delta = PROJ[j] * (1.1 - H[j] * H[j]);
        if (delta > clip) delta = clip; else if (delta < -clip) delta = -clip;
        W1[i * hs + j] -= lr * delta * X[i];
        W2[i * hs + j] -= lr * delta * (X[i] * X[i]);
    }
}

__global__ void kernel_vectorized_dfa_update_single(double* W1, const double* X, const double* PROJ, const double* H, int in_s, int hs, double lr, double clip) {
    int i = blockIdx.y; int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < hs && i < in_s) {
        double delta = PROJ[j] * (1.1 - H[j] * H[j]);
        if (delta > clip) delta = clip; else if (delta < -clip) delta = -clip;
        W1[i * hs + j] -= lr * delta * X[i];
    }
}

__global__ void kernel_bias_dfa_update(double* B_lat, const double* PROJ, const double* H, int hs, double lr, double clip) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < hs) {
        double delta = PROJ[j] * (1.1 - H[j] * H[j]);
        if (delta > clip) delta = clip; else if (delta < -clip) delta = -clip;
        B_lat[j] -= lr * delta;
    }
}

__global__ void kernel_output_dfa_update(double* Why, double* by, const double* H, const double* E, int hs, int vs, double lr, double clip) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < vs) {
        // Bias update (by)
        double de = E[j]; if (de > clip) de = clip; else if (de < -clip) de = -clip;
        by[j] -= lr * de;
        // Why update
        for (int i = 0; i < hs; ++i) {
            double dw = E[j] * H[i];
            if (dw > clip) dw = clip; else if (dw < -clip) dw = -clip;
            Why[i * vs + j] -= lr * dw;
        }
    }
}

__global__ void kernel_output_projection(const double* H, const double* Why, const double* by, double* YP, int hs, int vs) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < vs) {
        double s = by[j]; for(int i=0; i<hs; i++) s += H[i] * Why[i*vs + j];
        YP[j] = s;
    }
}

// Fix 4: Pure Silicon GRU Blend
__global__ void kernel_gru_blend(double* H, const double* Z, const double* HT, int hs) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < hs) H[j] = (1.0 - Z[j]) * H[j] + Z[j] * HT[j];
}

__global__ void kernel_poly_matmul_vec_tiled(const double* X, const double* W1, const double* W2, double* Y, int vs_sz, int hs_sz) {
    extern __shared__ double sX[];
    int tide = threadIdx.x; int j = blockIdx.x * blockDim.x + tide;
    for (int i = tide; i < vs_sz; i += blockDim.x) sX[i] = X[i];
    __syncthreads();
    if (j < hs_sz) {
        double sum = 0.0;
        for (int i = 0; i < vs_sz; ++i) {
            double v = sX[i];
            sum += v * W1[i * hs_sz + j] + (v * v) * W2[i * hs_sz + j];
        }
        Y[j] += sum;
    }
}

__global__ void kernel_compute_gate(const double* X, const double* Wg, double b, double* out, int vs_sz) {
    double s = b; for(int i=0; i<vs_sz; i++) s += X[i] * Wg[i];
    *out = 1.0 / (1.0 + exp(-s));
}

__global__ void kernel_matadd(double* C, const double* A, int sz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sz) C[idx] += A[idx];
}

__global__ void kernel_matmult(double* C, const double* A, int sz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sz) C[idx] *= A[idx];
}

__global__ void kernel_sigmoid(double* x, int sz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sz) x[idx] = 1.0 / (1.0 + exp(-x[idx]));
}

__global__ void kernel_tanh(double* x, int sz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sz) x[idx] = tanh(x[idx]);
}

__global__ void kernel_group_rmsnorm(double* x, int hs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int grp = 8; int start = (idx / grp) * grp;
    if (idx < hs) {
        double ssq = 0; for(int i=0; i<grp; i++) { double v = x[start+i]; ssq += v*v; }
        double rms = sqrt(ssq / grp + 1e-8); x[idx] /= rms;
    }
}

__global__ void kernel_poly_hdc_tiled(const double* e, const double* W, double* p, int d, int hs, double conf) {
    extern __shared__ double sH[];
    int ti = threadIdx.x; int j = blockIdx.x * blockDim.x + ti;
    for(int i=ti; i<d; i+=blockDim.x) sH[i] = e[i];
    __syncthreads();
    if(j < hs) {
        double s = 0; for(int i=0; i<d; i++) s += W[i*hs+j] * sH[i];
        p[j] = s * conf;
    }
}

struct CUDAMatrix {
    int r, c; double* d; std::vector<double> h;
    CUDAMatrix(int rows, int cols) : r(rows), c(cols), h(rows*cols, 0.0) {
        cudaMalloc(&d, r*c*sizeof(double)); cudaMemset(d, 0, r*c*sizeof(double));
    }
    ~CUDAMatrix() { if(d) cudaFree(d); }
    void randomize(std::mt19937& gen, double limit) {
        std::uniform_real_distribution<double> dist(-limit, limit);
        for(double& v : h) v = dist(gen); to_device();
    }
    void to_device() { cudaMemcpy(d, h.data(), h.size()*sizeof(double), cudaMemcpyHostToDevice); }
    void to_host() { cudaMemcpy(h.data(), d, h.size()*sizeof(double), cudaMemcpyDeviceToHost); }
};

class SovereignAbsoluteCore {
public:
    int vs, hs, np; double thr;
    CUDAMatrix Wz1, Wz2, Wz1a, Wz2a, bz;
    CUDAMatrix Wr1, Wr2, Wr1a, Wr2a, br;
    CUDAMatrix Wh1, Wh2, Wh1a, Wh2a, bh;
    CUDAMatrix Whdc_l, Whdc_a, Wg_l, Wg_a, by_l;
    CUDAMatrix Why_l, Why_a, WB, dX, dE, dZ, dR, dHT, dHDC, dH, dEXTR, dYP, dPROJ;
    double* dGS;

    SovereignAbsoluteCore(int v, int h, int n, std::mt19937& gen)
        : vs(v), hs(h), np(n), thr(0.01),
          Wz1(v, h), Wz2(v, h), Wz1a(v, h), Wz2a(v, h), bz(1, h),
          Wr1(v, h), Wr2(v, h), Wr1a(v, h), Wr2a(v, h), br(1, h),
          Wh1(v, h), Wh2(v, h), Wh1a(v, h), Wh2a(v, h), bh(1, h),
          Whdc_l(DIM, h), Whdc_a(DIM, h), Wg_l(v, 1), Wg_a(v, 1), by_l(1, v),
          Why_l(h, v), Why_a(h, v), WB(h, v),
          dX(1, v), dE(1, v), dZ(1, h), dR(1, h), dHT(1, h), dHDC(1, h), dH(1, h), dEXTR(1, DIM), dYP(1, v), dPROJ(1, h) {
        
        double s = std::sqrt(1.0/h);
        Wz1.randomize(gen, s); Wz2.randomize(gen, s*0.1); Wr1.randomize(gen, s); Wr2.randomize(gen, s*0.1);
        Wh1.randomize(gen, s); Wh2.randomize(gen, s*0.1); bz.randomize(gen, 0.1); br.randomize(gen, 0.1); 
        bh.randomize(gen, 0.1); by_l.randomize(gen, 0.1); Whdc_l.randomize(gen, std::sqrt(1.0/DIM)); 
        Wg_l.randomize(gen, s); Why_l.randomize(gen, s); WB.randomize(gen, 1.0);
        cudaMalloc(&dGS, sizeof(double));
    }
    ~SovereignAbsoluteCore() { cudaFree(dGS); }

    void prepare() {
        int tpb = 256;
        kernel_quantize_adaptive<<< (vs*hs+tpb-1)/tpb, tpb >>>(Wz1.d, Wz1a.d, vs*hs, thr);
        kernel_quantize_adaptive<<< (vs*hs+tpb-1)/tpb, tpb >>>(Wz2.d, Wz2a.d, vs*hs, thr);
        kernel_quantize_adaptive<<< (vs*hs+tpb-1)/tpb, tpb >>>(Wr1.d, Wr1a.d, vs*hs, thr);
        kernel_quantize_adaptive<<< (vs*hs+tpb-1)/tpb, tpb >>>(Wr2.d, Wr2a.d, vs*hs, thr);
        kernel_quantize_adaptive<<< (vs*hs+tpb-1)/tpb, tpb >>>(Wh1.d, Wh1a.d, vs*hs, thr);
        kernel_quantize_adaptive<<< (vs*hs+tpb-1)/tpb, tpb >>>(Wh2.d, Wh2a.d, vs*hs, thr);
        kernel_quantize_adaptive<<< (hs*vs+tpb-1)/tpb, tpb >>>(Why_l.d, Why_a.d, hs*vs, thr);
        kernel_quantize_adaptive<<< (DIM*hs+tpb-1)/tpb, tpb >>>(Whdc_l.d, Whdc_a.d, DIM*hs, thr);
        kernel_quantize_adaptive<<< (vs+tpb-1)/tpb, tpb >>>(Wg_l.d, Wg_a.d, vs, thr);
    }

    double train_absolute(const std::vector<int>& seq, const std::vector<std::vector<HDVector>>& signals, int target, double lr) {
        prepare();
        int tpb = 256; int blk_h = (hs + tpb - 1) / tpb;
        cudaMemset(dH.d, 0, hs*sizeof(double)); // Pure initial state

        for (int t = 0; t < (int)seq.size(); ++t) {
            dX.h.assign(vs, 0.0); dX.h[seq[t]] = 1.0; dX.to_device();
            std::vector<double> ex(DIM, 0.0); for(int p=0; p<np; p++) for(int k=0; k<DIM; k++) ex[k] += (0.5/np) * signals[t][p][k];
            dEXTR.h = ex; dEXTR.to_device();
            
            // Bias baseline
            cudaMemcpy(dZ.d, bz.d, hs*sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dR.d, br.d, hs*sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dHT.d, bh.d, hs*sizeof(double), cudaMemcpyDeviceToDevice);
            
            kernel_poly_matmul_vec_tiled<<<blk_h, tpb, vs*sizeof(double)>>>(dX.d, Wz1a.d, Wz2a.d, dZ.d, vs, hs);
            kernel_poly_matmul_vec_tiled<<<blk_h, tpb, vs*sizeof(double)>>>(dX.d, Wr1a.d, Wr2a.d, dR.d, vs, hs);
            kernel_poly_matmul_vec_tiled<<<blk_h, tpb, vs*sizeof(double)>>>(dX.d, Wh1a.d, Wh2a.d, dHT.d, vs, hs);
            
            kernel_sigmoid<<<blk_h, tpb>>>(dZ.d, hs); kernel_sigmoid<<<blk_h, tpb>>>(dR.d, hs);
            kernel_compute_gate<<<1, 1>>>(dX.d, Wg_a.d, -1.0, dGS, vs);
            double conf; cudaMemcpy(&conf, dGS, sizeof(double), cudaMemcpyDeviceToHost);
            kernel_poly_hdc_tiled<<<blk_h, tpb, DIM*sizeof(double)>>>(dEXTR.d, Whdc_a.d, dHDC.d, DIM, hs, conf);
            
            kernel_matmult<<<blk_h, tpb>>>(dHDC.d, dR.d, hs); 
            kernel_matadd<<<blk_h, tpb>>>(dHT.d, dHDC.d, hs);
            kernel_group_rmsnorm<<<blk_h, tpb>>>(dHT.d, hs); kernel_tanh<<<blk_h, tpb>>>(dHT.d, hs);
            
            // Fix 4: Silicon GRU Blend (No D2H transfers!)
            kernel_gru_blend<<<blk_h, tpb>>>(dH.d, dZ.d, dHT.d, hs);
        }

        kernel_output_projection<<< (vs+tpb-1)/tpb, tpb >>>(dH.d, Why_a.d, by_l.d, dYP.d, hs, vs);
        dYP.to_host(); double loss = 0; 
        for(int j=0; j<vs; j++) {
            double t_j = (j == target) ? 0.9 : 0.1;
            dE.h[j] = dYP.h[j] - t_j; loss += 0.5 * dE.h[j] * dE.h[j];
        }
        dE.to_device(); double clip = 0.05;

        // Fix 1-2: Full Autonomous DFA (Projection vector + Biases)
        kernel_dfa_projection<<<blk_h, tpb>>>(WB.d, dE.d, dPROJ.d, vs, hs);
        
        dim3 grd(blk_h, vs);
        kernel_vectorized_dfa_update_dual<<<grd, tpb>>>(Wz1.d, Wz2.d, dX.d, dPROJ.d, dH.d, vs, hs, lr, clip);
        kernel_vectorized_dfa_update_dual<<<grd, tpb>>>(Wr1.d, Wr2.d, dX.d, dPROJ.d, dH.d, vs, hs, lr, clip);
        kernel_vectorized_dfa_update_dual<<<grd, tpb>>>(Wh1.d, Wh2.d, dX.d, dPROJ.d, dH.d, vs, hs, lr, clip);
        
        kernel_bias_dfa_update<<<blk_h, tpb>>>(bz.d, dPROJ.d, dH.d, hs, lr, clip);
        kernel_bias_dfa_update<<<blk_h, tpb>>>(br.d, dPROJ.d, dH.d, hs, lr, clip);
        kernel_bias_dfa_update<<<blk_h, tpb>>>(bh.d, dPROJ.d, dH.d, hs, lr, clip);
        
        kernel_vectorized_dfa_update_single<<<dim3(blk_h, DIM), tpb>>>(Whdc_l.d, dEXTR.d, dPROJ.d, dH.d, DIM, hs, lr, clip);
        kernel_vectorized_dfa_update_single<<<dim3(blk_h, vs), tpb>>>(Wg_l.d, dX.d, dPROJ.d, dH.d, vs, hs, lr, clip);
        
        kernel_output_dfa_update<<< (vs+tpb-1)/tpb, tpb >>>(Why_l.d, by_l.d, dH.d, dE.d, hs, vs, lr, clip);
        
        cudaDeviceSynchronize(); return loss / vs;
    }
};

HDVector bnd(const HDVector& a, const HDVector& b) { HDVector v(DIM); for(int i=0; i<DIM; i++) v[i]=a[i]*b[i]; return v; }
HDVector prm(const HDVector& a, int s=1) { HDVector v(DIM); for(int i=0; i<DIM; i++) v[(i+s)%DIM]=a[i]; return v; }

int main() {
    std::cout << "==========================================================\n";
    std::cout << "[SYSTEM]: SOVEREIGN ABSOLUTE (v4.6)\n";
    std::cout << "[SYSTEM]: 100% SILICON GRU CORE | FULL AUTONOMY\n";
    std::cout << "==========================================================\n";
    std::string data = "engine_evaluate_math_5x5[TOOL]_"; std::mt19937 gen(42);
    std::map<char, HDVector> itm; for(char c : data) if(!itm.count(c)) {
        HDVector v(DIM); std::uniform_int_distribution<> d(0,1); for(int i=0; i<DIM; i++) v[i]=(d(gen)?1:-1); itm[c]=v;
    }
    int vs = itm.size(); std::map<char, int> c2i; int ci=0; for(auto const& p : itm) c2i[p.first] = ci++;
    std::vector<int> tokens; for(char c : data) tokens.push_back(c2i[c]);

    std::vector<std::vector<int>> r_in; std::vector<std::vector<std::vector<HDVector>>> h_in; std::vector<int> r_t;
    for(size_t i=0; i<tokens.size()-5; ++i) {
        std::vector<int> t_seq; std::vector<std::vector<HDVector>> h_seq; HDVector q(DIM, 1);
        for(int j=0; j<5; j++) { t_seq.push_back(tokens[i+j]); q = bnd(prm(q,1), itm[data[i+j]]); h_seq.push_back({q}); }
        r_in.push_back(t_seq); h_in.push_back(h_seq); r_t.push_back(tokens[i+5]);
    }

    SovereignAbsoluteCore eng(vs, 128, 1, gen); double lr = 0.05;
    for(int epoch=0; epoch<50; ++epoch) {
        double total_l = 0; int n = r_in.size();
        for(int i=0; i<n; i++) total_l += eng.train_absolute(r_in[i], h_in[i], r_t[i], lr);
        if(eng.thr < 0.05) eng.thr += 0.001;
        if(eng.thr > 0.03 && lr > 0.01) lr = 0.01; if(eng.thr > 0.045 && lr > 0.001) lr = 0.001;
        if(epoch % 10 == 0 || epoch == 49) std::cout << "Epoch " << epoch+1 << " | Absolute Loss: " << (total_l/n) << " | Thresh: " << eng.thr << "\n";
    }
    std::cout << "\n[SUCCESS]: Sovereign v4.6 Absolute Complete.\n";
    return 0;
}
