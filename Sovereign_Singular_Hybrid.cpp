#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <string>
#include <map>

// Optimized HDC Dimension to drastically speed up Matrix math in a CPU-only environment
#define DIM 2000

using HDVector = std::vector<int>;

HDVector generate_random_vector(std::mt19937& gen) {
    HDVector v(DIM);
    std::uniform_int_distribution<> dist(0, 1);
    for (int i = 0; i < DIM; i++) v[i] = dist(gen) == 1 ? 1 : -1;
    return v;
}
HDVector bind_vectors(const HDVector& a, const HDVector& b) {
    HDVector v(DIM);
    for (int i = 0; i < DIM; i++) v[i] = a[i] * b[i];
    return v;
}
HDVector permute_vector(const HDVector& a, int shifts = 1) {
    HDVector v(DIM);
    for (int i = 0; i < DIM; i++) v[(i + shifts) % DIM] = a[i];
    return v;
}
HDVector bundle_vectors(const std::vector<HDVector>& vectors) {
    std::vector<int> sum(DIM, 0);
    for (const auto& vec : vectors) {
        for (int i = 0; i < DIM; i++) sum[i] += vec[i];
    }
    HDVector bundled(DIM);
    for (int i = 0; i < DIM; i++) bundled[i] = sum[i] >= 0 ? 1 : -1;
    return bundled;
}

struct Matrix {
    int rows, cols;
    std::vector<double> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}
    double& at(int r, int c) { return data[r * cols + c]; }
    const double& at(int r, int c) const { return data[r * cols + c]; }
    void randomize(std::mt19937& gen, double limit) {
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (double& val : data) val = dist(gen);
    }
    void zero() { for (double& val : data) val = 0.0; }
};

void matadd(Matrix& C, const Matrix& A) {
    for (size_t i = 0; i < C.data.size(); ++i) C.data[i] += A.data[i];
}

void quantize_matrix(const Matrix& latent, Matrix& quant) {
    double sum_abs = 0;
    for (double v : latent.data) sum_abs += std::abs(v);
    double mean_abs = sum_abs / latent.data.size();
    
    double scale = 1.0 / (mean_abs + 1e-8);
    for (size_t i = 0; i < latent.data.size(); ++i) {
        double val = std::round(latent.data[i] * scale);
        if (val > 1.0) val = 1.0;
        else if (val < -1.0) val = -1.0;
        quant.data[i] = val;
    }
}

class SingularHybridCore {
public:
    int vocab_size, hidden_size;
    
    // Ternary Recurrent Weights
    Matrix W1_xh_lat, W2_xh_lat, W1_hh_lat, W2_hh_lat;
    Matrix W1_xh_act, W2_xh_act, W1_hh_act, W2_hh_act;
    
    // TRUE INTEGRATION: Bounding Matrix connecting HDC boolean output to RNN float input
    Matrix W_hdc_lat, W_hdc_act;
    
    Matrix b_h, W_hy, b_y;

    SingularHybridCore(int vsize, int hsize, std::mt19937& gen)
        : vocab_size(vsize), hidden_size(hsize),
          W1_xh_lat(vsize, hsize), W2_xh_lat(vsize, hsize),
          W1_hh_lat(hsize, hsize), W2_hh_lat(hsize, hsize),
          W1_xh_act(vsize, hsize), W2_xh_act(vsize, hsize),
          W1_hh_act(hsize, hsize), W2_hh_act(hsize, hsize),
          W_hdc_lat(DIM, hsize), W_hdc_act(DIM, hsize), // High-Dimension to Low-Dimension Gate
          b_h(1, hsize), W_hy(hsize, vsize), b_y(1, vsize) {
        
        double std_h = std::sqrt(1.0 / hsize);
        W1_xh_lat.randomize(gen, std_h);
        W2_xh_lat.randomize(gen, std_h * 0.1); 
        W1_hh_lat.randomize(gen, std_h);
        W2_hh_lat.randomize(gen, std_h * 0.1);
        
        // Massive Matrix requires tiny initialization limit to survive scale
        double std_hdc = std::sqrt(1.0 / DIM);
        W_hdc_lat.randomize(gen, std_hdc);
        
        b_h.randomize(gen, 0.0);
        W_hy.randomize(gen, std_h);
        b_y.randomize(gen, 0.0);
    }

    void prepare_forward() {
        quantize_matrix(W1_xh_lat, W1_xh_act);
        quantize_matrix(W2_xh_lat, W2_xh_act);
        quantize_matrix(W1_hh_lat, W1_hh_act);
        quantize_matrix(W2_hh_lat, W2_hh_act);
        quantize_matrix(W_hdc_lat, W_hdc_act);
    }

    void forward_step(const Matrix& x, const Matrix& h_prev, const HDVector& hdc_signal, Matrix& h_next, Matrix& y, double& out_rms, Matrix& pre_norm) {
        Matrix poly_xh(1, hidden_size);
        Matrix poly_hh(1, hidden_size);
        Matrix poly_hdc(1, hidden_size);
        
        // Native Algebraic Sequence
        for (int i = 0; i < vocab_size; ++i) {
            double v = x.at(0, i);
            double v2 = v * v;
            for (int j = 0; j < hidden_size; ++j) {
                poly_xh.at(0, j) += v * W1_xh_act.at(i, j) + v2 * W2_xh_act.at(i, j);
            }
        }
        for (int i = 0; i < hidden_size; ++i) {
            double v = h_prev.at(0, i);
            double v2 = v * v;
            for (int j = 0; j < hidden_size; ++j) {
                poly_hh.at(0, j) += v * W1_hh_act.at(i, j) + v2 * W2_hh_act.at(i, j);
            }
        }
        
        // HDC BOOLEAN INTEGRATION
        for(int i = 0; i < DIM; ++i) {
            double v = (double)hdc_signal[i];
            for(int j = 0; j < hidden_size; ++j) {
                poly_hdc.at(0, j) += v * W_hdc_act.at(i, j);
            }
        }
        
        // Physics of Singular Structure: Memory Output directly controls Pre-Activation logic
        for (int i = 0; i < hidden_size; ++i) pre_norm.at(0, i) = poly_xh.at(0, i) + poly_hh.at(0, i) + poly_hdc.at(0, i);

        // Custom RMS Stabilization choke 
        double ms = 0;
        for (int i = 0; i < hidden_size; ++i) ms += pre_norm.at(0, i) * pre_norm.at(0, i);
        double rms = std::sqrt(ms / hidden_size + 1e-8);
        out_rms = rms;

        for (int i = 0; i < hidden_size; ++i) {
            h_next.at(0, i) = std::tanh((pre_norm.at(0, i) / rms) + b_h.at(0, i));
        }

        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < vocab_size; ++j) {
                y.at(0, j) += h_next.at(0, i) * W_hy.at(i, j);
            }
        }
        matadd(y, b_y);
    }
    
    // Train the RNN while passing backprop Calculus INTO the HDC routing boolean logic
    double train_hybrid_sequence(const std::vector<int>& seq_idx, const std::vector<HDVector>& hdc_signals, int target_idx, double lr) {
        prepare_forward();

        int seq_len = seq_idx.size();
        std::vector<Matrix> hs(seq_len + 1, Matrix(1, hidden_size));
        std::vector<Matrix> pre_norms(seq_len, Matrix(1, hidden_size));
        std::vector<double> rms_vals(seq_len, 0.0);
        Matrix y_pred(1, vocab_size);
        std::vector<Matrix> xs;

        for (int t = 0; t < seq_len; ++t) {
            Matrix x_t(1, vocab_size); 
            x_t.at(0, seq_idx[t]) = 1.0; 
            xs.push_back(x_t);
            y_pred.zero(); 
            forward_step(x_t, hs[t], hdc_signals[t], hs[t+1], y_pred, rms_vals[t], pre_norms[t]);
        }

        double error_sq_sum = 0;
        std::vector<double> dy(vocab_size, 0.0);
        for(int j=0; j<vocab_size; ++j) {
            double target_val = (j == target_idx) ? 1.0 : 0.0;
            double diff = y_pred.at(0, j) - target_val;
            dy[j] = diff;
            error_sq_sum += diff * diff;
        }

        Matrix dW_hy(hidden_size, vocab_size);
        Matrix db_y(1, vocab_size);
        Matrix dW1_xh(vocab_size, hidden_size), dW2_xh(vocab_size, hidden_size);
        Matrix dW1_hh(hidden_size, hidden_size), dW2_hh(hidden_size, hidden_size);
        Matrix dW_hdc(DIM, hidden_size);
        Matrix db_h(1, hidden_size);

        for (int j = 0; j < vocab_size; ++j) db_y.at(0, j) = dy[j];
        for (int i = 0; i < hidden_size; ++i) {
            for(int j=0; j < vocab_size; ++j) dW_hy.at(i, j) = hs[seq_len].at(0, i) * dy[j];
        }

        Matrix dh_next(1, hidden_size);
        for(int i = 0; i < hidden_size; i++) {
            double sum = 0;
            for(int j=0; j<vocab_size; ++j) sum += W_hy.at(i, j) * dy[j];
            dh_next.at(0, i) = sum;
        }

        for (int t = seq_len - 1; t >= 0; --t) {
            Matrix dh(1, hidden_size); 
            
            std::vector<double> dnorm(hidden_size, 0.0);
            for (int i = 0; i < hidden_size; ++i) {
                double dtanh = 1.0 - hs[t+1].at(0, i) * hs[t+1].at(0, i);
                double dres = dh_next.at(0, i) * dtanh;
                db_h.at(0, i) += dres;
                dnorm[i] = dres;
            }

            double rms = rms_vals[t];
            double inv_rms = 1.0 / rms;
            double sum_dnorm_z_hat = 0;
            for(int i=0; i<hidden_size; ++i) sum_dnorm_z_hat += dnorm[i] * (pre_norms[t].at(0, i) * inv_rms);

            std::vector<double> dpre(hidden_size, 0.0);
            for(int i = 0; i < hidden_size; ++i) {
                double z_hat = pre_norms[t].at(0, i) * inv_rms;
                dpre[i] = inv_rms * (dnorm[i] - z_hat * (sum_dnorm_z_hat / hidden_size));

                for(int v=0; v<vocab_size; ++v) {
                    double v_x = xs[t].at(0, v);
                    double v_x2 = v_x * v_x;
                    dW1_xh.at(v, i) += v_x * dpre[i];
                    dW2_xh.at(v, i) += v_x2 * dpre[i];
                }
                
                for(int j=0; j<hidden_size; j++) {
                    double h_prev_val = hs[t].at(0, j);
                    dW1_hh.at(j, i) += h_prev_val * dpre[i];
                    dW2_hh.at(j, i) += (h_prev_val * h_prev_val) * dpre[i];
                    
                    double poly_deriv = W1_hh_act.at(j, i) + 2.0 * W2_hh_act.at(j, i) * h_prev_val;
                    dh.at(0, j) += dpre[i] * poly_deriv;
                }
                
                // BACKPROPAGATION INTO HDC GATE: Learning to trust the Boolean Boolean Extraction Matrix
                for(int k=0; k<DIM; k++) {
                    double hdc_val = (double)hdc_signals[t][k];
                    dW_hdc.at(k, i) += hdc_val * dpre[i];
                }
            }
            dh_next = dh;
        }

        auto apply_grad = [lr](Matrix& W, const Matrix& dW) {
            for (size_t i = 0; i < W.data.size(); ++i) W.data[i] -= lr * dW.data[i];
        };
        apply_grad(W_hy, dW_hy); apply_grad(b_y, db_y);
        apply_grad(W1_xh_lat, dW1_xh); apply_grad(W2_xh_lat, dW2_xh);
        apply_grad(W1_hh_lat, dW1_hh); apply_grad(W2_hh_lat, dW2_hh);
        apply_grad(W_hdc_lat, dW_hdc);
        apply_grad(b_h, db_h);

        return error_sq_sum / vocab_size; 
    }
    
    double validate_mse(const std::vector<int>& seq_idx, const std::vector<HDVector>& hdc_signals, int target_idx) {
        prepare_forward();
        std::vector<Matrix> hs(seq_idx.size() + 1, Matrix(1, hidden_size));
        Matrix y_pred(1, vocab_size);
        for (size_t t = 0; t < seq_idx.size(); ++t) {
            Matrix x_t(1, vocab_size); 
            x_t.at(0, seq_idx[t]) = 1.0;
            y_pred.zero();
            double dump_rms;
            Matrix dump_pre(1, hidden_size);
            forward_step(x_t, hs[t], hdc_signals[t], hs[t+1], y_pred, dump_rms, dump_pre);
        }
        
        double error_sq_sum = 0;
        for(int j=0; j<vocab_size; ++j) {
            double target_val = (j == target_idx) ? 1.0 : 0.0;
            double diff = y_pred.at(0, j) - target_val;
            error_sq_sum += diff * diff;
        }
        return error_sq_sum / vocab_size;
    }
};

int main() {
    std::cout << "--- PHASE 9: SINGULAR HYBRID CORE (TRUE NEURO-SYMBOLIC INTEGRATION) ---\n";
    std::cout << "[SYSTEM]: Instantiating 2000-D HDC Memory Array...\n";
    
    std::string text_logic = "engine_evaluate_";
    std::string symbol_logic = "math_5x5[TOOL]_";
    
    std::string data = "";
    for(int i=0; i<20; i++) {
        data += text_logic;
        data += symbol_logic;
    }
    
    std::mt19937 gen(42);
    
    // BUILD HDC ENVIRONMENT
    std::map<char, HDVector> item_memory;
    for (char c : data) {
        if (item_memory.find(c) == item_memory.end()) item_memory[c] = generate_random_vector(gen);
    }
    int vocab_size = item_memory.size();

    // Context Array (Memory Pool) formulation
    std::vector<HDVector> memory_pool;
    for (size_t i = 0; i < data.size() - 3; ++i) {
        char c1 = data[i]; char c2 = data[i+1]; char c3 = data[i+2]; char value = data[i+3];
        HDVector key1 = bind_vectors(permute_vector(item_memory[c1], 2), permute_vector(item_memory[c2], 1));
        HDVector key = bind_vectors(key1, item_memory[c3]);
        memory_pool.push_back(bind_vectors(key, item_memory[value]));
    }
    std::cout << "[SYSTEM]: Encoding 1-Bit HDC Superstate. Compressing...\n";
    HDVector SUPER_VECTOR = bundle_vectors(memory_pool);

    // PREPARE NEURAL SEQUENCES
    std::vector<int> tokens;
    std::map<char, int> c2i;
    int c_idx = 0;
    for (auto const& [k, val] : item_memory) { c2i[k] = c_idx; c_idx++; }
    for(char c : data) tokens.push_back(c2i[c]);

    int seq_length = 3; // Trigram Alignment 
    std::vector<std::vector<int>> rnn_inputs;
    std::vector<std::vector<HDVector>> hdc_inputs;
    std::vector<int> rnn_targets;

    for (size_t i = 0; i < tokens.size() - seq_length; ++i) {
        std::vector<int> t_seq;
        std::vector<HDVector> hdc_seq;
        
        for(int j=0; j<seq_length; j++) {
            t_seq.push_back(tokens[i+j]);
            
            // To provide aligned context for the Unified Forward Step, 
            // We pass an unbind query based on text context natively through the matrix
            if (i+j >= 2) {
                char c1 = data[i+j-2]; char c2 = data[i+j-1]; char c3 = data[i+j];
                HDVector key1 = bind_vectors(permute_vector(item_memory[c1], 2), permute_vector(item_memory[c2], 1));
                HDVector query_key = bind_vectors(key1, item_memory[c3]);
                HDVector noisy_retrieval = bind_vectors(SUPER_VECTOR, query_key);
                hdc_seq.push_back(noisy_retrieval);
            } else {
                hdc_seq.push_back(generate_random_vector(gen)); // Empty context padding
            }
        }
        rnn_inputs.push_back(t_seq);
        hdc_inputs.push_back(hdc_seq);
        rnn_targets.push_back(tokens[i + seq_length]);
    }

    int train_size = (int)(rnn_inputs.size() * 0.8);
    
    // BUILD UNIFIED CORE
    std::cout << "[SYSTEM]: Commencing True Integration Training. Fusing Gradient Engine with Memory Signals...\n\n";
    SingularHybridCore hybrid(vocab_size, 64, gen);
    double lr = 0.05; 
    
    for (int epoch = 0; epoch < 250; ++epoch) {
        double total_loss = 0;
        for (int i = 0; i < train_size; ++i) {
            total_loss += hybrid.train_hybrid_sequence(rnn_inputs[i], hdc_inputs[i], rnn_targets[i], lr);
        }
        double mse = total_loss / train_size;
        
        if (epoch % 50 == 0 || epoch == 249) {
            std::cout << "Epoch " << epoch + 1 << " | Unified Model MSE: " << std::fixed << std::setprecision(6) << mse << "\n";
        }
    }
    
    // Validating Final Matrix Merge success
    double test_loss = 0;
    for (int i = train_size; i < (int)rnn_inputs.size(); ++i) {
        test_loss += hybrid.validate_mse(rnn_inputs[i], hdc_inputs[i], rnn_targets[i]);
    }
    int total_tests = rnn_inputs.size() - train_size;
    test_loss /= total_tests;

    std::cout << "\n--- FINAL HYBRID VALIDATION ---\n";
    std::cout << "Test Target MSE Loss: " << std::fixed << std::setprecision(6) << test_loss << "\n";
    
    if (test_loss < 0.01) {
        std::cout << "\n[SUCCESS]: Theoretical Fusion Conquered! Calculus gradients successfully absorbed and routed 1-Bit boolean memory extractions without matrix collapse. MSE < 0.01\n";
    } else {
        std::cout << "\n[FAILED]: Architecture crashed reconstructing vocabulary sequence via boolean collision.\n";
    }

    return 0;
}
