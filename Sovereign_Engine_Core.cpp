#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <string>
#include <map>

struct Matrix {
    int rows, cols;
    std::vector<double> data;

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

class LLMCore {
public:
    int vocab_size, hidden_size;
    Matrix W1_xh_lat, W2_xh_lat, W1_hh_lat, W2_hh_lat;
    Matrix W1_xh_act, W2_xh_act, W1_hh_act, W2_hh_act;
    Matrix b_h, W_hy, b_y;

    LLMCore(int vsize, int hsize, std::mt19937& gen)
        : vocab_size(vsize), hidden_size(hsize),
          W1_xh_lat(vsize, hsize), W2_xh_lat(vsize, hsize),
          W1_hh_lat(hsize, hsize), W2_hh_lat(hsize, hsize),
          W1_xh_act(vsize, hsize), W2_xh_act(vsize, hsize),
          W1_hh_act(hsize, hsize), W2_hh_act(hsize, hsize),
          b_h(1, hsize), W_hy(hsize, vsize), b_y(1, vsize) {
        
        double std_h = std::sqrt(1.0 / hsize);
        W1_xh_lat.randomize(gen, std_h);
        W2_xh_lat.randomize(gen, std_h * 0.1); 
        W1_hh_lat.randomize(gen, std_h);
        W2_hh_lat.randomize(gen, std_h * 0.1);
        b_h.randomize(gen, 0.0);
        W_hy.randomize(gen, std_h);
        b_y.randomize(gen, 0.0);
    }

    void prepare_forward() {
        quantize_matrix(W1_xh_lat, W1_xh_act);
        quantize_matrix(W2_xh_lat, W2_xh_act);
        quantize_matrix(W1_hh_lat, W1_hh_act);
        quantize_matrix(W2_hh_lat, W2_hh_act);
    }

    void forward_step(const Matrix& x, const Matrix& h_prev, Matrix& h_next, Matrix& y, double& out_rms, Matrix& pre_norm) {
        Matrix poly_xh(1, hidden_size);
        Matrix poly_hh(1, hidden_size);
        
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
        
        for (int i = 0; i < hidden_size; ++i) pre_norm.at(0, i) = poly_xh.at(0, i) + poly_hh.at(0, i);

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
    
    double train_sequence(const std::vector<int>& seq_idx, int target_idx, double lr) {
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
            forward_step(x_t, hs[t], hs[t+1], y_pred, rms_vals[t], pre_norms[t]);
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
        Matrix db_h(1, hidden_size);

        for (int j = 0; j < vocab_size; ++j) {
            db_y.at(0, j) = dy[j];
        }
        for (int i = 0; i < hidden_size; ++i) {
            for(int j=0; j < vocab_size; ++j) {
                dW_hy.at(i, j) = hs[seq_len].at(0, i) * dy[j];
            }
        }

        Matrix dh_next(1, hidden_size);
        for(int i = 0; i < hidden_size; i++) {
            double sum = 0;
            for(int j=0; j<vocab_size; ++j) {
                sum += W_hy.at(i, j) * dy[j];
            }
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
            }
            dh_next = dh;
        }

        auto apply_grad = [lr](Matrix& W, const Matrix& dW) {
            for (size_t i = 0; i < W.data.size(); ++i) W.data[i] -= lr * dW.data[i];
        };
        apply_grad(W_hy, dW_hy); apply_grad(b_y, db_y);
        apply_grad(W1_xh_lat, dW1_xh); apply_grad(W2_xh_lat, dW2_xh);
        apply_grad(W1_hh_lat, dW1_hh); apply_grad(W2_hh_lat, dW2_hh);
        apply_grad(b_h, db_h);

        return error_sq_sum / vocab_size; 
    }
    
    int predict(const std::vector<int>& seq_idx) {
        prepare_forward();
        std::vector<Matrix> hs(seq_idx.size() + 1, Matrix(1, hidden_size));
        Matrix y_pred(1, vocab_size);
        for (size_t t = 0; t < seq_idx.size(); ++t) {
            Matrix x_t(1, vocab_size); 
            x_t.at(0, seq_idx[t]) = 1.0;
            y_pred.zero();
            double dump_rms;
            Matrix dump_pre(1, hidden_size);
            forward_step(x_t, hs[t], hs[t+1], y_pred, dump_rms, dump_pre);
        }
        
        int best_idx = 0;
        double best_val = y_pred.at(0, 0);
        for(int j=1; j<vocab_size; ++j){
            if(y_pred.at(0,j) > best_val) {
                best_val = y_pred.at(0, j);
                best_idx = j;
            }
        }
        return best_idx;
    }
    
    double validate_mse(const std::vector<int>& seq_idx, int target_idx) {
        prepare_forward();
        std::vector<Matrix> hs(seq_idx.size() + 1, Matrix(1, hidden_size));
        Matrix y_pred(1, vocab_size);
        for (size_t t = 0; t < seq_idx.size(); ++t) {
            Matrix x_t(1, vocab_size); 
            x_t.at(0, seq_idx[t]) = 1.0;
            y_pred.zero();
            double dump_rms;
            Matrix dump_pre(1, hidden_size);
            forward_step(x_t, hs[t], hs[t+1], y_pred, dump_rms, dump_pre);
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
    std::cout << "--- PHASE 7: NEURO-SYMBOLIC TOOL HOOKING ---\n";
    
    // Conditional Dataset Phase 7
    std::string text_logic = "engine_running_";
    std::string symbol_logic = "calc_5x5[TOOL]_";
    
    std::string data = "";
    // Build an alternating mixed string pattern to teach contextual logic branching
    for(int i=0; i<30; i++) {
        data += text_logic;
        data += symbol_logic;
    }
    
    std::map<char, int> c2i;
    std::vector<char> i2c;
    for (char c : data) {
        if (c2i.find(c) == c2i.end()) {
            c2i[c] = i2c.size();
            i2c.push_back(c);
        }
    }
    int vocab_size = i2c.size();
    
    std::vector<int> tokens;
    for(char c : data) tokens.push_back(c2i[c]);

    std::mt19937 gen(42);
    int seq_length = 5; 
    
    std::vector<std::vector<int>> inputs;
    std::vector<int> targets;
    for (size_t i = 0; i < tokens.size() - seq_length; ++i) {
        std::vector<int> seq;
        for(int j=0; j<seq_length; j++) seq.push_back(tokens[i+j]);
        inputs.push_back(seq);
        targets.push_back(tokens[i + seq_length]);
    }

    int train_size = (int)(inputs.size() * 0.8);
    
    LLMCore model(vocab_size, 64, gen);
    double lr = 0.05; 
    
    for (int epoch = 0; epoch < 250; ++epoch) {
        double total_loss = 0;
        for (int i = 0; i < train_size; ++i) {
            total_loss += model.train_sequence(inputs[i], targets[i], lr);
        }
        double mse = total_loss / train_size;
        
        if (epoch % 50 == 0 || epoch == 249) {
            std::cout << "Epoch " << epoch + 1 << " | Train Vocab MSE Loss: " << std::fixed << std::setprecision(6) << mse << "\n";
        }
    }
    
    // Evaluation 
    double test_loss = 0;
    int correct_chars = 0;
    for (int i = train_size; i < (int)inputs.size(); ++i) {
        int pred_idx = model.predict(inputs[i]);
        if(pred_idx == targets[i]) correct_chars++;
        test_loss += model.validate_mse(inputs[i], targets[i]);
    }
    int total_tests = inputs.size() - train_size;
    test_loss /= total_tests;

    std::cout << "\n--- FINAL SYMBOLIC VALIDATION ---\n";
    std::cout << "Test Target MSE Loss: " << std::fixed << std::setprecision(6) << test_loss << "\n";
    
    if (test_loss < 0.01) {
        std::cout << "[SUCCESS]: Logic Paths Converged! MSE < 0.01\n\n";
        
        // C++ HARDCODED SYSTEM HOOK DEMONSTRATION
        std::cout << "--- EXECUTING AUTONOMOUS HOOK TEST ---\n";
        std::string test_prompt = "calc_";
        std::cout << "Injecting Neural Prompt: \"" << test_prompt << "\"\n";
        std::cout << "Model Output: ";
        
        std::vector<int> current_seq;
        for(char c : test_prompt) current_seq.push_back(c2i[c]);
        
        std::string generated = "";
        for(int step=0; step<10; step++) {
            int pred = model.predict(current_seq);
            char c = i2c[pred];
            generated += c;
            std::cout << c;
            
            // Check for Hardcoded Trigger Context intercept
            if (generated.find("[TOOL]") != std::string::npos) {
                std::cout << "\n\n>> [SYSTEM INTERCEPT]: Tool trigger detected in Neural Output.\n";
                std::cout << ">> [HARDCODE CALCULATING]: Executing exact math 5x5...\n";
                std::cout << ">> [SYMBOLIC RESULT]: 25\n";
                break;
            }
            
            // shift buffer
            current_seq.erase(current_seq.begin());
            current_seq.push_back(pred);
        }
        std::cout << "\n";
        
    } else {
        std::cout << "\n[FAILED]: Architecture crashed reconstructing vocabulary sequence.\n";
    }

    return 0;
}
