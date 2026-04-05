#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <random>
#include <cmath>

#define DIM 10000

using HDVector = std::vector<int>;

// Generate random 1-bit (+1 / -1) hypervector
HDVector generate_random_vector(std::mt19937& gen) {
    HDVector v(DIM);
    std::uniform_int_distribution<> dist(0, 1);
    for (int i = 0; i < DIM; i++) {
        v[i] = dist(gen) == 1 ? 1 : -1;
    }
    return v;
}

// BIND: Element-wise multiplication (equivalent to XOR for +1/-1)
HDVector bind_vectors(const HDVector& a, const HDVector& b) {
    HDVector v(DIM);
    for (int i = 0; i < DIM; i++) v[i] = a[i] * b[i];
    return v;
}

// PERMUTE: Circular shift array to represent SEQUENCE order
HDVector permute_vector(const HDVector& a, int shifts = 1) {
    HDVector v(DIM);
    for (int i = 0; i < DIM; i++) {
        v[(i + shifts) % DIM] = a[i];
    }
    return v;
}

// BUNDLE (Superposition): Adding vectors and taking the sign threshold
HDVector bundle_vectors(const std::vector<HDVector>& vectors) {
    std::vector<int> sum(DIM, 0);
    for (const auto& vec : vectors) {
        for (int i = 0; i < DIM; i++) {
            sum[i] += vec[i];
        }
    }
    
    HDVector bundled(DIM);
    for (int i = 0; i < DIM; i++) {
        bundled[i] = sum[i] >= 0 ? 1 : -1;
    }
    return bundled;
}

// Cosine Similarity to find nearest match
double cosine_similarity(const HDVector& a, const HDVector& b) {
    double dot = 0;
    for (int i = 0; i < DIM; i++) dot += a[i] * b[i];
    // Magnitudes are always exactly DIM for 1/-1 vectors
    return dot / DIM; 
}

int main() {
    std::cout << "--- PHASE 8: 1-BIT HYPERDIMENSIONAL COMPUTING ---\n";
    std::cout << "[INFO]: Creating 10,000-Dimension Superstate...\n\n";

    std::mt19937 gen(42);
    std::string text = "sovereign_hive_engine_symbolic_quantization";
    
    // 1. Create Base Item Memory (Dictionary)
    std::map<char, HDVector> item_memory;
    for (char c : text) {
        if (item_memory.find(c) == item_memory.end()) {
            item_memory[c] = generate_random_vector(gen);
        }
    }

    // 2. ONE-SHOT LEARNING: Encode Key-Value properties directly into Superposition
    std::vector<HDVector> memory_pool;
    
    // We will map Tetragrams: Context (Current Length 3) -> Next Character
    for (size_t i = 0; i < text.size() - 3; ++i) {
        char c1 = text[i];
        char c2 = text[i+1];
        char c3 = text[i+2];
        char value = text[i+3];

        // Encode Key: Bind Permuted(c1, 2) with Permuted(c2, 1) and c3
        HDVector key1 = bind_vectors(permute_vector(item_memory[c1], 2), permute_vector(item_memory[c2], 1));
        HDVector key = bind_vectors(key1, item_memory[c3]);
        
        // Encode state: Bind Key to Value
        HDVector state = bind_vectors(key, item_memory[value]);
        
        // Add to superposition pool
        memory_pool.push_back(state);
    }
    
    // 3. COLLAPSE TO SUPERSTATE RAM:
    // Literally squashing 40+ Tetragram Pathways into exactly ONE 10,000D 1-BIT Vector
    HDVector SUPER_VECTOR = bundle_vectors(memory_pool);

    std::cout << "[STATUS]: Training complete. Model absorbed entire sequence in ONE pass.\n";
    std::cout << "[STATUS]: Parameters collapsed to exactly 10,000 pure 1-Bit logic bounds.\n\n";

    // 4. SEQUENTIAL RETRIEVAL (Querying the Superstate)
    int correct = 0;
    int total = text.size() - 3;

    std::cout << "--- SUPERSTATE QUERY VALIDATION ---\n";
    for (size_t i = 0; i < text.size() - 3; ++i) {
        char c1 = text[i];
        char c2 = text[i+1];
        char c3 = text[i+2];
        char actual_target = text[i+3];
        
        // Re-generate the identical Key sequence
        HDVector key1 = bind_vectors(permute_vector(item_memory[c1], 2), permute_vector(item_memory[c2], 1));
        HDVector query_key = bind_vectors(key1, item_memory[c3]);
        
        // Unbind the Query from the massive SUPER_VECTOR memory overlay
        HDVector noisy_retrieval = bind_vectors(SUPER_VECTOR, query_key);
        
        // Find nearest matching character in Item Memory
        char best_match = '?';
        double best_sim = -1.0;
        for (const auto& pair : item_memory) {
            double sim = cosine_similarity(noisy_retrieval, pair.second);
            if (sim > best_sim) {
                best_sim = sim;
                best_match = pair.first;
            }
        }
        
        if (best_match == actual_target) correct++;
        
        // Print first 5 tests to show mapping
        if(i < 5) {
            std::cout << "Querying Prefix: '" << c1 << c2 << c3 << "' -> Extracted: '" << best_match << "' [Expected: '" << actual_target << "'] | Confidence: " << best_sim << "\n";
        }
    }
    
    double accuracy = (double)correct / total * 100.0;
    std::cout << "...\n\n--- HDC 1-BIT VALIDATION ---\n";
    std::cout << "Superstate Retrieval Accuracy: " << correct << " / " << total << " (" << accuracy << "%)\n";
    
    if (accuracy > 95.0) {
        std::cout << "\n[SUCCESS]: HDC Superstate Verified. Sequence successfully compressed and retrieved inside a localized vector without Epoch training.\n";
    } else {
        std::cout << "\n[FAILED]: Superstate degradation. Vector unable to unbind accurately.\n";
    }

    return 0;
}
