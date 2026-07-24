#ifndef __D_T__VOCABULARY_CONFIG__
#define __D_T__VOCABULARY_CONFIG__

#include <memory>
#include <flann/flann.hpp>

// Forward-declare the classes that will be created.
namespace DBoW3 {
    class Vocabulary;
    template<class TDescriptor> class AKMVocabulary;
}

// Include the full definitions of the classes the factory will build.
#include "Vocabulary.h"
#include "AKMVocabulary.h"

namespace DBoW3 {

//=============================================================================
// 1. PARAMETER STRUCTS
//=============================================================================

/// @brief Parameters for creating a DBoW3::Vocabulary instance.
struct VocabularyParameters {
    int k;
    int L;
    WeightingType weighting;
    ScoringType scoring;
    std::string file_path;  // Path to load vocabulary from, empty means create new
    
    // C++14 compatible constructor
    VocabularyParameters() : k(10), L(5), weighting(TF_IDF), scoring(L1_NORM), file_path("") {}
    
    VocabularyParameters(int k_val, int L_val, WeightingType w, ScoringType s, const std::string& path = "")
        : k(k_val), L(L_val), weighting(w), scoring(s), file_path(path) {}
};

/// @brief Parameters for creating a DBoW3::AKMVocabulary instance.
struct AKMVocabularyParameters {
    int vocab_size;
    int max_iter;
    int r;
    double var;
    int num_cores;
    std::string file_path;  // Path to load vocabulary from, empty means create new
    
    // C++14 compatible constructor
    AKMVocabularyParameters() : vocab_size(10000), max_iter(200), r(1), var(1.0), num_cores(1), file_path("") {}
    
    AKMVocabularyParameters(int vs, int mi, int r_val, double v, int nc = 1, const std::string& path = "")
        : vocab_size(vs), max_iter(mi), r(r_val), var(v), num_cores(nc), file_path(path) {}
};

enum VocabularyType {  // C++14 compatible enum (not enum class)
    VOCABULARY,
    AKM_VOCABULARY
};

/// @brief Main configuration struct that can hold either vocabulary type parameters
struct VocabularyConfig {
    VocabularyType type;
    union {
        VocabularyParameters vocab_params;
        AKMVocabularyParameters akm_params;
    };
    
    // Constructor for Vocabulary parameters
    VocabularyConfig(const VocabularyParameters& params) : type(VOCABULARY), vocab_params(params) {}
    
    // Constructor for AKMVocabulary parameters
    VocabularyConfig(const AKMVocabularyParameters& params) : type(AKM_VOCABULARY), akm_params(params) {}
    
    // Default constructor
    VocabularyConfig() : type(VOCABULARY), vocab_params() {}
    
    // Copy constructor
    VocabularyConfig(const VocabularyConfig& other) : type(other.type) {
        if (type == VOCABULARY) {
            new (&vocab_params) VocabularyParameters(other.vocab_params);
        } else {
            new (&akm_params) AKMVocabularyParameters(other.akm_params);
        }
    }
    
    // Assignment operator
    VocabularyConfig& operator=(const VocabularyConfig& other) {
        if (this != &other) {
            // Handle union member destruction and construction properly
            if (type == VOCABULARY) {
                vocab_params.~VocabularyParameters();
            } else {
                akm_params.~AKMVocabularyParameters();
            }
            
            type = other.type;
            
            if (type == VOCABULARY) {
                new (&vocab_params) VocabularyParameters(other.vocab_params);
            } else {
                new (&akm_params) AKMVocabularyParameters(other.akm_params);
            }
        }
        return *this;
    }
    
    // Destructor
    ~VocabularyConfig() {
        // Properly destruct union members
        if (type == VOCABULARY) {
            vocab_params.~VocabularyParameters();
        } else {
            akm_params.~AKMVocabularyParameters();
        }
    }
};


//=============================================================================
// 2. VOCABULARY FACTORY
//=============================================================================

// Alias for the specific AKMVocabulary type for clarity
using BinaryAKMVocabulary = AKMVocabulary<flann::Hamming<uint8_t>>;

// Generic factory template (undefined)
template<typename VocabType, typename ParamsType>
struct VocabularyFactory;

// Specialization for DBoW3::Vocabulary
template<>
struct VocabularyFactory<Vocabulary, VocabularyParameters> {
    static std::unique_ptr<Vocabulary> create(const VocabularyParameters& params) {
        if (!params.file_path.empty()) {
            return std::make_unique<Vocabulary>(params.file_path);
        }
        return std::make_unique<Vocabulary>(params.k, params.L, params.weighting, params.scoring);
    }
};

// Specialization for DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>
template<>
struct VocabularyFactory<BinaryAKMVocabulary, AKMVocabularyParameters> {
    static std::unique_ptr<BinaryAKMVocabulary> create(const AKMVocabularyParameters& params) {
        if (!params.file_path.empty()) {
            return std::make_unique<BinaryAKMVocabulary>(params.file_path);
        }
        return std::make_unique<BinaryAKMVocabulary>(params.vocab_size, params.max_iter, params.r, params.var, params.num_cores);
    }
};

} // namespace DBoW3

#endif // __D_T__VOCABULARY_CONFIG__