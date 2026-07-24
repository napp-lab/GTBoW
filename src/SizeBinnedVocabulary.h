#ifndef __D_T__SIZEBINNEDVOCABULARY__
#define __D_T__SIZEBINNEDVOCABULARY__

#include <vector>
#include <string>
#include <cstring>
#include <opencv2/core/core.hpp>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "Vocabulary.h"
#include "AKMVocabulary.h"
#include "BowVector.h"
#include "VocabularyConfig.h"
#include <filesystem>
#include <sys/stat.h>
#include <sys/types.h>

namespace DBoW3 {

template<typename VocabType>
class SizeBinnedVocabulary {
public:
    SizeBinnedVocabulary(const VocabularyConfig& config) : num_scales_(10) {
        config_ = config;
    }

    SizeBinnedVocabulary(const std::string& input_dir) {
        *this = load(input_dir);
    }

    // Move constructor
    SizeBinnedVocabulary(SizeBinnedVocabulary&& other) noexcept 
        : config_(std::move(other.config_)), num_scales_(other.num_scales_),
          scale_bins_(std::move(other.scale_bins_)), vocab_(std::move(other.vocab_)) {}

    // Move assignment operator
    SizeBinnedVocabulary& operator=(SizeBinnedVocabulary&& other) noexcept {
        if (this != &other) {
            config_ = std::move(other.config_);
            num_scales_ = other.num_scales_;
            scale_bins_ = std::move(other.scale_bins_);
            vocab_ = std::move(other.vocab_);
        }
        return *this;
    }

    // Delete copy constructor and copy assignment 
    SizeBinnedVocabulary(const SizeBinnedVocabulary&) = delete;
    SizeBinnedVocabulary& operator=(const SizeBinnedVocabulary&) = delete;

    ~SizeBinnedVocabulary() {}

    void setNumScales(int n) { num_scales_ = n; }
    int getNumScales() const { return num_scales_; }

    void create(const std::vector<cv::Mat>& features, const std::vector<float>& scales, 
               const std::vector<std::string>& img_names) {
        // Determine scale bins if not already set
        if (scale_bins_.empty()) {
            determineScaleBins(scales);
        }

        // Create features vector for vocabulary creation
        std::vector<std::vector<cv::Mat>> training_features;
        for (const auto& feature_mat : features) {
            std::vector<cv::Mat> feature_vec;
            for (int i = 0; i < feature_mat.rows; i++) {
                feature_vec.push_back(feature_mat.row(i));
            }
            training_features.push_back(feature_vec);
        }

        // Create vocabulary based on type using SFINAE approach
        createVocabularyImpl(training_features, typename std::is_same<VocabType, Vocabulary>::type{});
    }

    void transform(const cv::Mat& features, const std::vector<float>& scales,
                  std::vector<int>& assignments, std::vector<double>& weights, BowVector& bow_vec) const {
        if (features.empty() || scales.empty()) {
            assignments.clear();
            weights.clear();
            bow_vec.clear();
            return;
        }

        // Digitize scales into bins
        std::vector<int> digitized_idx(scales.size());
        for (size_t i = 0; i < scales.size(); ++i) {
            if (scale_bins_.size() == num_scales_) {
                // When using unique scales, find exact matches
                auto it = std::find(scale_bins_.begin(), scale_bins_.end(), scales[i]);
                if (it != scale_bins_.end()) {
                    digitized_idx[i] = it - scale_bins_.begin();
                } else {
                    // Find closest scale
                    digitized_idx[i] = std::lower_bound(scale_bins_.begin(), scale_bins_.end(), scales[i]) - scale_bins_.begin();
                    if (digitized_idx[i] >= num_scales_) digitized_idx[i] = num_scales_ - 1;
                }
            } else {
                // For percentile-based bins, use upper_bound to match numpy's right=True
                digitized_idx[i] = std::upper_bound(scale_bins_.begin(), scale_bins_.end(), scales[i]) - scale_bins_.begin();
                if (digitized_idx[i] >= num_scales_) digitized_idx[i] = num_scales_ - 1;
            }
        }

        // Process each bin
        const int total_words = getNumWords();
        std::vector<double> combined_bow_vec(total_words, 0.0);
        size_t vocab_offset = 0;
        // Pre-allocate assignment and weight vectors to support dynamic r (soft assignments factor)
        // We defer resizing until the first bin gives us r, or we default to r=1
        size_t r = 0;
        std::vector<int> all_assignments;
        std::vector<double> all_weights;

        for (int bin_idx = 0; bin_idx < num_scales_; ++bin_idx) {
            // Find features in this bin
            std::vector<int> bin_indices;
            for (size_t i = 0; i < digitized_idx.size(); ++i) {
                if (digitized_idx[i] == bin_idx) {
                    bin_indices.push_back(i);
                }
            }

            if (!bin_indices.empty()) {
                // Zero-copy feature binning: build a Mat that references
                // original rows via pointer array instead of copying.
                cv::Mat bin_features;
                bin_features.create(static_cast<int>(bin_indices.size()), features.cols, features.type());
                for (size_t i = 0; i < bin_indices.size(); ++i) {
                    // Use memcpy for contiguous row data - avoids cv::Mat::copyTo overhead
                    std::memcpy(bin_features.ptr(static_cast<int>(i)),
                                features.ptr(bin_indices[i]),
                                features.cols * features.elemSize());
                }

                // Transform features using SFINAE approach for different vocabulary types
                BowVector bin_bow_vec;
                std::vector<int> bin_assignments;
                std::vector<double> bin_weights;
                
                transformVocabularyImpl(bin_features, bin_assignments, bin_weights, bin_bow_vec,
                                      typename std::is_same<VocabType, Vocabulary>::type{});
                // Set r and allocate the arrays exactly once based on the first processed bin
                if (r == 0 && !bin_assignments.empty() && features.rows > 0) {
                    r = bin_assignments.size() / bin_indices.size();
                    all_assignments.resize(features.rows * r, -1);
                    all_weights.resize(features.rows * r, 0.0);
                }

                // L2-normalize this bin's BoW vector before combining, so each
                // scale bin contributes equally regardless of feature count.
                // Without this, bins with many features dominate the final vector
                // and suppress complementary information from less-populated bins.
                double bin_norm = 0.0;
                for (const auto& pair : bin_bow_vec) {
                    bin_norm += pair.second * pair.second;
                }
                bin_norm = std::sqrt(bin_norm);

                // Add to combined bow vector with shifted indices.
                // BoW weights are always non-negative (TF-IDF × L1-normalized soft weights);
                // a negative value indicates a bug upstream.
                for (const auto& pair : bin_bow_vec) {
                    double w = (bin_norm > 0.0) ? pair.second / bin_norm : pair.second;
                    if (w < 0) {
                        throw std::runtime_error(
                            "Negative BoW weight encountered for word " +
                            std::to_string(pair.first) + " in size bin " +
                            std::to_string(bin_idx));
                    }
                    WordId shifted_id = pair.first + vocab_offset;
                    if (shifted_id < static_cast<WordId>(total_words)) {
                        combined_bow_vec[shifted_id] += w;
                    }
                }

                // Add assignments and weights directly into their original interleaved positions
                for (size_t i = 0; i < bin_indices.size(); ++i) {
                    int original_idx = bin_indices[i];
                    for(size_t j = 0; j < r; ++j) {
                        all_assignments[original_idx * r + j] = bin_assignments[i * r + j] + vocab_offset;
                        all_weights[original_idx * r + j] = bin_weights[i * r + j];
                    }
                }
            }

            vocab_offset += vocab_->getWordSize();
        }

        // Normalize the combined bow vector
        double norm = 0.0;
        for (int i = 0; i < total_words; ++i) {
            norm += combined_bow_vec[i] * combined_bow_vec[i];
        }
        norm = std::sqrt(norm);

        if (norm > 0.0) {
            for (int i = 0; i < total_words; ++i) {
                if (combined_bow_vec[i] > 0.0) {
                    double normalized_weight = combined_bow_vec[i] / norm;
                    if (normalized_weight < 0) {
                        throw std::runtime_error(
                            "Negative normalized BoW weight encountered for word " +
                            std::to_string(i));
                    }
                    if (normalized_weight > 0.0) {
                        bow_vec.addWeight(i, normalized_weight);
                    }
                }
            }
        }

        // Set output assignments and weights
        assignments = std::move(all_assignments);
        weights = std::move(all_weights);
    }

private:
    // SFINAE-based vocabulary creation for Vocabulary type
    void createVocabularyImpl(const std::vector<std::vector<cv::Mat>>& training_features, std::true_type) {
        // This is for regular Vocabulary
        vocab_ = std::make_unique<VocabType>(
            config_.vocab_params.k,
            config_.vocab_params.L,
            config_.vocab_params.weighting,
            L2_NORM  // Force L2 normalization for standard vocabulary
        );
        vocab_->create(training_features);
    }

    // SFINAE-based vocabulary creation for AKMVocabulary type  
    void createVocabularyImpl(const std::vector<std::vector<cv::Mat>>& training_features, std::false_type) {
        // This is for AKMVocabulary
        vocab_ = std::make_unique<VocabType>(
            config_.akm_params.vocab_size,
            config_.akm_params.max_iter,
            config_.akm_params.r,
            config_.akm_params.var,
            config_.akm_params.num_cores
        );
        vocab_->create(training_features);
    }

    // SFINAE-based vocabulary transform for Vocabulary type
    void transformVocabularyImpl(const cv::Mat& features, std::vector<int>& assignments, 
                               std::vector<double>& weights, BowVector& bow_vec, std::true_type) const {
        // For regular Vocabulary, use the standard transform method
        vocab_->transform(features, bow_vec);
        
        // Extract assignments and weights from the bow vector
        assignments.clear();
        weights.clear();
        assignments.resize(features.rows, -1);
        weights.resize(features.rows, 1.0); // Default weight is 1.0, default used for Vocabulary which doesn't do soft assignments
        
        // For each feature, find its closest word and weight
        for (int i = 0; i < features.rows; ++i) {
            WordId word_id = vocab_->transform(features.row(i));
            assignments[i] = word_id;
            if (word_id >= 0) {
                bow_vec.addWeight(word_id, 1.0);
            }
        }
        // L2 normalize the bow vector is removed per Bug 4 fix
    }

    // SFINAE-based vocabulary transform for AKMVocabulary type
    void transformVocabularyImpl(const cv::Mat& features, std::vector<int>& assignments,
                               std::vector<double>& weights, BowVector& bow_vec, std::false_type) const {
        // For AKMVocabulary, use the extended transform method
        vocab_->transform(features, assignments, weights, bow_vec);
    }

public:

    void save(const std::string& output_dir) {
        // Create output directory if it doesn't exist
        mkdir(output_dir.c_str(), 0755);

        // Save the size binned vocabulary configuration
        cv::FileStorage fs(output_dir + "/size_binned_vocabulary.yml", cv::FileStorage::WRITE);
        fs << "num_scales" << static_cast<int>(num_scales_);
        fs << "scale_bins" << scale_bins_;
        fs.release();

        // Save the vocabulary
        if (vocab_) {
            if (std::is_same<VocabType, Vocabulary>::value) {
                // For standard vocabulary, save to a file
                vocab_->save(output_dir + "/vocabulary.voc");
            } else {
                // For AKMVocabulary, save to a directory
                mkdir(output_dir.c_str(), 0755);
                vocab_->save(output_dir);
            }
        }
    }

    void load(const std::string& input_dir) {
        cv::FileStorage fs(input_dir + "/size_binned_vocabulary.yml", cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Failed to open file: " + input_dir + "/size_binned_vocabulary.yml");
        }

        fs["num_scales"] >> num_scales_;
        fs["scale_bins"] >> scale_bins_;
        fs.release();

        // Load the vocabulary
        if (std::is_same<VocabType, Vocabulary>::value) {
            vocab_ = std::make_unique<VocabType>(input_dir + "/vocabulary.voc");
        } else {
            vocab_ = std::make_unique<VocabType>(input_dir);
        }

        // // Print scale bins for debugging
        // std::cout << "C++ vocabulary scale_bins: ";
        // for (size_t i = 0; i < scale_bins_.size(); ++i) {
        //     std::cout << scale_bins_[i];
        //     if (i < scale_bins_.size() - 1) {
        //         std::cout << ", ";
        //     }
        // }
        // std::cout << std::endl;
    }

    static SizeBinnedVocabulary load(const std::string& input_dir, const VocabularyConfig& config) {
        cv::FileStorage fs(input_dir + "/size_binned_vocabulary.yml", cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Failed to open file: " + input_dir + "/size_binned_vocabulary.yml");
        }

        int num_scales;
        std::vector<float> scale_bins;

        fs["num_scales"] >> num_scales;
        fs["scale_bins"] >> scale_bins;
        fs.release();

        SizeBinnedVocabulary vocab(config);
        vocab.num_scales_ = num_scales;
        vocab.scale_bins_ = scale_bins;

        // Load the vocabulary using appropriate method
        if (std::is_same<VocabType, Vocabulary>::value) {
            vocab.vocab_ = std::make_unique<VocabType>(input_dir + "/vocabulary.voc");
        } else {
            // For AKMVocabulary, use static load method to avoid recursive constructor call
            auto loaded_vocab = VocabType::load(input_dir);
            vocab.vocab_ = std::make_unique<VocabType>(std::move(loaded_vocab));
            // Apply num_cores from config to loaded vocabulary
            vocab.setNumCoresImpl(config.akm_params.num_cores, std::false_type{});
        }
        
        // Print scale bins for debugging
        std::cout << "C++ vocabulary scale_bins: ";
        for (size_t i = 0; i < vocab.scale_bins_.size(); ++i) {
            std::cout << vocab.scale_bins_[i];
            if (i < vocab.scale_bins_.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;

        return vocab;
    }

    int getNumWords() const {
        return vocab_ ? vocab_->getWordSize() * num_scales_ : 0;
    }

    int getWordSize() const {
        return getNumWords();
    }

    bool isCreated() const {
        return vocab_ != nullptr;
    }

    // SFINAE-based setNumCores: no-op for standard Vocabulary, forwards to AKMVocabulary
    void setNumCores(int num_cores) {
        setNumCoresImpl(num_cores, typename std::is_same<VocabType, Vocabulary>::type{});
    }

private:
    void setNumCoresImpl(int /*num_cores*/, std::true_type) {
        // No-op for standard Vocabulary (HKM does not use FLANN cores)
    }
    void setNumCoresImpl(int num_cores, std::false_type) {
        if (vocab_) {
            vocab_->setNumCores(num_cores);
        }
    }

    void determineScaleBins(const std::vector<float>& scales) {
        if (scales.empty()) {
            throw std::runtime_error("No scales provided for binning");
        }

        // Get unique scales
        std::set<float> unique_scales_set(scales.begin(), scales.end());
        std::vector<float> unique_scales(unique_scales_set.begin(), unique_scales_set.end());
        
        if (unique_scales.size() < num_scales_) {
            // When we have fewer unique scales than desired bins, match Python exactly:
            // Python: self.scale_bins = np.unique(scales) and self.num_scales = len(self.scale_bins)
            // Use all unique scales as bin edges for np.digitize with right=True
            num_scales_ = unique_scales.size();
            scale_bins_ = unique_scales;
        } else {
            // Original percentile-based logic for bin edges - matches Python's np.percentile
            std::vector<float> sorted_scales = scales;
            std::sort(sorted_scales.begin(), sorted_scales.end());
            
            scale_bins_.clear();
            scale_bins_.resize(num_scales_ - 1);
            for (int i = 0; i < num_scales_ - 1; ++i) {
                float percentile = (i + 1) * 100.0f / num_scales_;
                float index_float = percentile * (sorted_scales.size() - 1) / 100.0f;
                size_t index_lower = static_cast<size_t>(std::floor(index_float));
                size_t index_upper = static_cast<size_t>(std::ceil(index_float));
                
                if (index_lower == index_upper || index_upper >= sorted_scales.size()) {
                    scale_bins_[i] = sorted_scales[index_lower];
                } else {
                    // Linear interpolation (like numpy.percentile)
                    float weight = index_float - index_lower;
                    scale_bins_[i] = sorted_scales[index_lower] * (1.0f - weight) + 
                                    sorted_scales[index_upper] * weight;
                }
            }
        }

        // // Print scale bins for debugging
        // std::cout << "C++ vocabulary scale_bins: ";
        // for (size_t i = 0; i < scale_bins_.size(); ++i) {
        //     std::cout << scale_bins_[i];
        //     if (i < scale_bins_.size() - 1) {
        //         std::cout << ", ";
        //     }
        // }
        // std::cout << std::endl;
    }

    VocabularyConfig config_;
    int num_scales_;
    std::vector<float> scale_bins_;
    std::unique_ptr<VocabType> vocab_;
};

} // namespace DBoW3

#endif // __D_T__SIZEBINNEDVOCABULARY__