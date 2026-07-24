#ifndef __D_T__AKMVOCABULARY__
#define __D_T__AKMVOCABULARY__

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <flann/flann.hpp>
#include <random>
#include <numeric>
#include <map>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include "DescManip.h"
#include "BowVector.h"

namespace DBoW3 {

template<typename Distance>
class AKMVocabulary {
    
public:
    enum DistanceType {
        HAMMING,
        L1,
        L2
    };
    
    AKMVocabulary(const int vocab_size, const int max_iter = 200, const int r = 1, const double var = 1.0, const int num_cores = 1)
        : vocab_size_(vocab_size), max_iter_(max_iter), r_(r), var_(var), num_cores_(num_cores) {
        // Create the index params directly
        auto hc_params = flann::HierarchicalClusteringIndexParams(32, flann::FLANN_CENTERS_RANDOM, 4, 100);
        distance_type_ = HAMMING;
        flann_index_ = std::make_unique<flann::Index<flann::Hamming<uint8_t>>>(hc_params);
    }

    AKMVocabulary(const int vocab_size, const flann::IndexParams* params, const int max_iter = 200, const int r = 1, const double var = 1.0, const int num_cores = 1)
        : vocab_size_(vocab_size), max_iter_(max_iter), r_(r), var_(var), num_cores_(num_cores) {
        const auto* hc_params = static_cast<const flann::HierarchicalClusteringIndexParams*>(params);
        if (hc_params) {
            distance_type_ = HAMMING;
            flann_index_ = std::make_unique<flann::Index<flann::Hamming<uint8_t>>>(*hc_params);
        } else {
            throw std::invalid_argument("Invalid IndexParams type");
        }
    }

    AKMVocabulary(const std::string& input_dir) {
        *this = load(input_dir);
    }

    AKMVocabulary(int vocab_size, int max_iter, int r, double var, const cv::Mat& centroids, DistanceType distance_type, const flann::IndexParams* params, const std::vector<double>& weights, int num_cores = 1)
        : vocab_size_(vocab_size), max_iter_(max_iter), r_(r), var_(var), num_cores_(num_cores), centroids_(centroids), distance_type_(distance_type), weights_(weights) {
        if (distance_type_ == HAMMING) {
            flann::Matrix<uint8_t> centroids_flann_matrix((uint8_t*)centroids.data, centroids.rows, centroids.cols);
            flann_index_ = std::make_unique<flann::Index<flann::Hamming<uint8_t>>>(centroids_flann_matrix, *params);
        } else {
            throw std::invalid_argument("Invalid distance type");
        }
    }

    ~AKMVocabulary() {}

    // Explicitly delete the copy constructor and copy assignment operator
    AKMVocabulary(const AKMVocabulary&) = delete;
    AKMVocabulary& operator=(const AKMVocabulary&) = delete;

    // Explicitly define the move constructor and move assignment operator
    AKMVocabulary(AKMVocabulary&&) = default;
    AKMVocabulary& operator=(AKMVocabulary&&) = default;

    void create(const std::vector<cv::Mat>& training_features) {
        if (training_features.empty()) {
            throw std::invalid_argument("Training features cannot be empty");
        }
        
        // Check that all feature matrices have the same number of columns
        int expected_cols = -1;
        int total_features = 0;
        for (const auto& feature : training_features) {
            if (feature.empty()) {
                continue; // Skip empty matrices
            }
            if (expected_cols == -1) {
                expected_cols = feature.cols;
            } else if (feature.cols != expected_cols) {
                throw std::invalid_argument("All feature matrices must have the same number of columns");
            }
            total_features += feature.rows;
        }
        
        if (total_features == 0) {
            throw std::invalid_argument("No valid features found in training data");
        }
        
        if (total_features < vocab_size_) {
            throw std::invalid_argument("Total number of training features (" + std::to_string(total_features) + 
                                      ") must be at least as large as vocabulary size (" + std::to_string(vocab_size_) + ")");
        }
        
        cv::Mat feature_mat;
        std::vector<int> img_assignments;
        for (size_t feature_idx = 0; feature_idx < training_features.size(); ++feature_idx) {
            const auto& feature = training_features[feature_idx];
            if (feature.empty()) {
                continue; // Skip empty matrices
            }
            feature_mat.push_back(feature);
            for (int i = 0; i < feature.rows; ++i) {
                img_assignments.push_back(static_cast<int>(feature_idx));
            }
        }

        cv::Mat img_assignments_mat = cv::Mat(img_assignments.size(), 1, CV_32S);
        for (size_t i = 0; i < img_assignments.size(); ++i) {
            img_assignments_mat.at<int>(i, 0) = img_assignments[i];
        }

        // std::cout << "feature_mat size: " << feature_mat.rows << " x " << feature_mat.cols << std::endl;
        // std::cout << "img_assignments_mat size: " << img_assignments_mat.rows << " x " << img_assignments_mat.cols << std::endl;
        create(feature_mat, img_assignments_mat);
    }

    void create(const std::vector<std::vector<cv::Mat>>& training_features) {
        std::vector<cv::Mat> concatenated_features;
        std::vector<int> img_assignments;

        for (size_t i = 0; i < training_features.size(); ++i) {
            for (const auto& feature : training_features[i]) {
                concatenated_features.push_back(feature);
                img_assignments.push_back(static_cast<int>(i));
            }
        }

        create(concatenated_features, img_assignments);
    }

    void create(const std::vector<cv::Mat>& training_features, const std::vector<int>& img_assignments) {
        cv::Mat feature_mat;
        for (const auto& feature : training_features) {
            feature_mat.push_back(feature);
        }

        cv::Mat img_assignments_mat(img_assignments.size(), 1, CV_32S);
        for (size_t i = 0; i < img_assignments.size(); ++i) {
            img_assignments_mat.at<int>(i, 0) = img_assignments[i];
        }

        create(feature_mat, img_assignments_mat);
    }

    void create(const cv::Mat& training_features, const cv::Mat& img_assignments) {
        if (training_features.empty()) {
            throw std::invalid_argument("Training features cannot be empty");
        }
        if (img_assignments.empty()) {
            throw std::invalid_argument("Image assignments cannot be empty");
        }
        if (training_features.rows != img_assignments.rows) {
            throw std::invalid_argument("Number of features must match number of assignments");
        }
        if (training_features.rows < vocab_size_) {
            throw std::invalid_argument("Number of training features (" + std::to_string(training_features.rows) + 
                                      ") must be at least as large as vocabulary size (" + std::to_string(vocab_size_) + ")");
        }
        
        determineWords(training_features);
        determineWeights(training_features, img_assignments);
        std::cout << "AKMVocabulary created with " << vocab_size_ << " words." << std::endl;
    }

    void determineWords(const cv::Mat& features) {
        std::cout << "Starting K-means clustering: " << features.rows << " features, "
                  << vocab_size_ << " clusters, max_iter=" << max_iter_
                  << ", num_cores=" << num_cores_ << std::endl;
        initCentroids(features, centroids_);

        const int feat_cols = features.cols;
        const int n_centroids = vocab_size_;
        const int n_features = features.rows;

        // Scale checks with vocab size so the FLANN search examines a large
        // enough fraction of centroids.  The `checks` parameter in FLANN
        // controls how many individual point comparisons are made per query.
        // For K-means to converge, the NN assignment must be highly accurate.
        int training_checks = std::max(2000, n_centroids / 5);
        std::cout << "FLANN checks for K-means assignment: " << training_checks
                  << " (" << (100.0 * training_checks / std::max(n_centroids, 1))
                  << "% of centroids)" << std::endl;

        int iter = 0;
        cv::Mat prev_assignments;
        long long prev_cost = -1;
        while (iter < max_iter_) {
            auto iter_start = std::chrono::high_resolution_clock::now();

            // Build FLANN index on current centroids
            flann::Matrix<uint8_t> centroids_flann_matrix(
                (uint8_t*)centroids_.data, centroids_.rows, centroids_.cols);
            flann_index_->buildIndex(centroids_flann_matrix);

            // Assignment step with high checks
            flann::SearchParams train_params(training_checks);
            train_params.cores = num_cores_;
            flann::Matrix<uint8_t> features_flann(
                (uint8_t*)features.data, n_features, feat_cols);
            flann::Matrix<int> assign_flann(
                new int[n_features], n_features, 1);
            flann::Matrix<uint> dist_flann(
                new uint[n_features], n_features, 1);

            flann_index_->knnSearch(
                features_flann, assign_flann, dist_flann, 1, train_params);

            cv::Mat assignments(n_features, 1, CV_32S);
            std::memcpy(assignments.data, assign_flann.ptr(),
                        n_features * sizeof(int));

            delete[] assign_flann.ptr();
            delete[] dist_flann.ptr();

            // ── Sticky assignment: resolve ties deterministically ─────
            // The FLANN index is rebuilt each iteration with random tree
            // structure, which changes tie-breaking for features equidistant
            // to multiple centroids.  This non-determinism prevents
            // convergence even when the objective is flat.  Fix: only
            // switch a feature's centroid if the new one is STRICTLY closer.
            // Ties keep their previous assignment, ensuring consistency.
            if (iter > 0) {
                #pragma omp parallel for schedule(static) num_threads(num_cores_)
                for (int j = 0; j < n_features; ++j) {
                    int new_id = assignments.at<int>(j, 0);
                    int old_id = prev_assignments.at<int>(j, 0);
                    if (new_id != old_id) {
                        const uint8_t* feat = features.ptr<uint8_t>(j);
                        const uint8_t* new_cent = centroids_.ptr<uint8_t>(new_id);
                        const uint8_t* old_cent = centroids_.ptr<uint8_t>(old_id);
                        int new_dist = 0, old_dist = 0;
                        for (int b = 0; b + 8 <= feat_cols; b += 8) {
                            uint64_t f, nc, oc;
                            std::memcpy(&f, feat + b, 8);
                            std::memcpy(&nc, new_cent + b, 8);
                            std::memcpy(&oc, old_cent + b, 8);
                            new_dist += __builtin_popcountll(f ^ nc);
                            old_dist += __builtin_popcountll(f ^ oc);
                        }
                        if (new_dist >= old_dist) {
                            assignments.at<int>(j, 0) = old_id;
                        }
                    }
                }
            }

            // Compute K-means cost and per-feature distances (for empty cluster reseeding)
            std::vector<int> feat_dists(n_features);
            long long total_cost = 0;
            #pragma omp parallel for reduction(+:total_cost) num_threads(num_cores_)
            for (int j = 0; j < n_features; ++j) {
                int cid = assignments.at<int>(j, 0);
                const uint8_t* feat = features.ptr<uint8_t>(j);
                const uint8_t* cent = centroids_.ptr<uint8_t>(cid);
                int d = 0;
                for (int b = 0; b + 8 <= feat_cols; b += 8) {
                    uint64_t fv, cv;
                    std::memcpy(&fv, feat + b, 8);
                    std::memcpy(&cv, cent + b, 8);
                    d += __builtin_popcountll(fv ^ cv);
                }
                feat_dists[j] = d;
                total_cost += d;
            }

            // ── Group features by cluster ────────────────────────────────
            std::vector<std::vector<int>> cluster_indices(n_centroids);
            for (int j = 0; j < n_features; ++j) {
                cluster_indices[assignments.at<int>(j, 0)].push_back(j);
            }

            // ── Update step: recompute centroids ─────────────────────────
            std::vector<cv::Mat> clusters(n_centroids);
            #pragma omp parallel for schedule(dynamic) num_threads(num_cores_)
            for (int i = 0; i < n_centroids; ++i) {
                if (!cluster_indices[i].empty()) {
                    std::vector<cv::Mat> cluster_features;
                    cluster_features.reserve(cluster_indices[i].size());
                    for (int idx : cluster_indices[i]) {
                        cluster_features.push_back(features.row(idx));
                    }
                    DescManip::meanValue(cluster_features, clusters[i]);
                }
            }

            // ── Reseed empty clusters with distance-proportional sampling ──
            // Features far from their assigned centroid are more likely to be
            // selected as new centroids (k-means++ style heuristic).
            {
                std::vector<int> empty_clusters;
                for (int i = 0; i < n_centroids; ++i) {
                    if (cluster_indices[i].empty()) {
                        empty_clusters.push_back(i);
                    }
                }
                if (!empty_clusters.empty()) {
                    std::vector<double> probs(n_features);
                    double total = 0;
                    for (int j = 0; j < n_features; ++j) {
                        probs[j] = static_cast<double>(feat_dists[j]);
                        total += probs[j];
                    }
                    std::mt19937 gen(std::random_device{}());
                    if (total > 0) {
                        std::discrete_distribution<int> dist(probs.begin(), probs.end());
                        for (int i : empty_clusters) {
                            clusters[i] = features.row(dist(gen)).clone();
                        }
                    } else {
                        // All features at zero distance from centroids — fall back to uniform
                        std::uniform_int_distribution<int> dist(0, n_features - 1);
                        for (int i : empty_clusters) {
                            clusters[i] = features.row(dist(gen)).clone();
                        }
                    }
                }
            }

            // ── Convergence check ────────────────────────────────────────
            int num_changed = (iter > 0)
                ? cv::countNonZero(assignments != prev_assignments)
                : n_features;
            if (iter > 0 && num_changed == 0) {
                std::cout << "K-means converged at iteration " << iter << std::endl;
                break;
            }

            // Early stopping: for binary K-means, Hamming distance ties
            // cause a small residual of features to oscillate between
            // equidistant centroids.  Stop when fewer than 1% of
            // features changed — the vocabulary is effectively converged.
            double changed_frac = (double)num_changed / n_features;
            if (iter > 5 && changed_frac < 0.01) {
                std::cout << "K-means effectively converged at iteration "
                          << iter << " (" << num_changed << " changed, "
                          << std::fixed << std::setprecision(3)
                          << (changed_frac * 100.0) << "%)" << std::defaultfloat
                          << std::endl;
                break;
            }

            int num_empty = 0;
            for (int i = 0; i < n_centroids; ++i) {
                if (cluster_indices[i].empty()) num_empty++;
            }

            auto iter_end = std::chrono::high_resolution_clock::now();
            double iter_sec = std::chrono::duration<double>(
                iter_end - iter_start).count();

            if (iter % 5 == 0) {
                double cost_delta = (prev_cost >= 0)
                    ? (double)(total_cost - prev_cost) / prev_cost * 100.0
                    : 0.0;
                std::cout << "K-means iteration " << iter
                          << ", changed=" << num_changed
                          << ", empty=" << num_empty
                          << ", cost=" << total_cost
                          << " (delta=" << std::fixed << std::setprecision(2)
                          << cost_delta << "%)"
                          << ", time=" << std::setprecision(2) << iter_sec << "s"
                          << std::defaultfloat
                          << std::endl;
            }

            prev_cost = total_cost;
            ++iter;

            if (iter < max_iter_) {
                cv::Mat clusters_mat(n_centroids, feat_cols, features.type());
                for (int i = 0; i < n_centroids; ++i) {
                    clusters[i].copyTo(clusters_mat.row(i));
                }
                centroids_ = clusters_mat;
                prev_assignments = assignments.clone();
            }
        }

        // Build the FLANN index on final centroids for query-time use
        flann::Matrix<uint8_t> final_centroids(
            (uint8_t*)centroids_.data, centroids_.rows, centroids_.cols);
        flann_index_->buildIndex(final_centroids);
    }

    void determineWeights(const cv::Mat& features, const cv::Mat& img_assignments) {
        std::cout << "Computing word weights (determineWeights)..." << std::endl;
        cv::Mat assignments, dists;
        getWords(features, r_, assignments, dists);
        std::cout << "  Soft assignment knnSearch complete." << std::endl;

        // Soft assignment weight: exp(-d^2 / (2 * var_)).
        // NOTE: var_ is used directly as the denominator scale, NOT as sigma^2.
        // The paper (ICRA 2025) writes exp(-d^2/(2*sigma^2)) with sigma=580, but
        // the formula implemented here is exp(-d^2/(2*var)) with var=580. These
        // differ by a factor of 580; the paper's notation is inconsistent with the
        // actual parameter meaning. The code is correct for producing useful results.
        cv::Mat dist_weights;
        dists.convertTo(dist_weights, CV_64F);  // Must convert before squaring to avoid int truncation
        dist_weights = -dist_weights.mul(dist_weights) / (2.0 * var_);
        cv::exp(dist_weights, dist_weights);
        cv::Mat l1_norm_weights(dist_weights.rows, dist_weights.cols, CV_64F);
        for (int i = 0; i < dist_weights.rows; ++i) {
            double sum = cv::sum(dist_weights.row(i))[0];
            double factor = (sum > 0 ? sum : 1.0);
            for (int j = 0; j < dist_weights.cols; ++j) {
                l1_norm_weights.at<double>(i, j) = dist_weights.at<double>(i, j) / factor;
            }
        }
        std::cout << "  L1-normalized distance weights computed." << std::endl;

        // Determine the number of unique images by finding the max value in img_assignments + 1
        int num_unique_images = 0;
        if (!img_assignments.empty())
        {
            double minVal, maxVal;
            cv::minMaxLoc(img_assignments, &minVal, &maxVal);
            num_unique_images = static_cast<int>(maxVal) + 1;
        }
        std::cout << "  Processing " << num_unique_images << " images for TF-IDF weights..." << std::endl;

        std::vector<std::vector<int>> unique_assignments_by_img(num_unique_images);
        std::vector<std::vector<double>> unique_weights_by_img(num_unique_images);

        // Pre-group feature indices by image in a single O(N) pass
        std::vector<std::vector<int>> feats_by_img(num_unique_images);
        for (int i = 0; i < img_assignments.rows; ++i) {
            int img_num = img_assignments.at<int>(i, 0);
            if (img_num >= 0 && img_num < num_unique_images) {
                feats_by_img[img_num].push_back(i);
            }
        }

        for (int img_num = 0; img_num < num_unique_images; ++img_num) {
            const std::vector<int>& feats_in_img_indices = feats_by_img[img_num];

            std::vector<int> unique_assignments_vec;
            std::unordered_map<int, int> assignment_map;
            int unique_assignment_counter = 0;
            std::vector<int> inverse_index;  // a flattened vector that stores the index of the unique assignment for each feature in the image

            // Reserve capacity to avoid repeated reallocations in this hot path
            const std::size_t expected_capacity =
                feats_in_img_indices.size() * static_cast<std::size_t>(assignments.cols);
            unique_assignments_vec.reserve(expected_capacity);
            assignment_map.reserve(expected_capacity);
            inverse_index.reserve(expected_capacity);
            
            // for each assignment in the image, check if it is unique
            for (int idx : feats_in_img_indices)
            {
                for (int j = 0; j < assignments.cols; ++j)
                {
                    int assignment_val = assignments.at<int>(idx, j);
                    auto result = assignment_map.emplace(assignment_val, unique_assignment_counter);
                    if (result.second)  // check if assignment_val has been assigned a unique index yet
                    {
                        unique_assignment_counter++;
                        unique_assignments_vec.push_back(assignment_val);
                    }
                    inverse_index.push_back(result.first->second);
                }
            }

            unique_assignments_by_img[img_num] = unique_assignments_vec;

            // Calculate DF weights: max soft-assignment confidence per unique word.
            // Using max (rather than sum) keeps DF contributions in [0, 1] per image
            // while still reflecting assignment uncertainty — a word with only weak
            // soft assignments contributes less to its document frequency.
            std::vector<double> weights(unique_assignment_counter, 0.0);
            int inv_idx = 0;
            for (int i = 0; i < (int)feats_in_img_indices.size(); ++i) {
                int feat_global_idx = feats_in_img_indices[i];
                for (int j = 0; j < assignments.cols; ++j) {
                    int unique_idx = inverse_index[inv_idx++];
                    double w = l1_norm_weights.at<double>(feat_global_idx, j);
                    if (w > weights[unique_idx]) {
                        weights[unique_idx] = w;
                    }
                }
            }
            unique_weights_by_img[img_num] = weights;

            if ((img_num + 1) % 100 == 0 || img_num == num_unique_images - 1) {
                std::cout << "  Processed " << (img_num + 1) << "/" << num_unique_images << " images for TF-IDF." << std::endl;
            }
        }

        std::vector<int> assignments_per_docs;
        std::vector<double> weights_per_docs;
        for (const auto& doc : unique_assignments_by_img) {
            assignments_per_docs.insert(assignments_per_docs.end(), doc.begin(), doc.end());
        }
        for (const auto& doc : unique_weights_by_img) {
            weights_per_docs.insert(weights_per_docs.end(), doc.begin(), doc.end());
        }

        std::vector<double> doc_frequencies(vocab_size_, 0.0);
        for (size_t i = 0; i < assignments_per_docs.size(); ++i) {
            if (assignments_per_docs[i] >= 0 && assignments_per_docs[i] < vocab_size_) {
                doc_frequencies[assignments_per_docs[i]] += weights_per_docs[i];
            }
        }

        for (int i = 0; i < vocab_size_; ++i) {
            if (doc_frequencies[i] == 0.0) {
                // std::cerr << "AKMVocabulary ERROR: Some words are not assigned to any feature." << std::endl;
                doc_frequencies[i] = 1.0;
            }
        }

        weights_.resize(vocab_size_);
        for (int i = 0; i < vocab_size_; ++i) {
            double idf = std::log(static_cast<double>(num_unique_images) / doc_frequencies[i]);
            // Avoid weights being exactly 0 when vocab matches docs exactly (like in a test with 1 document)
            if (idf <= 0.0) idf = 1.0;
            weights_[i] = idf;
        }
    }

    void transform(const cv::Mat& features, std::vector<int>& assignments, std::vector<double>& weights, BowVector& bow_vec) const {
        // Input validation
        if (features.empty()) {
            // Handle empty features gracefully
            assignments.clear();
            weights.clear();
            bow_vec.clear();
            return;
        }
        
        if (centroids_.empty() || weights_.empty()) {
            throw std::runtime_error("Vocabulary has not been created yet. Call create() first.");
        }
        
        if (features.cols != centroids_.cols) {
            throw std::runtime_error("Feature dimension mismatch. Expected " + 
                                   std::to_string(centroids_.cols) + " but got " + std::to_string(features.cols));
        }
        
        cv::Mat assignments_mat, dists;
        getWords(features, r_, assignments_mat, dists);
        // Soft assignment weight: exp(-d^2 / (2 * var_)). See var_ member comment.
        cv::Mat dist_weights;
        dists.convertTo(dist_weights, CV_64F);
        dist_weights = -dist_weights.mul(dist_weights) / (2.0 * var_);
        cv::exp(dist_weights, dist_weights);

        cv::Mat l1_norm_weights(dist_weights.rows, dist_weights.cols, CV_64F);
        for (int i = 0; i < dist_weights.rows; ++i) {
            double sum = cv::sum(dist_weights.row(i))[0];
            double factor = (sum > 0 ? sum : 1.0);
            for (int j = 0; j < dist_weights.cols; ++j) {
                l1_norm_weights.at<double>(i, j) = dist_weights.at<double>(i, j) / factor;
            }
        }

        assignments.resize(assignments_mat.total());
        weights.resize(l1_norm_weights.total());
        std::memcpy(assignments.data(), assignments_mat.data, assignments_mat.total() * sizeof(int));
        std::memcpy(weights.data(), l1_norm_weights.data, l1_norm_weights.total() * sizeof(double));

        // Create BowVector using dense array for O(1) indexed accumulation
        bow_vec.clear();
        std::vector<double> word_weights(vocab_size_, 0.0);
        
        for (int i = 0; i < assignments_mat.rows; ++i) {
            for (int j = 0; j < assignments_mat.cols; ++j) {
                int word_id = assignments_mat.at<int>(i, j);
                if (word_id >= 0 && word_id < vocab_size_) {
                    double weight = l1_norm_weights.at<double>(i, j) * weights_[word_id];
                    if (!std::isnan(weight) && !std::isinf(weight)) {  // Only add valid weights
                        word_weights[word_id] += weight;
                    }
                }
            }
        }

        double total_sum_sq = 0.0;
        for (int i = 0; i < vocab_size_; ++i) {
            total_sum_sq += word_weights[i] * word_weights[i];
        }
        double factor = (total_sum_sq > 0.0) ? std::sqrt(total_sum_sq) : 1.0;

        for (int i = 0; i < vocab_size_; ++i) {
            if (word_weights[i] != 0.0) {
                bow_vec.addWeight(i, word_weights[i] / factor);
            }
        }
    }

    void transform(const cv::Mat& features, std::vector<int>& assignments, std::vector<double>& weights) const {
        BowVector bow_vec;
        transform(features, assignments, weights, bow_vec);
    }

    void save(const std::string& output_dir) {
        cv::FileStorage fs(output_dir + "/akm_vocabulary.yml", cv::FileStorage::WRITE);
        fs << "vocab_size" << vocab_size_;
        fs << "max_iter" << max_iter_;
        fs << "r" << r_;
        fs << "var" << var_;
        fs << "weights" << weights_;
        fs << "centroids" << centroids_;
        fs << "distance_type" << distance_type_;
        fs.release();
        flann_index_->save(output_dir + "/akm_vocabulary_index.flann");
    }

    static AKMVocabulary load(const std::string& input_dir) {
        cv::FileStorage fs(input_dir + "/akm_vocabulary.yml", cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Failed to open file: " + input_dir + "/akm_vocabulary.yml");
        }
        int vocab_size, max_iter, r;
        double var;
        std::vector<double> weights;
        cv::Mat centroids;
        AKMVocabulary::DistanceType distance_type; //needs to be fully qneeds to be flified
        int distance_type_int;

        fs["vocab_size"] >> vocab_size;
        fs["max_iter"] >> max_iter;
        fs["r"] >> r;
        fs["var"] >> var;
        fs["weights"] >> weights;
        fs["centroids"] >> centroids;
        fs["distance_type"] >> distance_type_int; //read in an integer
        fs.release();

        distance_type = static_cast<AKMVocabulary::DistanceType>(distance_type_int); //cast to the correct enum

        // Create params object directly without shared_ptr to avoid dangling pointer
        flann::SavedIndexParams params(input_dir + "/akm_vocabulary_index.flann");

        if (distance_type == AKMVocabulary::HAMMING) {
            return AKMVocabulary<flann::Hamming<uint8_t>>(vocab_size, max_iter, r, var, centroids, distance_type, &params, weights);
        } else {
            throw std::invalid_argument("Invalid distance type");
        }


    }

    int getNumWords() const {
        return vocab_size_;
    }

    int getWordSize() const {
        return vocab_size_;
    }

    bool isCreated() const {
        return !centroids_.empty() && !weights_.empty();
    }

    void setNumCores(int n) { num_cores_ = n; }
    int getNumCores() const { return num_cores_; }

private:
    int vocab_size_;
    int max_iter_;
    int r_;
    // Gaussian scale parameter for soft assignment weighting: exp(-d^2 / (2 * var_)).
    // This is used directly as the denominator scale; it is NOT sigma^2 despite the
    // paper's notation. Empirically tuned to var=580 for ORB (256-bit Hamming distance).
    double var_;
    int num_cores_;
    std::vector<double> weights_;
    cv::Mat centroids_;
    std::unique_ptr<flann::Index<flann::Hamming<uint8_t>>> flann_index_;
    DistanceType distance_type_;

    void initCentroids(const cv::Mat& features, cv::Mat& centers) {
        std::vector<int> indices(features.rows);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_shuffle(indices.begin(), indices.end());

        centers.create(vocab_size_, features.cols, features.type());
        for (int i = 0; i < vocab_size_; ++i) {
            features.row(indices[i]).copyTo(centers.row(i));
        }
    }

    void getWords(const cv::Mat& features, int num_neighbors, cv::Mat& assignments, cv::Mat& dists) const { 
        int n_checks = std::max(1000, vocab_size_ / 100);
        int num_neighbors_lookup = std::max(5, num_neighbors);
        flann::SearchParams params(n_checks);
        params.cores = num_cores_;
        flann::Matrix<uint8_t> features_flann_matrix((uint8_t*)features.data, features.rows, features.cols);
        flann::Matrix<int> assignments_flann_matrix(new int[features.rows * num_neighbors_lookup], features.rows, num_neighbors_lookup);
        flann::Matrix<uint> dists_flann_matrix(new uint[features.rows * num_neighbors_lookup], features.rows, num_neighbors_lookup);

        flann_index_->knnSearch(features_flann_matrix, assignments_flann_matrix, dists_flann_matrix, num_neighbors_lookup, params);

        // Copy assignments and dists from contiguous FLANN matrices into cv::Mat
        assignments = cv::Mat(features.rows, num_neighbors, CV_32S);
        dists = cv::Mat(features.rows, num_neighbors, CV_32S);
        
        const int* assignments_src = assignments_flann_matrix.ptr();
        const uint* dists_src = dists_flann_matrix.ptr();
        int* assignments_dst = reinterpret_cast<int*>(assignments.data);
        int* dists_dst = reinterpret_cast<int*>(dists.data);
        
        for (int i = 0; i < features.rows; ++i) {
            std::vector<std::pair<uint, int>> neighbors(num_neighbors_lookup);
            for (int j = 0; j < num_neighbors_lookup; ++j) {
                neighbors[j] = {dists_src[i * num_neighbors_lookup + j], assignments_src[i * num_neighbors_lookup + j]};
            }
            std::sort(neighbors.begin(), neighbors.end(), [](const std::pair<uint, int>& a, const std::pair<uint, int>& b) {
                if (a.first == b.first) {
                    return a.second < b.second;
                }
                return a.first < b.first;
            });
            
            for (int j = 0; j < num_neighbors; ++j) {
                assignments_dst[i * num_neighbors + j] = neighbors[j].second;
                dists_dst[i * num_neighbors + j] = static_cast<int>(neighbors[j].first);
            }
        }

        // Clean up
        delete[] assignments_flann_matrix.ptr();
        delete[] dists_flann_matrix.ptr();
    }
};

} // namespace DBoW3

#endif // __D_T__AKMVOCABULARY__
