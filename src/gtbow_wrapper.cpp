/*
 * Python binding structure derived from pyDBoW3.
 * Copyright (c) 2016 The Pybind Development Team, All rights reserved.
 * Derived portions retain thirdparty/PYDBOW3_LICENSE.txt; GTBoW additions are MIT.
 */
#include "BowVector.h"
#include "FeatureVector.h"


#include <iostream>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sys/types.h>
#include <tuple>
namespace py = pybind11;

#include "Database.h"
#include "QueryResults.h"
#include "Vocabulary.h"
#include "GTDatabase.h"
#include "GTQueryResults.h"
#include "AKMVocabulary.h"
#include "SizeBinnedVocabulary.h"
#include "VocabularyConfig.h"
#include "string"
#include <vector>
#include <unordered_map>

#include <flann/flann.hpp>

#include "ndarray_converter.h"
using namespace pybind11::literals;
using namespace std;
string version() { return "1.0.0"; }

/*
    const int k = 9;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;
    DBoW3::Vocabulary voc(k, L, weight, score);
 */

static std::map<std::string, DBoW3::WeightingType> all_weight_method{
    {"TF_IDF", DBoW3::WeightingType::TF_IDF},
    {"BINARY", DBoW3::WeightingType::BINARY},
    {"IDF", DBoW3::WeightingType::IDF},
    {"TF", DBoW3::WeightingType::TF},
    };
static std::map<std::string, DBoW3::ScoringType> all_score_method{
    {"L1_NORM", DBoW3::ScoringType::L1_NORM},
    {"L2_NORM", DBoW3::ScoringType::L2_NORM},
    {"DOT_PRODUCT", DBoW3::ScoringType::DOT_PRODUCT},
    {"BHATTACHARYYA", DBoW3::ScoringType::BHATTACHARYYA},
    {"KL", DBoW3::ScoringType::KL}};
static std::map<DBoW3::WeightingType, std::string> rvt_all_weight_method{
    {DBoW3::WeightingType::TF_IDF, "TF_IDF"},
    {DBoW3::WeightingType::BINARY, "BINARY"},
    {DBoW3::WeightingType::IDF, "IDF"},
    {DBoW3::WeightingType::TF, "TF"}};
static std::map<DBoW3::ScoringType, std::string> rvt_all_score_method{
    {DBoW3::ScoringType::L1_NORM, "L1_NORM"},
    {DBoW3::ScoringType::L2_NORM, "L2_NORM"},
    {DBoW3::ScoringType::DOT_PRODUCT, "DOT_PRODUCT"},
    {DBoW3::ScoringType::BHATTACHARYYA, "BHATTACHARYYA"},
    {DBoW3::ScoringType::KL, "KL"}};

class Vocabulary {
public:
  Vocabulary(int k = 10, int L = 6, const std::string &weight_method = "TF_IDF",
             const std::string &score_method = "L2_NORM", bool verbose = true) {
    assert(all_weight_method.count(weight_method));
    assert(all_score_method.count(score_method));

    std::cout << "Creating Vocabulary with k: " << k << " L: " << L
              << " weight_method: " << weight_method
              << " score_method: " << score_method << std::endl;

    voc = std::make_unique<DBoW3::Vocabulary>(k, L, all_weight_method[weight_method],
                                all_score_method[score_method]);
  }

  Vocabulary(const std::string &filename) {
    voc = std::make_unique<DBoW3::Vocabulary>(filename);
  }

  ~Vocabulary() {
    if (_verbose)
      std::cout << "Entering destructor" << std::endl;
    if (voc) {
      voc->clear();
    }
    if (_verbose)
      std::cout << "Exiting destructor" << std::endl;
  }

  // input is a list of feature matrices, one matrix for each image and each row represents a feature
  void create(std::vector<cv::Mat> &training_feat_vec) {
    int N = training_feat_vec.size();
    std::vector<std::vector<cv::Mat> > tf(N);
    for (int i = 0, sz = N; i < sz; i++) {
      tf[i].resize(training_feat_vec[i].rows);
      for (int j = 0; j < training_feat_vec[i].rows; j++) {
        tf[i][j] = training_feat_vec[i].row(j);
      }
      // tf[i] = training_feat_vec.row(i);
    }
    voc->create(tf);
  }

  void clear() { voc->clear(); }

  void load(const std::string &path) { voc->load(path); }

  void save(const std::string &path) {
    voc->save(path);
  }

  // std::tuple<std::map<unsigned int, double>,
  //            std::map<unsigned int, std::vector<unsigned int>>>
  // transform(const std::vector<cv::Mat> &training_feat_vec, int level) {
  //   DBoW3::BowVector bv;
  //   DBoW3::FeatureVector fv;
  //   voc->transform(training_feat_vec, bv, fv, level);

  //   std::map<unsigned int, std::vector<unsigned int>> ret2 =
  //       static_cast<std::map<unsigned int, std::vector<unsigned int>>>(fv);

  //   return std::make_tuple(bv, ret2);
  // }

    std::tuple<std::map<unsigned int, double>,
             std::map<unsigned int, std::vector<unsigned int>>>
  transform(const cv::Mat &features, int level) {
    if (!features.empty() && features.cols != (int)getDescriptorSize()) {
      throw std::invalid_argument("Descriptor size mismatch. Expected " + std::to_string(getDescriptorSize()) + " but got " + std::to_string(features.cols));
    }

    DBoW3::BowVector bv;
    DBoW3::FeatureVector fv;

    std::vector<cv::Mat> feat_vec;
    for (int i = 0; i < features.rows; i++) {
      feat_vec.push_back(features.row(i));
    }
    voc->transform(feat_vec, bv, fv, level);

    std::map<unsigned int, std::vector<unsigned int>> ret2 =
        static_cast<std::map<unsigned int, std::vector<unsigned int>>>(fv);

    return std::make_tuple(bv, ret2);
  }

  std::map<unsigned int, double> getTFVector(const cv::Mat &features, bool normalize=true) const
  {
    if (!features.empty() && features.cols != (int)getDescriptorSize()) {
        throw std::invalid_argument("Descriptor size mismatch. Expected " + std::to_string(getDescriptorSize()) + " but got " + std::to_string(features.cols));
    }
    std::vector<cv::Mat> feat_vec;
    for (int i = 0; i < features.rows; i++) {
      feat_vec.push_back(features.row(i));
    }
    DBoW3::BowVector bv = voc->getTFVector(feat_vec);
    if(normalize)
      bv.normalize(DBoW3::L1);
    std::map<unsigned int, double> toReturn;
    for (const auto& pair : bv) {
      toReturn[pair.first] = pair.second;
    }
    return toReturn;
  }

  bool _verbose = false;

  py::list getWord(uint32_t word_id) { 
    cv::Mat word = voc->getWord(word_id);
    // std::vector<uint8_t> toReturn;
    py::list toReturn;
    for (int j = 0; j < word.cols; j++) {
      // toReturn.push_back(word.at(0, j));
      toReturn.append(word.at<uint8_t>(0, j));
      }
    return toReturn;
  }


  std::map<uint32_t, uint32_t> nodeId2WordId() const {
    return voc->getNodeId2WordId();
  }
  uint32_t getWordId(uint32_t node_id) const {
    return voc->getWordId(node_id);
  }
  float getWordWeight(uint32_t word_id) const {
    return voc->getWordWeight(word_id);
  }
  uint32_t getDepth() const { return voc->getDepthLevels(); }
  uint32_t getDescriptorSize() const { return voc->getDescritorSize(); }
  uint32_t getWordSize() const { return voc->getWordSize(); }

  std::string log() {
    char str_buffer[256] = {};
    sprintf(str_buffer,
            "nwords: %d\ndepths: %d\ndescriptor size %d \nK: %d \nweight "
            "method: %s\nscore method: %s\n",
            getWordSize(), getDepth(), getDescriptorSize(),
            voc->getBranchingFactor(),
            rvt_all_weight_method[voc->getWeightingType()].c_str(),
            rvt_all_score_method[voc->getScoringType()].c_str());
    return std::string(str_buffer);
  }
  std::unique_ptr<DBoW3::Vocabulary> voc;
};


class Database {
public:
  Database(bool use_di, int di_levels = 0) {
    db = std::make_unique<DBoW3::Database>(use_di, di_levels);
  }
  Database(const Vocabulary &voc, bool use_di, int di_levels = 0) {
    db = std::make_unique<DBoW3::Database>(*voc.voc, use_di, di_levels);
  }
  ~Database() {
    if (_verbose)
      std::cout << "Entering destructor" << std::endl;
    if (_verbose)
      std::cout << "Exiting destructor" << std::endl;
  }
  int add(const cv::Mat &features) {
    // std::vector<DBoW3::BowVector> bow_vecs(features.size());
    // std::vector<DBoW3::FeatureVector> feat_vecs(features.size());
    // for (size_t i = 0; i < features.size(); i++) {
    //   voc->transform(features[i], bow_vecs[i], feat_vecs[i]);
    // }
    return db->add(features);
  }
  int addBowVec(std::map<unsigned int, double> bow_vec) {
    DBoW3::BowVector bv;
    for (const auto& pair : bow_vec) {
      bv.addWeight(pair.first, pair.second);
    }
    
    return db->add(bv);  // TODO: add feature vector for direct index access
  }
  // may have to do std::vector<std::vector<int>>
  py::list query(const cv::Mat &features, int max_results = 1, int max_id = -1) {
    DBoW3::QueryResults results;
    db->query(features, results, max_results, max_id);
    py::list toReturn;
    for (const auto& result : results) {
      py::tuple pyResult = py::make_tuple(result.Id, result.Score);
      toReturn.append(pyResult);
    }
    return toReturn;
  }
  py::list queryBowVec(const std::map<unsigned int, double> bow_vec_dict, int max_results = 1, int max_id = -1, const std::string &score_method = "L2_NORM") {
    DBoW3::BowVector bv;
    for (const auto& pair : bow_vec_dict) {
      bv.addWeight(pair.first, pair.second);
    }
    
    DBoW3::QueryResults results;
    db->query(bv, results, max_results, max_id, all_score_method[score_method]);
    py::list toReturn;
    for (const auto& result : results) {
      py::tuple pyResult = py::make_tuple(result.Id, result.Score);
      toReturn.append(pyResult);
    }
    return toReturn;
  }
  std::map<unsigned int, std::vector<unsigned int> > retrieveFeatures(unsigned int id) {
    return db->retrieveFeatures(id);
  }
  void setNumWords(unsigned int nwords){
    db->setNumWords(nwords);
  }
  void save(const std::string &path) {
    db->save(path);
  }
  void load(const std::string &path) {
    db->load(path);
  }
  void clear() {
    db->clear();
  }
  bool _verbose = false;
  std::unique_ptr<DBoW3::Database> db;
  // Vocabulary voc;
};


class FLANNVocabulary {
  public:
    FLANNVocabulary(int branching = 32, const std::string &centers_init = "RANDOM", int trees = 4, int max_leaf_size = 100,
                    const std::string &distance_type = "L2", const std::string &data_type = "float", bool verbose = true) {

      if (_verbose)
        std::cout << "Creating FLANNVocabulary with branching: " << branching << " centers_init: " << centers_init
                << " trees: " << trees << " max_leaf_size: " << max_leaf_size
                << " distance_type: " << distance_type << " data_type: " << data_type << std::endl;

      _verbose = verbose;

      if (distance_type != "Hamming" or data_type != "uint8")
        throw std::invalid_argument("Only Hamming distance and uint8 data type are supported currently.");

      flann_index = std::make_unique<flann::Index<flann::Hamming<uint8_t> > >(flann::HierarchicalClusteringIndexParams(branching, flann::FLANN_CENTERS_RANDOM, trees, max_leaf_size));
      
    }

    FLANNVocabulary(const std::string &path, const cv::Mat &training_feat_vec) {
      training_feat = std::make_unique<flann::Matrix<uint8_t> >((uint8_t *)training_feat_vec.data, training_feat_vec.rows, training_feat_vec.cols);
      flann_index = std::make_unique<flann::Index<flann::Hamming<uint8_t> > >(*training_feat, flann::SavedIndexParams(path));
      flann_index->buildIndex(*training_feat);
    }

    ~FLANNVocabulary() {
      if (_verbose)
        std::cout << "Entering destructor" << std::endl;
      if (flann_index != nullptr) {
        flann_index.reset();
      }
      if (training_feat != nullptr) {
        training_feat.reset();
      }
      if (_verbose)
        std::cout << "Exiting destructor" << std::endl;
    }

    void create(cv::Mat &training_feat_vec) {
      training_feat = std::make_unique<flann::Matrix<uint8_t> >((uint8_t *)training_feat_vec.data, training_feat_vec.rows, training_feat_vec.cols);
      flann_index->buildIndex(*training_feat);
    }

    // function that searches the index for the nearest neighbors of a set of query features
    // and returns the indices of the nearest neighbors and their distances
      py::tuple nn_index(const cv::Mat &query_feat_vec, int num_neighbors = 1, int num_cores = 0) {
        flann::Matrix<uint8_t> query_feat((uint8_t *)query_feat_vec.data, query_feat_vec.rows, query_feat_vec.cols);
        std::vector<int> indices_vec(query_feat.rows * num_neighbors);
        std::vector<uint> dists_vec(query_feat.rows * num_neighbors);
        flann::Matrix<int> indices(indices_vec.data(), query_feat.rows, num_neighbors);
        flann::Matrix<uint> dists(dists_vec.data(), query_feat.rows, num_neighbors);

        flann::SearchParams params = flann::SearchParams(1000);
        params.cores = num_cores;

      flann_index->knnSearch(query_feat, indices, dists, num_neighbors, params);

      py::list toReturnIndices;
      for (int i = 0; i < indices.rows; i++) {
        py::list indicesList;
        for (int j = 0; j < indices.cols; j++) {
          indicesList.append(indices[i][j]);
        }
        toReturnIndices.append(indicesList);
      }
      py::list toReturnDists;
      for (int i = 0; i < dists.rows; i++) {
        py::list distsList;
        for (int j = 0; j < dists.cols; j++) {
          distsList.append(dists[i][j]);
        }
        toReturnDists.append(distsList);
      }

      py::tuple toReturn = py::make_tuple(toReturnIndices, toReturnDists);
      return toReturn;
    }


  void save(const std::string &path) {
    flann_index->save(path);
  }

  cv::Mat bitwise_median(const cv::Mat &descriptors) {
    if (descriptors.type()!=CV_8U ){
        throw std::invalid_argument("Not a binary descriptor.");
    }

    // std::vector<cv::Mat> descriptors_list;
    // for (int i = 0; i < descriptors_list.rows; i++) {
    //   descriptors_list_list.push_back(descriptors_list.row(i));
    // }

    if (descriptors.empty()) {
      return cv::Mat();
    }

    // if(descriptors_list.empty()) return;

    if(descriptors.rows == 1)
    {
      cv::Mat mean = descriptors.clone();
      return mean;
    }

    //determine number of bytes of the binary descriptor
      // int L= DBoW3::getDescSizeBytes( descriptors.row(0));
      int L= descriptors.cols * descriptors.elemSize();
      vector<int> sum( L * 8, 0);

      for(size_t i = 0; i < descriptors.rows; ++i)
      {
          const cv::Mat &d = descriptors.row(i);
          const unsigned char *p = d.ptr<unsigned char>();

          for(int j = 0; j < d.cols; ++j, ++p)
          {
              if(*p & (1 << 7)) ++sum[ j*8     ];
              if(*p & (1 << 6)) ++sum[ j*8 + 1 ];
              if(*p & (1 << 5)) ++sum[ j*8 + 2 ];
              if(*p & (1 << 4)) ++sum[ j*8 + 3 ];
              if(*p & (1 << 3)) ++sum[ j*8 + 4 ];
              if(*p & (1 << 2)) ++sum[ j*8 + 5 ];
              if(*p & (1 << 1)) ++sum[ j*8 + 6 ];
              if(*p & (1))      ++sum[ j*8 + 7 ];
          }
      }

      cv::Mat mean = cv::Mat::zeros(1, L, CV_8U);
      unsigned char *p = mean.ptr<unsigned char>();

      const int N2 = (int)descriptors.rows / 2 + descriptors.rows % 2;
      for(size_t i = 0; i < sum.size(); ++i)
      {
          if(sum[i] >= N2)
          {
              // set bit
              *p |= 1 << (7 - (i % 8));
          }

          if(i % 8 == 7) ++p;
      }
      return mean;
  }

    bool _verbose = false;
    std::unique_ptr<flann::Index<flann::Hamming<uint8_t> > > flann_index;
    std::unique_ptr<flann::Matrix<uint8_t> > training_feat;
};

class GTDatabase {
public:
  GTDatabase(bool use_di, int di_levels = 0) {
    db = std::make_unique<DBoW3::GTDatabase>(use_di, di_levels);
  }

  GTDatabase(const Vocabulary &voc, bool use_di, int di_levels = 0) {
    db = std::make_unique<DBoW3::GTDatabase>(*voc.voc, use_di, di_levels);
  }

  ~GTDatabase() {
    if (_verbose)
      std::cout << "Entering GTDatabase destructor" << std::endl;
    if (_verbose)
      std::cout << "Exiting GTDatabase destructor" << std::endl;
  }

  int addBowVec(std::unordered_map<unsigned int, double> bow_vec_dict, const cv::Mat &assignments, const cv::Mat &weights, const cv::Mat &poses) {
    DBoW3::BowVector bv;
    for (const auto& pair : bow_vec_dict) {
      bv.addWeight(pair.first, pair.second);
    }

    return db->add(bv, assignments, weights, poses);
  }


  // Returns a tuple of two elements:
  // 1. A list of tuples, each containing the id, score, match_point_ids (array w/ first col query, 2nd col db), and match_point_weights (array w/ first col query, 2nd col db) of a result
  // 2. A matrix of poses corresponding to the db_point_ids in the results
  // match_point_ids are the id matches between the query and db points, match_point_weights are the soft assignment weights of each point
  py::tuple queryBowVecImpl(const std::unordered_map<unsigned int, double> bow_vec_dict, const cv::Mat &assignments,
                        const cv::Mat &weights, const cv::Mat &poses, int max_results,
                        int max_id, const std::string &score_method, int num_orientation_bins,
                        bool include_orientation_response) {
    if (score_method != "L2_NORM") {
      throw std::invalid_argument("Only L2_NORM score method is supported currently.");
    }

    DBoW3::BowVector bv;
    for (const auto& pair : bow_vec_dict) {
      bv.addWeight(pair.first, pair.second);
    }
    DBoW3::GTQueryResults results;

    db->queryL2(
      bv,
      results,
      max_results,
      max_id,
      assignments,
      weights,
      poses,
      num_orientation_bins
    );

    // get all the unique db point ids from the results
    std::set<unsigned int> uniquePointIds;
    for (const auto& result : results) {
      for (const auto& pointMatch : result.point_matches) {
        uniquePointIds.insert(pointMatch.db_point_id);
      }
    }

    std::pair<cv::Mat, std::unordered_map<unsigned int, unsigned int>> poses_and_indices = db->getPoses(uniquePointIds);
    cv::Mat db_poses = poses_and_indices.first;
    std::unordered_map<unsigned int, unsigned int> db_index_to_pose_index = poses_and_indices.second;

    py::list results_to_return;
    py::list orientation_metadata;
    for (const auto& result : results) {
      size_t num_matches = result.point_matches.size();
      cv::Mat match_point_ids(num_matches, 2, CV_32S);
      cv::Mat match_point_weights(num_matches, 2, CV_64F);
      int row = 0;
      for (const auto& pointMatch : result.point_matches) {
        match_point_ids.at<int>(row, 0) = pointMatch.query_point_id;
        match_point_ids.at<int>(row, 1) = db_index_to_pose_index[pointMatch.db_point_id];
        match_point_weights.at<double>(row, 0) = pointMatch.query_point_weight;
        match_point_weights.at<double>(row, 1) = pointMatch.db_point_weight;
        row++;
      }
      py::tuple t = py::make_tuple(result.Id, result.Score, match_point_ids, match_point_weights);
      results_to_return.append(t);

      if (include_orientation_response) {
        const int effective_bins = static_cast<int>(result.orientation_vote_mass.size());
        std::vector<double> bin_centers;
        bin_centers.reserve(effective_bins);
        if (effective_bins > 0) {
          const double bin_width = 2.0 * M_PI / effective_bins;
          for (int i = 0; i < effective_bins; ++i) {
            bin_centers.push_back(-M_PI + i * bin_width);
          }
        }

        py::dict metadata;
        metadata["id"] = result.Id;
        metadata["winning_bin"] = result.winning_orientation_bin;
        metadata["bin_centers_rad"] = bin_centers;
        metadata["vote_mass"] = result.orientation_vote_mass;
        metadata["match_counts"] = result.orientation_match_counts;
        metadata["orientation_enabled"] = effective_bins > 1;
        orientation_metadata.append(metadata);
      }
    }

    if (include_orientation_response) {
      return py::make_tuple(results_to_return, db_poses, orientation_metadata);
    }
    return py::make_tuple(results_to_return, db_poses);
  }

  py::tuple queryBowVec(const std::unordered_map<unsigned int, double> bow_vec_dict, const cv::Mat &assignments = cv::Mat(),
                        const cv::Mat &weights = cv::Mat(), const cv::Mat &poses = cv::Mat(), int max_results = 1,
                        int max_id = -1, const std::string &score_method = "L2_NORM", int num_orientation_bins=1) {
    return queryBowVecImpl(bow_vec_dict, assignments, weights, poses, max_results,
                           max_id, score_method, num_orientation_bins, false);
  }

  py::tuple queryBowVecWithOrientationResponse(
                        const std::unordered_map<unsigned int, double> bow_vec_dict,
                        const cv::Mat &assignments = cv::Mat(), const cv::Mat &weights = cv::Mat(),
                        const cv::Mat &poses = cv::Mat(), int max_results = 1, int max_id = -1,
                        const std::string &score_method = "L2_NORM", int num_orientation_bins=1) {
    return queryBowVecImpl(bow_vec_dict, assignments, weights, poses, max_results,
                           max_id, score_method, num_orientation_bins, true);
  }

  std::map<unsigned int, std::vector<unsigned int> > retrieveFeatures(unsigned int id) {
    return db->retrieveFeatures(id);
  }

  void setNumWords(unsigned int nwords) {
    db->setNumWords(nwords);
  }

  unsigned int size() const {
    return db->size();
  }

  void save(const std::string &path) {
    db->save(path);
  }

  void load(const std::string &path) {
    db->load(path);
  }

  void clear() {
    db->clear();
  }

  bool _verbose = false;
  std::unique_ptr<DBoW3::GTDatabase> db;
};

class VocabularyConfig {
public:
    VocabularyConfig() {
        vocab_type = "vocabulary";  // Changed to lowercase
        k = 10;
        L = 5;
        weight_method = "TF_IDF";
        score_method = "L2_NORM";
        // AKM defaults
        vocab_size = 1000; // Adjusted default
        max_iter = 200;
        r = 1;
        var = 1.0;
        num_cores = 1;
        file_path = ""; // For loading
    }

    void setVocabularyParameters(int k_val = 10, int L_val = 5, 
                               const std::string& weight_method_val = "TF_IDF",
                               const std::string& score_method_val = "L2_NORM",
                               const std::string& path_val = "") {
        this->vocab_type = "vocabulary";  // Changed to lowercase
        this->k = k_val;
        this->L = L_val;
        this->weight_method = weight_method_val;
        this->score_method = score_method_val;
        this->file_path = path_val;
    }

    void setAKMVocabularyParameters(int vocab_size_val = 1000, int max_iter_val = 200,
                                  int r_val = 1, double var_val = 1.0,
                                  const std::string& path_val = "",
                                  int num_cores_val = 1) {
        this->vocab_type = "akmvocabulary";  // Changed to lowercase
        this->vocab_size = vocab_size_val;
        this->max_iter = max_iter_val;
        this->r = r_val;
        this->var = var_val;
        this->file_path = path_val;
        this->num_cores = num_cores_val;
    }

    std::string vocab_type;
    // Vocabulary params
    int k;
    int L;
    std::string weight_method;
    std::string score_method;
    // AKMVocabulary params
    int vocab_size;
    int max_iter;
    int r;
    double var;
    int num_cores;
    // Common
    std::string file_path; // Used if loading a pre-trained vocab inside SizeBinnedVocab
};

// Helper to create DBoW3 library's VocabularyConfig struct from the wrapper's VocabularyConfig class
DBoW3::VocabularyConfig create_dbow_library_config(const VocabularyConfig& wrapper_config) {
    if (wrapper_config.vocab_type == "vocabulary") {
        DBoW3::VocabularyParameters params;
        params.k = wrapper_config.k;
        params.L = wrapper_config.L;
        params.weighting = all_weight_method.at(wrapper_config.weight_method);
        params.scoring = all_score_method.at(wrapper_config.score_method);
        params.file_path = wrapper_config.file_path;
        return DBoW3::VocabularyConfig(params);
    } else if (wrapper_config.vocab_type == "akmvocabulary") {
        DBoW3::AKMVocabularyParameters params;
        params.vocab_size = wrapper_config.vocab_size;
        params.max_iter = wrapper_config.max_iter;
        params.r = wrapper_config.r;
        params.var = wrapper_config.var;
        params.num_cores = wrapper_config.num_cores;
        params.file_path = wrapper_config.file_path;
        return DBoW3::VocabularyConfig(params);
    } else {
        throw std::runtime_error("Invalid vocabulary type in VocabularyConfig: " + wrapper_config.vocab_type);
    }
}


class SizeBinnedVocabulary {
public:
    SizeBinnedVocabulary(const VocabularyConfig& wrapper_config) {
        vocab_type_ = wrapper_config.vocab_type;
        // Don't create lib_config_ until needed to avoid union assignment issues
        wrapper_config_ = wrapper_config;  // Store wrapper config instead

        // Don't create the vocabulary objects immediately - wait until they're needed
    }

    // Delete copy constructor and copy assignment
    SizeBinnedVocabulary(const SizeBinnedVocabulary&) = delete;
    SizeBinnedVocabulary& operator=(const SizeBinnedVocabulary&) = delete;

    // Add move constructor and move assignment
    SizeBinnedVocabulary(SizeBinnedVocabulary&& other) noexcept 
        : vocab_type_(std::move(other.vocab_type_)),
          wrapper_config_(std::move(other.wrapper_config_)),
          vocab_standard_(std::move(other.vocab_standard_)),
          vocab_akm_(std::move(other.vocab_akm_)) {}
    
    SizeBinnedVocabulary& operator=(SizeBinnedVocabulary&& other) noexcept {
        if (this != &other) {
            vocab_type_ = std::move(other.vocab_type_);
            wrapper_config_ = std::move(other.wrapper_config_);
            vocab_standard_ = std::move(other.vocab_standard_);
            vocab_akm_ = std::move(other.vocab_akm_);
        }
        return *this;
    }

    ~SizeBinnedVocabulary() = default;

    void create(const std::vector<cv::Mat>& features, const std::vector<float>& scales, 
               const std::vector<std::string>& img_names) {
        ensureVocabularyCreated();
        if (vocab_type_ == "vocabulary") {
            if (!vocab_standard_) throw std::runtime_error("Standard vocabulary not initialized.");
            vocab_standard_->create(features, scales, img_names);
        } else if (vocab_type_ == "akmvocabulary") {
            if (!vocab_akm_) throw std::runtime_error("AKM vocabulary not initialized.");
            vocab_akm_->create(features, scales, img_names);
        } else {
            throw std::runtime_error("Unknown vocabulary type: " + vocab_type_);
        }
    }

    py::tuple transform(const cv::Mat& features, const std::vector<float>& scales) {
        ensureVocabularyCreated();
        std::vector<int> assignments;
        std::vector<double> weights;
        DBoW3::BowVector bow_vec;
        
        if (vocab_type_ == "vocabulary") {
            if (!vocab_standard_) throw std::runtime_error("Standard vocabulary not initialized.");
            vocab_standard_->transform(features, scales, assignments, weights, bow_vec);
        } else if (vocab_type_ == "akmvocabulary") {
            if (!vocab_akm_) throw std::runtime_error("AKM vocabulary not initialized.");
            vocab_akm_->transform(features, scales, assignments, weights, bow_vec);
        } else {
            throw std::runtime_error("Unknown vocabulary type: " + vocab_type_);
        }

        // Convert assignments and weights to cv::Mat so they're numpy arrays in Python
        int N = features.rows;
        int r = (N > 0) ? assignments.size() / N : 1;
        
        cv::Mat mat_assignments(N, r, CV_32S);
        cv::Mat mat_weights(N, r, CV_64F);
        
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < r; ++j) {
                mat_assignments.at<int>(i, j) = assignments[i * r + j];
                mat_weights.at<double>(i, j) = weights[i * r + j];
            }
        }
        
        std::map<unsigned int, double> bow_dict;
        for (const auto& pair : bow_vec) {
            bow_dict[pair.first] = pair.second;
        }

        return py::make_tuple(mat_assignments, mat_weights, bow_dict);
    }

    void save(const std::string& output_dir) {
        ensureVocabularyCreated();
        if (vocab_type_ == "vocabulary") {
            if (!vocab_standard_) throw std::runtime_error("Standard vocabulary not initialized.");
            vocab_standard_->save(output_dir);
        } else if (vocab_type_ == "akmvocabulary") {
            if (!vocab_akm_) throw std::runtime_error("AKM vocabulary not initialized.");
            vocab_akm_->save(output_dir);
        } else {
            throw std::runtime_error("Unknown vocabulary type: " + vocab_type_);
        }
    }

    static SizeBinnedVocabulary load(const std::string& input_dir, const VocabularyConfig& wrapper_config) {
        // Create the result object using move semantics
        SizeBinnedVocabulary result;
        result.vocab_type_ = wrapper_config.vocab_type;
        result.wrapper_config_ = wrapper_config;
        
        if (wrapper_config.vocab_type == "vocabulary") {
            DBoW3::VocabularyConfig lib_config = create_dbow_library_config(wrapper_config);
            result.vocab_standard_ = std::make_unique<DBoW3::SizeBinnedVocabulary<DBoW3::Vocabulary>>(lib_config);
            result.vocab_standard_->load(input_dir);
        } else if (wrapper_config.vocab_type == "akmvocabulary") {
            DBoW3::VocabularyConfig lib_config = create_dbow_library_config(wrapper_config);
            result.vocab_akm_ = std::make_unique<DBoW3::SizeBinnedVocabulary<DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>>>(lib_config);
            result.vocab_akm_->load(input_dir);
        } else {
            throw std::runtime_error("Invalid vocabulary type for loading: " + wrapper_config.vocab_type);
        }
        
        return result;
    }

    int getNumWords() const {
        // Don't call ensureVocabularyCreated() in const method, check if initialized
        if (vocab_type_ == "vocabulary") {
            if (!vocab_standard_) return 0;
            return vocab_standard_->getNumWords();
        } else if (vocab_type_ == "akmvocabulary") {
            if (!vocab_akm_) return 0;
            return vocab_akm_->getNumWords();
        } else {
            throw std::runtime_error("Unknown vocabulary type: " + vocab_type_);
        }
    }

    int getWordSize() const {
        // Don't call ensureVocabularyCreated() in const method, check if initialized
        if (vocab_type_ == "vocabulary") {
            if (!vocab_standard_) return 0;
            return vocab_standard_->getWordSize();
        } else if (vocab_type_ == "akmvocabulary") {
            if (!vocab_akm_) return 0;
            return vocab_akm_->getWordSize();
        } else {
            throw std::runtime_error("Unknown vocabulary type: " + vocab_type_);
        }
    }

    int getNumScales() const {
        // Don't call ensureVocabularyCreated() in const method, check if initialized
        if (vocab_type_ == "vocabulary") {
            if (!vocab_standard_) return 0;
            return vocab_standard_->getNumScales();
        } else if (vocab_type_ == "akmvocabulary") {
            if (!vocab_akm_) return 0;
            return vocab_akm_->getNumScales();
        } else {
            throw std::runtime_error("Unknown vocabulary type: " + vocab_type_);
        }
    }

    bool isCreated() const {
        if (vocab_type_ == "vocabulary") {
            return vocab_standard_ && vocab_standard_->isCreated();
        } else if (vocab_type_ == "akmvocabulary") {
            return vocab_akm_ && vocab_akm_->isCreated();
        } else {
            return false;
        }
    }

private:
    // Private default constructor for static load method
    SizeBinnedVocabulary() = default;

    std::string vocab_type_;
    VocabularyConfig wrapper_config_;  // Store wrapper config instead of lib config
    std::unique_ptr<DBoW3::SizeBinnedVocabulary<DBoW3::Vocabulary>> vocab_standard_;
    std::unique_ptr<DBoW3::SizeBinnedVocabulary<DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>>> vocab_akm_;

    void ensureVocabularyCreated() {
        if (vocab_type_ == "vocabulary" && !vocab_standard_) {
            DBoW3::VocabularyConfig lib_config = create_dbow_library_config(wrapper_config_);
            vocab_standard_ = std::make_unique<DBoW3::SizeBinnedVocabulary<DBoW3::Vocabulary>>(lib_config);
        } else if (vocab_type_ == "akmvocabulary" && !vocab_akm_) {
            DBoW3::VocabularyConfig lib_config = create_dbow_library_config(wrapper_config_);
            vocab_akm_ = std::make_unique<DBoW3::SizeBinnedVocabulary<DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>>>(lib_config);
        } else {
            // Trigger create_dbow_library_config to throw the expected error
            create_dbow_library_config(wrapper_config_);
        }
    }
};

PYBIND11_MODULE(pyGTBoW, m) {
  NDArrayConverter::init_numpy();
  m.doc() = "pybind11 of fbow"; // optional module docstring
  m.def("__version__", &version, "get the version of fbow");
  
  py::class_<Vocabulary>(m, "Vocabulary")
      .def(py::init<int, int, std::string, std::string, bool>(), "K"_a = 10,
           "L"_a = 6, "weight_method"_a = "TF_IDF",
           "score_method"_a = "L2_NORM", "verbose"_a = true)
      .def(py::init<const std::string &>(), "filename"_a)
      // .def("create", &Vocabulary::create, "features"_a, "num_of_docs"_a)
      .def("create", &Vocabulary::create, "features"_a)
      .def("clear", &Vocabulary::clear)
      .def("load", &Vocabulary::load, "path"_a)
      .def("save", &Vocabulary::save, "path"_a)
      .def("transform", &Vocabulary::transform, "feature"_a, "level"_a)
      .def("getTFVector", &Vocabulary::getTFVector, "features"_a, "normalize"_a = true)
      .def("getWordSize", &Vocabulary::getWordSize)
      .def("getWordWeight", &Vocabulary::getWordWeight, "word_id"_a)
      .def("getDescriptorSize", &Vocabulary::getDescriptorSize)
      .def("getDepth", &Vocabulary::getDepth)
      .def("getWord", &Vocabulary::getWord, "word_id"_a)
      .def("nodeId2WordId", &Vocabulary::nodeId2WordId)
      .def("getWordId", &Vocabulary::getWordId, "node_id"_a)
      .def("__str__", &Vocabulary::log)
      .def("clear", &Vocabulary::clear);

  py::class_<Database>(m, "Database")
      .def(py::init<bool, int>(), "use_di"_a = false, "di_levels"_a = 0)
      .def(py::init<const Vocabulary&, bool, int>(), "voc"_a, "use_di"_a = false, "di_levels"_a = 0)
      .def("add", &Database::add, "features"_a)
      .def("addBowVec", &Database::addBowVec, "bow_vec"_a)
      .def("query", &Database::query, "features"_a, "max_results"_a = 1, "max_id"_a = -1)
      .def("queryBowVec", &Database::queryBowVec, "bow_vec"_a, "max_results"_a = 1, "max_id"_a = -1, "scoring_type"_a = "L2_NORM")
      .def("retrieveFeatures", &Database::retrieveFeatures, "id"_a)
      .def("setNumWords", &Database::setNumWords, "num_words"_a)
      .def("save", &Database::save, "path"_a)
      .def("load", &Database::load, "path"_a)
      .def("clear", &Database::clear);

  py::class_<FLANNVocabulary>(m, "FLANNVocabulary")
      .def(py::init<int, std::string, int, int, std::string, std::string, bool>(), "branching"_a = 32,
           "centers_init"_a = "RANDOM", "trees"_a = 4, "max_leaf_size"_a = 100,
           "distance_type"_a = "L2", "data_type"_a = "float", "verbose"_a = true)
      .def(py::init<const std::string &, const cv::Mat &>(), "path"_a, "training_feat_vec"_a)
      .def("create", &FLANNVocabulary::create, "training_feat_vec"_a)
      .def("nn_index", &FLANNVocabulary::nn_index, "query_feat_vec"_a, "num_neighbors"_a = 1, "num_cores"_a = 0)
      .def("save", &FLANNVocabulary::save, "path"_a)
      .def("bitwise_median", &FLANNVocabulary::bitwise_median, "descriptors"_a);

  py::class_<GTDatabase>(m, "GTDatabase")
    .def(py::init<bool, int>(), "use_di"_a = false, "di_levels"_a = 0)
    .def(py::init<const Vocabulary&, bool, int>(), "voc"_a, "use_di"_a = false, "di_levels"_a = 0)
    .def("addBowVec", &GTDatabase::addBowVec, "bow_vec_dict"_a, "assignments"_a, "weights"_a, "poses"_a)
    .def("queryBowVec", &GTDatabase::queryBowVec, "bow_vec_dict"_a,
         "assignments"_a = cv::Mat(), "weights"_a = cv::Mat(), "poses"_a = cv::Mat(),
         "max_results"_a = 1, "max_id"_a = -1, "score_method"_a = "L2_NORM", "num_orientation_bins"_a = 1)
    .def("queryBowVecWithOrientationResponse", &GTDatabase::queryBowVecWithOrientationResponse, "bow_vec_dict"_a,
         "assignments"_a = cv::Mat(), "weights"_a = cv::Mat(), "poses"_a = cv::Mat(),
         "max_results"_a = 1, "max_id"_a = -1, "score_method"_a = "L2_NORM", "num_orientation_bins"_a = 1)
    .def("retrieveFeatures", &GTDatabase::retrieveFeatures, "id"_a)
    .def("setNumWords", &GTDatabase::setNumWords, "nwords"_a)
    .def("size", &GTDatabase::size)
    .def("save", &GTDatabase::save, "path"_a)
    .def("load", &GTDatabase::load, "path"_a)
    .def("clear", &GTDatabase::clear);

  py::class_<DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>>(m, "HammingAKMVocabulary")
      .def(py::init<int, int, int, double>(), "vocab_size"_a, "max_iter"_a = 200, "r"_a = 1, "var"_a = 1.0)
      .def(py::init<int, const flann::IndexParams*, int, int, double>(), "vocab_size"_a, "params"_a, "max_iter"_a = 200, "r"_a = 1, "var"_a = 1.0)
      .def(py::init<int, int, int, double, const cv::Mat&, DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>::DistanceType, const flann::IndexParams*, const std::vector<double>&>(),
           "vocab_size"_a, "max_iter"_a, "r"_a, "var"_a, "centroids"_a, "distance_type"_a, "params"_a, "weights"_a)
      .def("create", [](DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>& self, const std::vector<cv::Mat>& training_features) {
          self.create(training_features);
      }, "training_features"_a)
      .def("create", [](DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>& self, const std::vector<std::vector<cv::Mat>>& training_features) {
          self.create(training_features);
      }, "training_features"_a)
      .def("create", [](DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>& self, const std::vector<cv::Mat>& training_features, const std::vector<int>& img_assignments) {
          self.create(training_features, img_assignments);
      }, "training_features"_a, "img_assignments"_a)
      .def("create", [](DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>& self, const cv::Mat& training_features, const cv::Mat& img_assignments) {
          self.create(training_features, img_assignments);
      }, "training_features"_a, "img_assignments"_a)
      .def("determineWords", &DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>::determineWords, "features"_a)
      .def("determineWeights", &DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>::determineWeights, "features"_a, "img_assignments"_a)
      .def("transform", [](DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>& self, const cv::Mat& features) {
          std::vector<int> assignments;
          std::vector<double> weights;
          DBoW3::BowVector bow_vec;
          self.transform(features, assignments, weights, bow_vec);
          std::map<unsigned int, double> bow_map;
          for (const auto& pair : bow_vec) {
              bow_map[pair.first] = pair.second;
          }
            // Convert assignments and weights to cv::Mat before returning so they become numpy arrays
            cv::Mat assignments_mat(assignments.size(), 1, CV_32S);
            for (size_t i = 0; i < assignments.size(); ++i) {
              assignments_mat.at<int>(i, 0) = assignments[i];
            }
            cv::Mat weights_mat(weights.size(), 1, CV_64F);
            for (size_t i = 0; i < weights.size(); ++i) {
              weights_mat.at<double>(i, 0) = weights[i];
            }
            return py::make_tuple(assignments_mat, weights_mat, bow_map);
      }, "features"_a)
      .def("save", &DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>::save, "output_dir"_a)
      .def_static("load", &DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>::load, "input_dir"_a)
      .def("getNumWords", &DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>::getNumWords)
      .def("getWordSize", &DBoW3::AKMVocabulary<flann::Hamming<uint8_t>>::getWordSize);

  py::class_<VocabularyConfig>(m, "VocabularyConfig")
      .def(py::init<>())
      .def("setVocabularyParameters", &VocabularyConfig::setVocabularyParameters,
           "k"_a = 10, "L"_a = 5, "weight_method"_a = "TF_IDF", "score_method"_a = "L2_NORM",
           "path"_a = "")
      .def("setAKMVocabularyParameters", &VocabularyConfig::setAKMVocabularyParameters,
           "vocab_size"_a = 1000, "max_iter"_a = 200, "r"_a = 1, "var"_a = 1.0, "path"_a = "",
           "num_cores"_a = 1)
      .def_readwrite("vocab_type", &VocabularyConfig::vocab_type)
      .def_readwrite("k", &VocabularyConfig::k)
      .def_readwrite("L", &VocabularyConfig::L)
      .def_readwrite("weight_method", &VocabularyConfig::weight_method)
      .def_readwrite("score_method", &VocabularyConfig::score_method)
      .def_readwrite("vocab_size", &VocabularyConfig::vocab_size)
      .def_readwrite("max_iter", &VocabularyConfig::max_iter)
      .def_readwrite("r", &VocabularyConfig::r)
      .def_readwrite("var", &VocabularyConfig::var)
      .def_readwrite("num_cores", &VocabularyConfig::num_cores)
      .def_readwrite("file_path", &VocabularyConfig::file_path);

  py::class_<SizeBinnedVocabulary>(m, "SizeBinnedVocabulary")
      .def(py::init<const VocabularyConfig&>(), "config"_a)
      .def("create", [](SizeBinnedVocabulary& self, const cv::Mat& features, const std::vector<float>& scales, const std::vector<std::string>& img_names) {
          // img_names has one entry per descriptor.  Group features by image so
          // that SizeBinnedVocabulary::create receives one cv::Mat per image and
          // image identity is preserved for TF-IDF weight computation.
          std::unordered_map<std::string, int> name_to_idx;
          std::vector<std::string> unique_names;
          for (const auto& name : img_names) {
              if (name_to_idx.find(name) == name_to_idx.end()) {
                  name_to_idx[name] = (int)unique_names.size();
                  unique_names.push_back(name);
              }
          }
          std::vector<std::vector<int>> rows_per_image(unique_names.size());
          for (int i = 0; i < (int)img_names.size(); ++i) {
              rows_per_image[name_to_idx[img_names[i]]].push_back(i);
          }
          std::vector<cv::Mat> features_vec;
          features_vec.reserve(unique_names.size());
          for (size_t img_idx = 0; img_idx < unique_names.size(); ++img_idx) {
              const auto& rows = rows_per_image[img_idx];
              cv::Mat img_features((int)rows.size(), features.cols, features.type());
              for (size_t r = 0; r < rows.size(); ++r) {
                  features.row(rows[r]).copyTo(img_features.row((int)r));
              }
              features_vec.push_back(img_features);
          }
          // Release the GIL during training so Python's signal handling is
          // not blocked for the entire duration of the C++ call.
          py::gil_scoped_release release;
          self.create(features_vec, scales, img_names);
      }, "features"_a, "scales"_a, "img_names"_a)
      .def("transform", &SizeBinnedVocabulary::transform, "features"_a, "scales"_a)
      .def("save", &SizeBinnedVocabulary::save, "output_dir"_a)
      .def_static("load", &SizeBinnedVocabulary::load, "input_dir"_a, "config"_a)
      .def("getNumWords", &SizeBinnedVocabulary::getNumWords)
      .def("getWordSize", &SizeBinnedVocabulary::getWordSize)
      .def("getNumScales", &SizeBinnedVocabulary::getNumScales)
      .def("isCreated", &SizeBinnedVocabulary::isCreated);

}
