/*
 * GTBoW adaptation of DBoW3 Database.cpp, derived from DBoW2.
 * Original author: Dorian Galvez-Lopez; DBoW3 modifications by Rafael Muñoz-Salinas.
 * Copyright (c) 2015 Dorian Galvez-Lopez. http://doriangalvez.com
 * Copyright (c) 2016 Rafael Muñoz-Salinas. All rights reserved.
 * Third-party portions retain the terms in thirdparty/DBow3/LICENSE.txt
 * and thirdparty/DBow3/DBOW2_LICENSE.txt. Original GTBoW additions are MIT.
 */
#include "GTDatabase.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#if defined(TRACY_ENABLE)
#include <tracy/Tracy.hpp>
#else
#define ZoneScopedN(x)
#endif

namespace DBoW3{

namespace {
constexpr char kMagic[8] = {'G', 'T', 'B', 'O', 'W', 'D', 'B', '\0'};
constexpr uint32_t kVersion = 1;
constexpr uint64_t kMaxContainerSize = 100000000ULL;

template <typename T>
void writeValue(std::ostream& out, const T& value)
{
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!out) throw std::runtime_error("GTDatabase: failed while writing database");
}

template <typename T>
T readValue(std::istream& in)
{
  T value;
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) throw std::runtime_error("GTDatabase: truncated database file");
  return value;
}

size_t checkedSize(std::istream& in, const char* name)
{
  const uint64_t size = readValue<uint64_t>(in);
  if (size > kMaxContainerSize)
    throw std::runtime_error(std::string("GTDatabase: corrupt ") + name + " size");
  return static_cast<size_t>(size);
}

}

// --------------------------------------------------------------------------


GTDatabase::GTDatabase
  (bool use_di, int di_levels)
  : m_voc(NULL), m_use_di(use_di), m_dilevels(di_levels), m_nentries(0)
{
  if(di_levels < 0)
    throw std::invalid_argument("GTDatabase: di_levels must be >= 0");
}

// --------------------------------------------------------------------------

GTDatabase::GTDatabase
  (const Vocabulary &voc, bool use_di, int di_levels)
  : m_voc(NULL), m_use_di(use_di), m_dilevels(di_levels)
{
  if(di_levels < 0)
    throw std::invalid_argument("GTDatabase: di_levels must be >= 0");
  setVocabulary(voc);
  clear();
}

// --------------------------------------------------------------------------


GTDatabase::GTDatabase
  (const GTDatabase &db)
  : m_voc(NULL)
{
  *this = db;
}

// --------------------------------------------------------------------------


// GTDatabase::GTDatabase
//   (const std::string &filename)
//   : m_voc(NULL)
// {
//   load(filename);
// }

// // --------------------------------------------------------------------------


// GTDatabase::GTDatabase
//   (const char *filename)
//   : m_voc(NULL)
// {
//   load(filename);
// }

// --------------------------------------------------------------------------


GTDatabase::~GTDatabase(void)
{
  delete m_voc;
}

// --------------------------------------------------------------------------


GTDatabase& GTDatabase::operator=
  (const GTDatabase &db)
{
  if(this != &db)
  {
    m_dfile = db.m_dfile;
    m_dilevels = db.m_dilevels;
    m_ifile = db.m_ifile;
    m_pindex = db.m_pindex;
    m_nentries = db.m_nentries;
    m_use_di = db.m_use_di;
    delete m_voc;
    m_voc = db.m_voc ? new Vocabulary(*db.m_voc) : NULL;
    m_results_pool.clear();
    m_active_entries.clear();
    m_touched_flags.clear();
  }
  return *this;
}

// ---------------------------------------------------------------------------

std::vector<GTDatabase::FlatPointInfo> GTDatabase::getPointInfoFlat(const cv::Mat& assignments, const cv::Mat& weights, const cv::Mat& poses, unsigned int starting_idx /*= 0*/) const
{
  ZoneScopedN("GTDatabase::getPointInfoFlat");
  const int num_points = assignments.rows;
  if(weights.rows != num_points || poses.rows != num_points)
  {
      throw std::invalid_argument("GTDatabase: assignments, weights and poses must have the same number of rows");
  }
  const int num_assignments = assignments.cols;

  std::vector<FlatPointInfo> flat_info;
  flat_info.reserve(num_points * num_assignments);

  for (int img_point_num = 0; img_point_num < num_points; img_point_num++)
  {
    const int* assign_row = assignments.ptr<int>(img_point_num);
    const double* weight_row = weights.ptr<double>(img_point_num);
    const double orientation = poses.ptr<double>(img_point_num)[2];

    for (int assignment_num = 0; assignment_num < num_assignments; assignment_num++)
    {
      flat_info.push_back({
        static_cast<WordId>(assign_row[assignment_num]),
        starting_idx + static_cast<unsigned int>(img_point_num),
        weight_row[assignment_num],
        orientation
      });
    }
  }

  // Sort by word_id for efficient two-pointer iteration with sorted BowVector
  std::sort(flat_info.begin(), flat_info.end(),
      [](const FlatPointInfo& a, const FlatPointInfo& b) { return a.word_id < b.word_id; });

  return flat_info;
}

// --------------------------------------------------------------------------

EntryId GTDatabase::add(const DBoW3::BowVector &v, const cv::Mat &assignments, const cv::Mat &weights, const cv::Mat &poses)
{
  ZoneScopedN("GTDatabase::add");
  EntryId entry_id = m_nentries++;

  auto flat_info = getPointInfoFlat(assignments, weights, poses, m_pindex.size());

  // Add poses using raw pointer access
  for(int img_point_num = 0; img_point_num < poses.rows; img_point_num++)
  {
    const double* pose_row = poses.ptr<double>(img_point_num);
    m_pindex.emplace_back(std::array<double, 3>{pose_row[0], pose_row[1], pose_row[2]});
  }

  // Resize persistent active array for query results
  if (m_results_pool.size() < static_cast<size_t>(m_nentries)) {
    m_results_pool.resize(m_nentries);
    m_touched_flags.resize(m_nentries, false);
  }

  // Two-pointer merge: BowVector (sorted map) with flat_info (sorted by word_id)
  {
    ZoneScopedN("GTDatabase::add::TwoPointerMerge");
    size_t fi = 0;
    for(auto vit = v.begin(); vit != v.end(); ++vit)
    {
        const WordId word_id = vit->first;
        const WordValue& word_weight = vit->second;

        // Skip flat entries with word_id less than current BowVector word
        while (fi < flat_info.size() && flat_info[fi].word_id < word_id) fi++;

        // Collect all flat entries for this word_id into a PointInfo vector
        std::vector<PointInfo> points;
        while (fi < flat_info.size() && flat_info[fi].word_id == word_id) {
            points.push_back({flat_info[fi].point_id, flat_info[fi].weight, flat_info[fi].orientation});
            fi++;
        }

        IFRow& ifrow = m_ifile[word_id];
        ifrow.emplace_back(entry_id, word_weight, std::move(points));
    }
  }

  // Populate direct index (FeatureVector) if enabled
  if (m_use_di) {
    if (m_dfile.size() < static_cast<size_t>(m_nentries)) {
        m_dfile.resize(m_nentries);
    }
    FeatureVector& fv = m_dfile[entry_id];
    for(size_t i = 0; i < flat_info.size(); i++) {
        // Here flat_info contains point_id relative to the full m_pindex.
        // For FeatureVector we need the point ID relative to the image (0 to poses.rows - 1)
        unsigned int img_local_point_id = flat_info[i].point_id - (m_pindex.size() - poses.rows);

        WordId word_id = flat_info[i].word_id;
        NodeId node_id = word_id;
        if (m_dilevels > 0) {
            if (m_voc) {
                node_id = m_voc->getParentNode(word_id, m_dilevels);
            } else {
                throw std::runtime_error("GTDatabase: di_levels > 0 requires a vocabulary to be set.");
            }
        }
        fv[node_id].push_back(img_local_point_id);
    }
  }

  return entry_id;
}

// --------------------------------------------------------------------------


  void GTDatabase::setVocabulary
  (const Vocabulary& voc)
{
  delete m_voc;
  m_voc = new Vocabulary(voc);
  clear();
}

// --------------------------------------------------------------------------


  void GTDatabase::setVocabulary
  (const Vocabulary& voc, bool use_di, int di_levels)
{
  if(di_levels < 0)
    throw std::invalid_argument("GTDatabase: di_levels must be >= 0");
  m_use_di = use_di;
  m_dilevels = di_levels;
  delete m_voc;
  m_voc = new Vocabulary(voc);
  clear();
}

// --------------------------------------------------------------------------


 const Vocabulary*
GTDatabase::getVocabulary() const
{
  return m_voc;
}

// --------------------------------------------------------------------------


 void GTDatabase::clear()
{
  // resize vectors
  m_ifile.resize(0);
  if (m_voc != NULL) {
    m_ifile.resize(m_voc->size());
  }
  m_dfile.resize(0);
  m_pindex.clear();  // Clear the pose index as well
  m_nentries = 0;
  // Clear persistent query arrays
  m_results_pool.clear();
  m_active_entries.clear();
  m_touched_flags.clear();
}

// --------------------------------------------------------------------------


 void GTDatabase::setNumWords(unsigned int nwords)
{
  if (m_nentries != 0)
    throw std::logic_error("GTDatabase: cannot change word count after adding entries");
  m_ifile.resize(nwords);
}

// --------------------------------------------------------------------------

void GTDatabase::save(const std::string &filename) const
{
  std::ofstream out(filename, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("GTDatabase: cannot open file for writing: " + filename);

  out.write(kMagic, sizeof(kMagic));
  writeValue(out, kVersion);
  writeValue<uint8_t>(out, m_use_di ? 1 : 0);
  writeValue<int32_t>(out, static_cast<int32_t>(m_dilevels));
  writeValue<uint64_t>(out, static_cast<uint64_t>(m_nentries));
  writeValue<uint64_t>(out, static_cast<uint64_t>(m_ifile.size()));

  for (const auto& row : m_ifile) {
    writeValue<uint64_t>(out, static_cast<uint64_t>(row.size()));
    for (const auto& entry : row) {
      writeValue<uint32_t>(out, entry.entry_id);
      writeValue<double>(out, entry.word_weight);
      writeValue<uint64_t>(out, static_cast<uint64_t>(entry.point_info.size()));
      for (const auto& point : entry.point_info) {
        writeValue<uint32_t>(out, point.point_id);
        writeValue<double>(out, point.weight);
        writeValue<double>(out, point.orientation);
      }
    }
  }

  writeValue<uint64_t>(out, static_cast<uint64_t>(m_dfile.size()));
  for (const auto& feature_vector : m_dfile) {
    writeValue<uint64_t>(out, static_cast<uint64_t>(feature_vector.size()));
    for (const auto& node : feature_vector) {
      writeValue<uint32_t>(out, node.first);
      writeValue<uint64_t>(out, static_cast<uint64_t>(node.second.size()));
      for (const auto feature_id : node.second)
        writeValue<uint32_t>(out, feature_id);
    }
  }

  writeValue<uint64_t>(out, static_cast<uint64_t>(m_pindex.size()));
  for (const auto& pose : m_pindex)
    for (double value : pose) writeValue<double>(out, value);
}

void GTDatabase::load(const std::string &filename)
{
  std::ifstream in(filename, std::ios::binary);
  if (!in) throw std::runtime_error("GTDatabase: cannot open file for reading: " + filename);

  char magic[sizeof(kMagic)];
  in.read(magic, sizeof(magic));
  if (!in) throw std::runtime_error("GTDatabase: truncated database file");
  if (!std::equal(std::begin(magic), std::end(magic), std::begin(kMagic)))
    throw std::runtime_error("GTDatabase: invalid database magic");
  if (readValue<uint32_t>(in) != kVersion)
    throw std::runtime_error("GTDatabase: unsupported database version");

  const uint8_t use_di = readValue<uint8_t>(in);
  const int32_t di_levels = readValue<int32_t>(in);
  const uint64_t nentries = readValue<uint64_t>(in);
  if (use_di > 1 || di_levels < 0 || nentries > std::numeric_limits<int>::max())
    throw std::runtime_error("GTDatabase: corrupt database settings");

  InvertedFile ifile(checkedSize(in, "inverted index"));
  for (auto& row : ifile) {
    row.resize(checkedSize(in, "inverted row"));
    EntryId previous_id = 0;
    bool first = true;
    for (auto& entry : row) {
      entry.entry_id = readValue<uint32_t>(in);
      entry.word_weight = readValue<double>(in);
      if (entry.entry_id >= nentries || (!first && entry.entry_id <= previous_id))
        throw std::runtime_error("GTDatabase: invalid inverted index entry reference");
      first = false;
      previous_id = entry.entry_id;
      entry.point_info.resize(checkedSize(in, "point list"));
      for (auto& point : entry.point_info) {
        point.point_id = readValue<uint32_t>(in);
        point.weight = readValue<double>(in);
        point.orientation = readValue<double>(in);
      }
    }
  }

  DirectFile dfile(checkedSize(in, "direct index"));
  if (use_di && dfile.size() < nentries)
    throw std::runtime_error("GTDatabase: direct index has fewer entries than database");
  if (!use_di && !dfile.empty())
    throw std::runtime_error("GTDatabase: direct index present while disabled");
  for (auto& feature_vector : dfile) {
    const size_t nodes = checkedSize(in, "feature vector");
    for (size_t i = 0; i < nodes; ++i) {
      const NodeId node_id = readValue<uint32_t>(in);
      auto& features = feature_vector[node_id];
      features.resize(checkedSize(in, "feature list"));
      for (auto& feature_id : features)
        feature_id = readValue<uint32_t>(in);
    }
  }

  PoseIndex pindex(checkedSize(in, "pose index"));
  for (auto& pose : pindex)
    for (double& value : pose) value = readValue<double>(in);

  for (const auto& row : ifile)
    for (const auto& entry : row)
      for (const auto& point : entry.point_info)
        if (point.point_id >= pindex.size())
          throw std::runtime_error("GTDatabase: invalid pose index reference");

  if (in.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error("GTDatabase: trailing data after database payload");

  m_use_di = use_di != 0;
  m_dilevels = di_levels;
  m_nentries = static_cast<int>(nentries);
  m_ifile.swap(ifile);
  m_dfile.swap(dfile);
  m_pindex.swap(pindex);
  m_results_pool.clear();
  m_active_entries.clear();
  m_touched_flags.clear();
}

// --------------------------------------------------------------------------


void GTDatabase::allocate(int nd, int ni)
{
  // m_ifile already contains |words| items
  if(ni > 0)
  {
    for(auto rit = m_ifile.begin(); rit != m_ifile.end(); ++rit)
    {
      int n = (int)rit->size();
      if(ni > n)
      {
        rit->resize(ni);
        rit->resize(n);
      }
    }
  }

  if(m_use_di && (int)m_dfile.size() < nd)
  {
    m_dfile.resize(nd);
  }
}

void GTDatabase::queryL2(const BowVector &vec,
  GTQueryResults &ret, int max_results, int max_id,
  const cv::Mat &assignments, const cv::Mat &weights, const cv::Mat &poses,
  int num_orientation_bins) const
{
  ZoneScopedN("GTDatabase::queryL2");
  if(num_orientation_bins == 0 || num_orientation_bins < -1)
  {
    throw std::invalid_argument("Number of orientation bins must be greater than 0 or -1");
  }
  num_orientation_bins = num_orientation_bins == -1 ? 1 : num_orientation_bins;

  BowVector::const_iterator vit;

  const int num_result_bins = num_orientation_bins;
  const double inv_bin_width = num_orientation_bins / (2.0 * M_PI);
  const double half_bins = num_orientation_bins / 2.0;
  const double twoPi = 2.0 * M_PI;
  
  // --- Persistent active array (Optimization 3) ---
  // Ensure pool is sized for all entries
  if (m_results_pool.size() < static_cast<size_t>(m_nentries)) {
    m_results_pool.resize(m_nentries);
    m_touched_flags.resize(m_nentries, false);
  }
  // Reset only previously touched entries
  for (const auto& eid : m_active_entries) {
    m_results_pool[eid].clear();
    m_touched_flags[eid] = false;
  }
  m_active_entries.clear();

  // --- Flat sorted feature indexing (Optimization 2) ---
  std::vector<FlatPointInfo> query_flat_info;
  {
    ZoneScopedN("GTDatabase::queryL2::GetPointInfoFlat");
    query_flat_info = getPointInfoFlat(assignments, weights, poses);
  }

  {
    ZoneScopedN("GTDatabase::queryL2::WordIteration");
  size_t fi = 0;
  for (vit = vec.begin(); vit != vec.end(); ++vit) // for each query bow_vec word
  {
    const WordId word_id = vit->first;
    const WordValue& qvalue = vit->second;

    // Advance flat index past words less than current BowVector word
    while (fi < query_flat_info.size() && query_flat_info[fi].word_id < word_id) fi++;

    // Identify the slice [fi_start, fi_end) of query points for this word
    size_t fi_start = fi;
    double sum_query_word_weight = 0;
    while (fi < query_flat_info.size() && query_flat_info[fi].word_id == word_id) {
      sum_query_word_weight += query_flat_info[fi].weight;
      fi++;
    }
    size_t fi_end = fi;

    if (fi_start == fi_end) continue;  // No query points for this word

    // Check if word_id is valid for this database
    if (word_id >= m_ifile.size()) {
      continue; // Skip this word if it's out of bounds
    }

    const IFRow& row = m_ifile[word_id];

    // IFRows are sorted in ascending entry_id order

    for (auto rit = row.begin(); rit != row.end(); ++rit) // for each image that contains the word
    {
      const EntryId entry_id = rit->entry_id;
      const WordValue& dvalue = rit->word_weight;

      if ((int)entry_id < max_id || max_id == -1)
      {

        double sum_db_word_weight = 0;
        // sum up the weights of the db points that correspond to the word
        for (const auto& db_point_info : rit->point_info)
        {
          sum_db_word_weight += db_point_info.weight;
        }

        // Pre-compute reciprocals to avoid repeated division in inner loop
        const double inv_sum_query = qvalue / sum_query_word_weight;
        const double inv_sum_db = dvalue / sum_db_word_weight;

        // Use persistent active array instead of unordered_map find/insert
        if (!m_touched_flags[entry_id]) {
          m_touched_flags[entry_id] = true;
          m_active_entries.push_back(entry_id);
          m_results_pool[entry_id].assign(num_result_bins, GTResult(entry_id, 0));
        }
        auto& entry_results = m_results_pool[entry_id];

        for (size_t qi = fi_start; qi < fi_end; qi++)
        {
          const auto& qpi = query_flat_info[qi];
          // Hoist query factor out of inner loop
          const double query_factor = -(qpi.weight * inv_sum_query);
          for (const auto& db_point_info : rit->point_info)
          {
            double value = query_factor * (db_point_info.weight * inv_sum_db); // minus sign baked into query_factor

            double orientation_diff = qpi.orientation - db_point_info.orientation;
            // Branchless wrapping to [-pi, pi] (Optimization 4)
            orientation_diff -= (orientation_diff > M_PI) * twoPi;
            orientation_diff += (orientation_diff < -M_PI) * twoPi;

            int orientation_bin = (int)(orientation_diff * inv_bin_width + half_bins + 0.5);
            // Handle edge case where orientation diff is exactly pi
            if (orientation_bin >= num_orientation_bins)
               orientation_bin = 0;
            else if (orientation_bin < 0)
               orientation_bin = 0;

            entry_results[orientation_bin].Score += value;
            entry_results[orientation_bin].point_matches.emplace_back(qpi.point_id, qpi.weight, db_point_info.point_id, db_point_info.weight);
          }  
        }
        
      }

    } // for each inverted row
  } // for each query word
  }

  // Collect results from active entries
  {
    ZoneScopedN("GTDatabase::queryL2::CollectResults");
  ret.reserve(m_active_entries.size());
  for (const auto& entry_id : m_active_entries)
  {
    auto& bins = m_results_pool[entry_id];
    // Find the bin with the smallest (most negative) score — the best-matching
    // orientation bin. The strict comparison intentionally preserves the
    // existing first-bin tie breaking.
    double min_score = std::numeric_limits<double>::max();
    int min_score_idx = 0;
    std::vector<double> vote_mass(num_result_bins, 0.0);
    std::vector<unsigned int> match_counts(num_result_bins, 0);
    for (int i = 0; i < num_result_bins; i++)
    {
      vote_mass[i] = std::max(0.0, -bins[i].Score);
      match_counts[i] = static_cast<unsigned int>(bins[i].point_matches.size());
      if (bins[i].Score < min_score)
      {
        min_score = bins[i].Score;
        min_score_idx = i;
      }
    }
    GTResult selected = std::move(bins[min_score_idx]);
    selected.winning_orientation_bin = min_score_idx;
    selected.orientation_vote_mass = std::move(vote_mass);
    selected.orientation_match_counts = std::move(match_counts);
    ret.push_back(std::move(selected));
  }
  }

  // resulting "scores" are now in [-1 best .. 0 worst]

  // sort vector in ascending order of score
  {
    ZoneScopedN("GTDatabase::queryL2::SortAndScoreConversion");
  std::sort(ret.begin(), ret.end());
  // (ret is inverted now --the lower the better--)

  // cut vector
  if(max_results > 0 && (int)ret.size() > max_results)
    ret.resize(max_results);

  // complete and scale score to [0 worst .. 1 best]
  // ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i)
    //		for all i | v_i != 0 and w_i != 0 )
    // (Nister, 2006)
    GTQueryResults::iterator qit;
  for(qit = ret.begin(); qit != ret.end(); qit++)
  {
    if(qit->Score <= -1.0) // rounding error
      qit->Score = 1.0;
    else
      qit->Score = 1.0 - sqrt(1.0 + qit->Score); // [0..1]
      // the + sign is ok, it is due to - sign in
      // value = - qvalue * dvalue
  }
  }

}

std::pair<cv::Mat, std::unordered_map<unsigned int, unsigned int>> GTDatabase::getPoses(const std::set<unsigned int>& point_indices) const
{
  cv::Mat poses(point_indices.size(), 3, CV_64F);
  std::unordered_map<unsigned int, unsigned int> index_map;

  unsigned int new_index = 0;
  for (unsigned int db_point_idx : point_indices)
  {
    if (db_point_idx < m_pindex.size())
    {
      poses.at<double>(new_index, 0) = m_pindex[db_point_idx][0];
      poses.at<double>(new_index, 1) = m_pindex[db_point_idx][1];
      poses.at<double>(new_index, 2) = m_pindex[db_point_idx][2];
      index_map[db_point_idx] = new_index;
      new_index++;
    }
  }

  // Trim to the number of valid rows actually filled to avoid returning
  // uninitialized data if any requested point indices were out of bounds.
  if (new_index < (unsigned int)point_indices.size())
  {
    poses = poses.rowRange(0, new_index).clone();
  }

  return std::make_pair(poses, index_map);
}

const FeatureVector& GTDatabase::retrieveFeatures
  (EntryId id) const
{
  if(id >= size())
    throw std::out_of_range("GTDatabase::retrieveFeatures: entry id out of range");
  return m_dfile[id];
}

std::ostream& operator<<(std::ostream &os,
  const GTDatabase &db)
{
  os << "Database: Entries = " << db.size() << ", "
    "Using direct index = " << (db.usingDirectIndex() ? "yes" : "no");

  if(db.usingDirectIndex())
    os << ", Direct index levels = " << db.getDirectIndexLevels();

  os << ". " << *db.getVocabulary();
  return os;
}

}
