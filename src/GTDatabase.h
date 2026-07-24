/*
 * GTBoW adaptation of DBoW3 Database.h, derived from DBoW2.
 * Original author: Dorian Galvez-Lopez; DBoW3 modifications by Rafael Muñoz-Salinas.
 * Copyright (c) 2015 Dorian Galvez-Lopez. http://doriangalvez.com
 * Copyright (c) 2016 Rafael Muñoz-Salinas. All rights reserved.
 * Third-party portions retain the terms in thirdparty/DBow3/LICENSE.txt
 * and thirdparty/DBow3/DBOW2_LICENSE.txt. Original GTBoW additions are MIT.
 */

#ifndef __D_T_GTDATABASE__
#define __D_T_GTDATABASE__

#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <list>
#include <set>
#include <unordered_map>

#include "Vocabulary.h"
#include "GTQueryResults.h"
#include "ScoringObject.h"
#include "BowVector.h"
#include "FeatureVector.h"
#include "exports.h"

namespace DBoW3 {

 ///   GTDatabase
class DBOW_API GTDatabase
{
public:

  /// Struct to hold info about each point that contributed to a word for an img entry
  struct PointInfo
  {
    unsigned int point_id;
    double weight;
    double orientation;
  };

  /// Flat point info for sorted feature indexing (avoids hash-map overhead)
  struct FlatPointInfo
  {
    WordId word_id;
    unsigned int point_id;
    double weight;
    double orientation;
  };

  /**
   * Creates an empty database without vocabulary
   * @param use_di a direct index is used to store feature indexes
   * @param di_levels levels to go up the vocabulary tree to select the 
   *   node id to store in the direct index when adding images
   */
  explicit GTDatabase(bool use_di = true, int di_levels = 0);

  /**
   * Creates a database with the given vocabulary
   * @param T class inherited from Vocabulary
   * @param voc vocabulary
   * @param use_di a direct index is used to store feature indexes
   * @param di_levels levels to go up the vocabulary tree to select the 
   *   node id to store in the direct index when adding images
   */

  explicit GTDatabase(const Vocabulary &voc, bool use_di = true,
    int di_levels = 0);

  /**
   * Copy constructor. Copies the vocabulary too
   * @param db object to copy
   */
  GTDatabase(const GTDatabase &db);

  // /** 
  //  * Creates the database from a file
  //  * @param filename
  //  */
  // GTDatabase(const std::string &filename);

  // /** 
  //  * Creates the database from a file
  //  * @param filename
  //  */
  // GTDatabase(const char *filename);

  /**
   * Destructor
   */
  virtual ~GTDatabase(void);

  /**
   * Copies the given database and its vocabulary
   * @param db database to copy
   */
  GTDatabase& operator=(
    const GTDatabase &db);

  /**
   * Sets the vocabulary to use and clears the content of the database.
   * @param T class inherited from Vocabulary
   * @param voc vocabulary to copy
   */
  void setVocabulary(const Vocabulary &voc);
  
  /**
   * Sets the vocabulary to use and the direct index parameters, and clears
   * the content of the database
   * @param T class inherited from Vocabulary
   * @param voc vocabulary to copy
   * @param use_di a direct index is used to store feature indexes
   * @param di_levels levels to go up the vocabulary tree to select the 
   *   node id to store in the direct index when adding images
   */

  void setVocabulary(const Vocabulary& voc, bool use_di, int di_levels = 0);
  
  /**
   * Returns a pointer to the vocabulary used
   * @return vocabulary
   */
  const Vocabulary* getVocabulary() const;

  /** 
   * Allocates some memory for the direct and inverted indexes
   * @param nd number of expected image entries in the database 
   * @param ni number of expected words per image
   * @note Use 0 to ignore a parameter
   */
  void allocate(int nd = 0, int ni = 0);

  /**
    * Organizes the assignments and weights into point info for each word for all the points in an image
    * @param assignments matrix that has the assignment of points to words
    * @param weights matrix that weights the assignments of the words
    * @param poses matrix holding the pose information for the points
    * @param starting_idx starting index for the point_ids
    */
    std::vector<FlatPointInfo> getPointInfoFlat(const cv::Mat& assignments, const cv::Mat& weights, const cv::Mat& poses, unsigned int starting_idx = 0) const;


/**
    * Adss an entry to the database and returns its index
    * @param vec bow vector
    * @param fec feature vector to add the entry. Only necessary if using the
    *   direct index
    * @param assignments matrix that has the assignment of points to words
    * @param weights matrix that weights the assignments of the words
    * @param poses matrix holding the pose information for the points
    * @return id of new entry
    */
  EntryId add(const BowVector &v,
     const cv::Mat &assignments = cv::Mat(),
     const cv::Mat &weights = cv::Mat(),
     const cv::Mat &poses = cv::Mat());

  /**
   * Empties the database
   */
  void clear();

  /**
   * Sets the number of words that the database will use (needed if no explicit vocabulary is given)
   * @param nwords
   */
  void setNumWords(unsigned int nwords);

  /**
   * Returns the number of entries in the database 
   * @return number of entries in the database
   */
  unsigned int size() const{  return m_nentries;}

  /**
   * Saves the database indexes and settings to a versioned binary file.
   * The vocabulary itself is not serialized; databases using di_levels > 0
   * must be paired with the same vocabulary when adding new entries.
   */
  void save(const std::string &filename) const;

  /**
   * Loads a database saved by save(), replacing the current contents.
   */
  void load(const std::string &filename);

  
  /**
   * Checks if the direct index is being used
   * @return true iff using direct index
   */
    bool usingDirectIndex() const{  return m_use_di;}
  
  /**
   * Returns the di levels when using direct index
   * @return di levels
   */
    int getDirectIndexLevels() const{  return m_dilevels;}
  

  /**
   * Gets the poses of the specified points in the database
   * @param point_indices set of point indices to get the poses for
   * @return a pair, the first being the poses and the second being a map of db_point_idx to their new row in the poses matrix
  */
  std::pair<cv::Mat, std::unordered_map<unsigned int, unsigned int>> getPoses(const std::set<unsigned int>& point_indices) const;

  /**
   * Returns the a feature vector associated with a database entry
   * @param id entry id (must be < size())
   * @return const reference to map of nodes and their associated features in
   *   the given entry
   */
  const FeatureVector& retrieveFeatures(EntryId id) const;

  // --------------------------------------------------------------------------

  /**
   * Writes printable information of the database
   * @param os stream to write to
   * @param db
   */
 DBOW_API friend   std::ostream& operator<<(std::ostream &os,
                                    const GTDatabase &db);



public:
  
  /// Query with L2 scoring
  void queryL2(const BowVector &vec, GTQueryResults &ret, 
    int max_results, int max_id, const cv::Mat &assignments,
    const cv::Mat &weights, const cv::Mat &poses,
    int num_orientation_bins) const;
  

protected:

  /* Inverted file declaration */

/// Item of IFRow
    struct IFEntry
    {
      /// Entry id
      EntryId entry_id;
      
      /// Word weight in this entry
      WordValue word_weight;

      /// A vector of point_ids, weights, and orientations
      std::vector<PointInfo> point_info;
        /**
         * Creates an empty entry
         */
        IFEntry() {}

        /**
         * Creates an inverted file entry with a single point
         * @param eid entry id
         * @param wv word weight
         * @param point_id point id
         */
        IFEntry(EntryId eid, WordValue wv, unsigned int point_id) :
          entry_id(eid), word_weight(wv) {
          point_info.push_back({ point_id, 0.0f, 0.0f });
        }

        /**
         * Creates an inverted file pair with a vector of point_info
         * @param eid entry id
         * @param wv word weight
         * @param point_info vector of point information
         */
        IFEntry(EntryId eid, WordValue wv, const std::vector<PointInfo>& point_info) :
          entry_id(eid), word_weight(wv), point_info(point_info) {}

        IFEntry(EntryId eid, WordValue wv, std::vector<PointInfo>&& point_info) :
          entry_id(eid), word_weight(wv), point_info(std::move(point_info)) {}

        /**
         * Add a point and its weight to the entry
         * @param point_id
         * @param weight
         * @param orientation
         */
        void addPoint(unsigned int point_id, double weight, double orientation) {
          point_info.push_back({ point_id, weight, orientation });
        }
        
        /**
         * L1 normalizes the weights of the points in the database.
         */
        void normalizeWeights() {
          double sum = 0.0f;
          for (auto& point : point_info) {
            sum += point.weight;
          }
          for (auto& point : point_info) {
            point.weight /= sum;
          }
        }

        /**
         * Compares the entry ids
         * @param eid
         * @return true iff this entry id is the same as eid
         */
        inline bool operator==(EntryId eid) const { return entry_id == eid; }

    };

  /// Row of InvertedFile
  typedef std::vector<IFEntry> IFRow;
  // IFRows are sorted in ascending entry_id order
  
  /// Inverted index
  typedef std::vector<IFRow> InvertedFile; 
  // InvertedFile[word_id] --> inverted file of that word
  
  /* Direct file declaration */

  /// Direct index
  typedef std::vector<FeatureVector> DirectFile;
  // DirectFile[entry_id] --> [ directentry, ... ]

  // Pose Index [point_id](x, y, orientation)
typedef std::vector<std::array<double, 3>> PoseIndex;

protected:

  /// Associated vocabulary
  Vocabulary *m_voc;
  
  /// Flag to use direct index
  bool m_use_di;
  
  /// Levels to go up the vocabulary tree to select nodes to store
  /// in the direct index
  int m_dilevels;
  
  /// Inverted file (must have size() == |words|)
  InvertedFile m_ifile;
  
  /// Direct file (resized for allocation)
  DirectFile m_dfile;

  /// Pose Index
    PoseIndex m_pindex;
  
  /// Number of valid entries in m_dfile
  int m_nentries;

  /// Persistent active array for queryL2 (mutable because queryL2 is const)
  mutable std::vector<std::vector<GTResult>> m_results_pool;
  mutable std::vector<EntryId> m_active_entries;
  mutable std::vector<bool> m_touched_flags;
  
};



// --------------------------------------------------------------------------

} // namespace DBoW3

#endif
