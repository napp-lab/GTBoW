/*
 * GTBoW adaptation of DBoW3 QueryResults.h, derived from DBoW2.
 * Original author: Dorian Galvez-Lopez; DBoW3 modifications by Rafael Muñoz-Salinas.
 * Copyright (c) 2015 Dorian Galvez-Lopez. http://doriangalvez.com
 * Copyright (c) 2016 Rafael Muñoz-Salinas. All rights reserved.
 * Third-party portions retain the terms in thirdparty/DBow3/LICENSE.txt
 * and thirdparty/DBow3/DBOW2_LICENSE.txt. Original GTBoW additions are MIT.
 */
#ifndef __D_T_GTQUERY_RESULTS__
#define __D_T_GTQUERY_RESULTS__

#include <vector>
#include "exports.h"
#include "QueryResults.h"

namespace DBoW3 {

/// Single result of a query
class DBOW_API GTResult
{
public:
  
  /// Entry id
  EntryId Id;
  
  /// Score obtained
  double Score;
  
  struct PointMatch
  {
    unsigned int query_point_id;
    double query_point_weight;
    unsigned int db_point_id;
    double db_point_weight;

    PointMatch(unsigned int qid, double qweight, unsigned int did, double dweight)
        : query_point_id(qid), query_point_weight(qweight), db_point_id(did), db_point_weight(dweight){}
  };

  std::vector<PointMatch> point_matches;

  /// Orientation-voting diagnostics for the candidate. These fields do not
  /// affect scoring and are populated by GTDatabase::queryL2.
  int winning_orientation_bin = -1;
  std::vector<double> orientation_vote_mass;
  std::vector<unsigned int> orientation_match_counts;

  /// debug
  int nWords; // words in common
  // !!! this is filled only by Bhatt score!
  // (and for BCMatching, BCThresholding then)
  
  double bhatScore, chiScore;
  /// debug
  
  // only done by ChiSq and BCThresholding 
  double sumCommonVi;
  double sumCommonWi;
  double expectedChiScore;
  /// debug

  /**
   * Empty constructors
   */
  inline GTResult(){}
  
  /**
   * Creates a result with the given data
   * @param _id entry id
   * @param _score score
   */
  inline GTResult(EntryId _id, double _score): Id(_id), Score(_score){}

  // /**
  //   * Creates a result with the given data and initializes the number of elements in point_matches
  //   * @param _id entry id
  //   * @param _score score
  //   * @param numElements initial number of elements in point_matches
  //   */
  //   inline GTResult(EntryId _id, double _score, size_t numElements): Id(_id), Score(_score)
  //   {
  //       point_matches.reserve(numElements);
  //   }

  /**
   * Compares the scores of two results
   * @return true iff this.score < r.score
   */
  inline bool operator<(const GTResult &r) const
  {
    return this->Score < r.Score;
  }

  /**
   * Compares the scores of two results
   * @return true iff this.score > r.score
   */
  inline bool operator>(const GTResult &r) const
  {
    return this->Score > r.Score;
  }

  /**
   * Compares the entry id of the result
   * @return true iff this.id == id
   */
  inline bool operator==(EntryId id) const
  {
    return this->Id == id;
  }
  
  /**
   * Compares the score of this entry with a given one
   * @param s score to compare with
   * @return true iff this score < s
   */
  inline bool operator<(double s) const
  {
    return this->Score < s;
  }
  
  /**
   * Compares the score of this entry with a given one
   * @param s score to compare with
   * @return true iff this score > s
   */
  inline bool operator>(double s) const
  {
    return this->Score > s;
  }
  
  /**
   * Compares the score of two results
   * @param a
   * @param b
   * @return true iff a.Score > b.Score
   */
  static inline bool gt(const GTResult &a, const GTResult &b)
  {
    return a.Score > b.Score;
  }
  
  /**
   * Compares the scores of two results
   * @return true iff a.Score > b.Score
   */
  inline static bool ge(const GTResult &a, const GTResult &b)
  {
    return a.Score > b.Score;
  }
  
  /**
   * Returns true iff a.Score >= b.Score
   * @param a
   * @param b
   * @return true iff a.Score >= b.Score
   */
  static inline bool geq(const GTResult &a, const GTResult &b)
  {
    return a.Score >= b.Score;
  }
  
  /**
   * Returns true iff a.Score >= s
   * @param a
   * @param s
   * @return true iff a.Score >= s
   */
  static inline bool geqv(const GTResult &a, double s)
  {
    return a.Score >= s;
  }
  
  
  /**
   * Returns true iff a.Id < b.Id
   * @param a
   * @param b
   * @return true iff a.Id < b.Id
   */
  static inline bool ltId(const GTResult &a, const GTResult &b)
  {
    return a.Id < b.Id;
  }
  
  /**
   * Prints a string version of the result
   * @param os ostream
   * @param ret GTResult to print
   */
  friend std::ostream & operator<<(std::ostream& os, const GTResult& ret );
};

/// Multiple results from a query
class GTQueryResults: public std::vector<GTResult>
{
public:

  /** 
   * Multiplies all the scores in the vector by factor
   * @param factor
   */
  inline void scaleScores(double factor);
  
  /**
   * Prints a string version of the results
   * @param os ostream
   * @param ret GTQueryResults to print
   */
  DBOW_API friend std::ostream & operator<<(std::ostream& os, const GTQueryResults& ret );
  
  /**
   * Saves a matlab file with the results 
   * @param filename 
   */
  void saveM(const std::string &filename) const;
  
};

// --------------------------------------------------------------------------

inline void GTQueryResults::scaleScores(double factor)
{
  for(GTQueryResults::iterator qit = begin(); qit != end(); ++qit) 
    qit->Score *= factor;
}

// --------------------------------------------------------------------------

} // namespace TemplatedBoW
  
#endif
