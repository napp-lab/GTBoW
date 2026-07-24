import unittest
import cv2
import numpy as np
import glob
import os
import shutil
from pyGTBoW import Vocabulary, Database

class TestDatabase(unittest.TestCase):
    def setUp(self):
        # Path to save/load vocabulary and database (in tests_output directory)
        tests_output_dir = os.path.join(os.path.dirname(__file__), "tests_output")
        os.makedirs(tests_output_dir, exist_ok=True)
        self.vocab_path = os.path.join(tests_output_dir, "test_db_standard_vocabulary.voc")

        # Try to load existing vocabulary
        if os.path.exists(self.vocab_path):
            try:
                self.voc = Vocabulary(self.vocab_path)
                print("Loaded existing vocabulary from", self.vocab_path)
            except Exception as e:
                print(f"Failed to load vocabulary: {e}")
                print("Creating new vocabulary...")
                self.voc = self._create_vocabulary()
        else:
            print("Creating new vocabulary...")
            self.voc = self._create_vocabulary()

        # Initialize Database with the vocabulary
        self.db = Database(self.voc, use_di=True, di_levels=0)

        # Store dataset info for tests
        self.data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "dataset")
        self.image_pattern = os.path.join(self.data_dir, "*.png")
        self.orb = cv2.ORB_create(nfeatures=100)

    def _create_vocabulary(self):
        """Helper method to create and train a vocabulary"""
        # Initialize the vocabulary with standard params
        voc = Vocabulary(K=10, L=4)  # Reduced depth for faster training

        # Path to the dataset directory
        self.data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "dataset")
        self.image_pattern = os.path.join(self.data_dir, "*.png")

        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=100)

        # Extract ORB features from all images and train vocabulary
        image_files = glob.glob(self.image_pattern)
        all_descriptors = []

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)

            if descriptors is not None and len(keypoints) > 0 and descriptors.shape[1] == 32:
                all_descriptors.append(descriptors)

        if len(all_descriptors) == 0:
            raise RuntimeError("No valid descriptors found in any image.")

        print(f"Using {len(all_descriptors)} images for standard vocabulary creation.")

        # Create vocabulary
        voc.create(all_descriptors)

        # Save the vocabulary for future use
        os.makedirs(os.path.dirname(self.vocab_path), exist_ok=True)
        voc.save(self.vocab_path)
        print("Saved vocabulary to", self.vocab_path)

        return voc

    def tearDown(self):
        # Clean up the saved vocabulary after all tests if necessary
        # Keeping it for next tests speeds up execution, so we only remove if specified or left alone
        pass

    def test_database_creation(self):
        """Test basic Database creation and properties"""
        # Test creation without vocabulary
        db_empty = Database(use_di=False, di_levels=0)
        self.assertIsInstance(db_empty, Database)

        # Test creation with vocabulary
        db_with_vocab = Database(self.voc, use_di=True, di_levels=2)
        self.assertIsInstance(db_with_vocab, Database)

    def test_add_and_query_features(self):
        """Test adding raw features and querying them"""
        image_files = glob.glob(self.image_pattern)
        self.assertGreater(len(image_files), 0, "No test images found")

        # Process first image
        img = cv2.imread(image_files[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        self.assertIsNotNone(descriptors, "No descriptors found in test image")

        # Ensure correct type for C++ wrapper (CV_8U)
        if descriptors.dtype != np.uint8:
            descriptors = descriptors.astype(np.uint8)

        # Add entry to Database
        entry_id = self.db.add(descriptors)
        self.assertIsInstance(entry_id, int)
        self.assertGreaterEqual(entry_id, 0)

        # Query with the same descriptors
        results = self.db.query(descriptors, max_results=5, max_id=-1)

        # Check results format
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "No query results returned")

        # Check first result
        result = results[0]
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)  # (id, score)

        result_id, score = result
        self.assertEqual(result_id, entry_id)  # Should match the entry we added
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_addBowVec_and_basic_query(self):
        """Test adding BowVec entries and basic querying"""
        image_files = glob.glob(self.image_pattern)
        self.assertGreater(len(image_files), 0, "No test images found")

        # Process first image
        img = cv2.imread(image_files[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        self.assertIsNotNone(descriptors, "No descriptors found in test image")

        # Get bow_vec from vocabulary
        bow_vec, feat_vec = self.voc.transform(descriptors, 0)

        # Add entry to Database
        entry_id = self.db.addBowVec(bow_vec)
        self.assertIsInstance(entry_id, int)
        self.assertGreaterEqual(entry_id, 0)

        # Query with the same BowVector
        results = self.db.queryBowVec(bow_vec, max_results=5, max_id=-1, scoring_type="L2_NORM")

        # Check results format
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "No query results returned")

        # Check first result
        result = results[0]
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)  # (id, score)

        result_id, score = result
        self.assertEqual(result_id, entry_id)  # Should match the entry we added
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_multiple_entries_and_retrieval(self):
        """Test adding multiple entries and retrieving them"""
        image_files = sorted(glob.glob(self.image_pattern))
        entry_ids = []
        bow_vectors = []

        # Add multiple images to the database
        for i, img_path in enumerate(image_files[:3]):  # Use first 3 images
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)

            if descriptors is not None and len(keypoints) > 0:
                bow_vec, feat_vec = self.voc.transform(descriptors, 0)
                bow_vectors.append(bow_vec)

                # Add to database
                entry_id = self.db.addBowVec(bow_vec)
                entry_ids.append(entry_id)

        self.assertGreater(len(entry_ids), 0, "No entries were added to the database")

        # Test querying with each added entry
        for i, (entry_id, bow_vec) in enumerate(zip(entry_ids, bow_vectors)):
            results = self.db.queryBowVec(bow_vec, max_results=len(entry_ids), max_id=-1, scoring_type="L2_NORM")

            # Check that we get results
            self.assertGreater(len(results), 0)

            # The best match should be the same entry (assuming perfect match)
            best_result = results[0]
            best_id, best_score = best_result
            self.assertEqual(best_id, entry_id)

    def test_clear_database(self):
        """Test clearing the database"""
        img = cv2.imread(glob.glob(self.image_pattern)[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        bow_vec, feat_vec = self.voc.transform(descriptors, 0)

        entry_id = self.db.addBowVec(bow_vec)

        # Query should return results
        results = self.db.queryBowVec(bow_vec, max_results=5)
        self.assertGreater(len(results), 0, "Should have results before clearing")

        # Clear database
        self.db.clear()

        # Query should return no results
        results = self.db.queryBowVec(bow_vec, max_results=5)
        self.assertEqual(len(results), 0, "Should have no results after clearing")

    def test_empty_query_handling(self):
        """Test handling of empty or invalid queries"""
        empty_bow_vec = {}

        # This should not crash and should return empty results
        results = self.db.queryBowVec(empty_bow_vec, max_results=5)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0, "Empty query should return no results")

    def test_setNumWords(self):
        """Test setNumWords functionality"""
        self.db.setNumWords(1000)

    def test_max_results_and_max_id_limits(self):
        """Test max_results and max_id parameters"""
        image_files = sorted(glob.glob(self.image_pattern))
        entry_ids = []
        bow_vectors = []

        for img_path in image_files[:3]:
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)

            if descriptors is not None and len(keypoints) > 0:
                bow_vec, feat_vec = self.voc.transform(descriptors, 0)
                bow_vectors.append(bow_vec)

                entry_id = self.db.addBowVec(bow_vec)
                entry_ids.append(entry_id)

        if len(entry_ids) >= 2:
            # Test max_results parameter
            results = self.db.queryBowVec(bow_vectors[0], max_results=1)
            self.assertLessEqual(len(results), 1, "max_results=1 should return at most 1 result")

            results = self.db.queryBowVec(bow_vectors[0], max_results=2)
            self.assertLessEqual(len(results), 2, "max_results=2 should return at most 2 results")

            # Test max_id parameter
            if len(entry_ids) >= 2:
                results = self.db.queryBowVec(bow_vectors[0], max_results=10, max_id=entry_ids[0])
                # Should only return entries with ID <= max_id
                for result in results:
                    result_id = result[0]
                    self.assertLessEqual(result_id, entry_ids[0], "Result ID should be <= max_id")

    def test_score_normalization(self):
        """Test that scores are properly normalized between 0 and 1"""
        img = cv2.imread(glob.glob(self.image_pattern)[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        bow_vec, feat_vec = self.voc.transform(descriptors, 0)
        entry_id = self.db.addBowVec(bow_vec)

        # Query with exact match - should get high score
        results = self.db.queryBowVec(bow_vec, max_results=5)

        self.assertGreater(len(results), 0, "Should have results")

        for result in results:
            result_id, score = result
            self.assertGreaterEqual(score, 0.0, "Score should be >= 0")
            self.assertLessEqual(score, 1.0, "Score should be <= 1")

        best_score = results[0][1]
        self.assertGreater(best_score, 0.8, "Exact match should have high score")

    def test_retrieve_features(self):
        """Test retrieving features from the database"""
        img = cv2.imread(glob.glob(self.image_pattern)[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        bow_vec, feat_vec = self.voc.transform(descriptors, 0)

        # Add raw features to database to properly populate Direct Index
        entry_id = self.db.add(descriptors)
        self.assertIsInstance(entry_id, int)
        self.assertGreaterEqual(entry_id, 0)

        # Retrieve features for the valid entry
        features_dict = self.db.retrieveFeatures(entry_id)

        self.assertIsInstance(features_dict, dict)
        self.assertGreater(len(features_dict), 0, "Retrieved features should not be empty")

        for node_id, feature_indices in features_dict.items():
            self.assertIsInstance(node_id, int)
            self.assertIsInstance(feature_indices, list)
            for idx in feature_indices:
                self.assertIsInstance(idx, int)
                self.assertGreaterEqual(idx, 0)

        # Retrieve features for an invalid entry (should now raise IndexError)
        invalid_id = entry_id + 100
        with self.assertRaises(IndexError):
            self.db.retrieveFeatures(invalid_id)


    def test_save_load_database(self):
        """Test saving and loading the database"""
        tests_output_dir = os.path.join(os.path.dirname(__file__), "tests_output")
        db_path = os.path.join(tests_output_dir, "test_database_save.yml.gz")

        # Add some data to original db
        img = cv2.imread(glob.glob(self.image_pattern)[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        bow_vec, feat_vec = self.voc.transform(descriptors, 0)
        self.db.addBowVec(bow_vec)

        # Save db
        self.db.save(db_path)
        self.assertTrue(os.path.exists(db_path), "Database file should exist after saving")

        # Load into new db
        new_db = Database(self.voc, use_di=True, di_levels=0)
        new_db.load(db_path)

        # Query new db
        results = new_db.queryBowVec(bow_vec, max_results=1)
        self.assertGreater(len(results), 0, "Loaded database should return results")
        self.assertGreater(results[0][1], 0.8, "Loaded database should have correct matching scores")

        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == '__main__':
    unittest.main(verbosity=2)
