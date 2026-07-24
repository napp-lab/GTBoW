import unittest
import cv2
import numpy as np
import glob
import os
import shutil
from pyGTBoW import SizeBinnedVocabulary, VocabularyConfig, GTDatabase

class TestGTDatabase(unittest.TestCase):
    def setUp(self):
        # Path to save/load vocabulary and database (in tests_output directory)
        tests_output_dir = os.path.join(os.path.dirname(__file__), "tests_output")
        os.makedirs(tests_output_dir, exist_ok=True)
        self.vocab_path = os.path.join(tests_output_dir, "test_gtdb_sizebinned_vocabulary")
        
        # Create vocabulary configuration
        self.config = VocabularyConfig()
        self.config.setVocabularyParameters(k=10, L=5, weight_method="TF_IDF", score_method="L2_NORM")
        
        # Try to load existing vocabulary
        if os.path.exists(self.vocab_path):
            try:
                self.voc = SizeBinnedVocabulary.load(self.vocab_path, self.config)
                print("Loaded existing vocabulary from", self.vocab_path)
            except Exception as e:
                print(f"Failed to load vocabulary: {e}")
                print("Creating new vocabulary...")
                self.voc = self._create_vocabulary()
        else:
            print("Creating new vocabulary...")
            self.voc = self._create_vocabulary()

        # Initialize GTDatabase without vocabulary (will set vocabulary later using setNumWords)
        self.gtdb = GTDatabase(use_di=True, di_levels=0)
        self.gtdb.setNumWords(self.voc.getWordSize())
        
        # Store dataset info for tests
        self.data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "dataset")
        self.image_pattern = os.path.join(self.data_dir, "*.png")
        self.orb = cv2.ORB_create(nfeatures=100)

    def _create_vocabulary(self):
        """Helper method to create and train a vocabulary"""
        # Initialize the vocabulary
        voc = SizeBinnedVocabulary(self.config)
        
        # Path to the dataset directory
        self.data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "dataset")
        self.image_pattern = os.path.join(self.data_dir, "*.png")
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=100)

        # Extract ORB features from all images and train vocabulary
        image_files = glob.glob(self.image_pattern)
        all_descriptors = []
        all_scales = []
        img_names = []

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            if descriptors is not None and len(keypoints) > 0 and descriptors.shape[1] == 32:
                all_descriptors.append(descriptors)
                # Use keypoint size as scale
                scales = [kp.size for kp in keypoints]
                all_scales.extend(scales)
                img_names.extend([os.path.basename(img_path)] * len(keypoints))

        if len(all_descriptors) == 0:
            raise RuntimeError("No valid descriptors found in any image.")

        # Create feature matrix
        feature_mat = np.vstack(all_descriptors)
        scales = np.array(all_scales, dtype=np.float32)

        print(f"Using {feature_mat.shape[0]} descriptors for vocabulary creation.")
        print(f"Scale range: [{scales.min():.2f}, {scales.max():.2f}]")

        # Create vocabulary
        voc.create(feature_mat, scales, img_names)
        
        # Save the vocabulary for future use
        os.makedirs(self.vocab_path, exist_ok=True)
        voc.save(self.vocab_path)
        print("Saved vocabulary to", self.vocab_path)

        return voc

    def tearDown(self):
        # Clean up the saved vocabulary after all tests
        if os.path.exists(self.vocab_path):
            shutil.rmtree(self.vocab_path)

    def test_gtdatabase_creation(self):
        """Test basic GTDatabase creation and properties"""
        # Test creation without vocabulary
        gtdb_empty = GTDatabase(use_di=False, di_levels=0)
        self.assertIsInstance(gtdb_empty, GTDatabase)
        
        # Test creation with vocabulary size set
        gtdb_with_vocab_size = GTDatabase(use_di=True, di_levels=2)
        gtdb_with_vocab_size.setNumWords(self.voc.getWordSize())
        self.assertIsInstance(gtdb_with_vocab_size, GTDatabase)

    def test_addBowVec_and_basic_query(self):
        """Test adding BowVec entries and basic querying"""
        # Get features from a test image
        image_files = glob.glob(self.image_pattern)
        self.assertGreater(len(image_files), 0, "No test images found")
        
        # Process first image
        img = cv2.imread(image_files[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        self.assertIsNotNone(descriptors, "No descriptors found in test image")
        self.assertGreater(len(keypoints), 0, "No keypoints found in test image")
        
        # Get features and scales for SizeBinnedVocabulary
        num_features = descriptors.shape[0]
        scales = np.array([kp.size for kp in keypoints], dtype=np.float32)
        
        # Get assignments, weights, and bow_vec from vocabulary
        assignments, weights, bow_vec = self.voc.transform(descriptors, scales)
        
        # Create poses: (x, y, orientation) for each keypoint
        poses = np.zeros((num_features, 3), dtype=np.float64)
        for i, kp in enumerate(keypoints):
            poses[i, 0] = kp.pt[0]  # x coordinate
            poses[i, 1] = kp.pt[1]  # y coordinate
            poses[i, 2] = kp.angle * np.pi / 180.0  # orientation in radians
        
        # Add entry to GTDatabase
        entry_id = self.gtdb.addBowVec(bow_vec, assignments, weights, poses)
        self.assertIsInstance(entry_id, int)
        self.assertGreaterEqual(entry_id, 0)
        
        # Query with the same BowVector
        results, db_poses = self.gtdb.queryBowVec(bow_vec, assignments, weights, poses, 5, -1, "L2_NORM")
        
        # Check results format
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "No query results returned")
        
        # Check first result
        result = results[0]
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)  # (id, score, match_point_ids, match_point_weights)
        
        result_id, score, match_point_ids, match_point_weights = result
        self.assertEqual(result_id, entry_id)  # Should match the entry we added
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Check match matrices
        self.assertEqual(match_point_ids.shape[1], 2)  # Two columns: query_id, db_id
        self.assertEqual(match_point_weights.shape[1], 2)  # Two columns: query_weight, db_weight
        self.assertEqual(match_point_ids.shape[0], match_point_weights.shape[0])
        
        # Check db_poses
        self.assertIsInstance(db_poses, np.ndarray)
        self.assertEqual(db_poses.shape[1], 3)  # x, y, orientation

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
                # Get features and scales for SizeBinnedVocabulary
                scales = np.array([kp.size for kp in keypoints], dtype=np.float32)
                
                # Get assignments, weights, and bow_vec from vocabulary
                assignments, weights, bow_vec = self.voc.transform(descriptors, scales)
                bow_vectors.append(bow_vec)
                
                # Create pose data
                num_features = descriptors.shape[0]
                poses = np.zeros((num_features, 3), dtype=np.float64)
                
                for j, kp in enumerate(keypoints):
                    poses[j, 0] = kp.pt[0]
                    poses[j, 1] = kp.pt[1]
                    poses[j, 2] = kp.angle * np.pi / 180.0
                
                # Add to database
                entry_id = self.gtdb.addBowVec(bow_vec, assignments, weights, poses)
                entry_ids.append(entry_id)
        
        self.assertGreater(len(entry_ids), 0, "No entries were added to the database")
        

        # Test querying with each added entry
        for i, (entry_id, bow_vec) in enumerate(zip(entry_ids, bow_vectors)):
            # Need to provide non-empty valid assignments and weights to get matches.
            # We will use exactly what we inserted to guarantee a perfect match.
            # However, we only have bow_vec saved here! Let's just create a dummy query with random data matching the dict keys of bow_vec.
            query_assignments = np.array(list(bow_vec.keys()), dtype=np.int32).reshape(-1, 1)
            query_weights = np.ones((len(query_assignments), 1), dtype=np.float64)
            query_poses = np.zeros((len(query_assignments), 3), dtype=np.float64)
            
            results, db_poses = self.gtdb.queryBowVec(bow_vec, query_assignments, query_weights, query_poses, len(entry_ids), -1, "L2_NORM")
            
            # Check that we get results
            self.assertGreater(len(results), 0)

            
            # The best match should be the same entry (assuming perfect match)
            best_result = results[0]
            best_id, best_score, _, _ = best_result
            self.assertEqual(best_id, entry_id)

    def test_geometric_verification_scoring(self):
        """Test that GTDatabase provides geometric verification through point matches"""
        # Add a test entry
        img = cv2.imread(glob.glob(self.image_pattern)[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        num_features = min(descriptors.shape[0], 20)  # Limit for testing
        limited_descriptors = descriptors[:num_features]
        limited_keypoints = keypoints[:num_features]
        scales = np.array([kp.size for kp in limited_keypoints], dtype=np.float32)
        
        # Get assignments, weights, and bow_vec from vocabulary
        assignments, weights, bow_vec = self.voc.transform(limited_descriptors, scales)
        
        poses = np.zeros((num_features, 3), dtype=np.float64)
        for i, kp in enumerate(limited_keypoints):
            poses[i, 0] = kp.pt[0]
            poses[i, 1] = kp.pt[1]
            poses[i, 2] = kp.angle * np.pi / 180.0
        
        entry_id = self.gtdb.addBowVec(bow_vec, assignments, weights, poses)
        
        # Query with similar but slightly different poses
        query_poses = poses.copy()
        query_poses[:, :2] += np.random.normal(0, 2, (num_features, 2))  # Add small noise to positions
        query_poses[:, 2] += np.random.normal(0, 0.1, num_features)  # Add small noise to orientations
        
        results, db_poses = self.gtdb.queryBowVec(bow_vec, assignments, weights, query_poses, 5, -1, "L2_NORM")
        
        # Check that we get point matches
        self.assertGreater(len(results), 0)
        result_id, score, match_point_ids, match_point_weights = results[0]
        
        # Should have point matches
        self.assertGreater(match_point_ids.shape[0], 0, "No point matches found")
        
        # Check that match_point_ids are valid indices
        query_ids = match_point_ids[:, 0]
        db_ids = match_point_ids[:, 1]
        
        self.assertTrue(np.all(query_ids >= 0), "Invalid query point IDs")
        self.assertTrue(np.all(query_ids < num_features), "Query point IDs out of range")
        self.assertTrue(np.all(db_ids >= 0), "Invalid database point IDs")
        
        # Check that weights are positive and normalized-ish
        query_weights = match_point_weights[:, 0]
        db_weights = match_point_weights[:, 1]
        
        self.assertTrue(np.all(query_weights >= 0), "Negative query weights found")
        self.assertTrue(np.all(db_weights >= 0), "Negative database weights found")

    def test_orientation_binning(self):
        """Test orientation binning functionality"""
        # Add a test entry
        img = cv2.imread(glob.glob(self.image_pattern)[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        num_features = min(descriptors.shape[0], 15)
        limited_descriptors = descriptors[:num_features]
        limited_keypoints = keypoints[:num_features]
        scales = np.array([kp.size for kp in limited_keypoints], dtype=np.float32)
        
        # Get assignments, weights, and bow_vec from vocabulary
        assignments, weights, bow_vec = self.voc.transform(limited_descriptors, scales)
        
        poses = np.zeros((num_features, 3), dtype=np.float64)
        
        # Set up poses with specific orientations
        for i in range(num_features):
            poses[i, 0] = i * 10  # x
            poses[i, 1] = i * 10  # y
            poses[i, 2] = (i * np.pi / 4) % (2 * np.pi)  # Regular orientation spacing
        
        entry_id = self.gtdb.addBowVec(bow_vec, assignments, weights, poses)
        
        # Test with different numbers of orientation bins
        for num_bins in [1, 4, 8]:
            results, db_poses = self.gtdb.queryBowVec(
                bow_vec, assignments, weights, poses, 5, -1, "L2_NORM", num_bins
            )
            
            self.assertGreater(len(results), 0, f"No results with {num_bins} orientation bins")
            
            # Should still get point matches
            result_id, score, match_point_ids, match_point_weights = results[0]
            self.assertGreater(match_point_ids.shape[0], 0, f"No point matches with {num_bins} bins")

    def test_clear_database(self):
        """Test clearing the database"""
        # Add an entry
        img = cv2.imread(glob.glob(self.image_pattern)[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        num_features = min(descriptors.shape[0], 10)
        limited_descriptors = descriptors[:num_features]
        limited_keypoints = keypoints[:num_features]
        scales = np.array([kp.size for kp in limited_keypoints], dtype=np.float32)
        
        # Get assignments, weights, and bow_vec from vocabulary
        assignments, weights, bow_vec = self.voc.transform(limited_descriptors, scales)
        
        poses = np.zeros((num_features, 3), dtype=np.float64)
        
        for i in range(num_features):
            poses[i, 0] = i * 10
            poses[i, 1] = i * 10
            poses[i, 2] = 0
        
        entry_id = self.gtdb.addBowVec(bow_vec, assignments, weights, poses)
        
        # Query should return results
        results, _ = self.gtdb.queryBowVec(bow_vec, assignments, weights, poses, 5, -1, "L2_NORM")
        self.assertGreater(len(results), 0, "Should have results before clearing")
        
        # Clear database
        self.gtdb.clear()
        
        # Query should return no results
        results, _ = self.gtdb.queryBowVec(bow_vec, assignments, weights, poses, 5, -1, "L2_NORM")
        self.assertEqual(len(results), 0, "Should have no results after clearing")

    def test_retrieve_features(self):
        """Test retrieving features from the GTDatabase"""
        # Add an entry
        img = cv2.imread(glob.glob(self.image_pattern)[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        num_features = min(descriptors.shape[0], 10)
        limited_descriptors = descriptors[:num_features]
        limited_keypoints = keypoints[:num_features]
        scales = np.array([kp.size for kp in limited_keypoints], dtype=np.float32)

        # Get assignments, weights, and bow_vec from vocabulary
        assignments, weights, bow_vec = self.voc.transform(limited_descriptors, scales)

        poses = np.zeros((num_features, 3), dtype=np.float64)

        for i in range(num_features):
            poses[i, 0] = i * 10
            poses[i, 1] = i * 10
            poses[i, 2] = 0

        entry_id = self.gtdb.addBowVec(bow_vec, assignments, weights, poses)
        self.assertIsInstance(entry_id, int)
        self.assertGreaterEqual(entry_id, 0)

        # Retrieve features for the valid entry
        features_dict = self.gtdb.retrieveFeatures(entry_id)

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
            self.gtdb.retrieveFeatures(invalid_id)

    def test_empty_query_handling(self):
        """Test handling of empty or invalid queries"""
        # Test with empty BowVector
        empty_bow_vec = {}
        empty_assignments = np.array([], dtype=np.int32).reshape(0, 1)
        empty_weights = np.array([], dtype=np.float64).reshape(0, 1)
        empty_poses = np.array([], dtype=np.float64).reshape(0, 3)
        
        # This should not crash and should return empty results
        results, db_poses = self.gtdb.queryBowVec(
            empty_bow_vec, empty_assignments, empty_weights, empty_poses, 5, -1, "L2_NORM"
        )
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0, "Empty query should return no results")

    def test_setNumWords(self):
        """Test setNumWords functionality"""
        # Test setting number of words
        self.gtdb.setNumWords(1000)
        
        # This should not raise an exception
        # The actual effect is internal to the database structure

    def test_max_results_and_max_id_limits(self):
        """Test max_results and max_id parameters"""
        # Add multiple entries
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
                num_features = min(descriptors.shape[0], 10)
                limited_descriptors = descriptors[:num_features]
                limited_keypoints = keypoints[:num_features]
                scales = np.array([kp.size for kp in limited_keypoints], dtype=np.float32)
                
                # Get assignments, weights, and bow_vec from vocabulary
                assignments, weights, bow_vec = self.voc.transform(limited_descriptors, scales)
                bow_vectors.append(bow_vec)
                
                poses = np.random.rand(num_features, 3) * 100
                
                entry_id = self.gtdb.addBowVec(bow_vec, assignments, weights, poses)
                entry_ids.append(entry_id)
        
        if len(entry_ids) >= 2:
            # Test max_results parameter
            query_assignments = np.zeros((5, 1), dtype=np.int32)
            query_weights = np.ones((5, 1), dtype=np.float64)
            query_poses = np.random.rand(5, 3) * 100
            
            # Query with max_results=1
            results, _ = self.gtdb.queryBowVec(
                bow_vectors[0], query_assignments, query_weights, query_poses, 1, -1, "L2_NORM"
            )
            self.assertLessEqual(len(results), 1, "max_results=1 should return at most 1 result")
            
            # Query with max_results=2
            results, _ = self.gtdb.queryBowVec(
                bow_vectors[0], query_assignments, query_weights, query_poses, 2, -1, "L2_NORM"
            )
            self.assertLessEqual(len(results), 2, "max_results=2 should return at most 2 results")
            
            # Test max_id parameter
            if len(entry_ids) >= 2:
                results, _ = self.gtdb.queryBowVec(
                    bow_vectors[0], query_assignments, query_weights, query_poses,
                    10, entry_ids[0], "L2_NORM"
                )
                # Should only return entries with ID <= max_id
                for result in results:
                    result_id = result[0]
                    self.assertLessEqual(result_id, entry_ids[0], "Result ID should be <= max_id")

    def test_score_normalization(self):
        """Test that scores are properly normalized between 0 and 1"""
        # Add a test entry
        img = cv2.imread(glob.glob(self.image_pattern)[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        num_features = min(descriptors.shape[0], 10)
        limited_descriptors = descriptors[:num_features]
        limited_keypoints = keypoints[:num_features]
        scales = np.array([kp.size for kp in limited_keypoints], dtype=np.float32)
        
        # Get assignments, weights, and bow_vec from vocabulary
        assignments, weights, bow_vec = self.voc.transform(limited_descriptors, scales)
        
        poses = np.zeros((num_features, 3), dtype=np.float64)
        
        for i in range(num_features):
            poses[i, 0] = i * 10
            poses[i, 1] = i * 10
            poses[i, 2] = 0
        
        entry_id = self.gtdb.addBowVec(bow_vec, assignments, weights, poses)
        
        # Query with exact match - should get high score
        results, _ = self.gtdb.queryBowVec(bow_vec, assignments, weights, poses, 5, -1, "L2_NORM")
        
        self.assertGreater(len(results), 0, "Should have results")
        
        for result in results:
            result_id, score, _, _ = result
            self.assertGreaterEqual(score, 0.0, "Score should be >= 0")
            self.assertLessEqual(score, 1.0, "Score should be <= 1")
            
        # The exact match should have a high score (close to 1.0)
        best_score = results[0][1]
        self.assertGreater(best_score, 0.8, "Exact match should have high score")

    def test_pose_coordinate_consistency(self):
        """Test that pose coordinates are handled consistently"""
        # Add entry with known poses
        test_poses = np.array([
            [100.0, 200.0, 0.0],      # Point 1: x=100, y=200, angle=0
            [150.0, 250.0, np.pi/2], # Point 2: x=150, y=250, angle=90deg
            [200.0, 300.0, np.pi],   # Point 3: x=200, y=300, angle=180deg
        ], dtype=np.float64)
        
        # Create minimal test data
        bow_vec = {0: 0.5, 1: 0.5}  # Simple bow vector
        assignments = np.array([[0], [1], [0]], dtype=np.int32)
        weights = np.ones((3, 1), dtype=np.float64)
        
        entry_id = self.gtdb.addBowVec(bow_vec, assignments, weights, test_poses)
        
        # Query and check that poses are returned correctly
        results, db_poses = self.gtdb.queryBowVec(bow_vec, assignments, weights, test_poses, 1, -1, "L2_NORM")
        
        self.assertGreater(len(results), 0, "Should have results")
        
        # Check that db_poses has correct shape and values
        self.assertEqual(db_poses.shape[1], 3, "db_poses should have 3 columns")
        self.assertGreater(db_poses.shape[0], 0, "db_poses should have at least one row")
        
        # The poses should be retrievable and match what we stored
        result_id, score, match_point_ids, match_point_weights = results[0]
        
        if match_point_ids.shape[0] > 0:
            # Check that point IDs reference valid poses
            db_point_ids = match_point_ids[:, 1]
            self.assertTrue(np.all(db_point_ids >= 0), "Database point IDs should be non-negative")
            self.assertTrue(np.all(db_point_ids < db_poses.shape[0]), "Database point IDs should be valid indices")


if __name__ == '__main__':
    unittest.main(verbosity=2)
