import unittest
import cv2
import numpy as np
import glob
import os
import shutil
from pyGTBoW import SizeBinnedVocabulary, VocabularyConfig

class TestSizeBinnedVocabulary(unittest.TestCase):
    def setUp(self):
        # Path to save/load vocabulary (in tests_output directory)
        tests_output_dir = os.path.join(os.path.dirname(__file__), "tests_output")
        os.makedirs(tests_output_dir, exist_ok=True)
        self.vocab_path = os.path.join(tests_output_dir, "test_sizebinned_vocabulary")
        self.akm_vocab_path = os.path.join(tests_output_dir, "test_sizebinned_akmvocabulary")
        
        # Create vocabulary configurations
        self.config = VocabularyConfig()
        
        self.akm_config = VocabularyConfig()
        
        print(f"Standard vocab path: {self.vocab_path}")
        print(f"AKM vocab path: {self.akm_vocab_path}")
        
        # Configure standard vocabulary
        self.config.setVocabularyParameters(k=10, L=5, weight_method="TF_IDF", score_method="L2_NORM")
        
        # Configure AKM vocabulary  
        self.akm_config.setAKMVocabularyParameters(vocab_size=500, max_iter=100, r=1, var=200.0)
        
        # Check if standard vocabulary directory exists
        if os.path.exists(self.vocab_path):
            try:
                self.voc = SizeBinnedVocabulary.load(self.vocab_path, self.config)
                print("Loaded existing standard vocabulary from", self.vocab_path)
            except Exception as e:
                print(f"Failed to load standard vocabulary: {e}")
                print("Creating new standard vocabulary...")
                self.voc = self._create_vocabulary(self.config, self.vocab_path)
        else:
            print("Standard vocabulary directory does not exist, creating new vocabulary...")
            self.voc = self._create_vocabulary(self.config, self.vocab_path)

        # Check if AKM vocabulary directory exists
        if os.path.exists(self.akm_vocab_path):
            try:
                self.akm_voc = SizeBinnedVocabulary.load(self.akm_vocab_path, self.akm_config)
                print("Loaded existing AKM vocabulary from", self.akm_vocab_path)
            except Exception as e:
                print(f"Failed to load AKM vocabulary: {e}")
                print("Creating new AKM vocabulary...")
                self.akm_voc = self._create_vocabulary(self.akm_config, self.akm_vocab_path)
        else:
            print("AKM vocabulary directory does not exist, creating new AKM vocabulary...")
            self.akm_voc = self._create_vocabulary(self.akm_config, self.akm_vocab_path)

    def _create_vocabulary(self, config, vocab_path):
        """Helper method to create and train a vocabulary"""
        # Initialize the vocabulary
        voc = SizeBinnedVocabulary(config)
        
        # Path to the dataset directory (moved to project root/data)
        # When running from tests directory, go up one level to project root
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
        os.makedirs(vocab_path, exist_ok=True)
        voc.save(vocab_path)
        print("Saved vocabulary to", vocab_path)

        return voc

    def tearDown(self):
        # Clean up the saved vocabularies after all tests
        for path in [self.vocab_path, self.akm_vocab_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    def test_vocabulary_creation_standard(self):
        """Test basic properties of standard vocabulary"""
        self.assertTrue(self.voc.getWordSize() > 0, "Standard vocabulary size should be greater than 0")
        self.assertTrue(self.voc.isCreated(), "Standard vocabulary should be created")

    def test_vocabulary_creation_akm(self):
        """Test basic properties of AKM vocabulary"""
        self.assertTrue(self.akm_voc.getWordSize() > 0, "AKM vocabulary size should be greater than 0")
        self.assertTrue(self.akm_voc.isCreated(), "AKM vocabulary should be created")

    def test_vocabulary_types_comparison(self):
        """Compare behavior between standard and AKM vocabularies"""
        # Create test features
        features = np.random.randint(0, 255, (50, 32), dtype=np.uint8)
        scales = np.random.uniform(1, 100, (50,))
        
        # Transform with both vocabularies
        std_assignments, std_weights, std_bow = self.voc.transform(features, scales)
        akm_assignments, akm_weights, akm_bow = self.akm_voc.transform(features, scales)
        
        # Both should produce valid outputs
        self.assertEqual(len(std_assignments), 50)
        self.assertEqual(len(akm_assignments), 50)
        self.assertEqual(len(std_weights), 50)
        self.assertEqual(len(akm_weights), 50)
        
        # Both bow vectors should be L2 normalized
        std_norm = sum(w*w for w in std_bow.values())
        akm_norm = sum(w*w for w in akm_bow.values())
        self.assertAlmostEqual(std_norm, 1.0, places=5)
        self.assertAlmostEqual(akm_norm, 1.0, places=5)

    def test_save_load_vocabulary_standard(self):
        """Test save/load for standard vocabulary"""
        self._test_save_load_vocabulary_type(self.voc, self.config, "standard")

    def test_save_load_vocabulary_akm(self):
        """Test save/load for AKM vocabulary"""
        self._test_save_load_vocabulary_type(self.akm_voc, self.akm_config, "akm")

    def _test_save_load_vocabulary_type(self, voc, config, vocab_type):
        """Helper method to test save/load for a specific vocabulary type"""
        # Create a temporary path for saving (in tests_output directory)
        tests_output_dir = os.path.join(os.path.dirname(__file__), "tests_output")
        temp_path = os.path.join(tests_output_dir, f"temp_vocab_{vocab_type}")
        os.makedirs(temp_path, exist_ok=True)
        
        try:
            # Save vocabulary
            voc.save(temp_path)
            self.assertTrue(os.path.exists(os.path.join(temp_path, "size_binned_vocabulary.yml")), 
                          f"{vocab_type} vocabulary YAML file should exist")
            
            # Different vocabulary types save differently
            if vocab_type == "standard":
                self.assertTrue(os.path.exists(os.path.join(temp_path, "vocabulary.voc")), 
                              "Standard vocabulary file should exist")
            else:  # AKM vocabulary saves multiple files
                # Check for typical AKM vocabulary files (these may vary)
                files_in_temp = os.listdir(temp_path)
                vocab_files = [f for f in files_in_temp if f != "size_binned_vocabulary.yml"]
                self.assertGreater(len(vocab_files), 0, "AKM vocabulary should save additional files")
            
            # Load vocabulary
            loaded_voc = SizeBinnedVocabulary.load(temp_path, config)
            
            # Test if loaded vocabulary matches original
            self.assertEqual(voc.getWordSize(), loaded_voc.getWordSize(), 
                           f"Loaded {vocab_type} vocabulary size should match original")
            
            # Test transform on both vocabularies with same input
            test_features = np.random.randint(0, 255, (10, 32), dtype=np.uint8)
            test_scales = np.random.uniform(1, 100, (10,))
            
            orig_assignments, orig_weights, orig_bow = voc.transform(test_features, test_scales)
            loaded_assignments, loaded_weights, loaded_bow = loaded_voc.transform(test_features, test_scales)
            
            # Compare results
            np.testing.assert_array_equal(orig_assignments, loaded_assignments, 
                                        f"{vocab_type} assignments should match between original and loaded vocabulary")
            np.testing.assert_array_almost_equal(orig_weights, loaded_weights, decimal=6,
                                               err_msg=f"{vocab_type} weights should match between original and loaded vocabulary")
            self.assertEqual(orig_bow, loaded_bow, 
                           f"{vocab_type} BowVector should match between original and loaded vocabulary")
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)

    def test_transform_features_standard(self):
        """Test transform features with standard vocabulary"""
        self._test_transform_features_type(self.voc, "standard")

    def test_transform_features_akm(self):
        """Test transform features with AKM vocabulary"""
        self._test_transform_features_type(self.akm_voc, "akm")

    def _test_transform_features_type(self, voc, vocab_type):
        """Helper method to test transform for a specific vocabulary type"""
        # Get one image for testing transform
        image_file = glob.glob(self.image_pattern)[0]
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        self.assertIsNotNone(descriptors, f"No descriptors found in test image for {vocab_type}.")
        self.assertEqual(descriptors.shape[1], 32, f"Test image descriptors do not have 32 columns (ORB) for {vocab_type}")
        
        # Get scales from keypoints
        scales = np.array([kp.size for kp in keypoints], dtype=np.float32)
        
        # Test transform
        print(f"Transforming {descriptors.shape[0]} descriptors with {vocab_type} vocabulary...")
        assignments, weights, bow_vec = voc.transform(descriptors, scales)
        
        # Check if assignments and weights are valid
        self.assertTrue(len(assignments) > 0, f"{vocab_type} assignments should not be empty")
        self.assertTrue(len(weights) > 0, f"{vocab_type} weights should not be empty")
        
        # Check BowVector
        self.assertTrue(isinstance(bow_vec, dict), f"{vocab_type} BowVector should be a dictionary")
        self.assertTrue(len(bow_vec) > 0, f"{vocab_type} BowVector should not be empty")
        
        # Check that all weights in BowVector are positive
        for word_id, weight in bow_vec.items():
            # import pdb; pdb.set_trace()  # Debugging line to inspect BowVector
            self.assertGreaterEqual(weight, 0, f"{vocab_type} BowVector weight for word {word_id} should be non-negative")
            self.assertLess(word_id, voc.getWordSize(), f"{vocab_type} Word ID {word_id} should be less than vocabulary size")
            
        # Check L2 normalization of BowVector
        sum_squared = sum(weight * weight for weight in bow_vec.values())
        self.assertAlmostEqual(sum_squared, 1.0, places=6, 
                             msg=f"{vocab_type} BowVector should be L2 normalized (sum of squared weights should be 1.0)")

    def test_empty_feature_handling(self):
        """Test handling of empty or invalid feature inputs"""
        # Test with empty features
        empty_features = np.array([], dtype=np.uint8).reshape(0, 32)
        empty_scales = np.array([], dtype=np.float32)
        
        assignments, weights, bow_vec = self.voc.transform(empty_features, empty_scales)
        self.assertIsNone(assignments)
        self.assertIsNone(weights)
        self.assertEqual(len(bow_vec), 0)
        
        # Test with single feature
        single_feature = np.random.randint(0, 255, (1, 32), dtype=np.uint8)
        single_scale = np.array([10.0], dtype=np.float32)
        
        assignments, weights, bow_vec = self.voc.transform(single_feature, single_scale)
        self.assertEqual(len(assignments), 1)
        self.assertEqual(len(weights), 1)
        self.assertIsInstance(bow_vec, dict)

    def test_different_scale_ranges(self):
        """Test vocabulary behavior with different scale ranges"""
        # Create test features
        features = np.random.randint(0, 255, (50, 32), dtype=np.uint8)
        
        # Test with different scale ranges
        scale_ranges = [
            (1, 10),    # Small scales
            (10, 100),  # Medium scales
            (100, 1000) # Large scales
        ]
        
        for min_scale, max_scale in scale_ranges:
            scales = np.random.uniform(min_scale, max_scale, (50,))
            assignments, weights, bow_vec = self.voc.transform(features, scales)
            
            self.assertEqual(len(assignments), 50)
            self.assertEqual(len(weights), 50)
            self.assertIsInstance(bow_vec, dict)
            
            # Check that assignments are valid
            for assignment in assignments:
                self.assertGreaterEqual(assignment, 0)
                self.assertLess(assignment, self.voc.getWordSize())

    def test_scale_binning_consistency(self):
        """Test that similar scales are assigned to similar bins"""
        # Create test features
        features = np.random.randint(0, 255, (100, 32), dtype=np.uint8)
        
        # Create two sets of scales that are very close to each other
        base_scales = np.random.uniform(10, 100, (100,))
        similar_scales = base_scales + np.random.uniform(-0.1, 0.1, (100,))
        
        # Get assignments for both sets
        base_assignments, _, _ = self.voc.transform(features, base_scales)
        similar_assignments, _, _ = self.voc.transform(features, similar_scales)
        
        # Calculate how many assignments are the same
        same_assignments = np.sum(base_assignments == similar_assignments)
        similarity_ratio = same_assignments / len(base_assignments)
        
        # Most assignments should be the same for similar scales
        self.assertGreater(similarity_ratio, 0.8, 
                          "Similar scales should be assigned to similar bins")

    def test_vocabulary_consistency(self):
        """Test that multiple transforms of the same features produce consistent results"""
        # Create test features and scales
        features = np.random.randint(0, 255, (50, 32), dtype=np.uint8)
        scales = np.random.uniform(1, 100, (50,))
        
        # Transform multiple times
        results1 = self.voc.transform(features, scales)
        results2 = self.voc.transform(features, scales)
        results3 = self.voc.transform(features, scales)
        
        # Results should be identical
        np.testing.assert_array_equal(results1[0], results2[0], "Assignments should be consistent")
        np.testing.assert_array_equal(results1[0], results3[0], "Assignments should be consistent")
        np.testing.assert_array_almost_equal(results1[1], results2[1], decimal=10, 
                                           err_msg="Weights should be consistent")
        self.assertEqual(results1[2], results2[2], "BowVectors should be consistent")

    def test_bow_vector_properties(self):
        """Test mathematical properties of BowVector output"""
        # Get multiple bow vectors for testing
        image_files = sorted(glob.glob(self.image_pattern))
        bow_vectors = []
        
        for img_path in image_files[:5]:  # Test with first 5 images
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            if descriptors is not None and descriptors.shape[1] == 32:
                scales = np.array([kp.size for kp in keypoints], dtype=np.float32)
                assignments, weights, bow_vec = self.voc.transform(descriptors, scales)
                bow_vectors.append(bow_vec)
        
        self.assertGreater(len(bow_vectors), 0, "Should have at least one bow vector")
        
        for i, bow_vec in enumerate(bow_vectors):
            # Test L2 normalization
            sum_squared = sum(weight * weight for weight in bow_vec.values())
            self.assertAlmostEqual(sum_squared, 1.0, places=5, 
                                 msg=f"BowVector {i} should be L2 normalized")
            
            # Test non-negative weights
            for word_id, weight in bow_vec.items():
                self.assertGreaterEqual(weight, 0, f"All weights should be non-negative in bow_vec {i}")
                self.assertIsInstance(word_id, int, f"Word IDs should be integers in bow_vec {i}")
                self.assertLess(word_id, self.voc.getWordSize(), f"Word ID should be valid in bow_vec {i}")

    def test_akm_vocabulary_creation(self):
        # Test basic properties of AKM vocabulary
        self.assertTrue(self.akm_voc.getWordSize() > 0, "AKM Vocabulary size should be greater than 0")
        self.assertTrue(self.akm_voc.isCreated(), "AKM Vocabulary should be created")

    def test_save_load_akm_vocabulary(self):
        # Create a temporary path for saving AKM vocabulary
        temp_path = os.path.join(os.path.dirname(__file__), "temp_akm_vocab")
        os.makedirs(temp_path, exist_ok=True)
        
        try:
            # Save AKM vocabulary
            self.akm_voc.save(temp_path)
            self.assertTrue(os.path.exists(os.path.join(temp_path, "size_binned_vocabulary.yml")), 
                          "AKM Vocabulary YAML file should exist")
            self.assertTrue(os.path.exists(os.path.join(temp_path, "akm_vocabulary.yml")), 
                          "AKM Vocabulary file should exist")
            
            # Load AKM vocabulary
            loaded_akm_voc = SizeBinnedVocabulary.load(temp_path, self.akm_config)
            
            # Test if loaded AKM vocabulary matches original
            self.assertEqual(self.akm_voc.getWordSize(), loaded_akm_voc.getWordSize(), 
                           "Loaded AKM vocabulary size should match original")
            
            # Test transform on both AKM vocabularies with same input
            test_features = np.random.randint(0, 255, (10, 32), dtype=np.uint8)
            test_scales = np.random.uniform(1, 100, (10,))
            
            orig_assignments, orig_weights, orig_bow = self.akm_voc.transform(test_features, test_scales)
            loaded_assignments, loaded_weights, loaded_bow = loaded_akm_voc.transform(test_features, test_scales)
            
            # Compare results
            np.testing.assert_array_equal(orig_assignments, loaded_assignments, 
                                        "AKM Assignments should match between original and loaded vocabulary")
            np.testing.assert_array_almost_equal(orig_weights, loaded_weights, decimal=6,
                                               err_msg="AKM Weights should match between original and loaded vocabulary")
            self.assertEqual(orig_bow, loaded_bow, 
                           "AKM BowVector should match between original and loaded vocabulary")
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)

       


    def test_invalid_load(self):
        """Test loading from an invalid path."""
        config = VocabularyConfig()
        voc = SizeBinnedVocabulary(config)
        with self.assertRaises(Exception):
            voc.load("/this/path/does/not/exist")

    def test_invalid_vocab_type_raises_error(self):
        """Test that using an invalid vocab_type in VocabularyConfig raises an error."""
        config = VocabularyConfig()
        config.vocab_type = "invalid_vocab_type"
        voc = SizeBinnedVocabulary(config)

        features = np.random.randint(0, 255, (20, 32), dtype=np.uint8)
        scales = np.ones((20,), dtype=np.float32) * 5.0
        image_paths = ["img_0.png"] * 20

        with self.assertRaisesRegex(RuntimeError, "Invalid vocabulary type in VocabularyConfig: invalid_vocab_type"):
            voc.create(features, scales, image_paths)

    def test_sizebinned_empty_scale_range(self):
        """Test size binned vocabulary where features have identical scales."""
        config = VocabularyConfig()
        config.setVocabularyParameters(k=2, L=2, weight_method='TF_IDF', score_method='L2_NORM')


        voc = SizeBinnedVocabulary(config)

        features = []
        scales = []
        image_paths = []

        for i in range(2):
            # 20 features per image, all same scale
            img_feats = np.random.randint(0, 255, (20, 32), dtype=np.uint8)
            img_scales = np.ones((20,), dtype=np.float32) * 5.0
            features.append(img_feats)
            scales.extend(img_scales)
            image_paths.extend([f"img_{i}.png"] * 20)

        # The creation should handle zero range in scale gracefully or bin everything to 1 scale
        voc.create(np.vstack(features), scales, image_paths)
        self.assertEqual(voc.getNumScales(), 1)

if __name__ == "__main__":
    unittest.main(verbosity=2)
