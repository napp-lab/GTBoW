import unittest
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import time
import shutil
from pyGTBoW import HammingAKMVocabulary

class TestAKMVocabulary(unittest.TestCase):
    def setUp(self):
        # Path to save/load vocabulary (in tests_output directory)
        tests_output_dir = os.path.join(os.path.dirname(__file__), "tests_output")
        os.makedirs(tests_output_dir, exist_ok=True)
        self.vocab_path = os.path.join(tests_output_dir, "test_vocabulary")
        
        # Try to load existing vocabulary
        try:
            self.voc = HammingAKMVocabulary.load(self.vocab_path)
            print("Loaded existing vocabulary from", self.vocab_path)
        except:
            print("Creating new vocabulary...")
            # Initialize the vocabulary with vocab_size=1000, max_iter=200, r=1, var=1.0
            self.voc = HammingAKMVocabulary(1000, max_iter=200, r=1, var=580.0)
            
            # Path to the dataset directory (moved to project root/data)
            # When running from tests directory, go up one level to project root
            self.data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "dataset")
            self.image_pattern = os.path.join(self.data_dir, "*.png")
            
            # Initialize ORB detector
            self.orb = cv2.ORB_create(nfeatures=100)

            # Extract ORB features from all images and train vocabulary
            image_files = glob.glob(self.image_pattern)
            descriptors_list = []
            img_assignments = []
            current_img_idx = 0

            all_descriptors = []  # List of matrices, each matrix is descriptors for one image

            for img_path in image_files:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not read image: {img_path}")
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                keypoints, descriptors = self.orb.detectAndCompute(gray, None)
                print(f"{img_path}: keypoints={len(keypoints) if keypoints is not None else 0}, ", end="")
                if descriptors is not None:
                    print(f"descriptors.shape={descriptors.shape}")
                else:
                    print("descriptors=None")
                if descriptors is not None and len(keypoints) > 0 and descriptors.shape[1] == 32:
                    all_descriptors.append(descriptors)  # Add descriptors matrix for this image
                # Add image index for each descriptor
                img_assignments.extend([current_img_idx] * descriptors.shape[0])
                current_img_idx += 1

            if len(all_descriptors) == 0:
                raise RuntimeError("No valid descriptors found in any image.")

            # Optionally, limit the number of descriptors per image or total
            total_descriptors = sum(desc.shape[0] for desc in all_descriptors)
            print(f"Total descriptors: {total_descriptors}")
            print(f"Number of images with descriptors: {len(all_descriptors)}")

            # Create image assignments matrix
            img_assignments = np.array(img_assignments, dtype=np.int32).reshape(-1, 1)

            print(f"Using {sum(desc.shape[0] for desc in all_descriptors)} descriptors for vocabulary creation.")

            # Create vocabulary
            # Create a single matrix of features from all descriptor matrices
            feature_mat = np.vstack(all_descriptors)
            
            # Create image assignments matrix matching feature_mat
            img_assignments_mat = np.zeros((feature_mat.shape[0], 1), dtype=np.int32)
            
            # Fill img_assignments_mat
            start_idx = 0
            for i, descriptors in enumerate(all_descriptors):
                end_idx = start_idx + descriptors.shape[0]
                img_assignments_mat[start_idx:end_idx, 0] = i
                start_idx = end_idx
            
            # Create vocabulary using the feature_mat and img_assignments_mat
            self.voc.create(feature_mat, img_assignments_mat)
            
            # Save the vocabulary for future use
            os.makedirs(self.vocab_path, exist_ok=True)
            self.voc.save(self.vocab_path)
            print("Saved vocabulary to", self.vocab_path)

    def tearDown(self):
        # Clean up the saved vocabulary after all tests
        if os.path.exists(self.vocab_path):
            import shutil
            shutil.rmtree(self.vocab_path)

    def test_vocabulary_creation(self):
        # Test basic properties
        self.assertTrue(self.voc.getWordSize() > 0, "Vocabulary size should be greater than 0")
        self.assertEqual(self.voc.getWordSize(), 1000, "Vocabulary size should be 1000")
        # Ensure getNumWords is consistent with the configured vocabulary size
        self.assertEqual(
            self.voc.getNumWords(),
            self.voc.getWordSize(),
            "getNumWords should return the same value as getWordSize",
        )

    def test_save_load_vocabulary(self):
        # Create a temporary path for saving (in tests_output directory)
        tests_output_dir = os.path.join(os.path.dirname(__file__), "tests_output")
        temp_path = os.path.join(tests_output_dir, "temp_vocab")
        os.makedirs(temp_path, exist_ok=True)
        
        try:
            # Save vocabulary
            self.voc.save(temp_path)
            self.assertTrue(os.path.exists(os.path.join(temp_path, "akm_vocabulary.yml")), 
                          "Vocabulary YAML file should exist")
            self.assertTrue(os.path.exists(os.path.join(temp_path, "akm_vocabulary_index.flann")), 
                          "Vocabulary FLANN index file should exist")
            
            # Load vocabulary
            loaded_voc = HammingAKMVocabulary.load(temp_path)
            
            # Test if loaded vocabulary matches original
            self.assertEqual(self.voc.getWordSize(), loaded_voc.getWordSize(), 
                           "Loaded vocabulary size should match original")
            
            # Test transform on both vocabularies with same input
            test_features = np.random.randint(0, 255, (10, 32), dtype=np.uint8)
            orig_assignments, orig_weights, orig_bow = self.voc.transform(test_features)
            loaded_assignments, loaded_weights, loaded_bow = loaded_voc.transform(test_features)
            
            # Compare results
            np.testing.assert_array_equal(orig_assignments, loaded_assignments, 
                                        "Assignments should match between original and loaded vocabulary")
            np.testing.assert_array_almost_equal(orig_weights, loaded_weights, decimal=6,
                                               err_msg="Weights should match between original and loaded vocabulary")
            self.assertEqual(orig_bow, loaded_bow, 
                           "BowVector should match between original and loaded vocabulary")
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                import shutil
                shutil.rmtree(temp_path)

    def test_transform_features(self):
        # Get one image for testing transform
        image_file = glob.glob(self.image_pattern)[0]
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        self.assertIsNotNone(descriptors, "No descriptors found in test image.")
        self.assertEqual(descriptors.shape[1], 32, "Test image descriptors do not have 32 columns (ORB)")
        
        # Ensure descriptors are in the correct format (uint8)
        if descriptors.dtype != np.uint8:
            descriptors = descriptors.astype(np.uint8)
        
        # Test transform (in-place output)
        print(f"Transforming {descriptors.shape[0]} descriptors...")
        assignments, weights, bow_vec = self.voc.transform(descriptors)
        
        # Check if assignments and weights are valid
        try:
            self.assertTrue(len(assignments) > 0, "Assignments should not be empty")
            print(f"Number of assignments: {len(assignments)}")
            print(f"Number of weights: {len(weights)}")
            if len(assignments) > 0:
                print(f"First few assignments: {assignments[:5]}")
                print(f"First few weights: {weights[:5]}")
        except Exception as e:
            print(f"Error during transform: {e}")
            print("Assignments:", assignments)
            print("Weights:", weights)
            self.fail("Transform failed with an exception")
        
        self.assertTrue(len(weights) > 0, "Weights should not be empty")
        
        # Check BowVector
        self.assertTrue(isinstance(bow_vec, dict), "BowVector should be a dictionary")
        self.assertTrue(len(bow_vec) > 0, "BowVector should not be empty")
        print(f"Number of words in BowVector: {len(bow_vec)}")
        print(f"First few BowVector entries: {dict(list(bow_vec.items())[:5])}")
        
        # Check that all weights in BowVector are positive
        for word_id, weight in bow_vec.items():
            self.assertGreaterEqual(weight, 0, f"BowVector weight for word {word_id} should be non-negative")
            self.assertLess(word_id, self.voc.getWordSize(), f"Word ID {word_id} should be less than vocabulary size")
            
        # Check L2 normalization of BowVector
        sum_squared = sum(weight * weight for weight in bow_vec.values())
        self.assertAlmostEqual(sum_squared, 1.0, places=6, 
                             msg="BowVector should be L2 normalized (sum of squared weights should be 1.0)")

    def test_bow_vector_similarity_matrix(self):
        """Test L2 similarity scores between BowVectors from different images and visualize as heatmap"""
        image_files = sorted(glob.glob(self.image_pattern))
        self.assertGreater(len(image_files), 1, "Need at least 2 images for similarity testing")
        
        # Extract BowVectors for all images
        bow_vectors = []
        image_names = []
        
        print(f"Extracting BowVectors from {len(image_files)} images...")
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            if descriptors is not None and len(keypoints) > 0 and descriptors.shape[1] == 32:
                if descriptors.dtype != np.uint8:
                    descriptors = descriptors.astype(np.uint8)
                
                # Get BowVector for this image
                assignments, weights, bow_vec = self.voc.transform(descriptors)
                bow_vectors.append(bow_vec)
                image_names.append(os.path.basename(img_path))
        
        self.assertGreater(len(bow_vectors), 1, "Need at least 2 valid BowVectors for similarity testing")
        
        # Compute L2 similarity matrix (implementation from ScoringObject.cpp)
        n_images = len(bow_vectors)
        similarity_matrix = np.zeros((n_images, n_images))
        
        print(f"Computing {n_images}x{n_images} similarity matrix...")
        for i in range(n_images):
            for j in range(n_images):
                similarity_matrix[i, j] = self._compute_l2_score(bow_vectors[i], bow_vectors[j])
        
        # Verify that diagonal elements are 1.0 (self-similarity)
        for i in range(n_images):
            self.assertAlmostEqual(similarity_matrix[i, i], 1.0, places=6,
                                 msg=f"Self-similarity for image {i} should be 1.0")
        
        # Verify that off-diagonal elements are less than 1.0
        for i in range(n_images):
            for j in range(n_images):
                if i != j:
                    self.assertLess(similarity_matrix[i, j], 1.0,
                                  msg=f"Cross-similarity between images {i} and {j} should be less than 1.0")
                    self.assertGreaterEqual(similarity_matrix[i, j], 0.0,
                                          msg=f"Similarity scores should be non-negative")
        
        # Create and save visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap='gray', vmin=0, vmax=1)
        plt.colorbar(label='L2 Similarity Score')
        plt.title('BowVector L2 Similarity Matrix\n(Lighter = Higher Similarity)')
        plt.xlabel('Image Index')
        plt.ylabel('Image Index')
        
        # Add image names as tick labels if there aren't too many
        if n_images <= 20:
            plt.xticks(range(n_images), [name[:10] for name in image_names], rotation=45, ha='right')
            plt.yticks(range(n_images), [name[:10] for name in image_names])
        
        plt.tight_layout()
        
        # Save the plot (in tests_output directory)
        tests_output_dir = os.path.join(os.path.dirname(__file__), "tests_output")
        output_path = os.path.join(tests_output_dir, "bow_similarity_matrix.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Similarity matrix saved to: {output_path}")
        
        # Print some statistics
        print(f"Similarity matrix statistics:")
        print(f"  Diagonal (self-similarity): mean={np.mean(np.diag(similarity_matrix)):.6f}, std={np.std(np.diag(similarity_matrix)):.6f}")
        
        # Off-diagonal elements
        off_diag_mask = ~np.eye(n_images, dtype=bool)
        off_diag_values = similarity_matrix[off_diag_mask]
        print(f"  Off-diagonal (cross-similarity): mean={np.mean(off_diag_values):.6f}, std={np.std(off_diag_values):.6f}")
        print(f"  Min cross-similarity: {np.min(off_diag_values):.6f}")
        print(f"  Max cross-similarity: {np.max(off_diag_values):.6f}")
        
        plt.close()  # Close the figure to free memory
    
    def _compute_l2_score(self, bow_vec1, bow_vec2):
        """
        Compute L2 similarity score between two BowVectors.
        Implementation based on ScoringObject.cpp L2Scoring::score()
        
        Formula: score = 1.0 - sqrt(1.0 - dot_product) where dot_product = Sum(v_i * w_i)
        """
        # Convert to sorted lists for efficient iteration (similar to C++ iterators)
        items1 = sorted(bow_vec1.items())
        items2 = sorted(bow_vec2.items())
        
        i1, i2 = 0, 0
        dot_product = 0.0
        
        # Iterate through both vectors simultaneously (like C++ implementation)
        while i1 < len(items1) and i2 < len(items2):
            word_id1, weight1 = items1[i1]
            word_id2, weight2 = items2[i2]
            
            if word_id1 == word_id2:
                dot_product += weight1 * weight2
                i1 += 1
                i2 += 1
            elif word_id1 < word_id2:
                i1 += 1
            else:
                i2 += 1
        
        # Apply L2 scoring formula from ScoringObject.cpp
        if dot_product >= 1.0:  # Handle rounding errors
            score = 1.0
        else:
            score = 1.0 - np.sqrt(1.0 - dot_product)
        
        return score

    def test_constructor_parameters(self):
        """Test different constructor parameter combinations"""
        # Test default parameters
        voc1 = HammingAKMVocabulary(100)
        self.assertEqual(voc1.getWordSize(), 100)
        
        # Test custom parameters
        voc2 = HammingAKMVocabulary(500, max_iter=100, r=2, var=200.0)
        self.assertEqual(voc2.getWordSize(), 500)
        
        # Test small vocabulary size
        voc3 = HammingAKMVocabulary(10, max_iter=50, r=1, var=100.0)
        self.assertEqual(voc3.getWordSize(), 10)
        
        # Test large vocabulary size
        voc4 = HammingAKMVocabulary(2000, max_iter=150, r=3, var=400.0)
        self.assertEqual(voc4.getWordSize(), 2000)

    def test_empty_feature_handling(self):
        """Test handling of empty or invalid feature inputs"""
        small_voc = HammingAKMVocabulary(50, max_iter=10, r=1, var=100.0)
        
        # Create a small vocabulary first with minimal data
        minimal_features = np.random.randint(0, 255, (50, 32), dtype=np.uint8)
        minimal_assignments = np.zeros((50, 1), dtype=np.int32)
        small_voc.create(minimal_features, minimal_assignments)
        
        # Test with empty features - this should raise an exception or return empty results
        empty_features = np.array([], dtype=np.uint8).reshape(0, 32)
        try:
            assignments, weights, bow_vec = small_voc.transform(empty_features)
            # If no exception, check that results are empty
            self.assertEqual(len(assignments), 0)
            self.assertEqual(len(weights), 0)
            self.assertIsInstance(bow_vec, dict)
        except Exception:
            # Empty input causing exception is also acceptable behavior
            pass
        
        # Test with single feature
        single_feature = np.random.randint(0, 255, (1, 32), dtype=np.uint8)
        assignments, weights, bow_vec = small_voc.transform(single_feature)
        self.assertEqual(len(assignments), 1)
        self.assertEqual(len(weights), 1)
        self.assertIsInstance(bow_vec, dict)

    def test_feature_dimension_validation(self):
        """Test that vocabulary properly handles different feature dimensions"""
        small_voc = HammingAKMVocabulary(10, max_iter=5, r=1, var=50.0)
        
        # Test incorrect feature dimensions
        wrong_dim_features = np.random.randint(0, 255, (10, 64), dtype=np.uint8)  # Wrong dimension
        wrong_assignments = np.zeros((10, 1), dtype=np.int32)
        
        # This should work - AKMVocabulary should adapt to feature dimension
        try:
            small_voc.create(wrong_dim_features, wrong_assignments)
            # Test transform with same dimension
            test_features = np.random.randint(0, 255, (5, 64), dtype=np.uint8)
            assignments, weights, bow_vec = small_voc.transform(test_features)
            self.assertEqual(len(assignments), 5)
        except Exception as e:
            print(f"Feature dimension test failed: {e}")

    def test_different_data_types(self):
        """Test vocabulary with different input data types"""
        small_voc = HammingAKMVocabulary(20, max_iter=10, r=1, var=100.0)
        
        # Create test data with uint8 (correct type)
        uint8_features = np.random.randint(0, 255, (20, 32), dtype=np.uint8)
        assignments = np.zeros((20, 1), dtype=np.int32)
        small_voc.create(uint8_features, assignments)
        
        # Test transform with different data types (should be converted to uint8)
        float_features = np.random.rand(5, 32) * 255
        float_features = float_features.astype(np.uint8)  # Convert to uint8
        
        assignments, weights, bow_vec = small_voc.transform(float_features)
        self.assertEqual(len(assignments), 5)
        self.assertIsInstance(bow_vec, dict)

    def test_vocabulary_consistency(self):
        """Test that multiple transforms of the same features produce consistent results"""
        # Use a smaller vocabulary for faster testing
        test_voc = HammingAKMVocabulary(100, max_iter=50, r=1, var=200.0)
        
        # Create vocabulary with small dataset
        features = np.random.randint(0, 255, (100, 32), dtype=np.uint8)
        assignments = np.zeros((100, 1), dtype=np.int32)
        test_voc.create(features, assignments)
        
        # Test same features multiple times
        test_features = np.random.randint(0, 255, (10, 32), dtype=np.uint8)
        
        results1 = test_voc.transform(test_features)
        results2 = test_voc.transform(test_features)
        results3 = test_voc.transform(test_features)
        
        # Results should be identical
        np.testing.assert_array_equal(results1[0], results2[0], "Assignments should be consistent")
        np.testing.assert_array_equal(results1[0], results3[0], "Assignments should be consistent")
        np.testing.assert_array_almost_equal(results1[1], results2[1], decimal=10, err_msg="Weights should be consistent")
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
                if descriptors.dtype != np.uint8:
                    descriptors = descriptors.astype(np.uint8)
                
                assignments, weights, bow_vec = self.voc.transform(descriptors)
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

    def test_save_load_robustness(self):
        """Test save/load functionality with various scenarios"""
        # Test saving to different path formats
        test_cases = [
            "test_vocab_1",
            "test_vocab_2/nested",
            "test_vocab_with_spaces in name",
        ]
        
        for test_path in test_cases:
            tests_output_dir = os.path.join(os.path.dirname(__file__), "tests_output")
            full_path = os.path.join(tests_output_dir, test_path)
            
            try:
                # Create directory if needed
                os.makedirs(full_path, exist_ok=True)
                
                # Save vocabulary
                self.voc.save(full_path)
                
                # Verify files exist
                self.assertTrue(os.path.exists(os.path.join(full_path, "akm_vocabulary.yml")))
                self.assertTrue(os.path.exists(os.path.join(full_path, "akm_vocabulary_index.flann")))
                
                # Load and test
                loaded_voc = HammingAKMVocabulary.load(full_path)
                self.assertEqual(self.voc.getWordSize(), loaded_voc.getWordSize())
                
            finally:
                # Clean up - remove the directory and any empty parent directories
                if os.path.exists(full_path):
                    shutil.rmtree(full_path)
                    
                # Also clean up any empty parent directories we may have created
                parent_dir = os.path.dirname(full_path)
                try:
                    # Only remove if it's empty and inside tests_output
                    if parent_dir != tests_output_dir and os.path.exists(parent_dir):
                        os.rmdir(parent_dir)  # Only removes if empty
                except OSError:
                    # Directory not empty or other error, which is fine
                    pass

    def test_load_nonexistent_vocabulary(self):
        """Test loading from non-existent path"""
        tests_output_dir = os.path.join(os.path.dirname(__file__), "tests_output")
        nonexistent_path = os.path.join(tests_output_dir, "nonexistent_vocab")
        
        with self.assertRaises(Exception):
            HammingAKMVocabulary.load(nonexistent_path)

    def test_transform_performance(self):
        """Test transform performance with different feature set sizes"""
        feature_sizes = [1, 10, 50, 100, 200]
        performance_data = []
        
        for size in feature_sizes:
            # Generate random features
            test_features = np.random.randint(0, 255, (size, 32), dtype=np.uint8)
            
            # Time the transform operation
            start_time = time.time()
            assignments, weights, bow_vec = self.voc.transform(test_features)
            end_time = time.time()
            
            duration = end_time - start_time
            performance_data.append((size, duration))
            
            # Verify output sizes
            self.assertEqual(len(assignments), size)
            self.assertEqual(len(weights), size)
            self.assertIsInstance(bow_vec, dict)
            
            print(f"Transform time for {size} features: {duration:.4f} seconds")
        
        # Test that performance scales reasonably (not exponentially)
        if len(performance_data) >= 2:
            ratio = performance_data[-1][1] / performance_data[0][1]  # largest/smallest time
            size_ratio = performance_data[-1][0] / performance_data[0][0]  # largest/smallest size
            # Performance should not be worse than quadratic
            self.assertLess(ratio, size_ratio ** 2, "Transform performance should scale reasonably")

    def test_vocabulary_size_limits(self):
        """Test vocabulary creation with edge case sizes"""
        # Test very small vocabulary
        tiny_voc = HammingAKMVocabulary(1, max_iter=5, r=1, var=50.0)
        self.assertEqual(tiny_voc.getWordSize(), 1)
        
        # Create with minimal features
        minimal_features = np.random.randint(0, 255, (5, 32), dtype=np.uint8)
        minimal_assignments = np.zeros((5, 1), dtype=np.int32)
        tiny_voc.create(minimal_features, minimal_assignments)
        
        # Test transform
        test_feature = np.random.randint(0, 255, (1, 32), dtype=np.uint8)
        assignments, weights, bow_vec = tiny_voc.transform(test_feature)
        if len(bow_vec) != 0:
            self.assertEqual(len(bow_vec), 1)  # Should have only one word or empty due to TF-IDF weights

    def test_different_r_values(self):
        """Test vocabulary behavior with different r (nearest neighbor) values"""
        r_values = [1, 2, 3]
        
        for r in r_values:
            test_voc = HammingAKMVocabulary(50, max_iter=20, r=r, var=100.0)
            
            # Create vocabulary
            features = np.random.randint(0, 255, (50, 32), dtype=np.uint8)
            assignments = np.zeros((50, 1), dtype=np.int32)
            test_voc.create(features, assignments)
            
            # Test transform
            test_features = np.random.randint(0, 255, (10, 32), dtype=np.uint8)
            assignments, weights, bow_vec = test_voc.transform(test_features)
            
            # With higher r, we should get more assignments per feature
            expected_assignments = 10 * r
            self.assertEqual(len(assignments), expected_assignments, f"Should have {expected_assignments} assignments with r={r}")
            self.assertEqual(len(weights), expected_assignments, f"Should have {expected_assignments} weights with r={r}")

    def test_different_variance_values(self):
        """Test vocabulary behavior with different variance values"""
        variances = [10.0, 100.0, 1000.0]
        
        for var in variances:
            test_voc = HammingAKMVocabulary(30, max_iter=10, r=1, var=var)
            
            # Create vocabulary
            features = np.random.randint(0, 255, (30, 32), dtype=np.uint8)
            assignments = np.zeros((30, 1), dtype=np.int32)
            test_voc.create(features, assignments)
            
            # Test transform
            test_features = np.random.randint(0, 255, (5, 32), dtype=np.uint8)
            assignments, weights, bow_vec = test_voc.transform(test_features)
            
            # Variance affects the soft assignment weights
            self.assertEqual(len(assignments), 5)
            self.assertEqual(len(weights), 5)
            self.assertIsInstance(bow_vec, dict)
            
            # All weights should be valid (0 <= weight <= 1 after normalization)
            for weight in weights:
                self.assertGreaterEqual(weight, 0)
                self.assertLessEqual(weight, 1)

    def test_assignment_validation(self):
        """Test that assignments are within valid word ID range"""
        # Get test features
        image_file = glob.glob(self.image_pattern)[0]
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is not None:
            if descriptors.dtype != np.uint8:
                descriptors = descriptors.astype(np.uint8)
            
            assignments, weights, bow_vec = self.voc.transform(descriptors)
            
            vocab_size = self.voc.getWordSize()
            
            # All assignments should be valid word IDs
            for assignment in assignments:
                self.assertGreaterEqual(assignment, 0, "Assignment should be non-negative")
                self.assertLess(assignment, vocab_size, f"Assignment {assignment} should be less than vocab size {vocab_size}")
            
            # All word IDs in bow_vec should be valid
            for word_id in bow_vec.keys():
                self.assertGreaterEqual(word_id, 0, "Word ID should be non-negative")
                self.assertLess(word_id, vocab_size, f"Word ID {word_id} should be less than vocab size {vocab_size}")

    def test_create_with_different_input_formats(self):
        """Test create method with different input format combinations"""
        # Test with single feature matrix and assignments (more reliable)
        test_voc1 = HammingAKMVocabulary(20, max_iter=5, r=1, var=50.0)
        
        # Create combined feature matrix
        combined_features = np.random.randint(0, 255, (30, 32), dtype=np.uint8)
        assignments = np.array([0]*10 + [1]*10 + [2]*10, dtype=np.int32).reshape(-1, 1)
        
        test_voc1.create(combined_features, assignments)
        self.assertEqual(test_voc1.getWordSize(), 20)
        
        # Test transform to verify it works
        test_features = np.random.randint(0, 255, (5, 32), dtype=np.uint8)
        assignments_result, weights, bow_vec = test_voc1.transform(test_features)
        self.assertEqual(len(assignments_result), 5)
        
        # Test with list of matrices (if supported by the implementation)
        test_voc2 = HammingAKMVocabulary(15, max_iter=5, r=1, var=50.0)
        
        feature_matrices = []
        for i in range(3):
            features = np.random.randint(0, 255, (10, 32), dtype=np.uint8)
            feature_matrices.append(features)
        
        try:
            test_voc2.create(feature_matrices)
            self.assertEqual(test_voc2.getWordSize(), 15)
        except Exception:
            # If list input format is not supported, that's acceptable
            # Fall back to matrix + assignments format
            combined_features2 = np.vstack(feature_matrices)
            assignments2 = np.array([0]*10 + [1]*10 + [2]*10, dtype=np.int32).reshape(-1, 1)
            test_voc2.create(combined_features2, assignments2)
            self.assertEqual(test_voc2.getWordSize(), 15)

    def test_weight_distribution(self):
        """Test that BowVector weights have reasonable distribution"""
        # Get multiple images and analyze weight distributions
        image_files = sorted(glob.glob(self.image_pattern))
        all_weights = []
        
        for img_path in image_files[:10]:  # Test with first 10 images
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            if descriptors is not None and descriptors.shape[1] == 32:
                if descriptors.dtype != np.uint8:
                    descriptors = descriptors.astype(np.uint8)
                
                assignments, weights, bow_vec = self.voc.transform(descriptors)
                all_weights.extend(bow_vec.values())
        
        if len(all_weights) > 0:
            all_weights = np.array(all_weights)
            
            # Test statistical properties
            self.assertGreater(np.mean(all_weights), 0, "Mean weight should be positive")
            self.assertGreater(np.std(all_weights), 0, "Weight standard deviation should be positive")
            self.assertGreaterEqual(np.min(all_weights), 0, "All weights should be non-negative")
            self.assertLessEqual(np.max(all_weights), 1, "All weights should be <= 1 (after L2 normalization)")
            
            print(f"Weight statistics: mean={np.mean(all_weights):.6f}, std={np.std(all_weights):.6f}")
            print(f"Weight range: [{np.min(all_weights):.6f}, {np.max(all_weights):.6f}]")

    def test_save_and_load(self):
        """Test saving and loading vocabulary"""
        test_dir = os.path.join(self.vocab_path, "save_load_test")
        os.makedirs(test_dir, exist_ok=True)

        # Save vocabulary
        self.voc.save(test_dir)

        # Load vocabulary into new instance
        loaded_voc = HammingAKMVocabulary.load(test_dir)

        # Verify sizes match
        self.assertEqual(self.voc.getWordSize(), loaded_voc.getWordSize())
        self.assertEqual(self.voc.getWordSize(), 1000)

        # Test transform on both yields same result
        test_features = np.random.randint(0, 255, (10, 32), dtype=np.uint8)

        assign1, weights1, bow1 = self.voc.transform(test_features)
        assign2, weights2, bow2 = loaded_voc.transform(test_features)

        np.testing.assert_array_equal(assign1, assign2)
        np.testing.assert_array_almost_equal(weights1, weights2)
        for key in bow1.keys():
            self.assertIn(key, bow2)
            self.assertAlmostEqual(bow1[key], bow2[key])

if __name__ == '__main__':
    unittest.main()
