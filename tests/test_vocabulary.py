import unittest
import cv2
import numpy as np
import glob
import os
from pyGTBoW import Vocabulary

class TestVocabulary(unittest.TestCase):
    def setUp(self):
        # Path to save/load vocabulary (in tests_output directory)
        tests_output_dir = os.path.join(os.path.dirname(__file__), "tests_output")
        os.makedirs(tests_output_dir, exist_ok=True)
        self.vocab_path = os.path.join(tests_output_dir, "test_vocabulary_standard.voc")

        # Try to load existing vocabulary
        try:
            self.voc = Vocabulary(self.vocab_path)
            print("Loaded existing standard vocabulary from", self.vocab_path)
        except Exception:
            print("Creating new standard vocabulary...")
            # Initialize the vocabulary with default params (K=10, L=6)
            self.voc = Vocabulary(K=10, L=4)  # Reduced depth for faster test training

            # Path to the dataset directory (moved to project root/data)
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
                    all_descriptors.append(descriptors)  # Append matrix for this image

            if len(all_descriptors) == 0:
                raise RuntimeError("No valid descriptors found in any image.")

            print(f"Using descriptors from {len(all_descriptors)} images for standard vocabulary creation.")

            # Create vocabulary using the list of feature matrices
            self.voc.create(all_descriptors)

            # Save the vocabulary for future use
            self.voc.save(self.vocab_path)
            print("Saved standard vocabulary to", self.vocab_path)

    def tearDown(self):
        # Clean up the saved vocabulary after all tests
        if os.path.exists(self.vocab_path):
            os.remove(self.vocab_path)

    def test_vocabulary_creation(self):
        # Test basic properties
        self.assertTrue(self.voc.getWordSize() > 0, "Vocabulary size should be greater than 0")
        self.assertTrue(self.voc.getDepth() > 0, "Vocabulary depth should be greater than 0")
        self.assertEqual(self.voc.getDescriptorSize(), 32, "Descriptor size should be 32 (ORB)")

    def test_save_load_vocabulary(self):
        # Create a temporary path for saving
        tests_output_dir = os.path.join(os.path.dirname(__file__), "tests_output")
        temp_path = os.path.join(tests_output_dir, "temp_standard_vocab.voc")

        try:
            # Save vocabulary
            self.voc.save(temp_path)
            self.assertTrue(os.path.exists(temp_path), "Vocabulary file should exist")

            # Load vocabulary
            loaded_voc = Vocabulary(temp_path)

            # Test if loaded vocabulary matches original
            self.assertEqual(self.voc.getWordSize(), loaded_voc.getWordSize(),
                           "Loaded vocabulary size should match original")

            # Test transform on both vocabularies with same input
            test_features = np.random.randint(0, 255, (10, 32), dtype=np.uint8)
            orig_bow, orig_feat_vec = self.voc.transform(test_features, 0)
            loaded_bow, loaded_feat_vec = loaded_voc.transform(test_features, 0)

            # Compare results
            self.assertEqual(orig_bow, loaded_bow, "BowVector should match between original and loaded vocabulary")
            self.assertEqual(orig_feat_vec, loaded_feat_vec, "FeatureVector should match between original and loaded vocabulary")

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_transform_features(self):
        # Get one image for testing transform
        image_file = glob.glob(self.image_pattern)[0]
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        self.assertIsNotNone(descriptors, "No descriptors found in test image.")

        if descriptors.dtype != np.uint8:
            descriptors = descriptors.astype(np.uint8)

        # Test transform
        bow_vec, feat_vec = self.voc.transform(descriptors, 0)

        # Check BowVector
        self.assertTrue(isinstance(bow_vec, dict), "BowVector should be a dictionary")
        self.assertTrue(len(bow_vec) > 0, "BowVector should not be empty")

        # Check FeatureVector
        self.assertTrue(isinstance(feat_vec, dict), "FeatureVector should be a dictionary")
        self.assertTrue(len(feat_vec) > 0, "FeatureVector should not be empty")

        # Check that all weights in BowVector are non-negative
        for word_id, weight in bow_vec.items():
            self.assertGreaterEqual(weight, 0, f"BowVector weight for word {word_id} should be non-negative")
            self.assertLess(word_id, self.voc.getWordSize(), f"Word ID {word_id} should be less than vocabulary size")

    def test_get_tf_vector(self):
        # Test getting TF vector with and without normalization
        test_features = np.random.randint(0, 255, (20, 32), dtype=np.uint8)

        tf_vec_norm = self.voc.getTFVector(test_features, normalize=True)
        self.assertTrue(isinstance(tf_vec_norm, dict))

        if len(tf_vec_norm) > 0:
            sum_norm = sum(tf_vec_norm.values())
            self.assertAlmostEqual(sum_norm, 1.0, places=5, msg="Normalized TF vector should sum to 1.0 (L1 norm)")

        tf_vec_unnorm = self.voc.getTFVector(test_features, normalize=False)
        self.assertTrue(isinstance(tf_vec_unnorm, dict))
        if len(tf_vec_unnorm) > 0:
            for val in tf_vec_unnorm.values():
                self.assertGreaterEqual(val, 0)

    def test_word_properties(self):
        word_size = self.voc.getWordSize()
        if word_size > 0:
            word_id = 0
            # Test getting word weight
            weight = self.voc.getWordWeight(word_id)
            self.assertGreaterEqual(weight, 0.0)

            # Test node to word mapping
            mapping = self.voc.nodeId2WordId()
            self.assertTrue(isinstance(mapping, dict))

            if len(mapping) > 0:
                node_id = list(mapping.keys())[0]
                mapped_word_id = self.voc.getWordId(node_id)
                self.assertEqual(mapped_word_id, mapping[node_id])

            # Test getting word representation
            word_repr = self.voc.getWord(word_id)
            self.assertEqual(len(word_repr), 32)
            self.assertTrue(all(isinstance(x, int) for x in word_repr))

    def test_empty_feature_handling(self):
        empty_features = np.array([], dtype=np.uint8).reshape(0, 32)
        try:
            bow_vec, feat_vec = self.voc.transform(empty_features, 0)
            self.assertEqual(len(bow_vec), 0)
            self.assertEqual(len(feat_vec), 0)
        except Exception:
            # Exception is also acceptable for empty features
            pass

    def test_vocabulary_consistency(self):
        # Multiple transforms of same feature should yield same result
        test_features = np.random.randint(0, 255, (10, 32), dtype=np.uint8)

        bow1, feat1 = self.voc.transform(test_features, 0)
        bow2, feat2 = self.voc.transform(test_features, 0)

        self.assertEqual(bow1, bow2)
        self.assertEqual(feat1, feat2)

    def test_log_string(self):
        # Test __str__
        log_str = str(self.voc)
        self.assertTrue(isinstance(log_str, str))
        self.assertTrue(len(log_str) > 0)
        self.assertIn("nwords:", log_str)
        self.assertIn("depths:", log_str)
        self.assertIn("descriptor size", log_str)
        self.assertIn("K:", log_str)

    def test_invalid_load(self):
        # Test loading from invalid path
        with self.assertRaises(Exception):
            Vocabulary("/path/that/does/not/exist.voc")

    def test_vocabulary_clear(self):
        # Create a small new vocabulary to test clear()
        voc2 = Vocabulary(K=5, L=2)
        test_features = [np.random.randint(0, 255, (10, 32), dtype=np.uint8)]
        voc2.create(test_features)

        self.assertGreater(voc2.getWordSize(), 0)
        voc2.clear()
        self.assertEqual(voc2.getWordSize(), 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
