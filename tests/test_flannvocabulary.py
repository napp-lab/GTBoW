import unittest
import numpy as np
import cv2
import os
import shutil
from pyGTBoW import FLANNVocabulary

class TestFLANNVocabulary(unittest.TestCase):
    def setUp(self):
        # Path for saving vocabulary
        self.tests_output_dir = os.path.join(os.path.dirname(__file__), "tests_output")
        os.makedirs(self.tests_output_dir, exist_ok=True)
        self.vocab_path = os.path.join(self.tests_output_dir, "test_flann_index.flann")

    def tearDown(self):
        # Clean up
        if os.path.exists(self.vocab_path):
            os.remove(self.vocab_path)

    def test_invalid_distance_or_data_type(self):
        """
        Test that FLANNVocabulary throws a ValueError when instantiated with
        unsupported distance_type or data_type.
        """
        # Test defaults (which are 'L2' and 'float')
        with self.assertRaisesRegex(ValueError, "Only Hamming distance and uint8 data type are supported currently."):
            FLANNVocabulary()

        # Test valid distance but invalid data type
        with self.assertRaisesRegex(ValueError, "Only Hamming distance and uint8 data type are supported currently."):
            FLANNVocabulary(distance_type="Hamming", data_type="float")

    def test_create_and_search(self):
        """Test creating an index and performing a nearest neighbor search."""
        vocab = FLANNVocabulary(distance_type="Hamming", data_type="uint8")
        
        # Create some random binary features (ORB-like)
        training_features = np.random.randint(0, 256, (100, 32), dtype=np.uint8)
        
        # Build index
        vocab.create(training_features)
        
        # Query with one of the training features
        query_feature = training_features[10:11, :]
        indices, dists = vocab.nn_index(query_feature, num_neighbors=1)
        
        self.assertEqual(len(indices), 1)
        self.assertEqual(len(dists), 1)
        self.assertEqual(indices[0][0], 10)
        self.assertEqual(dists[0][0], 0)  # Exact match should have 0 distance

    def test_save_and_load(self):
        """Test saving an index and loading it back."""
        # Note: the wrapper's load constructor expects training features as well
        training_features = np.random.randint(0, 256, (100, 32), dtype=np.uint8)
        vocab = FLANNVocabulary(distance_type="Hamming", data_type="uint8")
        vocab.create(training_features)
        
        # Save index
        vocab.save(self.vocab_path)
        self.assertTrue(os.path.exists(self.vocab_path))
        
        # Load index
        loaded_vocab = FLANNVocabulary(self.vocab_path, training_features)
        
        # Query loaded index
        query_feature = training_features[20:21, :]
        indices, dists = loaded_vocab.nn_index(query_feature, num_neighbors=1)
        
        self.assertEqual(indices[0][0], 20)
        self.assertEqual(dists[0][0], 0)

    def test_bitwise_median(self):
        """Test the bitwise_median utility method."""
        vocab = FLANNVocabulary(distance_type="Hamming", data_type="uint8")
        
        # Create features where median is easy to predict
        # 00001111 (15)
        # 11110000 (240)
        # 00001111 (15)
        # Median should be 00001111 (15)
        features = np.array([
            [15, 15],
            [240, 240],
            [15, 15]
        ], dtype=np.uint8)
        
        median = vocab.bitwise_median(features)
        self.assertIsInstance(median, np.ndarray)
        self.assertEqual(median.shape, (1, 2))
        self.assertEqual(median[0, 0], 15)
        self.assertEqual(median[0, 1], 15)

    def test_bitwise_median_empty(self):
        """Test bitwise_median with empty input."""
        vocab = FLANNVocabulary(distance_type="Hamming", data_type="uint8")
        empty_features = np.array([], dtype=np.uint8).reshape(0, 32)
        median = vocab.bitwise_median(empty_features)
        self.assertTrue(median is None or median.size == 0)

if __name__ == '__main__':
    unittest.main()
