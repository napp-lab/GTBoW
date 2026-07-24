import unittest
import numpy as np
import cv2
import os
from pyGTBoW import Vocabulary, Database, GTDatabase

class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        # Create a tiny vocabulary for testing
        self.voc = Vocabulary(K=5, L=2)
        train_features = [np.random.randint(0, 256, (20, 32), dtype=np.uint8)]
        self.voc.create(train_features)
        
        self.db = Database(self.voc, use_di=True)
        self.gtdb = GTDatabase(self.voc, use_di=True)

    def test_database_mismatched_dimensions(self):
        """Test that GTDatabase handles/rejects mismatched input dimensions."""
        bow_vec = {0: 0.5, 1: 0.5}
        assignments = np.array([[0, 1], [1, 0]], dtype=np.int32) # 2 points
        weights = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64) # 2 points
        poses = np.array([[0, 0, 0]], dtype=np.float64) # ONLY 1 POINT - MISMATCH!
        
        # This should ideally raise an error or at least not crash
        with self.assertRaises(Exception):
            self.gtdb.addBowVec(bow_vec, assignments, weights, poses)

    def test_empty_input_handling(self):
        """Test handling of empty matrices in various methods."""
        empty_mat = np.array([], dtype=np.uint8).reshape(0, 32)
        
        # Vocabulary.transform with empty input
        bow, feat = self.voc.transform(empty_mat, 0)
        self.assertEqual(len(bow), 0)
        self.assertEqual(len(feat), 0)
        
        # Database.query with empty input
        results = self.db.query(empty_mat)
        self.assertEqual(len(results), 0)

    def test_invalid_descriptor_size(self):
        """Test that passing descriptors with wrong dimensions is handled."""
        wrong_dim_features = np.random.randint(0, 256, (10, 64), dtype=np.uint8) # 64 instead of 32
        
        # This should raise an error because the vocabulary expects 32-byte descriptors
        with self.assertRaises(Exception):
            self.voc.transform(wrong_dim_features, 0)

    def test_null_vocabulary_operations(self):
        """Test operations on an empty/uninitialized vocabulary."""
        empty_voc = Vocabulary(K=10, L=6) # Created but not trained
        self.assertEqual(empty_voc.getWordSize(), 0)
        
        test_features = np.random.randint(0, 256, (5, 32), dtype=np.uint8)
        # Transforming with untrained vocab should handle gracefully (return empty or error)
        try:
            bow, feat = empty_voc.transform(test_features, 0)
            self.assertEqual(len(bow), 0)
        except:
            pass # Error is also acceptable

if __name__ == '__main__':
    unittest.main()
