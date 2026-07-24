#!/usr/bin/env python3
"""
Simple example demonstrating SizeBinnedVocabulary and GTDatabase usage.

This example shows:
1. Creating a SizeBinnedVocabulary from image features
2. Saving and loading the vocabulary
3. Using GTDatabase for image retrieval
4. Querying with an image and analyzing results
"""

import cv2
import numpy as np
import glob
import os
import tempfile
import shutil
from pyGTBoW import SizeBinnedVocabulary, VocabularyConfig, GTDatabase

def main():
    print("=== SizeBinnedVocabulary and GTDatabase Example ===\n")
    
    # Set up paths
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "dataset")
    image_pattern = os.path.join(data_dir, "*.png")
    vocab_path = tempfile.mkdtemp(prefix="example_vocab_")
    
    try:
        # 1. Create vocabulary configuration
        print("1. Setting up vocabulary configuration...")
        branching_factor = 3
        depth = 3
        config = VocabularyConfig()
        config.setVocabularyParameters(
            k=branching_factor,  # Smaller number of clusters per level
            L=depth,             # Fewer levels in the vocabulary tree
            weight_method="TF_IDF",  # Term frequency weighting
            score_method="L2_NORM"   # Scoring method
        )
        print(f"   - Vocabulary tree: k={branching_factor}, L={depth}")
        print("   - Weight method: TF_IDF")
        print("   - Score method: L2_NORM")
        print(
            "   Note: Using a small vocabulary for demonstration purposes "
            "to ensure\n"
            "         word sharing between images and meaningful similarity scores\n"
        )
        
        # 2. Extract features from images
        print("2. Extracting ORB features from images...")
        orb = cv2.ORB_create(nfeatures=1000)
        image_files = sorted(glob.glob(image_pattern))
        
        all_descriptors = []
        all_scales = []
        img_names = []
        
        print(f"   Found {len(image_files)} images")
        
        for i, img_path in enumerate(image_files):
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            if descriptors is not None and len(keypoints) > 0:
                all_descriptors.append(descriptors)
                scales = [kp.size for kp in keypoints]
                all_scales.extend(scales)
                img_names.extend([os.path.basename(img_path)] * len(keypoints))
                
                if i < 5:  # Print details for first few images
                    print(f"   - {os.path.basename(img_path)}: {len(keypoints)} features")
        
        if len(all_descriptors) == 0:
            raise RuntimeError("No valid descriptors found in any image.")
        
        # Create feature matrix
        feature_mat = np.vstack(all_descriptors)
        scales = np.array(all_scales, dtype=np.float32)
        
        print(f"   Total features: {feature_mat.shape[0]}")
        print(f"   Scale range: [{scales.min():.2f}, {scales.max():.2f}]\n")
        
        # 3. Create and train vocabulary
        print("3. Creating SizeBinnedVocabulary...")
        voc = SizeBinnedVocabulary(config)
        voc.create(feature_mat, scales, img_names)
        
        # Print vocabulary information
        word_size = voc.getWordSize()
        print(f"   Vocabulary created with {word_size} words")
        print(f"   Vocabulary is created: {voc.isCreated()}\n")
        
        # 4. Save vocabulary
        print("4. Saving vocabulary...")
        voc.save(vocab_path)
        print(f"   Vocabulary saved to: {vocab_path}\n")
        
        # 5. Load vocabulary (demonstrate loading)
        print("5. Loading vocabulary...")
        loaded_voc = SizeBinnedVocabulary.load(vocab_path, config)
        print(f"   Loaded vocabulary with {loaded_voc.getWordSize()} words\n")
        
        # 6. Create and setup GTDatabase
        print("6. Setting up GTDatabase...")
        # SizeBinnedVocabulary word IDs can be used directly at level 0.
        # Higher direct-index levels require a hierarchical Vocabulary object.
        gtdb = GTDatabase(use_di=True, di_levels=0)
        gtdb.setNumWords(loaded_voc.getWordSize())
        print("   GTDatabase created with the direct index enabled (level 0)")
        print(f"   Number of words set to: {loaded_voc.getWordSize()}\n")
        
        # 7. Add images to database (use first 10 images)
        print("7. Adding images to database...")
        database_entries = []
        
        for i, img_path in enumerate(image_files[:10]):  # Use first 10 images
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            if descriptors is not None and len(keypoints) > 0:
                # Get features for vocabulary
                scales = np.array([kp.size for kp in keypoints], dtype=np.float32)
                
                # Transform using vocabulary
                assignments, weights, bow_vec = loaded_voc.transform(descriptors, scales)
                
                # Create pose data (x, y, orientation)
                poses = np.zeros((len(keypoints), 3), dtype=np.float64)
                for j, kp in enumerate(keypoints):
                    poses[j, 0] = kp.pt[0]  # x coordinate
                    poses[j, 1] = kp.pt[1]  # y coordinate
                    poses[j, 2] = kp.angle * np.pi / 180.0  # orientation in radians
                
                # Add to database
                entry_id = gtdb.addBowVec(bow_vec, assignments, weights, poses)
                database_entries.append({
                    'id': entry_id,
                    'filename': os.path.basename(img_path),
                    'num_features': len(keypoints)
                })
                
                print(f"   Added {os.path.basename(img_path)} (ID: {entry_id}, {len(keypoints)} features)")
        
        print(f"   Database now contains {len(database_entries)} images\n")
        
        # 8. Query with image 4 (img04.png)
        print("8. Querying database with img04.png...")
        query_img_path = os.path.join(data_dir, "img04.png")
        
        if os.path.exists(query_img_path):
            # Extract features from query image
            query_img = cv2.imread(query_img_path)
            query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
            query_keypoints, query_descriptors = orb.detectAndCompute(query_gray, None)
            
            if query_descriptors is not None and len(query_keypoints) > 0:
                # Transform query features
                query_scales = np.array([kp.size for kp in query_keypoints], dtype=np.float32)
                query_assignments, query_weights, query_bow_vec = loaded_voc.transform(query_descriptors, query_scales)
                
                # Create query poses
                query_poses = np.zeros((len(query_keypoints), 3), dtype=np.float64)
                for j, kp in enumerate(query_keypoints):
                    query_poses[j, 0] = kp.pt[0]
                    query_poses[j, 1] = kp.pt[1]
                    query_poses[j, 2] = kp.angle * np.pi / 180.0
                
                print(f"   Query image has {len(query_keypoints)} features")
                
                # Perform query
                results, db_poses = gtdb.queryBowVec(
                    query_bow_vec, query_assignments, query_weights, query_poses,
                    5,         # max_results - Top 5 results
                    -1,        # max_id - Include all entries
                    "L2_NORM",  # score_method
                    8           # num_orientation_bins
                )
                
                print(f"   Found {len(results)} matches\n")
                
                # 9. Analyze and display results
                print("9. Top query results:")
                print("   Rank | DB ID | Score  | Filename     | Point Matches | Avg Match Weight")
                print("   -----|-------|--------|--------------|---------------|------------------")
                
                for rank, (db_id, score, match_point_ids, match_point_weights) in enumerate(results):
                    # Find corresponding filename
                    filename = "Unknown"
                    for entry in database_entries:
                        if entry['id'] == db_id:
                            filename = entry['filename']
                            break
                    
                    num_matches = match_point_ids.shape[0]
                    avg_weight = match_point_weights.mean() if num_matches > 0 else 0.0
                    
                    print(f"   {rank+1:4d} | {db_id:5d} | {score:6.4f} | {filename:12s} | {num_matches:13d} | {avg_weight:16.4f}")
                
                # Additional analysis for top result
                if len(results) > 0:
                    top_result = results[0]
                    top_id, top_score, top_matches, top_weights = top_result
                    
                    print(f"\n   Top match analysis:")
                    print(f"   - Image: {[entry['filename'] for entry in database_entries if entry['id'] == top_id][0]}")
                    print(f"   - Similarity score: {top_score:.4f}")
                    print(f"   - Point correspondences: {top_matches.shape[0]}")
                    
                    if top_matches.shape[0] > 0:
                        query_match_ids = top_matches[:, 0]
                        db_match_ids = top_matches[:, 1]
                        
                        print(f"   - Query point IDs range: [{query_match_ids.min()}, {query_match_ids.max()}]")
                        print(f"   - DB point IDs range: [{db_match_ids.min()}, {db_match_ids.max()}]")
                        print(f"   - Weight range: [{top_weights.min():.4f}, {top_weights.max():.4f}]")
                else:
                    print("   No results found")
            else:
                print("   No features found in query image")
        else:
            print("   Query image img04.png not found")
        
        print(f"\n=== Example completed successfully! ===")
        
    except Exception as e:
        print(f"Error: {e}")
        raise
    
    finally:
        # Clean up temporary vocabulary directory
        if os.path.exists(vocab_path):
            shutil.rmtree(vocab_path)
            print(f"\nCleaned up temporary vocabulary directory: {vocab_path}")

if __name__ == "__main__":
    main()
