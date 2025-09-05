from glob import glob
import os
import os.path as osp
import numpy as np
import pdb
import json
import hashlib
import time

# Handle imports for both standalone and module usage
try:
    from .default import FlowDataset
except ImportError:
    # If running standalone, try absolute import
    try:
        from default import FlowDataset
    except ImportError:
        # Create a minimal FlowDataset class for testing
        print("Warning: FlowDataset not found, using minimal implementation for testing")
        class FlowDataset:
            def __init__(self, aug_params=None):
                self.image_list = []
                self.disp_list = []
                self.aug_params = aug_params
            
            def __len__(self):
                return len(self.image_list)

class FoundationStereo(FlowDataset):
    def __init__(self, aug_params=None,
                 root='/projects/NEUFR/data/FSD',
                 test_set=False,
                 validate_subset=False,
                 only_left=False,
                 split_folders='all',  # List of split folders or 'all' or single string
                 max_scenes_per_split=None,  # Limit scenes per split to manage memory
                 use_cache=True,
                 ):
        super(FoundationStereo, self).__init__(aug_params)
        
        # Generate cache filename
        cache_params = {
            'split_folders': split_folders,
            'max_scenes_per_split': max_scenes_per_split,
            'test_set': test_set,
            'validate_subset': validate_subset
        }
        
        cache_hash = hashlib.md5(str(cache_params).encode()).hexdigest()
        cache_file = f"{root}/foundation_stereo_cache_{cache_hash}.json"
        
        # Try loading from cache FIRST
        if use_cache and os.path.exists(cache_file):
            print(f"Loading dataset from cache: {cache_file}")
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if cache parameters match
                if cached_data.get('parameters') == cache_params:
                    # Load the cached lists directly
                    self.image_list = cached_data['image_list']
                    self.disp_list = cached_data['disp_list']
                    
                    print(f"Loaded {len(self.image_list)} samples from cache")
                    print(f"Cache created: {time.ctime(cached_data.get('created_time', 0))}")
                    
                    # Apply validation subset if needed
                    if validate_subset:
                        self._apply_validation_subset()
                    
                    return 
                else:
                    print("Cache parameters don't match, rebuilding...")
                    
            except Exception as e:
                print(f"Cache loading failed: {e}. Rebuilding dataset...")
        
        print("Building dataset from scratch (slow)...")
        
        # Handle split_folders parameter
        if split_folders is None:
            split_folders = ['0000000']  # Default to first split
        elif split_folders == 'all':
            # Use all available splits
            split_folders = self.list_available_splits(root)
            if not split_folders:
                raise ValueError(f"No split folders found in {root}")
        elif isinstance(split_folders, str):
            # Single split folder provided as string
            split_folders = [split_folders]
        
        print(f"Using split folders: {split_folders}")
        
        all_left_images = []
        all_right_images = []
        all_disparity_images = []
        
        # Iterate through each split folder
        for split_folder in split_folders:
            print(f"\nProcessing split: {split_folder}")
            
            # Get all scene directories for this split
            scene_pattern = osp.join(root, split_folder, '*', 'dataset', 'data')
            scene_dirs = sorted(glob(scene_pattern))
            
            if not scene_dirs:
                print(f"  Warning: No scene directories found in {split_folder}")
                continue
            
            # Limit scenes per split if specified
            if max_scenes_per_split:
                scene_dirs = scene_dirs[:max_scenes_per_split]
                print(f"  Limited to {len(scene_dirs)} scenes (max_scenes_per_split={max_scenes_per_split})")
            
            print(f"  Found {len(scene_dirs)} scenes")
            
            split_left_images = []
            split_right_images = []
            split_disparity_images = []
            
            # Collect all images from all scenes in this split
            for scene_dir in scene_dirs:
                scene_name = osp.basename(osp.dirname(osp.dirname(scene_dir)))
                
                # Get left RGB images
                left_rgb_dir = osp.join(scene_dir, 'left', 'rgb')
                left_rgb_pattern = osp.join(left_rgb_dir, '*')
                scene_left_images = sorted(glob(left_rgb_pattern))
                
                # Get corresponding right RGB images and disparity images
                for left_img in scene_left_images:
                    img_name = osp.basename(left_img)
                    
                    # Right RGB image (same filename in right/rgb/)
                    right_img = osp.join(scene_dir, 'right', 'rgb', img_name)
                    
                    # Disparity image (same filename in left/disparity/)
                    disp_base = osp.splitext(img_name)[0]
                    
                    # Try common disparity extensions
                    disp_extensions = ['.pfm', '.png', '.exr', '.tiff', '.tif']
                    disp_img = None
                    
                    for ext in disp_extensions:
                        disp_candidate = osp.join(scene_dir, 'left', 'disparity', disp_base + ext)
                        if osp.exists(disp_candidate):
                            disp_img = disp_candidate
                            break
                    
                    # Only add if all three files exist
                    if osp.exists(left_img) and osp.exists(right_img) and disp_img and osp.exists(disp_img):
                        split_left_images.append(left_img)
                        split_right_images.append(right_img)
                        split_disparity_images.append(disp_img)
            
            print(f"  Collected {len(split_left_images)} image triplets from {split_folder}")
            
            # Add to overall lists
            all_left_images.extend(split_left_images)
            all_right_images.extend(split_right_images)
            all_disparity_images.extend(split_disparity_images)
        
        print(f"\nTotal images collected: {len(all_left_images)} triplets from {len(split_folders)} splits")
        
        if len(all_left_images) == 0:
            raise ValueError("No valid image triplets found!")
        
        # Validation subset selection (similar to original)
        if validate_subset:
            state = np.random.get_state()
            np.random.seed(1000)
            val_idxs = set(np.random.permutation(len(all_left_images))[:min(400, len(all_left_images))])
            np.random.set_state(state)
        else:
            val_idxs = set()
        
        # Add images to dataset
        for idx, (img1, img2, disp) in enumerate(zip(all_left_images, all_right_images, all_disparity_images)):
            if (test_set and idx in val_idxs) or not test_set:
                self.image_list += [[img1, img2]]
                self.disp_list += [disp]
        
        print(f"Final dataset: {len(self.image_list)} image pairs for {'validation' if test_set else 'training'}")

        # Save cache
        if use_cache:
            print(f"Saving dataset to cache: {cache_file}")
            cache_data = {
                'image_list': self.image_list,
                'disp_list': self.disp_list,
                'created_time': time.time(),
                'parameters': cache_params,
                'total_samples': len(self.image_list)
            }
            
            try:
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                print("Cache saved successfully!")
            except Exception as e:
                print(f"Failed to save cache: {e}")

    @staticmethod
    def list_available_splits(root):
        """List all available split folders (0000000, 0000001, etc.)"""
        splits = []
        if osp.exists(root):
            for item in os.listdir(root):
                item_path = osp.join(root, item)
                if osp.isdir(item_path) and item.isdigit():
                    splits.append(item)
        return sorted(splits)

    @staticmethod
    def analyze_split(root, split_folder):
        """Analyze a specific split to understand its contents"""
        scene_pattern = osp.join(root, split_folder, '*', 'dataset', 'data')
        scene_dirs = sorted(glob(scene_pattern))
        
        print(f"=== Analysis of split {split_folder} ===")
        print(f"Found {len(scene_dirs)} scenes")
        
        total_images = 0
        scenes_with_issues = []
        
        for scene_dir in scene_dirs[:5]:  # Analyze first 5 scenes
            scene_name = osp.basename(osp.dirname(osp.dirname(scene_dir)))
            
            left_rgb_dir = osp.join(scene_dir, 'left', 'rgb')
            right_rgb_dir = osp.join(scene_dir, 'right', 'rgb')
            left_disp_dir = osp.join(scene_dir, 'left', 'disparity')
            
            if not osp.exists(left_rgb_dir):
                scenes_with_issues.append(f"{scene_name}: missing left/rgb")
                continue
            if not osp.exists(right_rgb_dir):
                scenes_with_issues.append(f"{scene_name}: missing right/rgb")
                continue
            if not osp.exists(left_disp_dir):
                scenes_with_issues.append(f"{scene_name}: missing left/disparity")
                continue
                
            left_images = len(glob(osp.join(left_rgb_dir, '*')))
            right_images = len(glob(osp.join(right_rgb_dir, '*')))
            disp_images = len(glob(osp.join(left_disp_dir, '*')))
            
            print(f"  {scene_name}: {left_images} left, {right_images} right, {disp_images} disp")
            total_images += left_images
            
            if left_images != right_images or left_images != disp_images:
                scenes_with_issues.append(f"{scene_name}: count mismatch")
        
        if len(scene_dirs) > 5:
            print(f"  ... and {len(scene_dirs) - 5} more scenes")
        
        print(f"Total estimated images: ~{total_images * len(scene_dirs) // min(5, len(scene_dirs))}")
        
        if scenes_with_issues:
            print(f"Scenes with issues: {len(scenes_with_issues)}")
            for issue in scenes_with_issues:
                print(f"  - {issue}")
        
        return len(scene_dirs), total_images

    @staticmethod
    def get_sample_paths(root, split_folder='0000000', scene_idx=0):
        """Get sample file paths for inspection"""
        scene_pattern = osp.join(root, split_folder, '*', 'dataset', 'data')
        scene_dirs = sorted(glob(scene_pattern))
        
        if scene_idx >= len(scene_dirs):
            print(f"Scene index {scene_idx} out of range. Found {len(scene_dirs)} scenes.")
            return None
            
        scene_dir = scene_dirs[scene_idx]
        scene_name = osp.basename(osp.dirname(osp.dirname(scene_dir)))
        
        left_rgb_dir = osp.join(scene_dir, 'left', 'rgb')
        sample_left = sorted(glob(osp.join(left_rgb_dir, '*')))
        
        if sample_left:
            img_name = osp.basename(sample_left[0])
            left_path = sample_left[0]
            right_path = osp.join(scene_dir, 'right', 'rgb', img_name)
            
            # Find disparity file
            disp_base = osp.splitext(img_name)[0]
            disp_extensions = ['.pfm', '.png', '.exr', '.tiff', '.tif']
            disp_path = None
            
            for ext in disp_extensions:
                disp_candidate = osp.join(scene_dir, 'left', 'disparity', disp_base + ext)
                if osp.exists(disp_candidate):
                    disp_path = disp_candidate
                    break
            
            print(f"Sample from scene '{scene_name}':")
            print(f"  Left:  {left_path}")
            print(f"  Right: {right_path}")
            print(f"  Disp:  {disp_path}")
            
            return left_path, right_path, disp_path
        
        return None


# # Usage examples and testing
# if __name__ == "__main__":
#     # Example usage - UPDATE THIS PATH TO YOUR ACTUAL DATASET PATH
#     root_path = '/projects/NEUFR/data/FSD'
    
#     # Check if the root path exists
#     if not os.path.exists(root_path):
#         print(f"Dataset root path does not exist: {root_path}")
#         print("Please update the root_path variable to point to your Foundation Stereo dataset")
        
#         # Try to find potential dataset paths
#         potential_paths = [
#             '/projects/NEUFR/dennis/FoundationStereo',
#             '/projects/NEUFR/dennis/Datasets/FoundationStereo',
#             '/projects/NEUFR/datasets/FoundationStereo',
#             '/data/FoundationStereo',
#         ]
        
#         print("\nChecking potential paths:")
#         for path in potential_paths:
#             if os.path.exists(path):
#                 print(f"  ✓ Found: {path}")
#                 root_path = path
#                 break
#             else:
#                 print(f"  ✗ Not found: {path}")
        
#         if not os.path.exists(root_path):
#             print("\nPlease provide the correct path to your Foundation Stereo dataset")
#             exit(1)
    
#     print(f"Using dataset path: {root_path}")
    
#     # List available splits
#     splits = FoundationStereo.list_available_splits(root_path)
#     print(f"Available splits: {splits}")
    
#     if not splits:
#         print("No splits found! Let's explore the directory structure:")
#         print(f"Contents of {root_path}:")
#         try:
#             for item in os.listdir(root_path):
#                 item_path = os.path.join(root_path, item)
#                 if os.path.isdir(item_path):
#                     print(f"  📁 {item}/")
#                 else:
#                     print(f"  📄 {item}")
#         except Exception as e:
#             print(f"Error reading directory: {e}")
#         exit(1)
    
#     # Analyze the first split
#     if splits:
#         print(f"\n=== Analyzing split {splits[0]} ===")
#         FoundationStereo.analyze_split(root_path, splits[0])
        
#         # Get sample paths
#         print(f"\n=== Sample paths ===")
#         FoundationStereo.get_sample_paths(root_path, splits[0])
        
#         # Create dataset instance
#         try:
#             print(f"\n=== Creating dataset ===")
#             dataset = FoundationStereo(root=root_path, test_set=False)
#             print(f"✅ Dataset created successfully with {len(dataset)} samples")
#         except Exception as e:
#             print(f"❌ Error creating dataset: {e}")
#             import traceback
#             traceback.print_exc()

#     @staticmethod
#     def discover_dataset_structure(root):
#         """
#         Helper function to discover the actual structure of Foundation Stereo dataset
#         """
#         print(f"Discovering dataset structure in {root}")
        
#         # Check common directory patterns
#         common_patterns = [
#             'train/left/*.png',
#             'train/left/*.jpg', 
#             'train/*_left.png',
#             'train/*_left.jpg',
#             'train/*/left/*.png',
#             'train/*/left/*.jpg',
#             'left/*.png',
#             'left/*.jpg',
#             '*_left.png',
#             '*_left.jpg'
#         ]
        
#         for pattern in common_patterns:
#             full_pattern = osp.join(root, pattern)
#             matches = glob(full_pattern)
#             if matches:
#                 print(f"Found {len(matches)} files with pattern: {pattern}")
#                 print(f"Example: {matches[0]}")
        
#         # List directory structure
#         if osp.exists(root):
#             print(f"\nDirectory structure in {root}:")
#             for item in os.listdir(root):
#                 item_path = osp.join(root, item)
#                 if osp.isdir(item_path):
#                     print(f"  {item}/")
#                     # List subdirectories
#                     try:
#                         subitems = os.listdir(item_path)[:5]  # Show first 5 items
#                         for subitem in subitems:
#                             subitem_path = osp.join(item_path, subitem)
#                             if osp.isdir(subitem_path):
#                                 print(f"    {subitem}/")
#                             else:
#                                 print(f"    {subitem}")
#                         if len(os.listdir(item_path)) > 5:
#                             print(f"    ... and {len(os.listdir(item_path)) - 5} more items")
#                     except PermissionError:
#                         print(f"    <permission denied>")
#                 else:
#                     print(f"  {item}")


# Usage example:
if __name__ == "__main__":
    crop_size = (320, 896)  # Adjust based on your image sizes
    aug_params = {'crop_size': crop_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
    
    # Create Foundation Stereo dataset
    foundation_dataset = FoundationStereo(
        aug_params=aug_params,
        root='/projects/NEUFR/data/FSD',  # Update this path
        split_folders='all',  # Start with one split, can add more later
        test_set=False,
        validate_subset=False,
        max_scenes_per_split = None,
        use_cache=True,
    )

    print(f"Foundation Stereo dataset length: {len(foundation_dataset)}")