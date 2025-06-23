from glob import glob
import os
import os.path as osp
import numpy as np
import pdb

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
                 split_folder='0000000',  # The split folder (0000000, 0000001, etc.)
                 ):
        super(FoundationStereo, self).__init__(aug_params)
        
        # Foundation Stereo structure: root/split_folder/scene_name/dataset/data/left|right/rgb|disparity/
        
        # Get all scene directories
        scene_pattern = osp.join(root, split_folder, '*', 'dataset', 'data')
        scene_dirs = sorted(glob(scene_pattern))
        
        if not scene_dirs:
            raise ValueError(f"No scene directories found with pattern: {scene_pattern}")
        
        print(f"Found {len(scene_dirs)} scenes in {split_folder}")
        
        left_images = []
        right_images = []
        disparity_images = []
        
        # Collect all images from all scenes
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
                # Note: disparity might have different extension (.pfm, .png, .exr, etc.)
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
                    left_images.append(left_img)
                    right_images.append(right_img)
                    disparity_images.append(disp_img)
                else:
                    missing = []
                    if not osp.exists(left_img): missing.append("left")
                    if not osp.exists(right_img): missing.append("right") 
                    if not disp_img or not osp.exists(disp_img): missing.append("disparity")
                    print(f"Warning: Missing files for {scene_name}/{img_name}: {', '.join(missing)}")
        
        print(f"Found {len(left_images)} complete image triplets")
        
        if len(left_images) == 0:
            raise ValueError("No valid image triplets found!")
        
        # Validation subset selection (similar to original)
        if validate_subset:
            state = np.random.get_state()
            np.random.seed(1000)
            val_idxs = set(np.random.permutation(len(left_images))[:min(400, len(left_images))])
            np.random.set_state(state)
        else:
            val_idxs = set()
        
        # Add images to dataset
        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (test_set and idx in val_idxs) or not test_set:
                self.image_list += [[img1, img2]]
                self.disp_list += [disp]
        
        print(f"Loaded {len(self.image_list)} image pairs for {'validation' if test_set else 'training'}")

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

    @staticmethod
    def discover_dataset_structure(root):
        """
        Helper function to discover the actual structure of Foundation Stereo dataset
        """
        print(f"Discovering dataset structure in {root}")
        
        # Check common directory patterns
        common_patterns = [
            'train/left/*.png',
            'train/left/*.jpg', 
            'train/*_left.png',
            'train/*_left.jpg',
            'train/*/left/*.png',
            'train/*/left/*.jpg',
            'left/*.png',
            'left/*.jpg',
            '*_left.png',
            '*_left.jpg'
        ]
        
        for pattern in common_patterns:
            full_pattern = osp.join(root, pattern)
            matches = glob(full_pattern)
            if matches:
                print(f"Found {len(matches)} files with pattern: {pattern}")
                print(f"Example: {matches[0]}")
        
        # List directory structure
        if osp.exists(root):
            print(f"\nDirectory structure in {root}:")
            for item in os.listdir(root):
                item_path = osp.join(root, item)
                if osp.isdir(item_path):
                    print(f"  {item}/")
                    # List subdirectories
                    try:
                        subitems = os.listdir(item_path)[:5]  # Show first 5 items
                        for subitem in subitems:
                            subitem_path = osp.join(item_path, subitem)
                            if osp.isdir(subitem_path):
                                print(f"    {subitem}/")
                            else:
                                print(f"    {subitem}")
                        if len(os.listdir(item_path)) > 5:
                            print(f"    ... and {len(os.listdir(item_path)) - 5} more items")
                    except PermissionError:
                        print(f"    <permission denied>")
                else:
                    print(f"  {item}")

# Usage example:
if __name__ == "__main__":
    # First discover the dataset structure
    root_path = '/projects/NEUFR/data/FSD'
    FoundationStereo.discover_dataset_structure(root_path)
    
    # Then create dataset instance
    dataset = FoundationStereo(root=root_path, test_set=False)