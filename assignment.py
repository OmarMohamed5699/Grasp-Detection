
import torch
from torchvision.io.image import read_image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import json
from scipy import ndimage










device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def clean_depth_image_fixed(depth, out_size=300):
    """
    Fixed version: No longer does redundant resize
    Assumes input is already the correct size
    """
    # Add border
    depth_crop = cv2.copyMakeBorder(depth, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    
    # Find NaN values
    depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    depth_nan_mask = cv2.dilate(depth_nan_mask, kernel, iterations=1)
    
    # Set NaN to 0
    depth_crop[depth_nan_mask == 1] = 0
    
    # Scale for inpainting
    depth_scale = np.abs(depth_crop).max()
    if depth_scale > 0:
        depth_crop = depth_crop.astype(np.float32) / depth_scale
    else:
        depth_crop = depth_crop.astype(np.float32)
    
    # Inpaint over NaN regions
    depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)
    
    # Remove border and rescale
    depth_crop = depth_crop[1:-1, 1:-1]
    depth_crop = depth_crop * depth_scale
    
    
    return depth_crop











def normalize_depth_fixed(depth):
    """
    Fixed depth normalization using proper min-max scaling
    
    OLD (WRONG):
        depth_norm = np.clip((depth - depth.mean()), -1, 1)
        Problem: Loses depth information, arbitrary clipping
    
    NEW (CORRECT):
        Scale valid depth values to [-1, 1] range
    """
    # Get valid (non-zero) depth values
    valid_mask = depth > 0
    
    if not valid_mask.any():
        print(" No valid depth values!")
        return np.zeros_like(depth, dtype=np.float32)
    
    valid_depth = depth[valid_mask]
    d_min = valid_depth.min()
    d_max = valid_depth.max()
    
    print(f"  Depth range: [{d_min:.1f}, {d_max:.1f}]")
    
    # Normalize to [0, 1] then to [-1, 1]
    depth_norm = np.zeros_like(depth, dtype=np.float32)
    
    if d_max > d_min:
        depth_norm[valid_mask] = ((depth[valid_mask] - d_min) / (d_max - d_min)) * 2.0 - 1.0
    else:
        depth_norm[valid_mask] = 0.0
    
    # Keep invalid regions at -1 (or 0)
    depth_norm[~valid_mask] = -1.0
    
    return depth_norm










def postprocess_segmentations(quality_seg, angle_seg, width_seg):
    """
    Apply filtering to GG-CNN outputs
    """
    filters = (2, 2, 1)
    
    # Width is scaled by 150 pixels
    width_seg = width_seg * 150.0
    
    # Apply Gaussian smoothing
    if filters[0]:
        quality_seg = ndimage.filters.gaussian_filter(quality_seg, filters[0])
    if filters[1]:
        angle_seg = ndimage.filters.gaussian_filter(angle_seg, filters[1])
    if filters[2]:
        width_seg = ndimage.filters.gaussian_filter(width_seg, filters[2])
    
    # Clip quality to [0, 0.999]
    quality_seg = np.clip(quality_seg, 0, 1.0 - 1e-3)
    
    return quality_seg, angle_seg, width_seg









def extract_detections(prediction, coco_categories, confidence_threshold=0.5):
    """Extract detections above confidence threshold"""
    detections = []
    
    labels = [coco_categories[i] for i in prediction["labels"]]
    scores = prediction['scores']
    boxes = prediction['boxes']
    
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            detections.append({
                "class": labels[i],
                "confidence": float(scores[i]),
                "bbox": boxes[i].tolist()
            })
    
    return detections











def isolate_object_with_mask(rgb_tensor, depth_tensor, prediction, target_class, 
                             coco_categories, mask_thresh=0.5):
    """
    FIXED VERSION: Converts depth to float before masking
    """
    
    # Find target object
    labels = [coco_categories[i] for i in prediction["labels"]]
    
    target_idx = None
    for i, label in enumerate(labels):
        if label.lower() == target_class.lower():
            target_idx = i
            break
    
    if target_idx is None:
        print(f"Object '{target_class}' not found!")
        return None, None, None, None, None
    
    # Get bounding box and mask
    bbox = prediction["boxes"][target_idx].tolist()
    target_mask = prediction["masks"][target_idx, 0]  # [H, W]
    
    print(f"Found {target_class} at index {target_idx}")
    print(f"Bbox: {bbox}")
    print(f"Mask shape: {target_mask.shape}")
    
    # Create binary mask
    binary_mask = (target_mask >= mask_thresh).float()
    
    # Crop RGB
    x_min, y_min, x_max, y_max = [int(x) for x in bbox]
    
    # Validate coordinates
    h, w = rgb_tensor.shape[1:]
    x_min = max(0, min(x_min, w-1))
    y_min = max(0, min(y_min, h-1))
    x_max = max(x_min+1, min(x_max, w))
    y_max = max(y_min+1, min(y_max, h))
    
    cropped_rgb = rgb_tensor[:, y_min:y_max, x_min:x_max]
    cropped_depth = depth_tensor[:, y_min:y_max, x_min:x_max]
    cropped_mask = binary_mask[y_min:y_max, x_min:x_max]
    
    # Convert depth to float BEFORE trying to modify it
    if cropped_depth.dtype != torch.float32:
        cropped_depth = cropped_depth.float()  # ← THIS IS THE KEY FIX!
    
    # Now apply mask
    if len(cropped_depth.shape) == 3:
        depth_2d = cropped_depth.squeeze(0)
    else:
        depth_2d = cropped_depth
    
    masked_depth = depth_2d.clone()
    masked_depth[cropped_mask == 0] = 0  # ← Now this works!
    
    # Update bbox
    bbox = [x_min, y_min, x_max, y_max]
    
    return cropped_rgb, cropped_depth, masked_depth, bbox, cropped_mask















def process_single_test(test_id, test_data, mrcnn, preprocess, coco_categories, 
                       ggcnn_model, device):
    """
    FIXED VERSION: Process single test with all corrections
    """
    
    try:
        # Parse test data
        if isinstance(test_data, list):
            im_name = test_data[0]
            target_object = test_data[1]
        else:
            im_name = test_data['file_name']
            target_object = test_data['object_name']
        
        print(f"\n{'='*60}")
        print(f"Processing {test_id}: {target_object}")
        print(f"{'='*60}")
        
        # Load images
        rgb_path = f'/home/omar/Desktop/ENN583/enn583/Project 2/{im_name}'
        depth_path = rgb_path.replace('rgb', 'depth')
        im_rgb = read_image(rgb_path)
        im_D = read_image(depth_path)
        
        print(f"Original image shape: {im_rgb.shape}")
        
        # Run Mask R-CNN
        batch = [preprocess(im_rgb).to(device)]
        with torch.no_grad():
            prediction = mrcnn(batch)[0]
        
        # Extract detections
        detections = extract_detections(prediction, coco_categories, confidence_threshold=0.5)
        print(f"Detected {len(detections)} objects: {[d['class'] for d in detections]}")
        
        # Find target object
        target = None
        for det in detections:
            if det['class'].lower() == target_object.lower():
                target = det
                print(f"✓ Found '{target_object}' with confidence {det['confidence']:.3f}")
                break
        
        if target is None:
            print(f"✗ '{target_object}' not found in detections")
            return "object not present"
        
        bbox = target['bbox']
        print(f"Bounding box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        
        # Isolate with mask
        result = isolate_object_with_mask(
            im_rgb, im_D, prediction, target_object, coco_categories, mask_thresh=0.5
        )
        
        if result[0] is None:
            return "object not present"
        
        cropped_rgb, cropped_depth, masked_depth, bbox, mask = result
        
        # Get actual cropped dimensions
        _, crop_h, crop_w = cropped_rgb.shape  # [C, H, W]
        print(f"Cropped size: {crop_h} × {crop_w}")
        
        # Convert to numpy
        cropped_depth_np = masked_depth.cpu().numpy()
        if len(cropped_depth_np.shape) == 2:
            cropped_depth_np = cropped_depth_np[:, :, np.newaxis]
        else:
            cropped_depth_np = cropped_depth_np.transpose(1, 2, 0)
        
        # Resize to 300×300
        crop_depth = cv2.resize(cropped_depth_np, (300, 300), interpolation=cv2.INTER_LINEAR)
        
        # Calculate scale factors
        scale_w = crop_w / 300.0
        scale_h = crop_h / 300.0
        print(f"Scale factors: w={scale_w:.4f}, h={scale_h:.4f}")
        
        if abs(scale_w - scale_h) > 0.1:
            print(f" Aspect ratio distorted! ratio={scale_w/scale_h:.3f}")
        
        # Clean depth (no redundant resize)
        depth_clean = clean_depth_image_fixed(crop_depth, out_size=300)
        
        depth_norm = normalize_depth_fixed(depth_clean)
        
        # Validate normalization
        print(f"  Normalized range: [{depth_norm.min():.3f}, {depth_norm.max():.3f}]")
        
        depth_tensor = torch.from_numpy(
            depth_norm.reshape(1, 1, 300, 300).astype(np.float32)
        ).to(device)  # ← FIXED!
        
        # Run GG-CNN
        with torch.no_grad():
            quality, cos, sin, width_out = ggcnn_model(depth_tensor)
        
        # Convert to numpy
        quality = quality.cpu().numpy()
        cos = cos.cpu().numpy()
        sin = sin.cpu().numpy()
        width_out = width_out.cpu().numpy()
        ang_radians = np.arctan2(sin, cos) / 2.0
        
        # Postprocess
        quality_final, ang_final, width_final = postprocess_segmentations(
            quality, ang_radians, width_out
        )
        
        # Find best grasp
        quality_final = quality_final.squeeze()
        width_final = width_final.squeeze()
        ang_final = ang_final.squeeze()
        
        # Validate quality map
        print(f"  Quality range: [{quality_final.min():.3f}, {quality_final.max():.3f}]")
        
        if quality_final.max() < 0.1:
            print("Very low quality scores! Grasp may be unreliable.")
        
        # Find best grasp point
        most_confident_pixels = np.dstack(
            np.unravel_index(
                np.argsort(quality_final.ravel())[::-1],
                quality_final.shape
            )
        )[0]
        
        max_pixel = most_confident_pixels[0]
        u_300 = max_pixel[1]  # x coordinate
        v_300 = max_pixel[0]  # y coordinate
        angle_300 = ang_final[v_300, u_300]
        width_300 = width_final[v_300, u_300]
        
        print(f"\nGrasp in 300×300 space:")
        print(f"  Position: (u={u_300}, v={v_300})")
        print(f"  Quality: {quality_final[v_300, u_300]:.4f}")
        print(f"  Width: {width_300:.2f} pixels")
        print(f"  Angle: {np.degrees(angle_300):.2f}°")
        
        # Transform to original coordinates
        u_crop = u_300 * scale_w
        v_crop = v_300 * scale_h
        
        x_min, y_min, x_max, y_max = bbox
        u_final = u_crop + x_min
        v_final = v_crop + y_min
        
        scale_avg = (scale_w + scale_h) / 2.0
        width_final = width_300 * scale_avg
        angle_final = angle_300
        
        print(f"\nFinal grasp in original image:")
        print(f"  Position: (u={u_final:.2f}, v={v_final:.2f})")
        print(f"  Width: {width_final:.2f} pixels")
        print(f"  Angle: {np.degrees(angle_final):.2f}°")
        
        # Validate final coordinates
        img_h, img_w = im_rgb.shape[1], im_rgb.shape[2]
        if not (0 <= u_final < img_w and 0 <= v_final < img_h):
            print(f" Grasp outside image bounds!")
            print(f"  Image size: {img_w} × {img_h}")
            print(f"  Grasp: ({u_final:.1f}, {v_final:.1f})")
        
        return (float(u_final), float(v_final), float(width_final), float(angle_final))
    
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return "object not present"

if __name__ == "__main__":
    # Load test prompts
    with open('/home/omar/Desktop/ENN583/enn583/Project 2/test_prompts.json', 'r') as f:
        test_prompts = json.load(f)
    
    # Initialize models
    print("Loading models.")
    weights_mrcnn = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    mrcnn = maskrcnn_resnet50_fpn_v2(weights=weights_mrcnn, box_score_thresh=0.1)
    mrcnn = mrcnn.to(device)
    mrcnn.eval()
    coco_categories = weights_mrcnn.meta["categories"]
    preprocess = weights_mrcnn.transforms()
    
    ggcnn_model = torch.load(
        '/home/omar/Desktop/ENN583/enn583/Week 11/ggcnn2_weights_cornell/epoch_50_cornell',
        weights_only=False,
        map_location=device
    )
    ggcnn_model.eval()
    print("✓ Models loaded!\n")
    
    # Process all tests
    results = {}
    successful = 0
    not_found = 0
    
    print("="*60)
    print("PROCESSING ALL TEST CASES (FULL IMAGE APPROACH)")
    print("="*60)
    
    for test_id in sorted(test_prompts.keys()):
        test_data = test_prompts[test_id]
        
        result = process_single_test(
            test_id,
            test_data,
            mrcnn,
            preprocess,
            coco_categories,
            ggcnn_model,
            device
        )
        
        results[test_id] = result
        
        if result == "object not present":
            not_found += 1
        else:
            successful += 1
        
        # Clear GPU cache periodically
        if (successful + not_found) % 10 == 0:
            torch.cuda.empty_cache()
    
    
    
    
    
    
    
    
    
    
    
    # Save results
    output_path = '/home/omar/Desktop/ENN583/enn583/Project 2/test_results.json'
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Total tests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Not found: {not_found}")
    print(f"Success rate: {100*successful/len(results):.1f}%")
    print(f"\nResults saved to: {output_path}")
    
    # Print sample results
    print("\n=== SAMPLE RESULTS ===")
    for i, (test_id, result) in enumerate(list(results.items())[:5]):
        print(f"{test_id}: {result}")