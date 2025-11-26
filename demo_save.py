# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
from glob import glob

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample, visualize_sample_together
from tqdm import tqdm


def main(args):

    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
    if len(segmentor_path):
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    # Get all folders in specified root directory
    folders_list = sorted([d for d in glob(os.path.join(args.folder_root, "*")) if os.path.isdir(d)])
    
    # Parallel processing: distribute folders based on rank
    if args.rank is not None and args.n_rank is not None:
        folders_list = folders_list[args.rank::args.n_rank]
        print(f"Rank {args.rank}/{args.n_rank}: Processing {len(folders_list)} folders")
    
    image_extensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"]

    for folder_idx, folder_path in enumerate(tqdm(folders_list)):
        folder_outputs = {}
        if os.path.exists(f"{folder_path}/sam3db.pth"):
            print(f"Folder {folder_path} already processed, skipping")
            continue

        for subfolder in ['images', 'input_images']:

            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.exists(subfolder_path):
                continue

            # Get all images in current subfolder
            images_list = sorted([
                os.path.join(subfolder_path, fname) for fname in os.listdir(subfolder_path)
                    if fname.split('.')[-1].lower() in image_extensions])
        
            for i, image_path in enumerate(images_list):
                outputs = estimator.process_one_image(
                    image_path,
                    bbox_thr=args.bbox_thresh,
                    use_mask=args.use_mask,
                )

                if len(outputs) == 0:
                    print(f"No human detected in {image_path}")
                    continue

                # Select largest bbox if multiple detections
                if len(outputs) > 1:
                    bbox_area_fn = lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    bbox_area = [bbox_area_fn(output['bbox']) for output in outputs]
                    max_idx = np.argmax(bbox_area)
                    output = outputs[max_idx]
                else:
                    output = outputs[0]
                
                # Only visualize for first 10 folders
                if folder_idx < 10:
                    # Create output folder structure
                    output_base = f"{folder_path}/tmp/{subfolder}"
                    os.makedirs(output_base, exist_ok=True)
                    
                    img = cv2.imread(image_path)
                    rend_img = visualize_sample_together(img, [output], estimator.faces)

                    cv2.imwrite(
                        f"{output_base}/{os.path.basename(image_path)[:-4]}.jpg",
                        rend_img.astype(np.uint8),
                    )
                
                # Add image filename and outputs to subfolder results
                key = os.path.relpath(image_path, folder_path)
                del output['pred_vertices']
                folder_outputs[key] = output

        # Save all outputs of the folder as a single .pth file
        torch.save(
            folder_outputs,
            f"{folder_path}/sam3db.pth"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Single Image Human Mesh Recovery (Save PTH)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo_save.py --folder_root ../../data/LHM-fitting/LHM-Human4DiT --checkpoint_path ./checkpoints/model.ckpt
                python demo_save.py --folder_root ./my_data --checkpoint_path ./checkpoints/model.ckpt --rank 0 --n_rank 4
                
                Input/Output Structure:
                Processes all folders in specified root directory
                Each folder should contain 'images/' and 'input_images/' subfolders
                {folder_root}/any_folder/ -> ./tmp/any_folder/
                  ├── images/ -> tmp/any_folder/images/
                  └── input_images/ -> tmp/any_folder/input_images/

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )
    parser.add_argument(
        "--folder_root",
        required=True,
        type=str,
        help="Root directory to search for numbered folders (e.g., ../../data/LHM-fitting/LHM-Human4DiT)",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_mhr_path)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    parser.add_argument(
        "--rank",
        default=None,
        type=int,
        help="Rank for parallel processing (0-based index)",
    )
    parser.add_argument(
        "--n_rank",
        default=None,
        type=int,
        help="Total number of ranks for parallel processing",
    )
    args = parser.parse_args()

    main(args)