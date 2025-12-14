import argparse
import yaml
import cv2
import os
import logging
from src.depth_engine import DepthEngine
from src.inpainting import Inpainter
from src.renderer import Renderer
from src.video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_single_image(args, config):
    logger.info(f"Processing image: {args.input}")
    
    # Init modules
    depth_engine = DepthEngine(model_type=config['model']['depth_type'])
    inpainter = Inpainter()
    renderer = Renderer()
    
    # Read image
    img = cv2.imread(args.input)
    if img is None:
        logger.error("Failed to read image")
        return

    # Pipeline
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth_map = depth_engine.estimate_depth(img_rgb)
    
    mask = depth_engine.generate_trimap(depth_map, args.focus_depth)
    bg_filled = inpainter.process(img, mask)
    
    result = renderer.render_frame(
        img, 
        bg_filled, 
        mask, 
        blur_radius=args.blur_radius,
        bokeh_threshold=config['rendering']['bokeh_threshold'],
        bokeh_intensity=config['rendering']['bokeh_intensity']
    )
    
    # Save
    output_path = args.output if args.output else f"output_{os.path.basename(args.input)}"
    cv2.imwrite(output_path, result)
    logger.info(f"Saved result to {output_path}")

def process_video(args, config):
    processor = VideoProcessor(config)
    output_path = args.output if args.output else f"output_{os.path.basename(args.input)}"
    processor.process_video(args.input, output_path, args.focus_depth, args.blur_radius)

def main():
    parser = argparse.ArgumentParser(description="Neural-TiltShift-Pro CLI")
    parser.add_argument("input", type=str, help="Input image or video path")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--mode", type=str, choices=['image', 'video'], default='image', help="Processing mode")
    parser.add_argument("--focus_depth", type=float, default=0.6, help="Focus depth (0.0-1.0)")
    parser.add_argument("--blur_radius", type=int, default=15, help="Blur radius")
    parser.add_argument("--config", type=str, default="configs/settings.yaml", help="Path to config file")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.mode == 'video' or args.input.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video(args, config)
    else:
        process_single_image(args, config)

if __name__ == "__main__":
    main()
