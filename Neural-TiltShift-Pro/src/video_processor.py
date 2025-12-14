import cv2
import numpy as np
import logging
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import os

from .depth_engine import DepthEngine
from .inpainting import Inpainter
from .renderer import Renderer

logger = logging.getLogger(__name__)

def _process_frame_task(args):
    """
    Worker function for multiprocessing.
    Must be top-level to be picklable.
    """
    frame, mask, blur_radius, bokeh_threshold, bokeh_intensity = args
    
    # Re-instantiate lightweight classes inside worker if needed, 
    # or use static methods. Here we instantiate.
    inpainter = Inpainter()
    renderer = Renderer()
    
    # Inpaint
    bg_filled = inpainter.process(frame, mask)
    
    # Render
    result = renderer.render_frame(
        frame, 
        bg_filled, 
        mask, 
        blur_radius=blur_radius,
        bokeh_threshold=bokeh_threshold,
        bokeh_intensity=bokeh_intensity
    )
    return result

class VideoProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.depth_engine = DepthEngine(
            model_type=config['model']['depth_type'],
            device=config['model']['device']
        )
        self.window_size = config['video']['smoothing_window']
        self.depth_buffer = deque(maxlen=self.window_size)

    def process_video(self, input_path: str, output_path: str, 
                      focus_depth: float, blur_radius: int):
        
        if not os.path.exists(input_path):
            logger.error(f"Input video not found: {input_path}")
            return

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Setup Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Multiprocessing Setup
        # Use fewer workers than CPU count to avoid OOM with large images
        max_workers = max(1, multiprocessing.cpu_count() - 2)
        executor = ProcessPoolExecutor(max_workers=max_workers)
        futures = []
        
        # Frame Loop
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Processing Video")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. Depth Estimation (GPU - Main Process)
            # Convert to RGB for model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_map = self.depth_engine.estimate_depth(frame_rgb)
            
            # 2. Rolling Average Smoothing
            self.depth_buffer.append(depth_map)
            avg_depth = np.mean(np.array(self.depth_buffer), axis=0)
            
            # 3. Generate Mask
            mask = self.depth_engine.generate_trimap(avg_depth, focus_depth)
            
            # 4. Submit Rendering Task (CPU - Worker Process)
            # We submit tasks and collect them in order. 
            # To keep memory low, we can process in chunks or wait if too many futures.
            task_args = (
                frame, 
                mask, 
                blur_radius, 
                self.config['rendering']['bokeh_threshold'],
                self.config['rendering']['bokeh_intensity']
            )
            future = executor.submit(_process_frame_task, task_args)
            futures.append(future)
            
            # Flow control: Don't let queue grow too large
            if len(futures) > max_workers * 2:
                # Wait for the oldest future
                result = futures.pop(0).result()
                out.write(result)
                pbar.update(1)
            
            frame_idx += 1

        # Process remaining futures
        for future in futures:
            result = future.result()
            out.write(result)
            pbar.update(1)
            
        pbar.close()
        cap.release()
        out.release()
        executor.shutdown()
        logger.info(f"Video processing complete. Saved to {output_path}")
