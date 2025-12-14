import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DetectedObject:
    """检测到的目标对象"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]  # (cx, cy)
    area: int


class SemanticAutoFocus:
    """
    语义自动对焦模块
    使用 YOLOv8 检测图像中的车辆和行人，自动选择最大目标作为对焦点
    """
    
    # COCO 数据集中感兴趣的类别 ID
    # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
    TARGET_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }
    
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        初始化语义自动对焦模块
        
        Args:
            model_name: YOLOv8 模型名称 (yolov8n/s/m/l/x.pt)
            confidence_threshold: 检测置信度阈值
        """
        logger.info(f"Initializing SemanticAutoFocus with model: {model_name}")
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            self.confidence_threshold = confidence_threshold
            logger.info("SemanticAutoFocus initialized successfully.")
        except ImportError:
            logger.error("ultralytics 库未安装。请运行: pip install ultralytics")
            raise ImportError("请安装 ultralytics: pip install ultralytics")
        except Exception as e:
            logger.error(f"Failed to initialize SemanticAutoFocus: {e}")
            raise e

    def detect_objects(self, img_rgb: np.ndarray) -> List[DetectedObject]:
        """
        检测图像中的目标对象（车辆和行人）
        
        Args:
            img_rgb: RGB 图像 (H, W, 3) numpy array
            
        Returns:
            detections: 检测到的目标列表，按面积降序排列
        """
        # YOLOv8 推理
        results = self.model(img_rgb, verbose=False)[0]
        
        detections = []
        
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            
            # 过滤：仅保留感兴趣的类别和高置信度目标
            if class_id not in self.TARGET_CLASSES:
                continue
            if confidence < self.confidence_threshold:
                continue
            
            # 提取边界框
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # 计算中心点和面积
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            
            detections.append(DetectedObject(
                class_id=class_id,
                class_name=self.TARGET_CLASSES[class_id],
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                area=area
            ))
        
        # 按面积降序排列
        detections.sort(key=lambda x: x.area, reverse=True)
        
        logger.info(f"Detected {len(detections)} target objects")
        return detections

    def get_focus_depth(self, 
                        img_rgb: np.ndarray, 
                        depth_map: np.ndarray,
                        fallback_depth: float = 0.6) -> Tuple[float, Optional[DetectedObject]]:
        """
        自动获取对焦深度
        
        检测图像中的车辆/行人，选择面积最大的目标，
        从深度图中读取其中心点的深度值作为对焦深度。
        
        Args:
            img_rgb: RGB 图像 (H, W, 3)
            depth_map: 归一化深度图 (H, W), 值域 0.0-1.0
            fallback_depth: 未检测到目标时的默认深度
            
        Returns:
            focus_depth: 对焦深度值 (0.0 - 1.0)
            target: 选中的目标对象（如果有）
        """
        detections = self.detect_objects(img_rgb)
        
        if not detections:
            logger.warning(f"No target detected. Using fallback depth: {fallback_depth}")
            return fallback_depth, None
        
        # 选择面积最大的目标
        largest_target = detections[0]
        cx, cy = largest_target.center
        
        # 边界检查
        h, w = depth_map.shape[:2]
        cx = np.clip(cx, 0, w - 1)
        cy = np.clip(cy, 0, h - 1)
        
        # 从深度图读取深度值
        # 为了鲁棒性，取中心点周围小区域的平均深度
        patch_size = 5
        y_start = max(0, cy - patch_size)
        y_end = min(h, cy + patch_size + 1)
        x_start = max(0, cx - patch_size)
        x_end = min(w, cx + patch_size + 1)
        
        depth_patch = depth_map[y_start:y_end, x_start:x_end]
        focus_depth = float(np.median(depth_patch))
        
        logger.info(
            f"Auto-focus on {largest_target.class_name} "
            f"(area={largest_target.area}, conf={largest_target.confidence:.2f}) "
            f"at ({cx}, {cy}), depth={focus_depth:.3f}"
        )
        
        return focus_depth, largest_target

    def visualize_detections(self, 
                             img_rgb: np.ndarray, 
                             detections: List[DetectedObject],
                             selected_idx: int = 0) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            img_rgb: RGB 图像
            detections: 检测结果列表
            selected_idx: 选中目标的索引（绿色高亮）
            
        Returns:
            vis_img: 可视化后的图像
        """
        vis_img = img_rgb.copy()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            cx, cy = det.center
            
            # 选中的目标用绿色，其他用蓝色
            color = (0, 255, 0) if i == selected_idx else (255, 100, 100)
            thickness = 3 if i == selected_idx else 2
            
            # 绘制边界框
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制中心点
            cv2.circle(vis_img, (cx, cy), 5, color, -1)
            
            # 绘制标签
            label = f"{det.class_name} {det.confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(vis_img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_img
