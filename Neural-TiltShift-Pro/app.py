import gradio as gr
import cv2
import numpy as np
import yaml
import os
from PIL import Image
import tempfile

from src.depth_engine import DepthEngine
from src.inpainting import Inpainter
from src.renderer import Renderer
from src.video_processor import VideoProcessor
from src.auto_focus import SemanticAutoFocus

# Load Config
with open("configs/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize Engines
depth_engine = DepthEngine(model_type=config['model']['depth_type'])
inpainter = Inpainter()
renderer = Renderer()
auto_focus = None  # 延迟初始化，避免启动时下载 YOLO 模型

def get_auto_focus():
    """延迟加载 SemanticAutoFocus"""
    global auto_focus
    if auto_focus is None:
        auto_focus = SemanticAutoFocus()
    return auto_focus

# Global state to store current depth map for click interaction
current_depth_map = None
current_image_rgb = None
region_start = None  # 存储区域选择的起点

def process_linear(input_img, focus_position, direction, focus_width, falloff_power, 
                   blur_radius, saturation):
    """
    纯几何线性对焦模式处理 (不依赖深度估计)
    
    适用于垂直俯拍、卫星图等深度估计失效的场景
    """
    if input_img is None:
        return None, None
    
    # Convert PIL to cv2
    img_rgb = np.array(input_img)
    img_cv2 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # 纯几何渲染 (无需深度估计)
    result, mask = renderer.render_linear(
        img_cv2,
        focus_position=float(focus_position),
        direction=direction,
        focus_width=float(focus_width),
        falloff_power=float(falloff_power),
        max_blur_radius=int(blur_radius),
        color_enhance=True,
        saturation_boost=float(saturation)
    )
    
    # 转换回 RGB
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # 可视化线性 Mask
    mask_vis = (mask * 255).astype(np.uint8)
    mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
    
    return result_rgb, mask_vis

def process_scheimpflug(input_img, focus_depth, tilt_x, tilt_y, shift_correction, shift_direction, 
                        depth_of_field, blur_radius, saturation):
    """
    沙姆定律 (Scheimpflug Principle) 模式处理
    
    支持:
    - 3D 旋转焦平面 (Virtual Tilt)
    - 透视校正 (Virtual Shift)
    """
    if input_img is None:
        return None, None, None
    
    # Convert PIL to cv2
    img_rgb = np.array(input_img)
    img_cv2 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # 1. 估计深度
    depth_map = depth_engine.estimate_depth(img_rgb)
    
    # 2. 沙姆定律渲染
    result, mask = renderer.render_scheimpflug(
        img_cv2,
        depth_map,
        focus_depth=float(focus_depth),
        tilt_angle_x=float(tilt_x),
        tilt_angle_y=float(tilt_y),
        shift_correction=float(shift_correction),
        shift_direction=shift_direction,
        depth_of_field=float(depth_of_field),
        max_blur_radius=int(blur_radius),
        color_enhance=True,
        saturation_boost=float(saturation)
    )
    
    # 转换回 RGB
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # 可视化深度图
    depth_vis = (depth_map * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)
    
    # 可视化焦平面 Mask
    mask_vis = (mask * 255).astype(np.uint8)
    mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
    
    return result_rgb, depth_vis, mask_vis

def process_image_with_region(input_img, x1, y1, x2, y2, focus_depth, blur_radius, transition_smoothness, saturation, auto_focus_enabled=False):
    """
    处理带有区域选择的图像
    使用四个数值输入框来指定区域坐标
    """
    global current_depth_map, current_image_rgb
    
    if input_img is None:
        return None, focus_depth, None

    # Convert PIL to cv2
    img_rgb = np.array(input_img)
    img_cv2 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    current_image_rgb = img_rgb
    h, w = img_rgb.shape[:2]
    
    # 1. Estimate Depth
    depth_map = depth_engine.estimate_depth(img_rgb)
    current_depth_map = depth_map
    
    detection_vis = None
    
    # 自动对焦模式
    if auto_focus_enabled:
        af = get_auto_focus()
        focus_depth, target = af.get_focus_depth(img_rgb, depth_map)
        if target:
            detections = af.detect_objects(img_rgb)
            detection_vis = af.visualize_detections(img_rgb, detections, selected_idx=0)
    # 区域选择对焦模式
    elif x1 is not None and y1 is not None and x2 is not None and y2 is not None:
        # 转换为整数并确保有效
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 确保坐标有效
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)
        
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
            
        # 提取选定区域的深度值，取中值作为对焦深度
        if x2 > x1 and y2 > y1:
            region_depth = depth_map[y1:y2+1, x1:x2+1]
            if region_depth.size > 0:
                focus_depth = float(np.median(region_depth))
                print(f"Region selected: ({x1}, {y1}) to ({x2}, {y2}), Focus Depth: {focus_depth:.3f}")
            
            # 可视化选择的区域
            detection_vis = img_rgb.copy()
            cv2.rectangle(detection_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(detection_vis, f"Focus Depth: {focus_depth:.2f}", (x1, max(y1-10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 2. Generate Mask
    mask = depth_engine.generate_trimap(depth_map, focus_depth)
    
    # 3. Inpaint
    bg_filled = inpainter.process(img_cv2, mask)
    
    # 4. Render
    result = renderer.render_frame(
        img_cv2, 
        bg_filled, 
        mask, 
        blur_radius=int(blur_radius),
        transition_smoothness=float(transition_smoothness),
        color_enhance=True,
        saturation_boost=float(saturation)
    )
    
    # Convert back to RGB
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return result_rgb, focus_depth, detection_vis

def process_image(input_img, focus_depth, blur_radius, transition_smoothness, saturation, auto_focus_enabled=False, evt: gr.SelectData = None):
    global current_depth_map
    
    if input_img is None:
        return None, focus_depth, None

    # Convert PIL to cv2
    img_rgb = np.array(input_img)
    img_cv2 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # 1. Estimate Depth
    depth_map = depth_engine.estimate_depth(img_rgb)
    current_depth_map = depth_map
    
    detection_vis = None
    
    # 自动对焦模式
    if auto_focus_enabled:
        af = get_auto_focus()
        focus_depth, target = af.get_focus_depth(img_rgb, depth_map)
        if target:
            detections = af.detect_objects(img_rgb)
            detection_vis = af.visualize_detections(img_rgb, detections, selected_idx=0)
    # 点击对焦模式
    elif evt is not None:
        x, y = evt.index
        if y < depth_map.shape[0] and x < depth_map.shape[1]:
            clicked_depth = depth_map[y, x]
            focus_depth = float(clicked_depth)
            print(f"Clicked at ({x}, {y}), Depth: {focus_depth}")
    
    # 2. Generate Mask
    mask = depth_engine.generate_trimap(depth_map, focus_depth)
    
    # 3. Inpaint
    bg_filled = inpainter.process(img_cv2, mask)
    
    # 4. Render
    result = renderer.render_frame(
        img_cv2, 
        bg_filled, 
        mask, 
        blur_radius=int(blur_radius),
        transition_smoothness=float(transition_smoothness),
        color_enhance=True,
        saturation_boost=float(saturation)
    )
    
    # Convert back to RGB
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return result_rgb, focus_depth, detection_vis

def process_video_ui(video_path, focus_depth, blur_radius):
    if video_path is None:
        return None
        
    processor = VideoProcessor(config)
    
    # Create temp output path
    output_path = tempfile.mktemp(suffix=".mp4")
    
    processor.process_video(
        video_path, 
        output_path, 
        focus_depth, 
        int(blur_radius)
    )
    
    return output_path

with gr.Blocks(title="Neural-TiltShift-Pro") as app:
    gr.Markdown("# 🏙️ Neural-TiltShift-Pro: 神经渲染移轴生成器")
    
    with gr.Tab("📏 几何线性模式 (Geometric)"):
        gr.Markdown("""
        ### 🎯 纯几何线性对焦 (不依赖深度估计)
        
        **适用场景**: 
        - ✅ **垂直俯拍街道/卫星图**: 深度估计会产生噪点
        - ✅ **建筑立面**: 缺乏透视关系
        - ✅ **扁平化设计/插画**: 无真实深度信息
        
        **工作原理**: 基于像素的几何位置（如从上到下、从左到右）生成线性渐变 Mask
        
        **使用提示**:
        - `垂直渐变`: 适合俯拍街道（上清晰下模糊 或 反之）
        - `水平渐变`: 适合左右构图
        - `径向渐变`: 中心清晰向外模糊（类似移轴环形效果）
        - `对角线渐变`: 适合斜向构图
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_linear = gr.Image(type="pil", label="上传图片")
                
                gr.Markdown("#### 🔧 焦点控制")
                focus_pos_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.01,
                                             label="焦点位置",
                                             info="0=最上/左, 1=最下/右")
                direction_radio = gr.Radio(
                    ["vertical", "horizontal", "radial", "diagonal_tlbr", "diagonal_trbl"],
                    value="vertical",
                    label="渐变方向",
                    info="垂直/水平/径向/对角线"
                )
                
                gr.Markdown("#### 🎨 过渡参数")
                focus_width_slider = gr.Slider(0.05, 0.8, value=0.2, step=0.01,
                                               label="焦点区域宽度",
                                               info="值越大清晰区域越宽")
                falloff_slider = gr.Slider(0.5, 5.0, value=2.0, step=0.1,
                                          label="衰减锐度",
                                          info="值越大过渡越陡峭")
                
                gr.Markdown("#### 🎨 渲染参数")
                blur_linear = gr.Slider(5, 50, value=25, step=1, label="最大模糊半径")
                sat_linear = gr.Slider(1.0, 2.0, value=1.4, step=0.1, label="饱和度增强")
                
                btn_linear = gr.Button("🚀 生成移轴效果", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_linear = gr.Image(label="渲染结果")
                mask_linear = gr.Image(label="线性 Mask 可视化")
        
        btn_linear.click(
            process_linear,
            inputs=[input_linear, focus_pos_slider, direction_radio, focus_width_slider,
                   falloff_slider, blur_linear, sat_linear],
            outputs=[output_linear, mask_linear]
        )
    
    with gr.Tab("📐 沙姆定律模式 (Scheimpflug)"):
        gr.Markdown("""
        ### 🎯 专业移轴摄影模拟
        
        **沙姆定律 (Scheimpflug Principle)** 是移轴镜头的核心光学原理：
        - **Virtual Tilt (俯仰/摇摆)**: 倾斜焦平面，实现"地面全清晰，高处模糊"等效果
        - **Virtual Shift (透视校正)**: 梯形校正，拉直建筑线条
        
        **使用提示**: 
        - `俯仰角 > 0`: 地面/近处清晰，天空/远处模糊
        - `摇摆角 > 0`: 左侧清晰，右侧模糊
        - `透视校正 > 0`: 校正仰拍时建筑上窄下宽的变形
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_scheimpflug = gr.Image(type="pil", label="上传图片")
                
                gr.Markdown("#### 🔧 焦平面控制 (Virtual Tilt)")
                with gr.Row():
                    tilt_x_slider = gr.Slider(-45, 45, value=15, step=1, 
                                              label="俯仰角 Tilt-X (°)", 
                                              info="正值=地面清晰/天空模糊")
                    tilt_y_slider = gr.Slider(-45, 45, value=0, step=1, 
                                              label="摇摆角 Tilt-Y (°)", 
                                              info="正值=左清晰/右模糊")
                
                focus_depth_sch = gr.Slider(0.0, 1.0, value=0.5, step=0.05, 
                                            label="焦点深度", info="焦平面基准深度")
                dof_slider = gr.Slider(0.05, 0.5, value=0.15, step=0.01, 
                                       label="景深范围", info="值越大焦点区域越宽")
                
                gr.Markdown("#### 📏 透视校正 (Virtual Shift)")
                with gr.Row():
                    shift_slider = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, 
                                             label="校正强度", 
                                             info="正值=校正仰拍变形")
                    shift_dir = gr.Radio(["vertical", "horizontal"], value="vertical", 
                                         label="校正方向")
                
                gr.Markdown("#### 🎨 渲染参数")
                blur_sch = gr.Slider(5, 50, value=25, step=1, label="最大模糊半径")
                sat_sch = gr.Slider(1.0, 2.0, value=1.4, step=0.1, label="饱和度增强")
                
                btn_scheimpflug = gr.Button("🚀 生成移轴效果", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_scheimpflug = gr.Image(label="渲染结果")
                with gr.Row():
                    depth_vis = gr.Image(label="深度图", scale=1)
                    mask_vis = gr.Image(label="焦平面 Mask", scale=1)
        
        btn_scheimpflug.click(
            process_scheimpflug,
            inputs=[input_scheimpflug, focus_depth_sch, tilt_x_slider, tilt_y_slider, 
                   shift_slider, shift_dir, dof_slider, blur_sch, sat_sch],
            outputs=[output_scheimpflug, depth_vis, mask_vis]
        )
    
    with gr.Tab("🖼️ 图片模式 (区域选择)"):
        gr.Markdown("### 📌 使用方法：输入对焦区域的左上角和右下角坐标（像素），或点击图片获取坐标")
        with gr.Row():
            with gr.Column():
                input_image_region = gr.Image(type="pil", label="上传图片 (点击可获取坐标)")
                
                gr.Markdown("**对焦区域坐标 (点击图片自动填充)**")
                with gr.Row():
                    x1_input = gr.Number(label="左上角 X", value=100)
                    y1_input = gr.Number(label="左上角 Y", value=100)
                with gr.Row():
                    x2_input = gr.Number(label="右下角 X", value=300)
                    y2_input = gr.Number(label="右下角 Y", value=300)
                
                with gr.Group():
                    auto_focus_checkbox = gr.Checkbox(label="🎯 智能对焦 (YOLOv8 自动检测车辆/行人)", value=False)
                    focus_slider = gr.Slider(0.0, 1.0, value=0.6, label="焦点深度 (Focus Depth)")
                    blur_slider = gr.Slider(1, 50, value=15, step=1, label="模糊强度 (Blur Radius)")
                    transition_slider = gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="过渡平滑度 (值越大边缘越柔和)")
                    sat_slider = gr.Slider(1.0, 2.0, value=1.4, label="饱和度增强 (Saturation)")
                
                btn_run = gr.Button("生成移轴效果", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="渲染结果")
                detection_image = gr.Image(label="选区/检测可视化")
        
        # 点击图片更新坐标的状态
        click_state = gr.State(value={"count": 0})
        
        def update_coords(evt: gr.SelectData, state):
            """点击图片时交替更新左上角和右下角坐标"""
            x, y = evt.index
            count = state.get("count", 0)
            new_state = {"count": count + 1}
            
            if count % 2 == 0:
                # 第一次点击：设置左上角
                return x, y, gr.update(), gr.update(), new_state
            else:
                # 第二次点击：设置右下角
                return gr.update(), gr.update(), x, y, new_state
        
        input_image_region.select(
            update_coords,
            inputs=[click_state],
            outputs=[x1_input, y1_input, x2_input, y2_input, click_state]
        )
        
        # Event Bindings
        btn_run.click(
            process_image_with_region,
            inputs=[input_image_region, x1_input, y1_input, x2_input, y2_input, 
                   focus_slider, blur_slider, transition_slider, sat_slider, auto_focus_checkbox],
            outputs=[output_image, focus_slider, detection_image]
        )
    
    with gr.Tab("🖼️ 点击对焦模式"):
        gr.Markdown("### 📌 使用方法：**点击**图片上的某个位置来选择对焦点")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="上传图片 (点击图片可手动对焦)")
                
                with gr.Group():
                    auto_focus_checkbox2 = gr.Checkbox(label="🎯 智能对焦 (YOLOv8 自动检测车辆/行人)", value=False)
                    focus_slider2 = gr.Slider(0.0, 1.0, value=0.6, label="焦点深度 (Focus Depth)")
                    blur_slider2 = gr.Slider(1, 50, value=15, step=1, label="模糊强度 (Blur Radius)")
                    transition_slider2 = gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="过渡平滑度 (值越大边缘越柔和)")
                    sat_slider2 = gr.Slider(1.0, 2.0, value=1.4, label="饱和度增强 (Saturation)")
                
                btn_run2 = gr.Button("生成移轴效果", variant="primary")
            
            with gr.Column():
                output_image2 = gr.Image(label="渲染结果")
                detection_image2 = gr.Image(label="目标检测结果 (智能对焦模式)", visible=True)
        
        # Event Bindings
        input_image.select(
            process_image,
            inputs=[input_image, focus_slider2, blur_slider2, transition_slider2, sat_slider2, auto_focus_checkbox2],
            outputs=[output_image2, focus_slider2, detection_image2]
        )
        
        btn_run2.click(
            process_image,
            inputs=[input_image, focus_slider2, blur_slider2, transition_slider2, sat_slider2, auto_focus_checkbox2],
            outputs=[output_image2, focus_slider2, detection_image2]
        )

    with gr.Tab("🎥 视频模式"):
        gr.Markdown("⚠️ 视频处理较慢，请耐心等待。支持 .mp4 格式。")
        with gr.Row():
            input_video = gr.Video(label="上传视频")
            output_video = gr.Video(label="处理结果")
        
        with gr.Row():
            v_focus = gr.Slider(0.0, 1.0, value=0.6, label="焦点深度")
            v_blur = gr.Slider(1, 30, value=10, label="模糊半径")
            
        btn_video = gr.Button("开始视频渲染", variant="primary")
        
        btn_video.click(
            process_video_ui,
            inputs=[input_video, v_focus, v_blur],
            outputs=[output_video]
        )

if __name__ == "__main__":
    app.launch()
