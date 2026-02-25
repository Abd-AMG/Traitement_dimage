import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

# ---------- Page configuration ----------
st.set_page_config(page_title="Computer Vision Teaching Lab", layout="wide")

# ---------- Mathematical Explanation Functions with Proper LaTeX ----------
def get_segmentation_formula(method):
    formulas = {
        "Thresholding": r"""
        **Binary Thresholding Formula:**
        $$
        g(x,y) = \begin{cases} 
        255 & \text{if } f(x,y) > T \\
        0 & \text{otherwise}
        \end{cases}
        $$
        
        Where:
        * $f(x,y)$ = original pixel value
        * $T$ = threshold value
        * $g(x,y)$ = output pixel value
        """,
        
        "Adaptive Thresholding": r"""
        **Adaptive Thresholding Formula:**
        $$
        T(x,y) = \mu(x,y) - C
        $$
        
        $$
        g(x,y) = \begin{cases}
        255 & \text{if } f(x,y) > T(x,y) \\
        0 & \text{otherwise}
        \end{cases}
        $$
        
        Where:
        * $\mu(x,y)$ = mean of local neighborhood
        * $C$ = constant subtracted
        """,
        
        "Otsu's Method": r"""
        **Otsu's Thresholding:**
        $$
        \sigma^2_B(t) = \omega_1(t)\omega_2(t)[\mu_1(t) - \mu_2(t)]^2
        $$
        
        Where:
        * $\omega_1$, $\omega_2$ = class probabilities
        * $\mu_1$, $\mu_2$ = class means
        * Optimal threshold $t^*$ maximizes $\sigma^2_B(t)$
        """,
        
        "Canny Edge Detection": r"""
        **Canny Edge Detection Steps:**
        
        1. **Gaussian Smoothing:**
        $$
        G(x,y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}
        $$
        
        2. **Gradient Calculation:**
        $$
        |\nabla f| = \sqrt{G_x^2 + G_y^2}
        $$
        $$
        \theta = \arctan\left(\frac{G_y}{G_x}\right)
        $$
        
        3. **Non-maximum Suppression**
        4. **Hysteresis Thresholding**
        """,
        
        "Sobel Operator": r"""
        **Sobel Operator:**
        
        $$
        G_x = \begin{bmatrix}
        -1 & 0 & 1 \\
        -2 & 0 & 2 \\
        -1 & 0 & 1
        \end{bmatrix} * A
        $$
        
        $$
        G_y = \begin{bmatrix}
        -1 & -2 & -1 \\
        0 & 0 & 0 \\
        1 & 2 & 1
        \end{bmatrix} * A
        $$
        
        **Gradient Magnitude:**
        $$
        G = \sqrt{G_x^2 + G_y^2}
        $$
        """,
        
        "Laplacian": r"""
        **Laplacian Operator:**
        
        $$
        \nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}
        $$
        
        **Discrete Approximation:**
        $$
        \begin{bmatrix}
        0 & -1 & 0 \\
        -1 & 4 & -1 \\
        0 & -1 & 0
        \end{bmatrix}
        $$
        """
    }
    return formulas.get(method, "Select a method to see formula")

def get_color_space_formula(space):
    formulas = {
        "RGB": r"""
        **RGB Color Space:**
        
        Linear combination of Red, Green, Blue:
        $$
        C = R + G + B
        $$
        
        Each channel: $0 \leq R,G,B \leq 255$
        """,
        
        "HSV": r"""
        **HSV Conversion Formulas:**
        
        Let $M = \max(R,G,B)$ and $m = \min(R,G,B)$, with $\delta = M - m$
        
        **Hue ($H$):**
        $$
        H = \begin{cases}
        60^\circ \times \left(\frac{G-B}{\delta} \mod 6\right) & \text{if } M = R \\[2ex]
        60^\circ \times \left(\frac{B-R}{\delta} + 2\right) & \text{if } M = G \\[2ex]
        60^\circ \times \left(\frac{R-G}{\delta} + 4\right) & \text{if } M = B
        \end{cases}
        $$
        
        **Saturation ($S$):**
        $$
        S = \begin{cases}
        0 & \text{if } M = 0 \\[1ex]
        \frac{\delta}{M} & \text{otherwise}
        \end{cases}
        $$
        
        **Value ($V$):**
        $$
        V = \max(R,G,B)
        $$
        """,
        
        "YCbCr": r"""
        **YCbCr Conversion (ITU-R BT.601):**
        
        $$
        \begin{bmatrix} Y \\ Cb \\ Cr \end{bmatrix} = 
        \begin{bmatrix} 
        0.299 & 0.587 & 0.114 \\
        -0.169 & -0.331 & 0.500 \\
        0.500 & -0.419 & -0.081
        \end{bmatrix}
        \begin{bmatrix} R \\ G \\ B \end{bmatrix} + 
        \begin{bmatrix} 0 \\ 128 \\ 128 \end{bmatrix}
        $$
        
        Where:
        * $Y$ = Luma (brightness component)
        * $Cb$ = Blue-difference chroma component
        * $Cr$ = Red-difference chroma component
        """
    }
    return formulas.get(space, "")

# ---------- Custom CSS for beautiful math frames ----------
st.markdown("""
<style>
    /* Math formula frames */
    .math-frame {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-left: 6px solid #4CAF50;
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1), 0 6px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .math-frame:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
    }
    
    /* Formula title */
    .formula-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #4CAF50;
    }
    
    /* Parameter explanation box */
    .param-box {
        background-color: #e8f4fd;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        border: 1px solid #2196F3;
    }
    
    /* Parameter list styling */
    .param-list {
        list-style-type: none;
        padding: 0;
    }
    
    .param-list li {
        margin: 8px 0;
        padding: 5px 10px;
        background: rgba(33, 150, 243, 0.1);
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
    
    /* Override Streamlit's default LaTeX rendering */
    .stMarkdown {
        font-size: 1.1rem;
    }
    
    /* Make LaTeX formulas bigger and clearer */
    .MathJax {
        font-size: 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Title with animation ----------
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 25px;">
    <h1 style="color: white; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🔬 Computer Vision Teaching Lab</h1>
    <p style="color: white; font-size: 1.2rem;">Master Image Processing Through Interactive Visualization</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; text-align: center; margin: 0;">🎮 Control Panel</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Image upload
    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
    
    # Choose lab topic
    lab_topic = st.selectbox(
        "📚 Select Topic",
        ["Color Space Segmentation", "Edge Detection", "Contour Analysis", "Thresholding Methods"]
    )
    
    st.markdown("---")
    st.markdown("### 📐 Theory Panel")
    show_theory = st.checkbox("Show Mathematical Formulas", True)

# Initialize default image if none uploaded
if uploaded_file is None:
    # Create a test image with shapes for demonstration
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Draw shapes
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue square
    cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(img, (450, 50), (550, 150), (0, 0, 255), -1)  # Red square
    cv2.ellipse(img, (200, 300), (80, 40), 0, 0, 360, (255, 255, 0), -1)  # Yellow ellipse
    cv2.circle(img, (400, 300), 60, (255, 0, 255), -1)  # Magenta circle
    
    # Add some text
    cv2.putText(img, "Computer Vision", (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.session_state['test_image'] = img_rgb
else:
    # Load user image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.session_state['test_image'] = img_rgb

# Main content area
img_rgb = st.session_state['test_image']

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["🎨 Color Space Segmentation", "🔍 Edge Detection", "📐 Contour Analysis"])

# ========== TAB 1: COLOR SPACE SEGMENTATION ==========
with tab1:
    st.header("🎨 Color Space Segmentation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="color: #2c3e50; margin-top: 0;">📷 Original Image</h4>
        </div>
        """, unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="color: #2c3e50; margin-top: 0;">🎯 Segmentation Result</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Color space selection
        color_space = st.radio(
            "Select Color Space",
            ["RGB", "HSV", "YCbCr"],
            horizontal=True,
            key="cs_selector"
        )
        
        # Channel selection for segmentation
        if color_space == "RGB":
            channel_names = ["Red", "Green", "Blue"]
            channel_colors = ["red", "green", "blue"]
            
            # Convert to RGB
            img_cs = img_rgb.copy()
            
            # Channel range selectors
            ranges = []
            cols = st.columns(3)
            for i, (name, col) in enumerate(zip(channel_names, cols)):
                with col:
                    st.markdown(f"**{name} Channel**")
                    min_val = st.slider(f"Min", 0, 255, 0, key=f"min_{name}", label_visibility="collapsed")
                    max_val = st.slider(f"Max", 0, 255, 255, key=f"max_{name}", label_visibility="collapsed")
                    ranges.append((min_val, max_val))
            
            # Create mask based on ranges
            mask = np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=bool)
            for i in range(3):
                mask &= (img_cs[:, :, i] >= ranges[i][0]) & (img_cs[:, :, i] <= ranges[i][1])
            
        elif color_space == "HSV":
            channel_names = ["Hue (0-179)", "Saturation (0-255)", "Value (0-255)"]
            
            # Convert to HSV
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            img_cs = img_hsv
            
            st.markdown("**🌈 Hue Range (circular)**")
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                hue_min = st.slider("Hue Min", 0, 179, 0, key="hue_min")
            with col_h2:
                hue_max = st.slider("Hue Max", 0, 179, 179, key="hue_max")
            
            st.markdown("**🎨 Saturation Range**")
            sat_min = st.slider("Saturation Min", 0, 255, 0, key="sat_min")
            sat_max = st.slider("Saturation Max", 0, 255, 255, key="sat_max")
            
            st.markdown("**💡 Value Range**")
            val_min = st.slider("Value Min", 0, 255, 0, key="val_min")
            val_max = st.slider("Value Max", 0, 255, 255, key="val_max")
            
            # Create mask (handle circular hue)
            if hue_min <= hue_max:
                mask = (img_cs[:, :, 0] >= hue_min) & (img_cs[:, :, 0] <= hue_max)
            else:
                mask = (img_cs[:, :, 0] >= hue_min) | (img_cs[:, :, 0] <= hue_max)
            
            mask &= (img_cs[:, :, 1] >= sat_min) & (img_cs[:, :, 1] <= sat_max)
            mask &= (img_cs[:, :, 2] >= val_min) & (img_cs[:, :, 2] <= val_max)
            
        else:  # YCbCr
            channel_names = ["Y (Luma)", "Cb (Blue Diff)", "Cr (Red Diff)"]
            
            # Convert to YCrCb
            img_ycc = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
            img_cs = img_ycc
            
            ranges = []
            cols = st.columns(3)
            for i, name in enumerate(channel_names):
                with cols[i]:
                    st.markdown(f"**{name}**")
                    min_val = st.slider(f"Min", 0, 255, 0, key=f"ycc_min_{i}", label_visibility="collapsed")
                    max_val = st.slider(f"Max", 0, 255, 255, key=f"ycc_max_{i}", label_visibility="collapsed")
                    ranges.append((min_val, max_val))
            
            # Create mask
            mask = np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=bool)
            for i in range(3):
                mask &= (img_cs[:, :, i] >= ranges[i][0]) & (img_cs[:, :, i] <= ranges[i][1])
        
        # Apply mask to get segmentation
        segmented = np.zeros_like(img_rgb)
        segmented[mask] = img_rgb[mask]
        
        # Create overlay visualization
        overlay = img_rgb.copy()
        overlay[~mask] = overlay[~mask] // 4  # Darken non-selected areas
        
        # Display result
        st.image(overlay, use_container_width=True)
        
        # Show mask statistics
        total_pixels = mask.size
        selected_pixels = np.sum(mask)
        percentage = (selected_pixels / total_pixels) * 100
        
        st.info(f"📊 **Selected Pixels:** {selected_pixels:,} ({percentage:.1f}% of image)")
    
    # Mathematical explanation with beautiful frames
    if show_theory:
        with st.expander("📐 **Mathematical Explanation**", expanded=True):
            st.markdown(f"""
            <div class="math-frame">
                <div class="formula-title">🎨 {color_space} Color Space Mathematics</div>
                {get_color_space_formula(color_space)}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="math-frame">
                <div class="formula-title">🎯 Segmentation Mathematics</div>
                
                **Binary Mask Generation:**
                $$
                M(x,y) = \begin{cases}
                1 & \text{if } L_1 \leq C_1(x,y) \leq U_1 \text{ and } \\[1ex]
                & \quad L_2 \leq C_2(x,y) \leq U_2 \text{ and } \\[1ex]
                & \quad L_3 \leq C_3(x,y) \leq U_3 \\[2ex]
                0 & \text{otherwise}
                \end{cases}
                $$
                
                Where:
                * $C_i(x,y)$ = value of channel $i$ at pixel $(x,y)$
                * $L_i$ = lower bound for channel $i$
                * $U_i$ = upper bound for channel $i$
                
                **Segmented Image:**
                $$
                I_{seg}(x,y) = I_{orig}(x,y) \cdot M(x,y)
                $$
                
                <div class="param-box">
                    <strong>📝 Interpretation:</strong> Pixels that satisfy ALL channel conditions
                    are kept (mask = 1), others are filtered out (mask = 0).
                </div>
            </div>
            """, unsafe_allow_html=True)

# ========== TAB 2: EDGE DETECTION ==========
with tab2:
    st.header("🔍 Edge Detection Methods")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="color: #2c3e50; margin-top: 0;">📷 Original Image</h4>
        </div>
        """, unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="color: #2c3e50; margin-top: 0;">⚡ Edge Detection Result</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Edge detection method selection
        edge_method = st.selectbox(
            "Select Edge Detection Method",
            ["Canny", "Sobel X", "Sobel Y", "Laplacian", "Prewitt", "Scharr"]
        )
        
        # Parameters based on method
        if edge_method == "Canny":
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                threshold1 = st.slider("Threshold 1 (low)", 0, 255, 50)
            with col_p2:
                threshold2 = st.slider("Threshold 2 (high)", 0, 255, 150)
            
            aperture_size = st.selectbox("Aperture Size (Sobel kernel)", [3, 5, 7], index=0)
            
            # Apply Canny
            edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture_size)
            
        elif edge_method in ["Sobel X", "Sobel Y"]:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                ksize = st.selectbox("Kernel Size", [1, 3, 5, 7], index=1)
            with col_p2:
                scale = st.slider("Scale factor", 1, 10, 1)
            
            dx = 1 if edge_method == "Sobel X" else 0
            dy = 1 if edge_method == "Sobel Y" else 0
            
            edges = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize, scale=scale)
            edges = np.absolute(edges)
            edges = np.uint8(np.clip(edges, 0, 255))
            
        elif edge_method == "Laplacian":
            ksize = st.selectbox("Kernel Size", [1, 3, 5, 7], index=1)
            edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
            edges = np.absolute(edges)
            edges = np.uint8(np.clip(edges, 0, 255))
            
        elif edge_method == "Prewitt":
            st.info("Prewitt uses fixed 3x3 kernels")
            # Custom Prewitt implementation
            kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            
            grad_x = cv2.filter2D(gray.astype(np.float32), -1, kernel_x)
            grad_y = cv2.filter2D(gray.astype(np.float32), -1, kernel_y)
            
            edges = np.sqrt(grad_x**2 + grad_y**2)
            edges = np.uint8(np.clip(edges, 0, 255))
            
        else:  # Scharr
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                dx = st.selectbox("X derivative order", [0, 1], index=1)
            with col_p2:
                dy = st.selectbox("Y derivative order", [0, 1], index=0)
            
            edges = cv2.Scharr(gray, cv2.CV_64F, dx, dy)
            edges = np.absolute(edges)
            edges = np.uint8(np.clip(edges, 0, 255))
        
        # Display edges
        st.image(edges, use_container_width=True, clamp=True)
        
        # Edge statistics
        edge_pixels = np.sum(edges > 0)
        st.info(f"🔹 **Edge pixels detected:** {edge_pixels:,} ({edge_pixels/edges.size*100:.1f}% of image)")
    
    # Mathematical explanation with beautiful frames
    if show_theory:
        with st.expander("📐 **Mathematical Explanation**", expanded=True):
            st.markdown(f"""
            <div class="math-frame">
                <div class="formula-title">🔬 {edge_method} Edge Detection</div>
                {get_segmentation_formula(edge_method)}
                
                <div class="param-box">
                    <strong>💡 Key Concept:</strong> Edges correspond to areas of rapid intensity change,
                    detected by computing image derivatives.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ========== TAB 3: CONTOUR ANALYSIS ==========
with tab3:
    st.header("📐 Contour Detection and Analysis")
    
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="color: #2c3e50; margin-top: 0;">⚙️ Preprocessing</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Thresholding method
        thresh_method = st.selectbox(
            "Threshold Method",
            ["Binary", "Adaptive Mean", "Adaptive Gaussian", "Otsu"]
        )
        
        if thresh_method == "Binary":
            thresh_val = st.slider("Threshold Value", 0, 255, 127)
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        elif thresh_method == "Adaptive Mean":
            block_size = st.slider("Block Size", 3, 31, 11, step=2)
            C = st.slider("Constant C", 0, 30, 2)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, block_size, C)
        elif thresh_method == "Adaptive Gaussian":
            block_size = st.slider("Block Size", 3, 31, 11, step=2)
            C = st.slider("Constant C", 0, 30, 2)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, block_size, C)
        else:  # Otsu
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        st.image(binary, caption="Binary Image (White = Object, Black = Background)", 
                use_container_width=True, clamp=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="color: #2c3e50; margin-top: 0;">🔍 Contour Detection</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Contour retrieval mode
        retrieval_mode = st.selectbox(
            "Retrieval Mode",
            ["EXTERNAL (outer contours only)", 
             "LIST (all contours)", 
             "TREE (hierarchical)", 
             "CCOMP (two-level hierarchy)"]
        )
        
        mode_map = {
            "EXTERNAL (outer contours only)": cv2.RETR_EXTERNAL,
            "LIST (all contours)": cv2.RETR_LIST,
            "TREE (hierarchical)": cv2.RETR_TREE,
            "CCOMP (two-level hierarchy)": cv2.RETR_CCOMP
        }
        
        # Contour approximation
        approx_method = st.selectbox(
            "Approximation Method",
            ["NONE (all points)", "SIMPLE (compress horizontal/vertical)"]
        )
        
        method_map = {
            "NONE (all points)": cv2.CHAIN_APPROX_NONE,
            "SIMPLE (compress horizontal/vertical)": cv2.CHAIN_APPROX_SIMPLE
        }
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            binary, 
            mode_map[retrieval_mode], 
            method_map[approx_method]
        )
        
        # Create visualization
        contour_img = img_rgb.copy()
        
        # Filter contours by area
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            min_area = st.slider("Min Area", 0, 1000, 50)
        with col_f2:
            max_area = st.slider("Max Area", 100, 10000, 5000)
        
        # Draw contours
        valid_contours = []
        contour_data = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                valid_contours.append(contour)
                
                # Calculate properties
                perimeter = cv2.arcLength(contour, True)
                moments = cv2.moments(contour)
                
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                else:
                    cx, cy = 0, 0
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Get convex hull
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                # Calculate shape descriptors
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Fit ellipse if possible
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(contour_img, ellipse, (255, 0, 255), 2)
                
                contour_data.append({
                    "ID": i,
                    "Area (pixels)": int(area),
                    "Perimeter": int(perimeter),
                    "Circularity": f"{circularity:.3f}",
                    "Solidity": f"{solidity:.3f}",
                    "Center": f"({cx}, {cy})",
                    "Bounding Box": f"{w}×{h}"
                })
        
        # Draw all valid contours
        cv2.drawContours(contour_img, valid_contours, -1, (0, 255, 0), 2)
        
        # Draw bounding boxes and centroids
        for contour in valid_contours:
            # Centroid
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(contour_img, (cx, cy), 5, (255, 0, 0), -1)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
        
        st.image(contour_img, caption=f"Detected Contours: {len(valid_contours)}", 
                use_container_width=True)
    
    # Contour properties table
    if contour_data:
        st.markdown("""
        <div style="background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 20px;">
            <h4 style="color: #2c3e50; margin-top: 0;">📊 Contour Properties</h4>
        </div>
        """, unsafe_allow_html=True)
        df = pd.DataFrame(contour_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Mathematical explanation with beautiful frames
    if show_theory:
        with st.expander("📐 **Mathematical Explanation**", expanded=True):
            st.markdown("""
            <div class="math-frame">
                <div class="formula-title">📐 Contour Mathematics</div>
                
                **Contour Area (using Green's Theorem):**
                $$
                A = \frac{1}{2} \left|\sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i)\right|
                $$
                where $(x_i, y_i)$ are the contour points in order, and $(x_{n+1}, y_{n+1}) = (x_1, y_1)$.
                
                **Contour Perimeter (arc length):**
                $$
                P = \sum_{i=1}^{n} \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}
                $$
                
                **Centroid (Center of Mass):**
                $$
                \bar{x} = \frac{1}{6A} \sum_{i=1}^{n} (x_i + x_{i+1})(x_i y_{i+1} - x_{i+1} y_i)
                $$
                $$
                \bar{y} = \frac{1}{6A} \sum_{i=1}^{n} (y_i + y_{i+1})(x_i y_{i+1} - x_{i+1} y_i)
                $$
                
                **Bounding Rectangle:**
                $$
                x_{\min} = \min_{i}(x_i), \quad x_{\max} = \max_{i}(x_i)
                $$
                $$
                y_{\min} = \min_{i}(y_i), \quad y_{\max} = \max_{i}(y_i)
                $$
                
                **Shape Descriptors:**
                * **Circularity:** $\displaystyle \frac{4\pi A}{P^2}$ (1 = perfect circle)
                * **Solidity:** $\displaystyle \frac{A}{A_{\text{convex hull}}}$ (1 = convex shape)
                
                <div class="param-box">
                    <strong>🎯 Teaching Tip:</strong> A perfect circle has circularity = 1. 
                    As the shape becomes more elongated, circularity approaches 0.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------- Theory Sidebar with beautiful formatting ----------
if show_theory:
    with st.sidebar:
        st.markdown("---")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); padding: 15px; border-radius: 10px; margin: 20px 0;">
            <h4 style="color: white; text-align: center; margin: 0;">🎓 Quick Theory Reference</h4>
        </div>
        """, unsafe_allow_html=True)
        
        topic = st.selectbox(
            "Select topic for quick reference",
            ["Segmentation", "Edge Detection", "Contour Analysis", "Color Spaces"]
        )
        
        if topic == "Segmentation":
            st.markdown("""
            <div style="background: white; padding: 15px; border-radius: 10px; border-left: 4px solid #4CAF50;">
                <strong>🔪 Segmentation Types:</strong>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>• <strong>Threshold-based</strong>: Pixels grouped by intensity</li>
                    <li>• <strong>Region-based</strong>: Connected similar pixels</li>
                    <li>• <strong>Edge-based</strong>: Boundaries between regions</li>
                    <li>• <strong>Clustering</strong>: K-means, Mean-shift</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        elif topic == "Edge Detection":
            st.markdown("""
            <div style="background: white; padding: 15px; border-radius: 10px; border-left: 4px solid #FF6B6B;">
                <strong>⚡ Edge Detection Steps:</strong>
                <ol style="padding-left: 20px;">
                    <li><strong>Smoothing</strong>: Reduce noise</li>
                    <li><strong>Gradient</strong>: Find intensity changes</li>
                    <li><strong>Non-max suppression</strong>: Thin edges</li>
                    <li><strong>Hysteresis</strong>: Connect edges</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
        elif topic == "Contour Analysis":
            st.markdown("""
            <div style="background: white; padding: 15px; border-radius: 10px; border-left: 4px solid #4ECDC4;">
                <strong>📐 Contour Properties:</strong>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>• <strong>Area</strong>: Region size</li>
                    <li>• <strong>Perimeter</strong>: Boundary length</li>
                    <li>• <strong>Circularity</strong>: $4\pi A / P^2$</li>
                    <li>• <strong>Convexity</strong>: Area/Convex area</li>
                    <li>• <strong>Solidity</strong>: Area/Convex hull area</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        elif topic == "Color Spaces":
            st.markdown("""
            <div style="background: white; padding: 15px; border-radius: 10px; border-left: 4px solid #667eea;">
                <strong>🎨 Color Space Uses:</strong>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>• <strong>RGB</strong>: Display, basic operations</li>
                    <li>• <strong>HSV</strong>: Color-based segmentation</li>
                    <li>• <strong>YCbCr</strong>: Compression, skin detection</li>
                    <li>• <strong>Lab</strong>: Perceptual uniformity</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ---------- Footer with teaching tips ----------
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin-top: 20px;">
    <h4 style="color: white; margin-top: 0;">💡 Teaching Tips for Each Concept:</h4>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
        <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
            <strong style="color: white;">🎨 Color Spaces</strong>
            <p style="color: white; font-size: 0.9rem;">Use HSV to isolate colored objects easily</p>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
            <strong style="color: white;">🔍 Edge Detection</strong>
            <p style="color: white; font-size: 0.9rem;">Compare Canny vs Sobel on the same image</p>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
            <strong style="color: white;">📐 Contours</strong>
            <p style="color: white; font-size: 0.9rem;">Show how circularity identifies shapes</p>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px;">
            <strong style="color: white;">📊 Math</strong>
            <p style="color: white; font-size: 0.9rem;">Explain each formula with visual examples</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)