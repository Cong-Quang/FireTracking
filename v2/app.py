import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from models import get_model
import os

# --- Constants and Setup ---
st.set_page_config(page_title="Fire Detection System", layout="wide", page_icon="🔥")

# Dictionary of available models
MODEL_OPTIONS = {
    "Custom CNN": "custom_cnn",
    "ResNet-50": "resnet50",
    "MobileNet-V2": "mobilenet_v2",
    "VGG-16": "vgg16",
    "EfficientNet-B0": "efficientnet"
}

# Transform for incoming inference images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = ['fire', 'non_fire']

# --- Helper Functions ---
@st.cache_resource
def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, num_classes=2)
    weights_path = f"checkpoints/{model_name}_best.pth"
    
    if not os.path.exists(weights_path):
        return None, f"Weights file not found: {weights_path}. Please train this model first."
        
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

def predict_image(image, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class_idx = torch.max(probabilities, 0)
        
    return classes[predicted_class_idx.item()], confidence.item(), probabilities.cpu().numpy()

# --- UI Setup ---
st.title("🔥 Hệ thống giám sát và phát hiện cháy sớm")
st.write("Hệ thống cảnh báo an toàn môi trường sử dụng các mô hình Deep Learning.")

st.sidebar.header("Cấu hình mô hình")
selected_model_display = st.sidebar.selectbox("Chọn mô hình dự đoán:", list(MODEL_OPTIONS.keys()))
internal_model_name = MODEL_OPTIONS[selected_model_display]

model, error = load_model(internal_model_name)

if error:
    st.sidebar.error(error)
    st.warning("Vui lòng huấn luyện mô hình và lưu weights vào thư mục `checkpoints` trước khi sử dụng.")
else:
    st.sidebar.success("✅ Mô hình đã được tải thành công!")

st.header("Upload hình ảnh giám sát")
uploaded_file = st.file_uploader("Chọn hình ảnh môi trường...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Hình ảnh tải lên', use_column_width=True)
    
    if st.button("Phân tích hình ảnh"):
        if model is None:
            st.error("Mô hình chưa sẵn sàng. Không thể phân tích.")
        else:
            with st.spinner('Đang xử lý phân tích...'):
                prediction, confidence, probs = predict_image(image, model)
                
                st.subheader("Kết quả dự đoán:")
                
                if prediction == 'fire':
                    st.error(f"⚠️ CẢNH BÁO: PHÁT HIỆN CHÁY! (Độ tin cậy: {confidence*100:.2f}%)")
                else:
                    st.success(f"✅ AN TOÀN: Không phát hiện cháy. (Độ tin cậy: {confidence*100:.2f}%)")
                    
                # Display probabilities chart
                st.write("Chi tiết xác suất:")
                st.bar_chart({classes[0]: [probs[0]], classes[1]: [probs[1]]})
