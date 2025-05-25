import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os
import torchvision.models as models 

CLASS_NAMES_DATASET_INTEL = {
    0: "buildings",
    1: "forest", 
    2: "glacier",
    3: "mountain",
    4: "sea",
    5: "street"
}

CLASS_NAMES_DATASET_MIT = {
    0: "airport_inside",
    1: "artstudio",
    2: "auditorium",
    3: "bakery",
    4: "bar",
    5: "bathroom",
    6: "bedroom",
    7: "bookstore",
    8: "bowling",
    9: "buffet",
    10: "casino", 
    11: "children_room",
    12: "church_inside",
    13: "classroom",
    14: "cloister",
    15: "closet",
    16: "clothing_store",
    17: "computer_room",
    18: "concert_hall",
    19: "corridor",
    20: "deli",
    21: "dental_office",
    22: "dining_room",
    23: "elevator",
    24: "fastfood_restaurant",
    25: "florist",
    26: "game_room",
    27: "garage",
    28: "greenhouse",
    29: "grocery_store",
    30: "gym",
    31: "hair_salon",
    32: "hospital",
    33: "inside_bus",
    34: "inside_subway",
    35: "jewelry_store",
    36: "kindergarten",
    37: "kitchen",
    38: "laboratorywet",
    39: "laundromat",
    40: "library",
    41: "living_room",
    42: "lobby",
    43: "lockeroom",
    44: "mall", 
    45: "meeting_room",
    46: "movie_theater",
    47: "museum",
    48: "nursery",
    49: "office",
    50: "operating_room",
    51: "pantry",
    52: "poolinside",
    53: "prison_cell",
    54: "restaurant",
    55: "restaurant_kitchen",
    56: "shoeshop",
    57: "stairscase",
    58: "studiomusic", 
    59: "subway",
    60: "toy_store",
    61: "train_station",
    62: "tv_studio",
    63: "video_store",
    64: "waiting_room",
    65: "warehouse",
    66: "winecellar"
}

# Danh sách các model có sẵn
AVAILABLE_MODELS = [
    "VGG16",
    "ResNet18", 
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "MobileNet",
    "VisionTransformer"
]

def get_pretrained_model(model_name, num_classes):
    if model_name == "VGG16":
        model = models.vgg16(pretrained=False)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "ResNet18":
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ResNet34":
        model = models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ResNet50":
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ResNet101":
        model = models.resnet101(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "MobileNetV3":
        model = models.mobilenet_v3_large(pretrained=False)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "VisionTransformer":
        model = models.vit_b_16(pretrained=False)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} không được hỗ trợ.")
    return model
# Preprocessing transform
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def load_model(model_name, dataset_choice):
    """
    Load model với state_dict tương ứng
    """
    try:
        if dataset_choice == "Intel Image Classification":
            num_classes = len(CLASS_NAMES_DATASET_INTEL)
            folder_name = "intel"
        elif dataset_choice == "MIT Indoor Scenes":
            num_classes = len(CLASS_NAMES_DATASET_MIT)
            folder_name = "mit"
        else:
            return None, "Dataset không hợp lệ. Vui lòng chọn lại."
        model = get_pretrained_model(model_name, num_classes)
        
        state_dict_path = os.path.join(folder_name, f"{model_name}.pth")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()  
            return model, None
        else:
            return None, f"Không tìm thấy file weights cho model {model_name} trong dataset {dataset_choice}. Vui lòng kiểm tra lại."
    except Exception as e:
        return None, f"Lỗi khi tải model: {str(e)}"

def predict_image(image, model_name, dataset_choice):
    """
    Hàm dự đoán ảnh
    """
    if image is None:
        return "Vui lòng tải lên một ảnh"
    
    try:
        # Load model
        model, error = load_model(model_name, dataset_choice)
        if error:
            return error
            
        # Preprocessing
        transform = get_transform()
        
        # Convert PIL Image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Transform image
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map class number to class name
        class_names = CLASS_NAMES_DATASET_INTEL if dataset_choice == "Intel Image Classification" else CLASS_NAMES_DATASET_MIT
        predicted_class_name = class_names.get(predicted_class, f"Unknown_Class_{predicted_class}")
        
        # Format kết quả
        result = f"""
        🎯 **Kết quả phân loại:**\n\n
        
        📋 **Model:** {model_name}\n
        📊 **Dataset:** {dataset_choice}\n
        🏷️ **Class dự đoán:** {predicted_class_name}\n
        📈 **Độ tin cậy:** {confidence:.2%}\n
        """
        
        return result
        
    except Exception as e:
        return f"Lỗi trong quá trình dự đoán: {str(e)}"

def create_interface():
    """
    Tạo giao diện Gradio
    """
    with gr.Blocks(title="Image Classification App", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("# 🖼️ Ứng Dụng Phân Loại Ảnh")
        gr.Markdown("Tải lên ảnh và chọn model để phân loại")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                image_input = gr.Image(
                    type="pil",
                    label="📷 Tải lên ảnh cần phân loại"
                )
                
                model_dropdown = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value="ResNet50",
                    label="🤖 Chọn Model",
                    info="Chọn kiến trúc model để sử dụng"
                )
                
                dataset_radio = gr.Radio(
                    choices=["Intel Image Classification", "MIT Indoor Scenes"],
                    value="Intel Image Classification", 
                    label="📊 Chọn Dataset",
                    info="Chọn dataset tương ứng với model đã train"
                )
                
                predict_btn = gr.Button(
                    "🚀 Phân Loại", 
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                # Output component
                result_output = gr.Markdown(
                    label="📋 Kết Quả",
                    value="Kết quả sẽ hiển thị ở đây..."
                )
        
        # Examples (nếu có)
        
        # Event handling
        predict_btn.click(
            fn=predict_image,
            inputs=[image_input, model_dropdown, dataset_radio],
            outputs=result_output
        )
        
        # Thông tin thêm
        with gr.Accordion("Thông tin chi tiết", open=False):
            gr.Markdown("""
            ### Các Model hỗ trợ:
            - **VGG16**
            - **ResNet18/34/50/101**
            - **MobileNet**
            - **ViT**
            
            ### Dataset:
            - **Intel Image Classification**
            - **MIT Indoor Scene**
            
            ### Lưu ý:
            - File weights phải có tên theo format: `{ModelName}.pth`, ví dụ: `ResNet50.pth`
            - Ảnh đầu vào sẽ được resize về 224x224 pixels
            - Kết quả bao gồm class dự đoán và độ tin cậy
            """)
    
    return demo

if __name__ == "__main__":
    # Tạo và chạy ứng dụng
    demo = create_interface()
    demo.launch(
        share=True,  
        server_port=7860,  
        show_error=True  
    )