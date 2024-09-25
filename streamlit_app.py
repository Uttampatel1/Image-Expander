import streamlit as st
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
from PIL import Image, ImageDraw
import numpy as np

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.pipe = None

@st.cache_resource
def load_models():
    config_file = hf_hub_download(
        "xinsir/controlnet-union-sdxl-1.0",
        filename="config_promax.json",
    )
    config = ControlNetModel_Union.load_config(config_file)
    controlnet_model = ControlNetModel_Union.from_config(config)
    model_file = hf_hub_download(
        "xinsir/controlnet-union-sdxl-1.0",
        filename="diffusion_pytorch_model_promax.safetensors",
    )
    state_dict = load_state_dict(model_file)
    model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
        controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
    )
    model.to(device="cuda", dtype=torch.float16)

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    ).to("cuda")

    pipe = StableDiffusionXLFillPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning",
        torch_dtype=torch.float16,
        vae=vae,
        controlnet=model,
        variant="fp16",
    ).to("cuda")

    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    
    return model, pipe

def can_expand(source_width, source_height, target_width, target_height, alignment):
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True

def infer(image, width, height, overlap_width, num_inference_steps, resize_option, custom_resize_size, prompt_input, alignment):
    source = image
    target_size = (width, height)
    overlap = overlap_width

    if source.width < target_size[0] and source.height < target_size[1]:
        scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
        new_width = int(source.width * scale_factor)
        new_height = int(source.height * scale_factor)
        source = source.resize((new_width, new_height), Image.LANCZOS)

    if source.width > target_size[0] or source.height > target_size[1]:
        scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
        new_width = int(source.width * scale_factor)
        new_height = int(source.height * scale_factor)
        source = source.resize((new_width, new_height), Image.LANCZOS)
    
    if resize_option == "Full":
        resize_size = max(source.width, source.height)
    elif resize_option == "1/2":
        resize_size = max(source.width, source.height) // 2
    elif resize_option == "1/3":
        resize_size = max(source.width, source.height) // 3
    elif resize_option == "1/4":
        resize_size = max(source.width, source.height) // 4
    else:  # Custom
        resize_size = custom_resize_size

    aspect_ratio = source.height / source.width
    new_width = resize_size
    new_height = int(resize_size * aspect_ratio)
    source = source.resize((new_width, new_height), Image.LANCZOS)

    if not can_expand(source.width, source.height, target_size[0], target_size[1], alignment):
        alignment = "Middle"

    if alignment == "Middle":
        margin_x = (target_size[0] - source.width) // 2
        margin_y = (target_size[1] - source.height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - source.height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - source.width
        margin_y = (target_size[1] - source.height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - source.width) // 2
        margin_y = 0
    elif alignment == "Bottom":
        margin_x = (target_size[0] - source.width) // 2
        margin_y = target_size[1] - source.height

    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    if alignment == "Middle":
        mask_draw.rectangle([
            (margin_x + overlap, margin_y + overlap),
            (margin_x + source.width - overlap, margin_y + source.height - overlap)
        ], fill=0)
    elif alignment == "Left":
        mask_draw.rectangle([
            (margin_x, margin_y),
            (margin_x + source.width - overlap, margin_y + source.height)
        ], fill=0)
    elif alignment == "Right":
        mask_draw.rectangle([
            (margin_x + overlap, margin_y),
            (margin_x + source.width, margin_y + source.height)
        ], fill=0)
    elif alignment == "Top":
        mask_draw.rectangle([
            (margin_x, margin_y),
            (margin_x + source.width, margin_y + source.height - overlap)
        ], fill=0)
    elif alignment == "Bottom":
        mask_draw.rectangle([
            (margin_x, margin_y + overlap),
            (margin_x + source.width, margin_y + source.height)
        ], fill=0)

    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)

    final_prompt = f"{prompt_input} , high quality, 4k"

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = st.session_state.pipe.encode_prompt(final_prompt, "cuda", True)

    # image = st.session_state.pipe(
    #     prompt_embeds=prompt_embeds,
    #     negative_prompt_embeds=negative_prompt_embeds,
    #     pooled_prompt_embeds=pooled_prompt_embeds,
    #     negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    #     image=cnet_image,
    #     num_inference_steps=num_inference_steps
    # )
    
    generated_images = []
    for image in st.session_state.pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
        num_inference_steps=num_inference_steps
    ):
        generated_images.append(image)
    
    

    image = generated_images[-1].convert("RGBA")
    cnet_image.paste(image, (0, 0), mask)

    return background, cnet_image

def main():
    st.title("Image Expander 📸")
    # st.text("App created by Uttam Patel 👨‍💻")
    st.write("This app uses AI to expand your images to desired ratios while maintaining the original content. 🤯")

    # Load models
    if st.session_state.model is None or st.session_state.pipe is None:
        with st.spinner("Loading models... ⏳"):
            st.session_state.model, st.session_state.pipe = load_models()

    # Input image
    input_image = st.file_uploader("Upload Input Image", type=["png", "jpg", "jpeg", "webp"])

    # Sidebar for settings
    st.sidebar.header("Settings ⚙️")
    target_ratio = st.sidebar.radio("Expected Ratio", ["9:16", "16:9", "1:1", "Custom"])
    
    if target_ratio == "9:16":
        width, height = 720, 1280
    elif target_ratio == "16:9":
        width, height = 1280, 720
    elif target_ratio == "1:1":
        width, height = 1024, 1024
    else:
        width = st.sidebar.slider("Width", 720, 1536, 720, 8)
        height = st.sidebar.slider("Height", 720, 1536, 1280, 8)

    alignment = st.sidebar.selectbox("Alignment", ["Middle", "Left", "Right", "Top", "Bottom"])
    prompt_input = st.sidebar.text_input("Prompt (Optional)")
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings 🔧"):
        num_inference_steps = st.slider("Steps", 4, 12, 8)
        overlap_width = st.slider("Mask overlap width", 1, 50, 42)
        resize_option = st.radio("Resize input image", ["Full", "1/2 ", "1/3", "1/4", "Custom"])
        custom_resize_size = st.slider("Custom resize size", 64, 1024, 512, 8, 
                                       disabled=(resize_option != "Custom"))

    if input_image:
        image = Image.open(input_image)
        st.image(image, caption="Input Image", use_column_width=True)

        if st.button("Generate 🔮"):
            with st.spinner("Generating... ⏳"):
                background, result = infer(image, width, height, overlap_width, num_inference_steps,
                                           resize_option, custom_resize_size, prompt_input, alignment)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(background, caption="Background", use_column_width=True)
                with col2:
                    st.image(result, caption="Generated Image", use_column_width=True)
                    

if __name__ == "__main__":
    main()
