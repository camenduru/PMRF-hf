# Some of the implementations below are adopted from
# https://huggingface.co/spaces/sczhou/CodeFormer and https://huggingface.co/spaces/wzhouxiff/RestoreFormerPlusPlus
import os

import matplotlib.pyplot as plt

if os.getenv("SPACES_ZERO_GPU") == "true":
    os.environ["SPACES_ZERO_GPU"] = "1"
os.environ["K_DIFFUSION_USE_COMPILE"] = "0"

import spaces
import cv2
from tqdm import tqdm
import gradio as gr
import random
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from realesrgan.utils import RealESRGANer

from lightning_models.mmse_rectified_flow import MMSERectifiedFlow

MAX_SEED = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("pretrained_models", exist_ok=True)
realesr_model_path = "pretrained_models/RealESRGAN_x4plus.pth"
if not os.path.exists(realesr_model_path):
    os.system(
        "wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -O pretrained_models/RealESRGAN_x4plus.pth"
    )


# # background enhancer with RealESRGAN
# model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
# half = True if torch.cuda.is_available() else False
# upsampler = RealESRGANer(scale=4, model_path=realesr_model_path, model=model, tile=400, tile_pad=10, pre_pad=0,
#                          half=half)


def set_realesrgan():
    use_half = False
    if torch.cuda.is_available():  # set False in CPU/MPS mode
        no_half_gpu_list = ["1650", "1660"]  # set False for GPUs that don't support f16
        if not True in [
            gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list
        ]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=use_half,
    )
    return upsampler


upsampler = set_realesrgan()
pmrf = MMSERectifiedFlow.from_pretrained(
    "ohayonguy/PMRF_blind_face_image_restoration"
).to(device=device)


def generate_reconstructions(pmrf_model, x, y, non_noisy_z0, num_flow_steps, device):
    source_dist_samples = pmrf_model.create_source_distribution_samples(
        x, y, non_noisy_z0
    )
    dt = (1.0 / num_flow_steps) * (1.0 - pmrf_model.hparams.eps)
    x_t_next = source_dist_samples.clone()
    t_one = torch.ones(x.shape[0], device=device)
    for i in tqdm(range(num_flow_steps)):
        num_t = (i / num_flow_steps) * (
            1.0 - pmrf_model.hparams.eps
        ) + pmrf_model.hparams.eps
        v_t_next = pmrf_model(x_t=x_t_next, t=t_one * num_t, y=y).to(x_t_next.dtype)
        x_t_next = x_t_next.clone() + v_t_next * dt

    return x_t_next.clip(0, 1)


def resize(img, size):
    # From https://github.com/sczhou/CodeFormer/blob/master/facelib/utils/face_restoration_helper.py
    h, w = img.shape[0:2]
    scale = size / min(h, w)
    h, w = int(h * scale), int(w * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


@torch.inference_mode()
@spaces.GPU()
def enhance_face(img, face_helper, has_aligned, num_flow_steps, scale=2):
    face_helper.clean_all()
    if has_aligned:  # The inputs are already aligned
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_helper.cropped_faces = [img]
    else:
        face_helper.read_image(img)
        face_helper.input_img = resize(face_helper.input_img, 640)
        face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
        face_helper.align_warp_face()
    if len(face_helper.cropped_faces) == 0:
        raise gr.Error("Could not identify any face in the image.")
    if has_aligned and len(face_helper.cropped_faces) > 1:
        raise gr.Error(
            "You marked that the input image is aligned, but multiple faces were detected."
        )

    # face restoration
    for i, cropped_face in tqdm(enumerate(face_helper.cropped_faces)):
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        output = generate_reconstructions(
            pmrf,
            torch.zeros_like(cropped_face_t),
            cropped_face_t,
            None,
            num_flow_steps,
            device,
        )
        restored_face = tensor2img(
            output.to(torch.float32).squeeze(0), rgb2bgr=True, min_max=(0, 1)
        )
        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face)

    if not has_aligned:
        # upsample the background
        # Now only support RealESRGAN for upsampling background
        bg_img = upsampler.enhance(img, outscale=scale)[0]
        face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img)
        return face_helper.cropped_faces, face_helper.restored_faces, restored_img
    else:
        return face_helper.cropped_faces, face_helper.restored_faces, None


@torch.inference_mode()
@spaces.GPU()
def inference(
    seed,
    randomize_seed,
    img,
    aligned,
    scale,
    num_flow_steps,
    progress=gr.Progress(track_tqdm=True),
):
    if img is None:
        raise gr.Error("Please upload an image before submitting.")
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    torch.manual_seed(seed)
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    h, w = img.shape[0:2]
    if h > 4500 or w > 4500:
        raise gr.Error("Image size too large.")

    face_helper = FaceRestoreHelper(
        scale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=True,
        device=device,
        model_rootpath=None,
    )

    has_aligned = aligned
    cropped_face, restored_faces, restored_img = enhance_face(
        img, face_helper, has_aligned, num_flow_steps=num_flow_steps, scale=scale
    )
    if has_aligned:
        output = restored_faces[0]
    else:
        output = restored_img

    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    for i, restored_face in enumerate(restored_faces):
        restored_faces[i] = cv2.cvtColor(restored_face, cv2.COLOR_BGR2RGB)
    torch.cuda.empty_cache()
    return output, restored_faces if len(restored_faces) > 1 else None


intro = """
<h1 style="font-weight: 1400; text-align: center; margin-bottom: 7px;">Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration</h1>
<h3 style="margin-bottom: 10px; text-align: center;">
    <a href="https://arxiv.org/abs/2410.00418">[Paper]</a>&nbsp;|&nbsp;
    <a href="https://pmrf-ml.github.io/">[Project Page]</a>&nbsp;|&nbsp;
    <a href="https://github.com/ohayonguy/PMRF">[Code]</a>
</h3>
"""
markdown_top = """
Gradio demo for the blind face image restoration version of [Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration](https://arxiv.org/abs/2410.00418). 
You may use this demo to enhance the quality of any image which contains faces.

Please refer to our project's page for more details: https://pmrf-ml.github.io/.

*Notes* : 

1. Our model is designed to restore aligned face images, where there is *only one* face in the image, and the face is centered and aligned. Here, however, we incorporate mechanisms that allow restoring the quality of *any* image that contains *any* number of faces. Thus, the resulting quality of such general images is not guaranteed.
2. If the faces in your image are not aligned, make sure that the checkbox "The input is an aligned face image" in *not* marked.
3. Too large images may result in out-of-memory error.

---
"""

article = r"""

If you find our work useful, please help to ⭐ our <a href='https://github.com/ohayonguy/PMRF' target='_blank'>GitHub repository</a>. Thanks! 
[![GitHub Stars](https://img.shields.io/github/stars/ohayonguy/PMRF?style=social)](https://github.com/ohayonguy/PMRF)

📝 **Citation**

```bibtex
@article{ohayon2024pmrf,
  author    = {Guy Ohayon and Tomer Michaeli and Michael Elad},
  title     = {Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration},
  journal   = {arXiv preprint arXiv:2410.00418},
  year      = {2024},
  url       = {https://arxiv.org/abs/2410.00418}
}
```

📋 **License**

This project is released under the <a rel="license" href="https://github.com/ohayonguy/PMRF/blob/master/LICENSE">MIT license</a>.

📧 **Contact**

If you have any questions, please feel free to contact me at <b>guyoep@gmail.com</b>.
"""

css = """
#col-container {
    margin: 0 auto;
    max-width: 512px;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.HTML(intro)
    gr.Markdown(markdown_top)

    with gr.Row():
        with gr.Column(scale=1):
            input_im = gr.Image(label="Input", type="filepath", show_label=True)
        with gr.Column(scale=1):
            result = gr.Image(label="Output", type="numpy", show_label=True, format="png")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                num_inference_steps = gr.Slider(
                    label="Number of Inference Steps", minimum=1, maximum=200, step=1, value=25, scale=1
                )
                upscale_factor = gr.Slider(
                    label="Scale factor (applicable to non-aligned face images)",
                    minimum=1,
                    maximum=4,
                    step=0.1,
                    value=1,
                    scale=1
                )
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42, scale=1)
            with gr.Row():
                with gr.Column(scale=1):
                    run_button = gr.Button(value="Submit", variant="primary")
                with gr.Column(scale=1):
                    clear_button = gr.ClearButton(value="Clear")
            with gr.Row():
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True, scale=1)
                aligned = gr.Checkbox(label="The input is an aligned face image", value=False, scale=1)

        with gr.Column(scale=1):
            gallery = gr.Gallery(
                label="Restored faces gallery", type="numpy", show_label=True, format="png"
            )

    clear_button.add(input_im)
    clear_button.add(result)
    clear_button.add(gallery)

    with gr.Row():
        examples = gr.Examples(
            examples=[
                [42, False, "examples/01.png", False, 1, 25],
                [42, False, "examples/03.jpg", False, 2, 25],
                [42, False, "examples/00000055.png", True, 1, 25],
                [42, False, "examples/00000085.png", True, 1, 25],
                [42, False, "examples/00000113.png", True, 1, 25],
                [42, False, "examples/00000137.png", True, 1, 25],
            ],
            fn=inference,
            inputs=[
                seed,
                randomize_seed,
                input_im,
                aligned,
                upscale_factor,
                num_inference_steps,
            ],
            outputs=[result, gallery],
            cache_examples="lazy",
        )

    gr.Markdown(article)
    gr.on(
        [run_button.click],
        fn=inference,
        inputs=[
            seed,
            randomize_seed,
            input_im,
            aligned,
            upscale_factor,
            num_inference_steps,
        ],
        outputs=[result, gallery],
        # show_api=False,
        # show_progress="minimal",
    )

demo.queue()
demo.launch(state_session_capacity=15)
