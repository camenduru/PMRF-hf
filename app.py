import os
if os.getenv('SPACES_ZERO_GPU') == "true":
    os.environ['SPACES_ZERO_GPU'] = "1"
os.environ['K_DIFFUSION_USE_COMPILE'] = "0"
import spaces
import cv2
import gradio as gr
import random
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils import img2tensor, tensor2img
from gradio_imageslider import ImageSlider
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from realesrgan.utils import RealESRGANer

from lightning_models.mmse_rectified_flow import MMSERectifiedFlow

torch.set_grad_enabled(False)

MAX_SEED = 1000000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs('pretrained_models', exist_ok=True)
realesr_model_path = 'pretrained_models/RealESRGAN_x4plus.pth'
if not os.path.exists(realesr_model_path):
    os.system(
        "wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -O pretrained_models/RealESRGAN_x4plus.pth")

# background enhancer with RealESRGAN
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=4, model_path=realesr_model_path, model=model, tile=400, tile_pad=10, pre_pad=0, half=half)

pmrf = MMSERectifiedFlow.from_pretrained('ohayonguy/PMRF_blind_face_image_restoration').to(device=device)

face_helper_dummy = FaceRestoreHelper(
    1,
    face_size=512,
    crop_ratio=(1, 1),
    det_model='retinaface_resnet50',
    save_ext='png',
    use_parse=True,
    device=device,
    model_rootpath=None)

os.makedirs('output', exist_ok=True)


def generate_reconstructions(pmrf_model, x, y, non_noisy_z0, num_flow_steps, device):
    source_dist_samples = pmrf_model.create_source_distribution_samples(x, y, non_noisy_z0)
    dt = (1.0 / num_flow_steps) * (1.0 - pmrf_model.hparams.eps)
    x_t_next = source_dist_samples.clone()
    t_one = torch.ones(x.shape[0], device=device)
    for i in range(num_flow_steps):
        num_t = (i / num_flow_steps) * (1.0 - pmrf_model.hparams.eps) + pmrf_model.hparams.eps
        v_t_next = pmrf_model(x_t=x_t_next, t=t_one * num_t, y=y).to(x_t_next.dtype)
        x_t_next = x_t_next.clone() + v_t_next * dt

    return x_t_next.clip(0, 1).to(torch.float32)

@torch.inference_mode()
@spaces.GPU()
def enhance_face(img, face_helper, has_aligned, num_flow_steps, only_center_face=False, paste_back=True, scale=2):
    face_helper.clean_all()
    if has_aligned:  # the inputs are already aligned
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_helper.cropped_faces = [img]
    else:
        face_helper.read_image(img)
        face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
        # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
        # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
        # align and warp each face
        face_helper.align_warp_face()
    # face restoration
    for cropped_face in face_helper.cropped_faces:
        # prepare data
        h, w = cropped_face.shape[0], cropped_face.shape[1]
        cropped_face = cv2.resize(cropped_face, (512, 512), interpolation=cv2.INTER_LINEAR)
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        dummy_x = torch.zeros_like(cropped_face_t)
        # with torch.autocast("cuda", dtype=torch.bfloat16):
        output = generate_reconstructions(pmrf, dummy_x, cropped_face_t, None, num_flow_steps, device)
        restored_face = tensor2img(output.to(torch.float32).squeeze(0), rgb2bgr=True, min_max=(0, 1))
        # restored_face = cropped_face
        restored_face = cv2.resize(restored_face, (h, w), interpolation=cv2.INTER_LINEAR)


        restored_face = restored_face.astype('uint8')
        face_helper.add_restored_face(restored_face)

    if not has_aligned and paste_back:
        # upsample the background
        if upsampler is not None:
            # Now only support RealESRGAN for upsampling background
            bg_img = upsampler.enhance(img, outscale=scale)[0]
        else:
            bg_img = None

        face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img)
        return face_helper.cropped_faces, face_helper.restored_faces, restored_img
    else:
        return face_helper.cropped_faces, face_helper.restored_faces, None


@torch.inference_mode()
@spaces.GPU()
def inference(seed, randomize_seed, img, aligned, scale, num_flow_steps):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    torch.manual_seed(seed)
    if scale > 4:
        scale = 4  # avoid too large scale value
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:  # for gray inputs
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[0:2]
    if h > 4500 or w > 4500:
        print('Image size too large.')
        return None, None

    if h < 300:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    face_helper = FaceRestoreHelper(
        scale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device,
        model_rootpath=None)

    has_aligned = True if aligned == 'Yes' else False
    _, restored_aligned, restored_img = enhance_face(img, face_helper, has_aligned, only_center_face=False,
                                                     paste_back=True, num_flow_steps=num_flow_steps, scale=scale)
    if has_aligned:
        output = restored_aligned[0]
    else:
        output = restored_img

    save_path = f'output/out.png'
    cv2.imwrite(save_path, output)

    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    orig_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_input = cv2.resize(orig_input, (output.shape[0], output.shape[1]), interpolation=cv2.INTER_LINEAR)
    return [[orig_input, output, seed], save_path]

intro = """
<h2 style="font-weight: 1400; text-align: center; margin-bottom: 7px;">Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration</h2>
<h3 style="margin-bottom: 10px; text-align: center;">
    <a href="https://arxiv.org/abs/2410.00418">[Paper]</a>&nbsp;|&nbsp;
    <a href="https://pmrf-ml.github.io/">[Project Page]</a>&nbsp;|&nbsp;
    <a href="https://github.com/ohayonguy/PMRF">[Code]</a>
</h3>
"""
markdown_top = """
Gradio demo for the blind face image restoration version of [Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration](https://arxiv.org/abs/2410.00418). 

Please refer to our project's page for more details: https://pmrf-ml.github.io/.

---

You may use this demo to enhance the quality of any image which contains faces.

1. If your input image has only one face and it is aligned, please mark "Yes" to the answer below. 
2. Otherwise, your image may contain any number of faces (>=1), and the quality of each face will be enhanced separately.

*Notes*: 

1. Our model is designed to restore aligned face images, but here we incorporate mechanisms that allow restoring the quality of any image that contains any number of faces. Thus, the resulting quality of such general images is not guaranteed.
2. Images that are too large won't work due to memory constraints.
"""

#
# title = "Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration"
#
# description = r"""
# Gradio demo for the blind face image restoration version of <a href='https://arxiv.org/abs/2410.00418' target='_blank'><b>Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration</b></a>.
#
# Please refer to our project's page for more details: https://pmrf-ml.github.io/.
#
# ---
#
# You may use this demo to enhance the quality of any image which contains faces.
#
# 1. If your input image has only one face and it is aligned, please mark "Yes" to the answer below.
# 2. Otherwise, your image may contain any number of faces (>=1), and the quality of each face will be enhanced separately.
#
# <b>NOTEs</b>:
#
# 1. Our model is designed to restore aligned face images, but here we incorporate mechanisms that allow restoring the quality of any image that contains any number of faces. Thus, the resulting quality of such general images is not guaranteed.
# 2. Images that are too large won't work due to memory constraints.
# """


article = r"""

If you find our work useful, please help to ⭐ our <a href='https://github.com/ohayonguy/PMRF' target='_blank'>GitHub repository</a>. Thanks! 
[![GitHub Stars](https://img.shields.io/github/stars/ohayonguy/PMRF?style=social)](https://github.com/ohayonguy/PMRF)

📝 **Citation**

If our work is useful for your research, please consider citing:
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
Redistribution and use for non-commercial purposes should follow this license.

📧 **Contact**

If you have any questions, please feel free to contact me at <b>guyoep@gmail.com</b>.
"""

css = """
#col-container {
    margin: 0 auto;
    max-width: 512px;
}
"""



with gr.Blocks(css=css) as demo:
    gr.HTML(intro)
    gr.Markdown(markdown_top)

    with gr.Row():
        run_button = gr.Button(value="Run")

    with gr.Row():
        with gr.Column(scale=4):
            input_im = gr.Image(label="Input Image", type="pil")
        with gr.Column(scale=1):
            num_inference_steps = gr.Slider(
                label="Number of Inference Steps",
                minimum=1,
                maximum=200,
                step=1,
                value=25,
            )
            upscale_factor = gr.Slider(
                label="Scale factor for the background upsampler. Applicable only to non-aligned face images.",
                minimum=1,
                maximum=4,
                step=0.1,
                value=1,
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=42,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            aligned = gr.Checkbox(label="The input is an aligned face image", value=True)

    with gr.Row():
        result = ImageSlider(label="Input / Output", type="numpy", interactive=True)
    with gr.Row():
        file = gr.File(label="Download the output image")

    # examples = gr.Examples(
    #     examples=[
    #     #    [42, False, "examples/image_1.jpg", 28, 4, 0.6],
    #     #     [42, False, "examples/image_2.jpg", 28, 4, 0.6],
    #     #    [42, False, "examples/image_3.jpg", 28, 4, 0.6],
    #     #     [42, False, "examples/image_4.jpg", 28, 4, 0.6],
    #     #    [42, False, "examples/image_5.jpg", 28, 4, 0.6],
    #     #    [42, False, "examples/image_6.jpg", 28, 4, 0.6],
    #     ],
    #     inputs=[
    #         seed,
    #         randomize_seed,
    #         input_im,
    #         num_inference_steps,
    #         upscale_factor,
    #         controlnet_conditioning_scale,
    #     ],
    #     fn=infer,
    #     outputs=result,
    #     cache_examples="lazy",
    # )

    # examples = gr.Examples(
    #     examples=[
    #         #[42, False, "examples/image_1.jpg", 28, 4, 0.6],
    #         [42, False, "examples/image_2.jpg", 28, 4, 0.6],
    #         #[42, False, "examples/image_3.jpg", 28, 4, 0.6],
    #         #[42, False, "examples/image_4.jpg", 28, 4, 0.6],
    #         [42, False, "examples/image_5.jpg", 28, 4, 0.6],
    #         [42, False, "examples/image_6.jpg", 28, 4, 0.6],
    #         [42, False, "examples/image_7.jpg", 28, 4, 0.6],
    #     ],
    #     inputs=[
    #         seed,
    #         randomize_seed,
    #         input_im,
    #         num_inference_steps,
    #         upscale_factor,
    #         controlnet_conditioning_scale,
    #     ],
    # )


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
        outputs=[result, file],
        show_api=False,
        # show_progress="minimal",
    )


# demo = gr.Interface(
#     inference, [
#         gr.Image(type="filepath", label="Input"),
#         gr.Radio(['Yes', 'No'], type="value", value='aligned', label='Is the input an aligned face image?'),
#         gr.Slider(label="Scale factor for the background upsampler. Applicable only to non-aligned face images.", minimum=1, maximum=4, value=2, step=0.1, interactive=True),
#         gr.Number(label="Number of flow steps. A higher value should result in better image quality, but will inference will take a longer time.", value=25),
#     ], [
#         gr.ImageSlider(type="numpy", label="Input / Output", interactive=True),
#         gr.File(label="Download the output image")
#     ],
#     title=title,
#     description=description,
#     article=article,
# )


demo.queue()
demo.launch(state_session_capacity=15, show_api=False, share=False)