import os

import cv2
import gradio as gr
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from realesrgan.utils import RealESRGANer
import spaces

from lightning_models.mmse_rectified_flow import MMSERectifiedFlow

torch.set_grad_enabled(False)

if os.getenv('SPACES_ZERO_GPU') == "true":
    os.environ['SPACES_ZERO_GPU'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists('pretrained_models'):
    os.makedirs('pretrained_models')
realesr_model_path = 'pretrained_models/RealESRGAN_x4plus.pth'
if not os.path.exists(realesr_model_path):
    os.system(
        "wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -O experiments/pretrained_models/RealESRGAN_x4plus.pth")

pmrf_model_path = 'blind_face_restoration_pmrf.ckpt'

# background enhancer with RealESRGAN
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=4, model_path=realesr_model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

pmrf = MMSERectifiedFlow.load_from_checkpoint('./blind_face_restoration_pmrf.ckpt',
                                              mmse_model_arch='swinir_L',
                                              mmse_model_ckpt_path=None,
                                              map_location='cpu').to(device)

os.makedirs('output', exist_ok=True)


@torch.inference_mode()
@spaces.GPU()
def enhance_face(img, face_helper, has_aligned, only_center_face=False, paste_back=True, scale=2):
    face_helper.clean_all()

    if has_aligned:  # the inputs are already aligned
        img = cv2.resize(img, (512, 512))
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
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        try:
            dummy_x = torch.zeros_like(cropped_face_t)
            output = pmrf.generate_reconstructions(dummy_x, cropped_face_t, None, 25, device)
            restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(0, 1))
        except RuntimeError as error:
            print(f'\tFailed inference for RestoreFormer: {error}.')
            restored_face = cropped_face

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
def inference(img, aligned, scale):
    if scale > 4:
        scale = 4  # avoid too large scale value
    try:

        extension = os.path.splitext(os.path.basename(str(img)))[1]
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        elif len(img.shape) == 2:  # for gray inputs
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if h > 3500 or w > 3500:
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

        try:
            has_aligned = True if aligned == 'aligned' else False
            _, restored_aligned, restored_img = enhance_face(img, face_helper, has_aligned, only_center_face=False,
                                                             paste_back=True)
            if has_aligned:
                output = restored_aligned[0]
            else:
                output = restored_img
        except RuntimeError as error:
            print('Error', error)

        try:
            if scale != 2:
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                h, w = img.shape[0:2]
                output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
        except Exception as error:
            print('wrong scale input.', error)
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        else:
            extension = 'jpg'
        save_path = f'output/out.{extension}'
        cv2.imwrite(save_path, output)

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output, save_path
    except Exception as error:
        print('global exception', error)
        return None, None


css = r"""
"""

demo = gr.Interface(
    inference, [
        gr.Image(type="filepath", label="Input"),
        gr.Radio(['aligned', 'unaligned'], type="value", value='unaligned', label='Image Alignment'),
        gr.Number(label="Rescaling factor", value=2),
    ], [
        gr.Image(type="numpy", label="Output (The whole image)"),
        gr.File(label="Download the output image")
    ],
)