import argparse
import cv2
import glob
import math
import numpy as np
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from torch.nn import functional as F


# ------------------------------ Utility: Self-Ensemble (x8 TTA) ------------------------------ #
# Rotations/flips inspired by common ESRGAN/EDSR test-time augmentation.
# Each transform returns a pair of (forward, inverse) callables.

def _tta_transforms():
    def identity(x):
        return x

    def flip_h(x):
        return torch.flip(x, dims=[-1])

    def flip_v(x):
        return torch.flip(x, dims=[-2])

    def rot90(x):
        return torch.rot90(x, k=1, dims=[-2, -1])

    def rot180(x):
        return torch.rot90(x, k=2, dims=[-2, -1])

    def rot270(x):
        return torch.rot90(x, k=3, dims=[-2, -1])

    # pairs: (forward, inverse)
    ops = [
        (identity, identity),
        (flip_h, flip_h),
        (flip_v, flip_v),
        (rot90, rot270),
        (rot180, rot180),
        (rot270, rot90),
        (lambda x: rot90(flip_h(x)), lambda x: flip_h(rot270(x))),
        (lambda x: rot90(flip_v(x)), lambda x: flip_v(rot270(x))),
    ]
    return ops


# ------------------------------ Main Script ------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/RealESRGAN_x4plus.pth', help='Path to the pre-trained model')
    parser.add_argument('--scale', type=int, default=4, help='Upsample scale factor')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--alpha_upsampler', type=str, default='realesrgan', help='Upsampler for alpha channels: realesrgan | bicubic')
    parser.add_argument('--ext', type=str, default='auto', help='Image extension: auto | jpg | png (auto keeps original)')

    # Quality / post-processing knobs
    parser.add_argument('--tta', action='store_true', help='Enable x8 test-time augmentation (better quality, slower)')
    parser.add_argument('--sharpness', type=float, default=0.0, help='Unsharp mask amount (0 to disable). Typical 0.15~0.35')
    parser.add_argument('--denoise', type=float, default=0.0, help='Non-local means denoise strength (0 to disable). Typical 3~7')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma correction on output luma (e.g., 0.95~1.05). 1.0 disables')

    # Precision / performance
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--half', action='store_true', help='Use half precision inference (faster on RTX, may slightly reduce quality)')
    group.add_argument('--fp32', action='store_true', help='Force float32 precision (default)')

    # Save params
    parser.add_argument('--jpeg_quality', type=int, default=95, help='JPEG quality (if saving .jpg)')
    parser.add_argument('--png_compression', type=int, default=3, help='PNG compression level 0-9 (lower=faster/larger)')

    args = parser.parse_args()

    upsampler = RealESRGANer(
        scale=args.scale,
        model_path=args.model_path,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        use_half=args.half,
        use_tta=args.tta,
    )

    os.makedirs('results/', exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        # ------------------------------ read image ------------------------------ #
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'\tWarning: failed to read {path}. Skipping.')
            continue
        if img.dtype == np.uint16 or np.max(img) > 255:
            max_range = 65535.0
            print('\tInput is a 16-bit image')
        else:
            max_range = 255.0
        img = img.astype(np.float32) / max_range

        if img.ndim == 2:  # gray
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            bgr = img[:, :, :3]
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if args.alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ------------------- process image (without the alpha channel) ------------------- #
        upsampler.pre_process(img)
        upsampler.process()
        output_img = upsampler.post_process().data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))  # RGB->BGR
        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            if args.alpha_upsampler == 'realesrgan':
                upsampler.pre_process(alpha)
                upsampler.process()
                output_alpha = upsampler.post_process().data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:
                h, w = alpha.shape[:2]
                output_alpha = cv2.resize(alpha, (w * args.scale, h * args.scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ optional post-processing ------------------------------ #
        if args.gamma != 1.0:
            if img_mode == 'RGBA':
                bgr = output_img[:, :, :3].astype(np.float32)
                bgr = np.clip(bgr, 0, 1)
                # gamma on luma to avoid hue shift
                ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
                y = np.clip(ycrcb[:, :, 0], 0, 1)
                y = np.power(y, args.gamma)
                ycrcb[:, :, 0] = y
                bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                output_img[:, :, :3] = bgr
            else:
                bgr = np.clip(output_img.astype(np.float32), 0, 1)
                ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
                y = np.clip(ycrcb[:, :, 0], 0, 1)
                y = np.power(y, args.gamma)
                ycrcb[:, :, 0] = y
                output_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        if args.sharpness > 0.0:
            # Unsharp masking: out = (1 + a)*img - a*blur(img)
            # Kernel auto from scale; clip to [0,1]
            k = max(3, int(round(3 * args.scale)) | 1)  # odd
            if img_mode == 'RGBA':
                base = output_img[:, :, :3].astype(np.float32)
                blur = cv2.GaussianBlur(base, (k, k), sigmaX=0)
                base = cv2.addWeighted(base, 1 + args.sharpness, blur, -args.sharpness, 0)
                base = np.clip(base, 0, 1)
                output_img[:, :, :3] = base
            else:
                base = output_img.astype(np.float32)
                blur = cv2.GaussianBlur(base, (k, k), sigmaX=0)
                output_img = cv2.addWeighted(base, 1 + args.sharpness, blur, -args.sharpness, 0)
                output_img = np.clip(output_img, 0, 1)

        # Denoise AFTER sharpening if requested (expects uint8 BGR/BGRA)
        if args.denoise > 0.0:
            if img_mode == 'RGBA':
                rgb = (np.clip(output_img[:, :, :3], 0, 1) * 255.0).round().astype(np.uint8)
                den = cv2.fastNlMeansDenoisingColored(rgb, None, h=args.denoise, hColor=args.denoise, templateWindowSize=7, searchWindowSize=21)
                output_img[:, :, :3] = den.astype(np.float32) / 255.0
            elif output_img.ndim == 3:
                bgr8 = (np.clip(output_img, 0, 1) * 255.0).round().astype(np.uint8)
                den = cv2.fastNlMeansDenoisingColored(bgr8, None, h=args.denoise, hColor=args.denoise, templateWindowSize=7, searchWindowSize=21)
                output_img = den.astype(np.float32) / 255.0

        # ------------------------------ save image ------------------------------ #
        if args.ext == 'auto':
            extension = extension[1:].lower()
        else:
            extension = args.ext.lower()
        if img_mode == 'RGBA':
            extension = 'png'  # preserve alpha

        save_path = f'results/{imgname}_{args.suffix}.{extension}'
        if img_mode == 'RGBA':
            # output_img in BGRA float [0,1]
            if output_img.dtype != np.float32:
                output_img = output_img.astype(np.float32)
            out_u8 = (np.clip(output_img, 0, 1) * 255.0).round().astype(np.uint8)
            cv2.imwrite(save_path, out_u8, [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression])
        else:
            # BGR float [0,1] or GRAY
            if max_range > 255.0:
                out_u16 = (np.clip(output_img, 0, 1) * 65535.0).round().astype(np.uint16)
                if extension == 'png':
                    cv2.imwrite(save_path, out_u16, [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression])
                else:
                    # Fallback to PNG for 16-bit
                    alt = f'results/{imgname}_{args.suffix}.png'
                    cv2.imwrite(alt, out_u16, [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression])
                    print(f"\tSaved as 16-bit PNG instead of {extension}: {alt}")
            else:
                out_u8 = (np.clip(output_img, 0, 1) * 255.0).round().astype(np.uint8)
                if extension in ['jpg', 'jpeg']:
                    cv2.imwrite(save_path, out_u8, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                elif extension == 'png':
                    cv2.imwrite(save_path, out_u8, [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression])
                else:
                    cv2.imwrite(save_path, out_u8)


class RealESRGANer:
    def __init__(self, scale, model_path, tile=0, tile_pad=10, pre_pad=10, use_half=False, use_tta=False):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.use_half = use_half and torch.cuda.is_available()
        self.use_tta = use_tta

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        loadnet = torch.load(model_path, map_location='cpu')
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        elif 'params' in loadnet:
            keyname = 'params'
        else:
            # allow loading a plain state_dict
            keyname = None
        if keyname is None:
            model.load_state_dict(loadnet, strict=True)
        else:
            model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.to(self.device)
        if self.use_half:
            self.model.half()

    def pre_process(self, img):
        # img: float32 numpy RGB in [0,1]
        tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))  # C,H,W
        tensor = tensor.unsqueeze(0)  # B,C,H,W
        if self.use_half:
            tensor = tensor.half()
        else:
            tensor = tensor.float()
        self.img = tensor.to(self.device)

        # symmetric pre-pad to reduce border ringing
        if self.pre_pad > 0:
            self.img = F.pad(self.img, (self.pre_pad, self.pre_pad, self.pre_pad, self.pre_pad), mode='reflect')

        # Make spatial dims divisible by scale to avoid resize artifacts
        self.mod_scale = self.scale if self.scale > 1 else 1
        if self.mod_scale is not None and self.mod_scale > 1:
            _, _, h, w = self.img.size()
            pad_h = (self.mod_scale - (h % self.mod_scale)) % self.mod_scale
            pad_w = (self.mod_scale - (w % self.mod_scale)) % self.mod_scale
            if pad_h or pad_w:
                self.mod_pad_h, self.mod_pad_w = pad_h, pad_w
                self.img = F.pad(self.img, (0, pad_w, 0, pad_h), mode='reflect')
            else:
                self.mod_pad_h = self.mod_pad_w = 0
        else:
            self.mod_pad_h = self.mod_pad_w = 0

    def _forward(self, x):
        with torch.no_grad():
            if self.use_tta:
                outs = []
                for fwd, inv in _tta_transforms():
                    xt = fwd(x)
                    yt = self.model(xt)
                    yt = inv(yt)
                    outs.append(yt)
                y = torch.stack(outs, dim=0).mean(dim=0)
            else:
                y = self.model(x)
        return y

    def process(self):
        try:
            if self.tile_size and self.tile_size > 0:
                self.output = self.tile_process()
            else:
                self.output = self._forward(self.img)
        except Exception as error:
            print('Error', error)
            self.output = self.img

    def tile_process(self):
        """Tile-based inference to reduce VRAM usage, with padding overlap to hide seams."""
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output = self.img.new_zeros((batch, channel, output_height, output_width))

        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size

                in_x0 = ofs_x
                in_x1 = min(ofs_x + self.tile_size, width)
                in_y0 = ofs_y
                in_y1 = min(ofs_y + self.tile_size, height)

                pad_x0 = max(in_x0 - self.tile_pad, 0)
                pad_x1 = min(in_x1 + self.tile_pad, width)
                pad_y0 = max(in_y0 - self.tile_pad, 0)
                pad_y1 = min(in_y1 + self.tile_pad, height)

                input_tile = self.img[:, :, pad_y0:pad_y1, pad_x0:pad_x1]

                # upscale tile with TTA if enabled
                out_tile = self._forward(input_tile)

                out_x0 = in_x0 * self.scale
                out_x1 = in_x1 * self.scale
                out_y0 = in_y0 * self.scale
                out_y1 = in_y1 * self.scale

                out_x0t = (in_x0 - pad_x0) * self.scale
                out_y0t = (in_y0 - pad_y0) * self.scale
                out_x1t = out_x0t + (in_x1 - in_x0) * self.scale
                out_y1t = out_y0t + (in_y1 - in_y0) * self.scale

                output[:, :, out_y0:out_y1, out_x0:out_x1] = out_tile[:, :, out_y0t:out_y1t, out_x0t:out_x1t]

                print(f'\tTile {y * tiles_x + x + 1}/{tiles_x * tiles_y}')

        return output

    def post_process(self):
        # remove mod pad
        if self.mod_scale is not None and self.mod_scale > 1:
            _, _, h, w = self.output.size()
            if self.mod_pad_h or self.mod_pad_w:
                self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove pre-pad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, self.pre_pad * self.scale:h - self.pre_pad * self.scale, self.pre_pad * self.scale:w - self.pre_pad * self.scale]
        return self.output


if __name__ == '__main__':
    main()
