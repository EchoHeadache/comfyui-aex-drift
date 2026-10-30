"""AEX Drift: hold an obvpm Timeline extend to the tone of the clip it continues, for MiniMax H3.

An extend's new frames drift from the frames it holds: shadows darken and spread, mids drop a code or two and contrast grows,
and every later extend continues from the drifted latent. Three nodes take it out of the latent:
AEX Drift First Stage Pins re-encodes a two-stage extend's first-stage pin from the extended clip's saved latent,
AEX Drift Handoff Match grades the second stage's new rows into its held ones before it samples, and
AEX Drift Seam Anchor grades an extend's new rows back to its held rows after the last stage has sampled them.
AEX Drift Check shows on the node how far the decoded take still moved off its held frames, and which knob to try next.
"""

import logging
import os

import torch

import comfy.model_management
import folder_paths
from comfy.latent_formats import MiniMaxH3Video
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import ComfyExtension, io, ui

ObvpmPins = io.Custom("OBVPM_H3_PINS")
ObvpmPinSpecs = io.Custom("OBVPM_H3_PINSPECS")


def first_stage_pins(latent, pins, pin_specs, vae):
    """Re-encode a first stage's extend pin from the extended clip's saved latent instead of its MP4.

    A first stage is smaller than the clip it continues, so obvpm's Pins From Pixels pins it from the clip's file: 8-bit, TV range,
    H.264, read back through swscale's truncating path (about a code darker, half again as many pixels clipped to black) and with
    any grade the save applied. The first stage sets the new frames' tone from that copy while the second stage holds the latent
    itself, so every extend steps away from its parent. Decoded from the latent and squashed in float, only the VAE's round trip is left."""
    pin = next((p for p in pins if p["place"] == "before" and p["spec"]["source_kind"] == "clip_pixels"), None)
    source = next((s["source"] for s in pin_specs if s["place"] == "before" and s["source_kind"] == "clip"), None)
    if pin is None or source is None:
        return latent
    saved = source["video_latent"]
    # the file's frames come after the frames the clip held from its own parent
    start = int(source["meta"].get("pinned_head_frames", 0) or 0) + pin["spec"]["source_start"]
    k = start // 17 * 5
    if start < 0 or start % 17 or k + pin["steps"] > saved.shape[2]:
        logging.warning("AEX Drift: the first stage's pin starts at raw frame %d, off the clip's saved latent grid; it keeps the MP4's frames.", start)
        return latent
    lo = max(0, k - 5)  # one cycle of run-in and the decoder's two steps of lookahead decode the window as the whole clip decodes it
    frames = vae.decode(saved[:, :, lo:k + pin["steps"] + 2])
    frames = frames.reshape((-1,) + frames.shape[-3:])[(k - lo) // 5 * 17:][:pin["covered"]]
    video, audio = latent["samples"].unbind()
    frames = torch.nn.functional.interpolate(frames.movedim(-1, 1), size=(video.shape[3] * 16, video.shape[4] * 16), mode="bicubic", antialias=True)
    video = video.clone()
    video[:, :, :pin["steps"]] = vae.encode(frames.clamp(0.0, 1.0).movedim(1, -1)).to(video)
    return dict(latent, samples=NestedTensor((video, audio)))


def handoff_match(latent, unpinned, strength):
    """Grade a two-stage take's new rows into its held rows before the second stage samples it.

    The second stage's Apply Pins swaps the held rows of the latent-upscaled first stage for the clip's exact latent, while the new
    rows keep the first stage's look: re-encoded at the small size and upscaled. The second stage then grows a new frame next to a
    held one that it does not quite match, and the join comes out with crushed shadows and more contrast. Both copies of the held rows
    show the same picture, so they measure that look: a gain and offset per channel of the low frequencies, applied to the new rows."""
    mask = latent.get("noise_mask")
    if unpinned is None or mask is None:
        return latent
    held = mask.unbind()[0][0, 0].amin(dim=(1, 2)) < 1  # the rows Apply Pins wrote
    if not held.any():
        return latent
    video, audio = latent["samples"].unbind()
    blur = lambda x: torch.nn.functional.avg_pool3d(x, (1, 5, 5), stride=1, padding=(0, 2, 2), count_include_pad=False)
    low = blur(video)
    exact, copy = low[:, :, held], blur(unpinned["samples"].unbind()[0])[:, :, held]
    spread = copy.std(dim=(0, 2, 3, 4))
    gain = torch.where(spread > 0.05, exact.std(dim=(0, 2, 3, 4)) / spread, 1.0)  # a blank held window (a fade) has no contrast to measure
    shift = exact.mean(dim=(0, 2, 3, 4)) - gain * copy.mean(dim=(0, 2, 3, 4))
    video = video.clone()
    video[:, :, ~held] += strength * ((gain - 1).view(1, -1, 1, 1, 1) * low + shift.view(1, -1, 1, 1, 1))[:, :, ~held]
    logging.info("AEX Drift: handoff match graded %d new latent steps to %d held ones (channel gains %.2f to %.2f).",
                 int((~held).sum()), int(held.sum()), gain.min(), gain.max())
    return dict(latent, samples=NestedTensor((video, audio)))


def seam_anchor(latent, strength, region, detail, still_tolerance, min_still):
    """Grade an extend's new rows back to the tone of its held rows once the last stage has sampled them.

    H3 drifts as it samples: over the first few dozen new frames the low mids darken a code or two and contrast grows by a
    percent or so, with one stage or two, and a root drifts within itself as well. An extend shows it next to its parent, and
    the next extend continues from the drifted latent. Where the picture holds still the new rows should look like the held
    ones, so the low frequencies get a gain and offset per channel and the detail a gain per channel, fitted there row by row
    and smoothed over time, in the latent the take is decoded and saved from. The low frequencies are fitted over windows about
    a fifth of the frame wide: the drift differs by region, and one grade for the whole picture left a door darkening. Where
    little of a window holds still, as on a moving subject, its grade fades out rather than borrowing the background's. A
    luma-only grade along the preview's factors moved the decoded tone too little: the decoder's tone lives in more channels
    than the preview's linear fit says."""
    mask = latent.get("noise_mask")
    if mask is None:
        return latent
    held = (mask.unbind()[0][0, 0].amin(dim=(1, 2)) < 1).tolist()  # the rows Apply Pins wrote
    h = held.index(False) if False in held else len(held)
    new = [t for t in range(h, len(held)) if not held[t]]
    if not h or not new:
        return latent
    # what the anchor did, for AEX Drift Check
    report = dict(strength=strength, region=region, detail=detail, still_tolerance=still_tolerance, min_still=min_still, share=None, graded=False)
    if h < 5:
        return dict(latent, aex_drift_anchor=report)
    # the latent comes in groups of 5 steps per 17 frames (one 1-frame step, then four 4-frame ones), so each new step is
    # measured against the held step at the same place in its group
    ref = [h - 5 + (t - h) % 5 for t in new]
    video, audio = latent["samples"].unbind()
    toward_luma = torch.tensor(MiniMaxH3Video.latent_rgb_factors).to(video) @ torch.tensor([0.2126, 0.7152, 0.0722]).to(video)
    luma = torch.einsum("bcthw,c->bthw", video, toward_luma)
    box = lambda x: torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
    change = (box(luma[:, new]) - box(luma[:, ref])) * 127.5  # in codes of preview luma
    # a drift of the whole picture by up to half the tolerance is no motion, so each new row's median change is taken out up to
    # that much: an extend that darkened by about 3 codes otherwise counted 6% of its picture as still and was left alone
    change = change - change.flatten(2).median(dim=2).values[..., None, None].clamp(-still_tolerance / 2, still_tolerance / 2)
    still = (change.abs() < still_tolerance).all(dim=1).all(dim=0)
    share = report["share"] = still.float().mean().item()
    if share < min_still:
        logging.info("AEX Drift: seam anchor left the take alone, only %d%% of the picture holds still.", share * 100)
        return dict(latent, aex_drift_anchor=report)
    blur = lambda x: torch.nn.functional.avg_pool3d(x, (1, 5, 5), stride=1, padding=(0, 2, 2), count_include_pad=False)
    low, target = blur(video[:, :, new]), blur(video[:, :, ref])
    x, t = low[..., still], target[..., still]
    spread = x.std(dim=(0, 3))
    whole = torch.where(spread > 0.05, t.std(dim=(0, 3)) / spread, 1.0)[..., None, None]
    gain, shift = whole, t.mean(dim=(0, 3))[..., None, None] - whole * x.mean(dim=(0, 3))[..., None, None]
    if region > 0:
        # the same fit over the still pixels of a window around each latent pixel, odd so it centers there
        k = int(region * video.shape[4]) | 1
        w = still.to(video)
        pool = lambda s: torch.nn.functional.avg_pool2d(s.flatten(0, -3), k, stride=1, padding=k // 2, count_include_pad=False).view(s.shape)
        n = pool(w[None])[0]
        mean = lambda s: (pool(s * w) / n.clamp_min(1e-6)).mean(dim=0)
        mx, mt = mean(low), mean(target)
        sx, st = (mean(low ** 2) - mx ** 2).clamp_min(0).sqrt(), (mean(target ** 2) - mt ** 2).clamp_min(0).sqrt()
        local = torch.where(sx > 0.05, st / sx, whole).clamp(0.5, 2.0)  # a window with only a few still pixels can fit wild gains
        # windows less than a quarter still fade toward no grade: in a test the whole picture's grade, fitted on the background,
        # turned a moving face pink, while left alone the face kept its color
        trust = (n / 0.25).clamp(max=1.0)
        gain, shift = trust * local + 1 - trust, trust * (mt - local * mx)
    high = video[:, :, new] - low
    spread = high[..., still].std(dim=(0, 3))
    sharp = torch.where(spread > 0.05, (video[:, :, ref] - target)[..., still].std(dim=(0, 3)) / spread, 1.0)[..., None, None]
    smooth = lambda s: torch.nn.functional.avg_pool3d(s[None], (9, 1, 1), stride=1, padding=(4, 0, 0), count_include_pad=False)[0]
    gain, shift, sharp = smooth(gain), smooth(shift), smooth(sharp)
    fix = (gain - 1) * low + shift
    video = video.clone()
    video[:, :, new] += strength * (fix + detail * (sharp - 1) * high)
    # which way the grade moves the still picture at strength 1: its preview luma in codes, the contrast of that luma, and the detail
    before = luma[:, new][..., still]
    after = before + torch.einsum("bcthw,c->bthw", fix, toward_luma)[..., still]
    report.update(graded=True, tone=(after - before).mean().item() * 127.5, contrast=(after.std(dim=-1) / before.std(dim=-1)).mean().item() - 1,
                  texture=sharp.mean().item() - 1)
    logging.info("AEX Drift: seam anchor graded %d new latent steps over %d%% still picture (channel gains %.2f to %.2f over the "
                 "whole picture, detail gains %.2f to %.2f).", len(new), share * 100, whole.min(), whole.max(), sharp.min(), sharp.max())
    return dict(latent, samples=NestedTensor((video, audio)), aex_drift_anchor=report)


def gauss(x, s):
    """Separable Gaussian blur of (N, C, H, W), reflected at the edges and cut at 4 sigma."""
    r = int(4 * s + 0.5)
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * s * s))
    k = (k / k.sum()).view(1, 1, 1, -1).expand(x.shape[1], 1, 1, -1)
    x = torch.nn.functional.conv2d(torch.nn.functional.pad(x, (r, r, 0, 0), mode="reflect"), k, groups=x.shape[1])
    return torch.nn.functional.conv2d(torch.nn.functional.pad(x, (0, 0, r, r), mode="reflect"), k.transpose(-1, -2), groups=x.shape[1])


MATERIALS = 8  # colour clusters the texture hold measures separately, before merging near-duplicates


def texture_hold(latent, vae, strength, window, floor):
    """Soften an extend's new frames back to the texture of its held frames, region by region, moving subjects included.

    Each extend comes out a little rougher than the frames it holds: blotchier skin, harder folds, more pronounced lines, mostly
    at 3 to 20 px. Seam Anchor holds it where the picture is still, but faces and clothes move, and the next extend continues from
    the rougher latent. Resampling the last steps, with the base model or a turbo LoRA, didn't soften it in tests. So the frames
    are decoded and split into bands, and the pixels into a few materials by their colour below 8 px. The energy of the 3-8 px and
    8-20 px bands is measured per material over windows of the frame, on the held frames and on each new frame; where a material
    came out rougher in a window, those two bands are scaled down to its held level there (or to its level over the whole held
    picture, if it wasn't in that window then), never below it and never by more than `floor`. The 1-3 px band (pores, grain) is
    left as sampled, so skin doesn't go waxy. Measuring per material keeps a rougher face from softening the hoodie or the wall
    in the same window. The change is carried into the new rows by encoding each corrected 17-frame clip and its uncorrected
    original and adding the difference, which leaves out the VAE round trip's own error."""
    mask = latent.get("noise_mask")
    if mask is None:
        return latent
    held = (mask.unbind()[0][0, 0].amin(dim=(1, 2)) < 1).tolist()  # the rows Apply Pins wrote
    h = held.index(False) if False in held else len(held)
    new = [t for t in range(h, len(held)) if not held[t]]
    if h < 5 or not new:
        return latent
    video, audio = latent["samples"].unbind()
    frames = vae.decode(video)
    frames = frames.reshape((-1,) + frames.shape[-3:])
    start = lambda r: r // 5 * 17 + (r % 5 and 1 + 4 * (r % 5 - 1))  # first frame of a latent row: 1, 4, 4, 4, 4 frames per group of 5
    ref = range(1, start(h - 2))  # held frames clear of the decoder's two steps of lookahead into the new rows
    first = start(new[0])
    dev = comfy.model_management.get_torch_device()
    luma = torch.tensor([0.2126, 0.7152, 0.0722], device=dev).view(1, 3, 1, 1)
    k = max(16, int(window * frames.shape[2]))
    load = lambda f: frames[f].to(dev).movedim(-1, 0)[None].float()
    pool = lambda x: torch.nn.functional.avg_pool2d(x, k, stride=k // 2, ceil_mode=True)

    def colour(x):  # Y and doubled Cb, Cr of the picture below 8 px, at 1/8 size
        small = torch.nn.functional.avg_pool2d(gauss(x, 8.0), 8)
        y = (luma * small).sum(1)
        return torch.stack([y, (small[:, 2] - y) / 1.8556 * 2, (small[:, 0] - y) / 1.5748 * 2], dim=1)

    sample = torch.cat([colour(load(f))[0].flatten(1).T for f in ref[::4]])
    sample = sample[torch.randperm(len(sample), generator=torch.Generator().manual_seed(0))[:20000].to(dev)]
    centers = sample[:1]
    for _ in range(MATERIALS - 1):  # farthest-point seeds, then k-means
        centers = torch.cat([centers, sample[torch.cdist(sample, centers).amin(1).argmax()][None]])
    for _ in range(25):
        nearest = torch.cdist(sample, centers).argmin(1)
        centers = torch.stack([sample[nearest == i].mean(0) if (nearest == i).any() else centers[i] for i in range(MATERIALS)])
    # a scene with fewer colours than that gets fewer materials: near-duplicate ones would split each window between them
    count = torch.bincount(torch.cdist(sample, centers).argmin(1), minlength=MATERIALS)
    kept = []
    for i in count.argsort(descending=True).tolist():
        if count[i] > len(sample) // 100 and all((centers[i] - centers[j]).norm() > 0.04 for j in kept):
            kept.append(i)
    centers, materials = centers[kept], len(kept)
    spread = torch.cdist(sample, centers).amin(1).median().clamp_min(0.01)

    def measure(x):
        """Soft material weights (M, H, W), and per window and material the band energy (M, 2, gh, gw) and the share of the window
        it was measured over (M, gh, gw). Each material is measured on its interior, the pixels it clearly owns at least 4 px from
        any other: the edges between materials (glasses, hairline, the rim of a sleeve) hold most of the band energy, and a face
        that got blotchier inside hardly moves the energy of a window that includes them."""
        d = ((colour(x)[:, None] - centers[None, :, :, None, None]) ** 2).sum(2)
        w = torch.nn.functional.interpolate(torch.softmax(-d / (2 * spread ** 2), dim=1), size=x.shape[-2:], mode="bilinear", align_corners=False)[0]
        own = -torch.nn.functional.max_pool2d(-(w > 0.6).to(w)[None], 9, stride=1, padding=4)[0]
        g1, g3, g8 = gauss(x, 1.0), gauss(x, 3.0), gauss(x, 8.0)
        b = (g1 - g3, g3 - g8)
        share = pool(own[None])[0]
        energy = torch.stack([pool(own[None] * (luma * v).sum(1, keepdim=True) ** 2)[0] for v in b], 1)
        return w, b, energy, share

    total, present = 0, 0
    for f in ref:
        _, _, e, n = measure(load(f))
        total, present = total + e, present + n
    there = (present / len(ref) > 0.1)[:, None]
    whole = total.sum((2, 3)) / present.sum((1, 2))[:, None].clamp_min(1e-6)  # each material's energy over the whole held picture
    known = (present.sum((1, 2)) > 0)[:, None, None, None]  # a material the held frames don't show has nothing to be held to
    target = torch.where(there, total / present[:, None].clamp_min(1e-6), whole[..., None, None]).sqrt()
    measured, gain = [], []
    for f in range(first, frames.shape[0]):
        _, _, e, n = measure(load(f))
        m = (e / n[:, None].clamp_min(1e-6)).sqrt()
        # a material that barely touches this window keeps its texture
        cut = torch.where((n > 0.1)[:, None] & known, 1 - (target / m.clamp_min(1e-6)).clamp(floor, 1.0), 0.0)
        # a subject that turns or moves carries its texture from window to window; a material is softened only as far as its
        # energy over the whole picture rose, to within 3% of the held level, so a cat that turned its head keeps its fur
        a, b = (e * cut ** 2).sum((2, 3)), (e * cut).sum((2, 3))
        c = e.sum((2, 3)) - whole * n.sum((1, 2))[:, None] * 0.97 ** 2
        keep = ((b - (b ** 2 - a * c).clamp_min(0).sqrt()) / a.clamp_min(1e-12)).clamp(0.0, 1.0)  # the share of the cut that lands there
        keep = torch.where(c <= 0, 0.0, torch.where(a - 2 * b + c >= 0, 1.0, keep))
        gain.append(1 - keep[..., None, None] * cut)
        measured.append(m)
    measured, gain = torch.stack(measured), torch.stack(gain)
    # smoothed over about a third of a second, so the softening doesn't flicker
    kt = torch.exp(-torch.arange(-12, 13, device=dev, dtype=gain.dtype) ** 2 / 32)
    g = torch.nn.functional.conv1d(torch.nn.functional.pad(gain.flatten(1).T[:, None], (12, 12), mode="replicate"), (kt / kt.sum()).view(1, 1, -1))
    gain = 1 + strength * (g[:, 0].T.view(gain.shape) - 1)

    def soften(f):
        x = load(f)
        out, g = x, gain[f - first]
        aim = measured[f - first] * g
        for i in range(3):
            w, b, e, n = measure(out)
            if i:  # the bilinear map and the blur share each window's gain with its neighbours, so measure again and take the rest
                g = torch.where((n > 0.1)[:, None], (aim / (e / n[:, None].clamp_min(1e-6)).sqrt().clamp_min(1e-6)).clamp(max=1.0), 1.0)
            gmap = torch.nn.functional.interpolate(g.flatten(0, 1)[None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0]
            gmap = (w[:, None] * gmap.view(materials, 2, *x.shape[-2:])).sum(0)
            # blurred, so the change reaches nothing finer than the mid band and the fine band stays as sampled
            out = out + gauss((gmap[0] - 1) * b[0] + (gmap[1] - 1) * b[1], 2.5)
        return out[0].movedim(0, -1).clamp(0.0, 1.0)

    video = video.clone()
    for c in range(new[0] // 5, (len(held) + 4) // 5):
        # the VAE encodes each 17-frame clip on its own and drops the last 3 latent steps of every call, so one frame of the next
        # clip goes along to keep this clip's 5 steps
        clip = frames[c * 17:c * 17 + 18]
        fixed = torch.stack([soften(f).to(clip) if first <= f < c * 17 + 17 else clip[f - c * 17] for f in range(c * 17, c * 17 + len(clip))])
        rows = [t for t in range(c * 5, c * 5 + 5) if t in new]
        diff = (vae.encode(fixed) - vae.encode(clip)).to(video)
        video[:, :, rows] += diff[:, :, [t - c * 5 for t in rows]]
    logging.info("AEX Drift: texture hold softened %d new frames toward %d held ones over %d materials (3-8 px band gains %.2f to 1, "
                 "8-20 px %.2f to 1, on average %.3f and %.3f).", frames.shape[0] - first, len(ref), materials,
                 gain[:, :, 0].min(), gain[:, :, 1].min(), gain[:, :, 0].mean(), gain[:, :, 1].mean())
    return dict(latent, samples=NestedTensor((video, audio)))


# what counts as a real residual, from the tests behind the README: with Seam Anchor a take stayed within half a code of luma, a point
# of near-black share and a tenth of a percent of contrast, and saturation rose up to 2%; without it the mids dropped one to two codes,
# contrast grew about a percent and texture contrast a few percent per extend
TONE, BLACKS, CONTRAST, TEXTURE, SATURATION, PATCH = 0.5, 1.0, 0.5, 2.0, 3.0, 1.0


def drift_check(images, latent):
    """Tell how far an extend's new frames moved off the frames it holds from the clip it continues, and which knob to try next.

    The held frames are the earlier clip's own latent rows, decoded in the same pass as the new ones, so neither side carries an
    8-bit H.264 round trip or a save grade the other lacks. Over the pixels that hold still (a 31 px box mean within 6 codes in
    every compared frame), each compared frame gets the shift of seven luma percentiles, from the near blacks to the highlights, in
    8-bit codes of the saved file's Y plane (BT.709, TV range), and its change of near-black share, contrast, texture contrast and
    saturation. The share at or below Y 18 is what shows blacks spreading: a dark scene clips its darkest few percent to Y 16, where
    p2 cannot move."""
    mask = latent.get("noise_mask")
    if mask is None:
        return "No noise mask on the latent: wire the latent these images were decoded from, after the sampler."
    held = (mask.unbind()[0][0, 0].amin(dim=(1, 2)) < 1).tolist()  # the rows Apply Pins wrote
    h = held.index(False) if False in held else len(held)
    if h < 3 or h == len(held):
        return "Nothing to compare: the latent holds no frames of an earlier clip ahead of new ones."
    start = lambda t: t // 5 * 17 + (0, 1, 5, 9, 13)[t % 5]  # first frame of step t: each group of 5 steps is 1 frame, then 4 per step
    if images.shape[0] != start(len(held)):
        return f"These images are not the latent's decode: {images.shape[0]} frames for {len(held)} latent steps. Wire the untrimmed decode."
    # the decoder looks two steps ahead, so the last held frame none of the new steps reach ends step h - 3
    ref, first = start(h - 2) - 1, start(h)
    after = sorted({k for k in (4, 8, 16, 32, 64, 120) if first + k < images.shape[0]} | {images.shape[0] - 1 - first})
    rgb = images[[ref] + [first + k for k in after]].float()
    y = rgb @ torch.tensor([0.2126, 0.7152, 0.0722]).to(rgb)
    chroma = torch.hypot((rgb[..., 2] - y) / 1.8556, (rgb[..., 0] - y) / 1.5748)
    y = 16 + 219 * y
    box = lambda x, k: torch.nn.functional.avg_pool2d(x, k, stride=1, padding=(k[0] // 2, k[1] // 2), count_include_pad=False)
    local = box(box(y[:, None], (1, 31)), (31, 1))[:, 0]
    still = ((local[1:] - local[0]).abs() < 6).all(dim=0)
    share = still.float().mean().item()
    if share < 0.01:
        return f"Only {share:.1%} of the picture holds still, too little to compare (a moving camera or a lighting change)."
    late = [i for i, k in enumerate(after, 1) if k >= 32] or [len(after)]  # the drift has settled by then
    # the median of each quarter of the frame each way that is at least a quarter still, to find a patch that drifts on its own
    H, W = still.shape
    patches = []
    for r in range(4):
        for c in range(4):
            rows, cols = slice(r * H // 4, (r + 1) * H // 4), slice(c * W // 4, (c + 1) * W // 4)
            if still[rows, cols].float().mean() >= 0.25:
                medians = y[:, rows, cols][:, still[rows, cols]].median(dim=1).values
                patches.append(((c + 0.5) / 4, (r + 0.5) / 4, medians[0].item(), (medians[late] - medians[0]).mean().item()))
    # the detail finer than Seam Anchor's 5 latent pixel blur, about 80 px, is what its detail knob grades; the blur averages
    # only still pixels, so a subject moving nearby stays out of it
    blur = lambda x: box(box(x[:, None], (1, 81)), (81, 1))[:, 0]
    weight = still.to(y)[None]
    texture = (y - blur(y * weight) / blur(weight).clamp_min(1e-6))[:, still].std(dim=1)
    q = torch.tensor([0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]).to(y)
    y, chroma = y[:, still], chroma[:, still]
    base = torch.quantile(y[0], q)
    black = (y <= 18).float().mean(dim=1) * 100
    lines = [f"Against the held frames, over the {share:.0%} of the picture that holds still.",
             "Luma shift in 8-bit codes, <=18 in points, the rest in %.",
             "after    p2   p10   p25   p50   p75   p90   p98  <=18  contrast  texture  saturation"]
    shifts, measured = [], []
    for i, k in enumerate(after, 1):
        shifts.append((torch.quantile(y[i], q) - base).tolist())
        measured.append(((black[i] - black[0]).item(), (y[i].std() / y[0].std() - 1).item() * 100, (texture[i] / texture[0] - 1).item() * 100,
                         (chroma[i].mean() / chroma[0].mean() - 1).item() * 100))
        lines.append(f"{k:+5d} " + " ".join(f"{d:+5.1f}" for d in shifts[-1]) + " {:+5.1f}    {:+5.1f}%   {:+5.1f}%      {:+5.1f}%".format(*measured[-1]))
    mean = lambda values: sum(values) / len(values)
    settled = dict(tone=mean([mean(shifts[i - 1][1:5]) for i in late]), p50=mean([shifts[i - 1][3] for i in late]),
                   **{name: mean([measured[i - 1][n] for i in late]) for n, name in enumerate(("blacks", "contrast", "texture", "saturation"))})
    # a patch drifts on its own when the whole picture's tone line, fitted through the p10-p90 shifts, doesn't predict its shift
    xs, ys = base[1:6].tolist(), [mean([shifts[i - 1][n] for i in late]) for n in range(1, 6)]
    slope = sum((x - mean(xs)) * (v - mean(ys)) for x, v in zip(xs, ys)) / max(sum((x - mean(xs)) ** 2 for x in xs), 1e-6)
    off = lambda held, shift: shift - (mean(ys) + slope * (held - mean(xs)))
    worst = max(((across, down, shift, off(held, shift)) for across, down, held, shift in patches), key=lambda p: (round(abs(p[3]), 1), abs(p[2])), default=None)
    lines += ["p2 near blacks, p10-p25 low mids, p50 mids, p75-p90 upper mids, p98 highlights.",
              f"<=18: the share at or below Y 18, {black[0].item():.1f}% in the held frame. texture: detail finer than about 80 px.",
              "after: frames after the held ones.", "",
              f"Suggestions from frames {after[late[0] - 1]:+d} on (starting points from our tests; A/B each one):"]
    lines += ["- " + s for s in advise(latent.get("aex_drift_anchor"), settled, worst, h, base[0].item(), base[-1].item())]
    text = "\n".join(lines)
    logging.info("AEX Drift check:\n%s", text)
    return text


def advise(anchor, settled, worst, held_steps, p2, p98):
    """Which knob to try next. A residual that still runs against the way Seam Anchor graded is drift it left in; one that runs
    the same way overshot. Without Seam Anchor's record there is no grade to weigh it against."""
    out = []
    if p2 <= 16.5:
        out.append("p2 sits on the Y 16 floor in the held frame: read <=18 for the near blacks.")
    if p98 >= 234.5:
        out.append("p98 sits on the Y 235 ceiling in the held frame: highlight shifts can't show.")
    tone, blacks, contrast, texture = settled["tone"], settled["blacks"], settled["contrast"], settled["texture"]
    sign = lambda v, small: 0 if abs(v) < small else (1 if v > 0 else -1)
    found = [text for text, value, small in ((f"p10-p75 {tone:+.1f} codes", tone, TONE), (f"<=18 {blacks:+.1f} points", blacks, BLACKS),
                                             (f"contrast {contrast:+.1f}%", contrast, CONTRAST)) if abs(value) >= small]
    if anchor is None:
        out.append("No Seam Anchor record on this latent (bypassed, or not wired before the check): these numbers are the take's own drift."
                   + (f" Seam Anchor would grade it ({', '.join(found)})." if found else ""))
    elif anchor["share"] is None:
        out.append(f"Seam Anchor needs 5 held latent steps and this take holds {held_steps}: hold 39 frames (12 steps).")
    elif not anchor["graded"]:
        out.append(f"Seam Anchor left this take alone: {anchor['share']:.0%} of the picture held still, under min_still {anchor['min_still']:g}. "
                   f"If the camera is locked off, lower min_still, or raise still_tolerance from {anchor['still_tolerance']:g} when slow motion or "
                   "steam keeps the background from counting; a moving camera or a lighting change can't be graded.")
    else:
        s, region, d = anchor["strength"], anchor["region"], anchor["detail"]
        lift, punch, crisp = sign(anchor["tone"], 0.25), sign(anchor["contrast"], 0.002), sign(anchor["texture"], 0.002)
        votes = [v for v in (-lift * sign(tone, TONE), lift * sign(blacks, BLACKS), -punch * sign(contrast, CONTRAST)) if v]
        if s == 0:
            out.append("Seam Anchor's strength is 0, which passes the take through." + (f" Left in: {', '.join(found)}." if found else ""))
        elif votes and all(v > 0 for v in votes):
            if s >= 2.0:
                out.append(f"Drift left in ({', '.join(found)}), and strength is already at its maximum, 2.")
            else:
                raised = round(min(s + 0.2, 2.0), 2)
                out.append(f"Drift left in ({', '.join(found)}): raise strength from {s:g} to {raised:g}." +
                           (" Above 1 grades past the measured drift; compare against 1." if raised > 1 else ""))
        elif votes and all(v < 0 for v in votes):
            out.append(f"Seam Anchor overshot ({', '.join(found)}): lower strength from {s:g} to {round(max(s - 0.2, 0.0), 2):g}.")
        elif votes:
            out.append(f"The shifts point different ways ({', '.join(found)}): the grade is uneven rather than too weak or too strong.")
        elif found:
            out.append(f"Seam Anchor found little of this to grade ({', '.join(found)}): the drift is where the picture moves.")
        if worst and abs(worst[3]) >= PATCH:
            where = (f"The picture around {worst[0]:.0%} across and {worst[1]:.0%} down moved {worst[2]:+.1f} codes, {worst[3]:+.1f} off the "
                     "whole picture's tone")
            if region == 0:
                out.append(f"{where}, and region 0 fits one grade to the whole picture: set region to 0.2.")
            elif region > 0.12:
                out.append(f"{where}: lower region from {region:g} to {max(0.12, round(region * 0.75, 2)):g}.")
            else:
                out.append(f"{where}, and region {region:g} is already at or under the smallest tested width, 0.12.")
        if crisp and abs(texture) >= TEXTURE:
            if crisp * texture < 0:
                out.append(f"Texture contrast left in ({texture:+.1f}%): raise detail from {d:g} to {min(d + 0.25, 2.0):g}.")
            else:
                out.append(f"Texture contrast went past the held frames ({texture:+.1f}%): lower detail from {d:g} to {max(d - 0.25, 0.0):g}.")
        if anchor["share"] < 0.25:
            out.append(f"Only {anchor['share']:.0%} of the picture held still for Seam Anchor: its grade rests on few pixels.")
    if anchor is not None:
        if anchor["region"] and not 0.12 <= anchor["region"] <= 0.2:
            out.append(f"region {anchor['region']:g} is outside the tested 0.12-0.2.")
        if anchor["still_tolerance"] > 6:
            out.append(f"still_tolerance {anchor['still_tolerance']:g} is above the tested 6: slow motion and steam can enter the fit.")
    if abs(settled["saturation"]) >= SATURATION:
        out.append(f"Saturation moved {settled['saturation']:+.1f}%. No knob grades it on its own; the tests saw up to +2% with Seam Anchor.")
    return out or ["On target: the late frames are within what the tests measured with Seam Anchor."]


class AEXDriftFirstStagePins(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AEXDriftFirstStagePins",
            display_name="AEX Drift First Stage Pins",
            category="aex-drift",
            description="Re-encodes a two-stage extend's first-stage pin from the extended clip's saved latent instead of its MP4, so the "
                        "first stage continues the frames the second stage holds rather than an 8-bit, re-graded copy of them. Takes without "
                        "an extend, or extending a clip with no sidecar, pass through.",
            inputs=[
                io.Latent.Input("latent", tooltip="From the first stage's H3 MCtx Apply Pins."),
                ObvpmPins.Input("pins", tooltip="From the first stage's H3 MCtx Apply Pins."),
                ObvpmPinSpecs.Input("pin_specs", tooltip="The Timeline's pin_specs, as Pins From Pixels gets them."),
                io.Vae.Input("vae", tooltip="H3 video VAE."),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, pins, pin_specs, vae) -> io.NodeOutput:
        return io.NodeOutput(first_stage_pins(latent, pins, pin_specs, vae))


class AEXDriftHandoffMatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AEXDriftHandoffMatch",
            display_name="AEX Drift Handoff Match",
            category="aex-drift",
            description="Grades the new rows of a two-stage take's second-stage latent into its held rows. Apply Pins swaps the exact "
                        "latent in for the first stage's upscaled copy of the held frames; the difference between the two copies is the "
                        "first stage's look, and it is taken out of the new rows before the second stage starts. Takes with nothing held, "
                        "and single-stage runs, pass through.",
            inputs=[
                io.Latent.Input("latent", tooltip="From the second stage's H3 MCtx Apply Pins."),
                io.Latent.Input("unpinned", optional=True, tooltip="The same latent before Apply Pins: H3 Stage Latent's output."),
                io.Float.Input("strength", default=0.9, min=0.0, max=2.0, step=0.05,
                               tooltip="How much of the first stage's look to take out of the new rows. 0 passes the latent through, as "
                                       "bypassing does; in tests the background came out 1-2 px squished vertically without it."),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, strength, unpinned=None) -> io.NodeOutput:
        return io.NodeOutput(handoff_match(latent, unpinned, strength))


class AEXDriftSeamAnchor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AEXDriftSeamAnchor",
            display_name="AEX Drift Seam Anchor",
            category="aex-drift",
            description="Grades an extend's new rows back to the tone of its held rows after sampling: where the picture holds still, "
                        "a gain and offset per channel of the low frequencies, fitted region by region, and a gain per channel of the "
                        "detail, row by row and smoothed over time, take out the model's darker low mids and added contrast. Takes with "
                        "nothing held, or with too little of the picture still (a moving camera, a lighting change), pass through.",
            inputs=[
                io.Latent.Input("latent", tooltip="From the last stage's sampler, with Apply Pins' noise mask."),
                io.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.05,
                               tooltip="How much of the measured correction to apply. 0 passes the take through; above 1 overshoots."),
                io.Float.Input("region", default=0.2, min=0.0, max=1.0, step=0.01,
                               tooltip="Width of the window each low-frequency grade is fitted over, as a share of the frame width (0.2 is "
                                       "17 latent pixels at 1408 wide). Smaller follows a region that drifts on its own, such as a door, "
                                       "but fits fewer pixels; 0 fits one grade to the whole picture."),
                io.Float.Input("detail", default=1.0, min=0.0, max=2.0, step=0.05,
                               tooltip="How much of the fine-contrast correction to apply. The model adds a few percent of texture "
                                       "contrast per extend; 0 leaves the detail as sampled."),
                io.Float.Input("still_tolerance", default=6.0, min=1.0, max=64.0, step=0.5,
                               tooltip="How far a pixel may change, in 8-bit codes of the preview, and still count as background the "
                                       "grade is measured on; a drift of the whole picture by up to half of it doesn't count. Higher lets "
                                       "slow motion and steam into the fit."),
                io.Float.Input("min_still", default=0.1, min=0.01, max=1.0, step=0.01,
                               tooltip="Share of the picture that must hold still for the grade to run; below it (a moving camera, a "
                                       "lighting change) the take passes through."),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, strength, region, detail, still_tolerance, min_still) -> io.NodeOutput:
        return io.NodeOutput(seam_anchor(latent, strength, region, detail, still_tolerance, min_still))


class AEXDriftTextureHold(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AEXDriftTextureHold",
            display_name="AEX Drift Texture Hold",
            category="aex-drift",
            description="Softens an extend's new frames back to the texture of its held frames, moving subjects included: window by window, "
                        "the 3-8 px and 8-20 px bands of each new frame are scaled down to the held frames' level where the take came out "
                        "rougher, never below it; the 1-3 px band is left as sampled. The change goes back into the latent, so the next "
                        "extend continues from it. Takes with nothing held pass through.",
            inputs=[
                io.Latent.Input("latent", tooltip="Seam Anchor's output, with Apply Pins' noise mask."),
                io.Vae.Input("vae", tooltip="H3 video VAE."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="How much of the measured softening to apply. 0 passes the take through."),
                io.Float.Input("window", default=0.08, min=0.02, max=0.5, step=0.01,
                               tooltip="Width of the windows the texture is measured over, as a share of the frame width (0.08 is 128 px at "
                                       "1600 wide). Smaller follows a face or a sleeve more closely but compares fewer pixels."),
                io.Float.Input("floor", default=0.6, min=0.3, max=1.0, step=0.05,
                               tooltip="The most a band may be scaled down in one take: 0.6 keeps at least 60% of its sampled energy."),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, vae, strength, window, floor) -> io.NodeOutput:
        return io.NodeOutput(texture_hold(latent, vae, strength, window, floor))


class AEXDriftCheck(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AEXDriftCheck",
            display_name="AEX Drift Check",
            category="aex-drift",
            description="Shows on the node how an extend's new frames moved off the frames it holds from the clip it continues: over the "
                        "picture that holds still, the shift of seven luma percentiles from the near blacks to the highlights, and the "
                        "change of contrast, texture and saturation, a few frames after the hold and at the end, and which Seam Anchor "
                        "knob to try next. Both sides come from the same decode. It can also save the report as a text file.",
            inputs=[
                io.Image.Input("images", tooltip="The untrimmed decode of the latent: H3 VAE Decode's frames, as the save node gets them."),
                io.Latent.Input("latent", tooltip="The latent those frames were decoded from, with Apply Pins' noise mask: Seam Anchor's output."),
                io.Boolean.Input("check", default=True, label_on="on", label_off="off", tooltip="Off skips the check."),
                io.Boolean.Input("save", default=False, label_on="on", label_off="off", tooltip="On also saves the report as a .txt file in the output folder."),
                io.String.Input("filename_prefix", default="aex-drift/check",
                                tooltip="The saved report's name in the output folder, as in the other save nodes: a subfolder before the last slash, then a counter and .txt are added."),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, latent, check, save, filename_prefix) -> io.NodeOutput:
        if not check:
            return io.NodeOutput(ui=ui.PreviewText("Check off."))
        text = drift_check(images, latent)
        if not save:
            return io.NodeOutput(ui=ui.PreviewText(text))
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        file = f"{filename}_{counter:05}.txt"
        with open(os.path.join(folder, file), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        return io.NodeOutput(ui={"text": (text,), "files": [ui.SavedResult(file, subfolder, io.FolderType.output)]})


class DriftExtension(ComfyExtension):
    async def get_node_list(self):
        return [AEXDriftFirstStagePins, AEXDriftHandoffMatch, AEXDriftSeamAnchor, AEXDriftTextureHold, AEXDriftCheck]
