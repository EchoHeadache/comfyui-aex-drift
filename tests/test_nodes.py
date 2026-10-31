"""The four nodes on CPU with fake latents and a fake VAE. Needs ComfyUI's Python environment:

    <ComfyUI>/.venv/bin/python -m unittest discover -s tests
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
COMFY = PACK.parents[1]
if str(COMFY) not in sys.path:
    sys.path.insert(0, str(COMFY))
from comfy.cli_args import args as comfy_args
comfy_args.cpu = True
import torch

import folder_paths
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV

spec = importlib.util.spec_from_file_location("aex_drift_pack", PACK / "__init__.py", submodule_search_locations=[str(PACK)])
pack = importlib.util.module_from_spec(spec)
sys.modules["aex_drift_pack"] = pack
spec.loader.exec_module(pack)
from aex_drift_pack import nodes as drift


def fake_latent(frames, seed, width=96, height=64):
    """The AV latent a render of this size decodes from, with distinct values."""
    generator = torch.Generator().manual_seed(seed)
    video, audio = EmptyMiniMaxH3LatentAV.execute(width=width, height=height, length=frames).args[0]["samples"].unbind()
    return {"samples": NestedTensor((torch.randn(video.shape, generator=generator), torch.randn(audio.shape, generator=generator)))}


class RowVAE:
    """A stand-in H3 VAE: channels 0-2 of a latent pixel are drawn as the colour of its 16 px block on each frame of its row (1 frame for
    a group's first row, then 4), with the third frame of every 4-frame row a little darker as the real decoder draws some rows; the
    other channels aren't drawn. Encoding takes each row's frames back to channels 0-2, as a 17-frame clip at a time."""

    def __init__(self, dark=0.01):
        self.dark, self.decoded, self.pairs, self.encoded = dark, [], [], 0

    def decode(self, latent):
        self.decoded.append(latent.shape[2])
        rows = [latent[0, :3, t] for t in range(latent.shape[2]) for _ in range(1 if t % 5 == 0 else 4)]
        frames = torch.stack(rows).repeat_interleave(16, -2).repeat_interleave(16, -1).movedim(1, -1).clone()
        for f in range(len(frames)):
            if f % 17 and (f % 17 - 1) % 4 == 2:
                frames[f] -= self.dark
        return frames[None]

    def encode(self, pixels):
        if self.encoded % 2 == 0:
            self.corrected = pixels
        else:
            self.pairs.append((self.corrected, pixels))
        self.encoded += 1
        blocks = torch.nn.functional.avg_pool2d(pixels.movedim(-1, 1), 16)
        out, f = [], 0
        while f < len(pixels):
            n = 1 if len(out) % 5 == 0 else 4
            out.append(blocks[f:f + n].mean(0))
            f += n
        latent = torch.zeros(1, 24, len(out), *blocks.shape[-2:])
        latent[0, :3] = torch.stack(out, 1)
        return latent


class DriftTest(unittest.TestCase):
    def test_the_first_stage_pin_is_re_encoded_from_the_saved_latent(self):
        # the 39 frame tail of a 157 frame file that follows 35 frames held from its own parent: raw frame 153, step 45
        saved = (torch.arange(57.0) / 100).view(1, 1, 57, 1, 1).expand(1, 24, 57, 4, 6)  # step n decodes to the gray n / 100
        specs = [{"place": "before", "source_kind": "clip", "source": {"video_latent": saved, "meta": {"pinned_head_frames": "35"}}}]
        pin = {"place": "before", "steps": 12, "covered": 39, "spec": {"source_kind": "clip_pixels", "source_start": 118}}
        stage = fake_latent(192, 0, width=64, height=48)
        held = torch.ones(1, 1, 57, 3, 4)
        held[:, :, :12] = 0.0
        stage["noise_mask"] = NestedTensor((held, torch.ones(1, 1, 2, 320)))

        class GridVAE:
            def decode(self, latent):
                self.rows = [round(v * 100) for v in latent[0, 0, :, 0, 0].tolist()]
                # each step decodes to its frames on the 17-frame grid: 1 for a cycle's first step, 4 for the others
                frames = torch.cat([torch.full((1 if n % 5 == 0 else 4,), v / 100) for n, v in enumerate(self.rows)])
                return frames.view(1, -1, 1, 1, 1).expand(1, -1, 64, 96, 3)

            def encode(self, pixels):
                self.pixels = pixels
                return torch.full((1, 24, 12) + tuple(s // 16 for s in pixels.shape[1:3]), 7.0)
        vae = GridVAE()
        out = drift.AEXDriftFirstStagePins.execute(latent=stage, pins=[pin], pin_specs=specs, vae=vae).args[0]
        self.assertEqual(vae.rows, list(range(40, 57)))  # one cycle of run-in; the clip ends before two steps of lookahead
        self.assertEqual(vae.pixels.shape, (39, 48, 64, 3))
        window = [n for n in range(45, 57) for _ in range(1 if n % 5 == 0 else 4)]
        self.assertTrue(torch.allclose(vae.pixels[:, 0, 0, 0], torch.tensor(window) / 100))
        video, audio = out["samples"].unbind()
        before_video, before_audio = stage["samples"].unbind()
        self.assertTrue((video[:, :, :12] == 7.0).all() and torch.equal(video[:, :, 12:], before_video[:, :, 12:]))
        self.assertTrue(audio is before_audio and out["noise_mask"] is stage["noise_mask"])
        # a root take, a prepend, a clip with no sidecar and a window off the latent grid keep the MP4's frames
        for pins, pin_specs in (([], specs), ([dict(pin, place="after")], specs), ([pin], []),
                                ([dict(pin, spec=dict(pin["spec"], source_start=115))], specs)):
            self.assertIs(drift.AEXDriftFirstStagePins.execute(latent=stage, pins=pins, pin_specs=pin_specs, vae=vae).args[0], stage)

    def test_the_handoff_match_grades_the_new_rows_into_the_held_ones(self):
        _, audio = fake_latent(192, 0, width=192, height=128)["samples"].unbind()
        torch.manual_seed(0)
        scene = 0.5 + 0.15 * torch.nn.functional.interpolate(torch.randn(1, 24, 3, 4), size=(8, 12), mode="bilinear").unsqueeze(2).expand(1, 24, 57, 8, 12)
        exact = scene + 0.01 * torch.randn(1, 24, 57, 8, 12)  # the exact rows have detail the copy lacks
        look = torch.tensor([0.9, 1.0, 1.1]).view(1, 3, 1, 1, 1), torch.tensor([-0.02, 0.01, 0.03]).view(1, 3, 1, 1, 1)
        upscaled = scene.clone()
        upscaled[:, :3] = (scene[:, :3] - look[1]) / look[0]  # the first stage's look on the channels the fake VAE draws
        upscaled[:, 3:] += 0.3  # and a step in channels it doesn't draw, as the detail the soft copy lacks sits in some channels' means
        pinned = upscaled.clone()
        pinned[:, :, :12] = exact[:, :, :12]
        held = torch.ones(1, 1, 57, 8, 12)
        held[:, :, :11] = 0.0
        held[:, :, 11] = 0.5  # a feathered edge is still a row Apply Pins wrote
        mask = NestedTensor((held, torch.ones(1, 1, 2, 320)))
        stage = {"samples": NestedTensor((pinned, audio)), "noise_mask": mask}
        vae = RowVAE()
        match = lambda latent, unpinned, strength=1.0, vae=vae: drift.AEXDriftHandoffMatch.execute(latent=latent, vae=vae, strength=strength, unpinned=unpinned).args[0]
        unpinned = {"samples": NestedTensor((upscaled, audio))}
        out = match(stage, unpinned)
        video, out_audio = out["samples"].unbind()
        self.assertTrue(torch.equal(video[:, :, :12], pinned[:, :, :12]) and out_audio is audio and out["noise_mask"] is mask)
        # the copy is decoded only as far as its held rows clear of the decode's end need
        self.assertEqual(vae.decoded, [57, 12])
        # the new rows' low frequencies take the held look in what the VAE draws, their detail is left alone, and nothing moves in the
        # channels it doesn't draw
        blur = lambda x: torch.nn.functional.avg_pool3d(x, (1, 5, 5), stride=1, padding=(0, 2, 2), count_include_pad=False)
        copy = upscaled[:, :3, 12:]
        self.assertLess((video[:, :3, 12:] - (blur(scene[:, :3, 12:]) + copy - blur(copy))).abs().max(), 0.003)
        self.assertTrue(torch.allclose(video[:, 3:, 12:], upscaled[:, 3:, 12:], atol=1e-6))
        half = match(stage, unpinned, 0.5)["samples"].unbind()[0]
        self.assertTrue(torch.allclose(half[:, :, 12:], (pinned[:, :, 12:] + video[:, :, 12:]) / 2, atol=1e-3))
        # a blank held window has no contrast to measure: only the offset moves
        blank, dim = pinned.clone(), upscaled.clone()
        blank[:, :3, :12], dim[:, :3] = 0.5, 0.4
        flat = match(dict(stage, samples=NestedTensor((blank, audio))), {"samples": NestedTensor((dim, audio))}, vae=RowVAE(dark=0.0))["samples"].unbind()[0]
        self.assertTrue(torch.allclose(flat[:, :3, 12:], blank[:, :3, 12:] + 0.1, atol=1e-4))
        # a single-stage run, a latent with nothing held and one with no mask pass through
        nothing = dict(stage, noise_mask=NestedTensor((torch.ones(1, 1, 57, 8, 12), torch.ones(1, 1, 2, 320))))
        for latent, other in ((stage, None), (nothing, stage), ({"samples": stage["samples"]}, stage)):
            self.assertIs(match(latent, other), latent)

    def test_the_seam_anchor_grades_the_new_rows_back_to_the_held_ones(self):
        _, audio = fake_latent(192, 0, width=384, height=256)["samples"].unbind()
        torch.manual_seed(0)
        scene = 0.5 + 0.1 * torch.nn.functional.interpolate(torch.randn(1, 24, 4, 6), size=(16, 24), mode="bilinear").unsqueeze(2)
        # a still picture whose rows differ by their place in each group of 5, so only a same-place held row matches a new one
        truth = scene * torch.tensor([1.0, 0.9, 0.95, 1.05, 1.1]).repeat(12)[:57].view(1, 1, 57, 1, 1)
        # the drift is a gain and offset of luma and chroma, the channels the anchor grades
        gain, shift = torch.tensor([0.97, 1.05, 1.03]).view(1, 3, 1, 1, 1), torch.tensor([-0.01, 0.005, 0.0]).view(1, 3, 1, 1, 1)
        ycc = lambda x: torch.einsum("ij,bjthw->bithw", drift.SPACE, x)
        rgb = lambda x: torch.einsum("ij,bjthw->bithw", torch.linalg.inv(drift.SPACE), x)
        drifted = truth.clone()
        drifted[:, :3, 12:] = rgb(gain * ycc(truth[:, :3, 12:]) + shift)
        held = torch.ones(1, 1, 57, 16, 24)
        held[:, :, :12] = 0.0
        mask = NestedTensor((held, torch.ones(1, 1, 2, 320)))
        take = {"samples": NestedTensor((drifted, audio)), "noise_mask": mask}
        knobs = dict(strength=1.0, region=0.7, detail=1.0, still_tolerance=6.0, min_still=0.1)
        vae = RowVAE()
        anchor = lambda latent, **kw: drift.AEXDriftSeamAnchor.execute(latent=latent, vae=vae, **dict(knobs, **kw)).args[0]
        out = anchor(take)
        video, out_audio = out["samples"].unbind()
        self.assertEqual({k: out["aex_drift_anchor"][k] for k in knobs}, knobs)
        self.assertTrue(out["aex_drift_anchor"]["graded"] and out["aex_drift_anchor"]["share"] == 1.0)
        self.assertTrue(torch.equal(video[:, :, :12], drifted[:, :, :12]) and out_audio is audio and out["noise_mask"] is mask)
        # the new rows are the held look again in what the VAE draws; the channels it doesn't draw keep their values
        self.assertLess((video[:, :3, 12:] - truth[:, :3, 12:]).abs().max(), 0.003)
        self.assertTrue(torch.allclose(video[:, 3:], drifted[:, 3:], atol=1e-6))
        # every frame of a latent row gets the same correction, so the grade can't flicker on a frame rhythm inside a row
        for corrected, clip in vae.pairs:
            change = corrected - clip
            for a in (a for a in (1, 5, 9, 13) if a + 4 <= len(change)):
                self.assertLess((change[a:a + 4] - change[a]).abs().max(), 1e-5)
        # half the strength goes half the way; without the detail grade only the low frequencies move
        self.assertTrue(torch.allclose(anchor(take, strength=0.5)["samples"].unbind()[0], (drifted + video) / 2, atol=1e-3))
        self.assertLess((anchor(take, detail=0.0)["samples"].unbind()[0][:, :3, 12:] - truth[:, :3, 12:]).abs().max(), 0.006)
        # a region that darkens on its own gets its own grade, and the far side of the picture keeps its tone; one grade for the
        # whole picture leaves both sides off
        darker = truth.clone()
        darker[:, :3, 12:, :, :12] -= 0.02
        darker = dict(take, samples=NestedTensor((darker, audio)))
        graded = anchor(darker)
        video = graded["samples"].unbind()[0]
        self.assertGreater(graded["aex_drift_anchor"]["tone"], 0)  # it lifted the luma, for the check to weigh what is left against
        self.assertLess((video[:, :3, 12:, :, :2] - truth[:, :3, 12:, :, :2]).abs().max(), 0.005)
        self.assertLess((video[:, :3, 12:, :, 20:] - truth[:, :3, 12:, :, 20:]).abs().max(), 0.005)
        whole = anchor(darker, region=0.0)["samples"].unbind()[0]
        self.assertGreater((whole[:, :3, 12:, :, :2] - truth[:, :3, 12:, :, :2]).abs().max(), 0.008)
        # no mask and a root pass through; a head of fewer than 7 held rows and a picture that does not hold still pass through
        # with a record of why
        for latent in ({"samples": take["samples"]}, dict(take, noise_mask=NestedTensor((torch.ones(1, 1, 57, 16, 24), torch.ones(1, 1, 2, 320))))):
            self.assertIs(anchor(latent), latent)
        short = held.clone()
        short[:, :, 6:] = 1.0
        moved = drifted.clone()
        moved[:, :3, 12:] += torch.rand(1, 1, 45, 16, 24) * 0.2  # everything changes from row to row
        for latent, share in ((dict(take, noise_mask=NestedTensor((short, torch.ones(1, 1, 2, 320)))), None), (dict(take, samples=NestedTensor((moved, audio))), 0.0)):
            out = anchor(latent)
            self.assertTrue(out["samples"] is latent["samples"] and not out["aex_drift_anchor"]["graded"] and out["aex_drift_anchor"]["share"] == share)
        # a dimming of the whole picture by about 7 codes is drift once half the tolerance of it is taken out, and is graded away;
        # one of about 11 codes counts as still only under a wider tolerance
        dim = lambda codes: dict(take, samples=NestedTensor((truth - codes / 255 * (torch.arange(57) >= 12).view(1, 1, -1, 1, 1) * (torch.arange(24) < 3).view(1, -1, 1, 1, 1), audio)))
        self.assertLess((anchor(dim(7))["samples"].unbind()[0][:, :3] - truth[:, :3]).abs().max(), 0.003)
        dimmer = dim(11)
        self.assertIs(anchor(dimmer)["samples"], dimmer["samples"])
        graded = anchor(dimmer, still_tolerance=10.0)
        self.assertTrue(graded["aex_drift_anchor"]["graded"] and (graded["samples"].unbind()[0][:, :3] - truth[:, :3]).abs().max() < 0.003)
        # where too little holds still, as on a moving subject, the grade fades out instead of borrowing the background's
        busy = drifted.clone()
        busy[:, :3, 12:, :, :12] += (torch.rand(1, 1, 45, 16, 12) - 0.5) * 0.3
        busy = dict(take, samples=NestedTensor((busy, audio)))
        graded = anchor(busy, detail=0.0)
        video, before = graded["samples"].unbind()[0], busy["samples"].unbind()[0]
        self.assertTrue(0.4 < graded["aex_drift_anchor"]["share"] < 0.6)
        self.assertLess((video[:, :3, 12:, :, :2] - before[:, :3, 12:, :, :2]).abs().max(), 1e-4)
        self.assertLess((video[:, :3, 12:, :, 20:] - truth[:, :3, 12:, :, 20:]).abs().mean(), (before[:, :3, 12:, :, 20:] - truth[:, :3, 12:, :, 20:]).abs().mean() / 3)

    def test_the_texture_hold_softens_the_new_frames_to_the_held_texture(self):
        take = fake_latent(192, 0, width=256, height=128)
        held = torch.ones(1, 1, 57, 8, 16)
        held[:, :, :12] = 0.0  # 39 frames held
        take["noise_mask"] = NestedTensor((held, torch.ones(1, 1, 2, 320)))
        torch.manual_seed(0)
        texture = drift.gauss(torch.randn(192, 1, 128, 256), 3.0)
        texture = (texture / texture.std()).expand(-1, 3, -1, -1)
        grain = torch.randn(192, 1, 128, 256).expand(-1, 3, -1, -1) * 0.01

        # two materials, skin-coloured on the left and gray on the right, each with its own texture
        colours = torch.tensor([[0.75, 0.55, 0.45], [0.4, 0.42, 0.45]]).repeat_interleave(128, 0).T.view(1, 3, 1, 256)

        class FrameVAE:
            def __init__(self, roughness, moved=False):
                rough = 0.03 * texture * torch.where(torch.arange(192) >= 39, roughness, 1.0).view(-1, 1, 1, 1)
                if moved:  # the new frames' texture gathers in the top half of the picture, as a subject that moves carries it
                    rough[39:] = rough[39:] * torch.cat([torch.full((64,), 1.3), torch.full((64,), 0.6)]).view(1, 1, -1, 1)
                self.frames = (colours + rough + grain).movedim(1, -1)
                self.encoded = []

            def decode(self, latent):
                return self.frames[None]

            def encode(self, pixels):
                self.encoded.append(pixels)
                steps = (len(pixels) + 16) // 17 * 5 - 3
                return pixels.mean().expand(1, 24, steps, 8, 16)

        mid = lambda x: drift.gauss(x, 1.0) - drift.gauss(x, 3.0)
        coarse = lambda x: drift.gauss(x, 3.0) - drift.gauss(x, 8.0)
        fine = lambda x: x - drift.gauss(x, 1.0)
        away = (torch.arange(256) - 127.5).abs() > 16  # clear of the edge between the two materials
        energy = lambda band, frames: band(frames.movedim(-1, 1))[..., away].std().item()
        vae = FrameVAE(1.2)
        hold = lambda latent, vae, **kw: drift.AEXDriftTextureHold.execute(latent=latent, vae=vae, **dict(dict(strength=1.0, window=0.25, floor=0.6), **kw)).args[0]
        out = hold(take, vae)
        video, audio = out["samples"].unbind()
        before_video, before_audio = take["samples"].unbind()
        self.assertTrue(torch.equal(video[:, :, :12], before_video[:, :, :12]) and audio is before_audio and out["noise_mask"] is take["noise_mask"])
        # each clip from the one holding the first new row is encoded corrected, then as decoded, with a frame of the next clip
        self.assertEqual([len(p) for p in vae.encoded], [18, 18] * 9 + [5, 5])
        fixed = torch.cat([p[:17] for p in vae.encoded[0::2]])  # frames 34 on
        new, original = fixed[5:], vae.frames[39:]
        self.assertTrue(torch.equal(fixed[:5], vae.frames[34:39]))
        # the 3-8 px and 8-20 px bands, 20% rougher than the held frames, come back to within 5% of them and not below; the
        # 1-3 px band stays within 5% of the sampled one
        for band in (mid, coarse):
            self.assertGreater(energy(band, original) / energy(band, vae.frames[1:34]), 1.15)
            self.assertTrue(0.98 < energy(band, new) / energy(band, vae.frames[1:34]) < 1.05)
        self.assertLess(abs(energy(fine, new) / energy(fine, original) - 1), 0.05)
        # the new rows take the difference the corrected frames encode to
        self.assertTrue(torch.allclose(video[:, :, 12:15] - before_video[:, :, 12:15], (vae.encoded[0].mean() - vae.encoded[1].mean()).expand(1, 24, 3, 8, 16), atol=1e-6))
        # never more than the floor in one take, and half the strength goes about half the way
        floored = FrameVAE(1.2)
        hold(take, floored, floor=0.9)
        ratio = energy(mid, torch.cat([p[:17] for p in floored.encoded[0::2]])[5:]) / energy(mid, floored.frames[39:])
        self.assertTrue(0.89 < ratio < 0.95)
        half = FrameVAE(1.2)
        hold(take, half, strength=0.5)
        softened = energy(mid, torch.cat([p[:17] for p in half.encoded[0::2]])[5:])
        self.assertAlmostEqual(softened, (energy(mid, new) + energy(mid, original)) / 2, delta=0.03 * softened)
        # texture that jumped far past what an extend's roughening does is new content, and is left as sampled
        jumped = FrameVAE(3.0)
        hold(take, jumped)
        kept = torch.cat([p[:17] for p in jumped.encoded[0::2]])[5:]
        for band in (mid, coarse):
            self.assertGreater(energy(band, kept) / energy(band, jumped.frames[39:]), 0.98)
        # a take that came out smoother than its held frames keeps its texture: the hold never sharpens, and only a window where a
        # material's edge shifted gets a touch softer
        smoother = FrameVAE(0.6)
        hold(take, smoother)
        kept = torch.cat([p[:17] for p in smoother.encoded[0::2]])[5:]
        for band in (mid, coarse):
            self.assertTrue(0.98 < energy(band, kept) / energy(band, smoother.frames[39:]) < 1.005)
        # texture that only moved, about as much of it over the whole picture as in the held frames, keeps all but the few percent the
        # 3% slack lets go where it gathered
        moved = FrameVAE(1.0, moved=True)
        hold(take, moved)
        kept = torch.cat([p[:17] for p in moved.encoded[0::2]])[5:]
        for band in (mid, coarse):
            top = lambda frames: band(frames.movedim(-1, 1))[..., 8:56, :][..., away].std().item()
            self.assertGreater(top(kept) / top(moved.frames[39:]), 0.93)
        # no mask and a root pass through
        for latent in ({"samples": take["samples"]}, dict(take, noise_mask=NestedTensor((torch.ones(1, 1, 57, 8, 16), torch.ones(1, 1, 2, 320))))):
            self.assertIs(hold(latent, vae), latent)

    def test_the_check_reports_the_new_frames_against_the_held_ones(self):
        take = fake_latent(192, 0)
        held = torch.ones(1, 1, 57, 4, 6)
        held[:, :, :12] = 0.0  # 39 frames held
        take["noise_mask"] = NestedTensor((held, torch.ones(1, 1, 2, 320)))
        torch.manual_seed(0)
        picture = 0.2 + 0.6 * torch.nn.functional.interpolate(torch.rand(1, 3, 4, 6), size=(64, 96), mode="bilinear").movedim(1, -1)
        check = lambda images, latent=take, on=True: drift.AEXDriftCheck.execute(images=images, latent=latent, check=on, save=False, filename_prefix="aex-drift/check").ui.as_dict()["text"][0]
        rows = lambda text: [[float(v.rstrip("%")) for v in line.split()] for line in text.splitlines() if line.split()[:1] and line.split()[0].lstrip("+-").isdigit()]
        darker = picture.repeat(192, 1, 1, 1)
        darker[39:] -= 2 / 219  # two codes of Y darker, the same color
        darker[39:, :16, :16] = 1.0  # a patch that changes after the hold is left out
        report = rows(check(darker))
        self.assertEqual([r[0] for r in report], [4, 8, 16, 32, 64, 120, 152])
        self.assertTrue(all(r[1:] == [-2.0] * 7 + [0.0, 0.0, 0.0, 0.0] for r in report))
        # blacks spreading show in the share at or below Y 18, where the clipped p2 cannot move
        crushed = picture.repeat(192, 1, 1, 1)
        crushed[:, :8, 32:] = 0.0  # 8% of the picture at black
        crushed[:, 8:16, 32:] = 4 / 219
        crushed[39:, 8:16, 32:] = 1 / 219  # another 8% sinks from Y 20 to Y 17
        self.assertTrue(all(r[1] == 0.0 and r[8] == 8.3 for r in rows(check(crushed))))
        punchier = picture.repeat(192, 1, 1, 1)
        punchier[39:] = 0.5 + 1.1 * (punchier[39:] - 0.5)
        self.assertTrue(all(r[-3:] == [10.0, 10.0, 10.0] for r in rows(check(punchier))))
        # the trimmed frames, a root, a latent with no mask and the switch off report no rows
        root = dict(take, noise_mask=NestedTensor((torch.ones(1, 1, 57, 4, 6), torch.ones(1, 1, 2, 320))))
        for text in (check(darker[35:]), check(darker, root), check(darker, {"samples": take["samples"]}), check(darker, on=False)):
            self.assertEqual(rows(text), [])
        # the save switch writes the report the node shows, numbered like the other save nodes
        with tempfile.TemporaryDirectory() as out:
            output_directory = folder_paths.get_output_directory()
            folder_paths.set_output_directory(out)
            try:
                saved = [drift.AEXDriftCheck.execute(images=darker, latent=take, check=True, save=True, filename_prefix="drift/take").ui for _ in range(2)]
            finally:
                folder_paths.set_output_directory(output_directory)
            self.assertEqual([s["files"][0]["filename"] for s in saved], ["take_00001.txt", "take_00002.txt"])
            self.assertEqual((Path(out) / "drift" / "take_00001.txt").read_text(encoding="utf-8"), saved[0]["text"][0] + "\n")

        # the suggestions weigh what is left against which way Seam Anchor graded, from its record on the latent
        record = lambda **kw: dict(dict(strength=1.0, region=0.2, detail=1.0, still_tolerance=6.0, min_still=0.1, share=0.9, graded=True,
                                        tone=3.0, contrast=-0.01, texture=-0.03), **kw)
        suggest = lambda images, **kw: check(images, dict(take, aex_drift_anchor=record(**kw))).split("Suggestions")[1]
        self.assertIn("would grade it (p10-p75 -2.0 codes)", check(darker).split("Suggestions")[1])
        self.assertIn("Drift left in (p10-p75 -2.0 codes): raise strength from 1 to 1.2. Above 1", suggest(darker))
        self.assertIn("raise strength from 0.8 to 1.", suggest(darker, strength=0.8))
        self.assertIn("already at its maximum", suggest(darker, strength=2.0))
        self.assertIn("overshot (p10-p75 -2.0 codes): lower strength from 1 to 0.8.", suggest(darker, tone=-3.0))
        self.assertIn("left this take alone: 5% of the picture held still, under min_still 0.1", suggest(darker, graded=False, share=0.05))
        self.assertIn("On target", suggest(picture.repeat(192, 1, 1, 1)))
        # the model's added contrast and texture, left in by a grade that took some out, and the saturation no knob grades
        punch = suggest(punchier)
        self.assertTrue("contrast +10.0%" in punch and "raise strength from 1 to 1.2" in punch)
        self.assertIn("Texture contrast left in (+10.0%): raise detail from 1 to 1.25.", punch)
        self.assertIn("Saturation moved +10.0%.", punch)
        self.assertIn("lower detail from 1 to 0.75", suggest(punchier, texture=0.03))
        # a patch that darkens on its own asks for smaller windows, down to the smallest tested
        patch = picture.repeat(192, 1, 1, 1)
        patch[39:, :, :48] -= 4 / 219
        self.assertIn("lower region from 0.2 to 0.15.", suggest(patch))
        self.assertIn("already at or under the smallest tested width", suggest(patch, region=0.12))
        self.assertIn("set region to 0.2", suggest(patch, region=0.0))
        # the floor, and knobs past what was tested
        self.assertIn("p2 sits on the Y 16 floor", suggest(crushed))
        knobs = suggest(picture.repeat(192, 1, 1, 1), region=0.3, still_tolerance=8.0)
        self.assertTrue("region 0.3 is outside the tested 0.12-0.2." in knobs and "still_tolerance 8 is above the tested 6" in knobs)


if __name__ == "__main__":
    unittest.main()
