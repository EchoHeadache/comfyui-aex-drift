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
        exact, audio = fake_latent(192, 0, width=192, height=128)["samples"].unbind()
        gain, shift = torch.linspace(0.8, 1.25, 24).view(1, -1, 1, 1, 1), torch.linspace(-0.3, 0.3, 24).view(1, -1, 1, 1, 1)
        upscaled = (exact - shift) / gain  # the first stage's look over every row
        pinned = upscaled.clone()
        pinned[:, :, :12] = exact[:, :, :12]
        held = torch.ones(1, 1, 57, 8, 12)
        held[:, :, :11] = 0.0
        held[:, :, 11] = 0.5  # a feathered edge is still a row Apply Pins wrote
        mask = NestedTensor((held, torch.ones(1, 1, 2, 320)))
        stage = {"samples": NestedTensor((pinned, audio)), "noise_mask": mask}
        out = drift.AEXDriftHandoffMatch.execute(latent=stage, strength=1.0, unpinned={"samples": NestedTensor((upscaled, audio))}).args[0]
        video, out_audio = out["samples"].unbind()
        blur = lambda x: torch.nn.functional.avg_pool3d(x, (1, 5, 5), stride=1, padding=(0, 2, 2), count_include_pad=False)
        self.assertTrue(torch.equal(video[:, :, :12], exact[:, :, :12]))
        new = upscaled[:, :, 12:]
        # the low frequencies take the held look; the detail is left alone
        self.assertTrue(torch.allclose(video[:, :, 12:], blur(exact[:, :, 12:]) + new - blur(new), atol=1e-4))
        self.assertTrue(out_audio is audio and out["noise_mask"] is mask)
        half = drift.AEXDriftHandoffMatch.execute(latent=stage, strength=0.5, unpinned={"samples": NestedTensor((upscaled, audio))}).args[0]
        self.assertTrue(torch.allclose(half["samples"].unbind()[0], (pinned + video) / 2, atol=1e-5))
        # a blank held window has no contrast to measure: only the offset moves
        blank, dim = pinned.clone(), upscaled.clone()
        blank[:, :, :12], dim[:, :, :12] = 0.5, 0.4
        flat = drift.AEXDriftHandoffMatch.execute(latent=dict(stage, samples=NestedTensor((blank, audio))), strength=1.0,
                                                   unpinned={"samples": NestedTensor((dim, audio))}).args[0]["samples"].unbind()[0]
        self.assertTrue(torch.equal(flat[:, :, :12], blank[:, :, :12]) and torch.allclose(flat[:, :, 12:], blank[:, :, 12:] + 0.1, atol=1e-5))
        # a single-stage run, a latent with nothing held and one with no mask pass through
        nothing = dict(stage, noise_mask=NestedTensor((torch.ones(1, 1, 57, 8, 12), torch.ones(1, 1, 2, 320))))
        for latent, unpinned in ((stage, None), (nothing, stage), ({"samples": stage["samples"]}, stage)):
            self.assertIs(drift.AEXDriftHandoffMatch.execute(latent=latent, strength=1.0, unpinned=unpinned).args[0], latent)

    def test_the_seam_anchor_grades_the_new_rows_back_to_the_held_ones(self):
        _, audio = fake_latent(192, 0, width=384, height=256)["samples"].unbind()
        torch.manual_seed(0)
        scene = torch.nn.functional.interpolate(torch.randn(1, 24, 4, 6), size=(16, 24), mode="bilinear").unsqueeze(2)
        # a still picture whose rows differ by their place in each group of 5, so only a same-place held row matches a new one
        truth = scene * torch.tensor([1.0, 0.7, 0.85, 1.15, 1.3]).repeat(12)[:57].view(1, 1, 57, 1, 1)
        gain, shift = torch.linspace(0.97, 1.03, 24).view(1, -1, 1, 1, 1), torch.linspace(-0.01, 0.01, 24).view(1, -1, 1, 1, 1)
        drifted = truth.clone()
        drifted[:, :, 12:] = gain * truth[:, :, 12:] + shift
        held = torch.ones(1, 1, 57, 16, 24)
        held[:, :, :12] = 0.0
        mask = NestedTensor((held, torch.ones(1, 1, 2, 320)))
        take = {"samples": NestedTensor((drifted, audio)), "noise_mask": mask}
        # region 0.7 fits 17-pixel windows across this 24-wide picture, as the default 0.2 does across 1408 wide
        knobs = dict(strength=1.0, region=0.7, detail=1.0, still_tolerance=6.0, min_still=0.1)
        anchor = lambda latent, **kw: drift.AEXDriftSeamAnchor.execute(latent=latent, **dict(knobs, **kw)).args[0]
        out = anchor(take)
        video, out_audio = out["samples"].unbind()
        self.assertEqual({k: out["aex_drift_anchor"][k] for k in knobs}, knobs)
        self.assertTrue(out["aex_drift_anchor"]["graded"] and out["aex_drift_anchor"]["share"] == 1.0)
        self.assertTrue(torch.equal(video[:, :, :12], truth[:, :, :12]))
        # the low frequencies and the detail are the held look again
        self.assertTrue(torch.allclose(video[:, :, 12:], truth[:, :, 12:], atol=1e-4))
        self.assertTrue(out_audio is audio and out["noise_mask"] is mask)
        # half the strength goes half the way; without the detail grade the sampler's detail stays
        blur = lambda x: torch.nn.functional.avg_pool3d(x, (1, 5, 5), stride=1, padding=(0, 2, 2), count_include_pad=False)
        self.assertTrue(torch.allclose(anchor(take, strength=0.5)["samples"].unbind()[0], (drifted + video) / 2, atol=1e-4))
        new = truth[:, :, 12:]
        self.assertTrue(torch.allclose(anchor(take, detail=0.0)["samples"].unbind()[0][:, :, 12:], blur(new) + gain * (new - blur(new)), atol=1e-4))
        # a region that darkens on its own gets its own grade, and the far side of the picture keeps its tone; one grade for the
        # whole picture leaves both sides about the full 0.02 off
        darker = truth.clone()
        darker[:, :, 12:, :, :12] -= 0.02
        darker = dict(take, samples=NestedTensor((darker, audio)))
        graded = anchor(darker)
        video = graded["samples"].unbind()[0]
        # the record says which way the grade moved the preview luma, for the check to weigh what is left against
        luma = torch.tensor(drift.MiniMaxH3Video.latent_rgb_factors) @ torch.tensor([0.2126, 0.7152, 0.0722])
        moved = torch.einsum("bcthw,c->bthw", video[:, :, 12:] - darker["samples"].unbind()[0][:, :, 12:], luma).mean().item() * 127.5
        self.assertAlmostEqual(graded["aex_drift_anchor"]["tone"], moved, delta=0.05 * abs(moved))
        self.assertLess((video[..., :2] - truth[..., :2]).abs().max(), 0.005)
        self.assertLess((video[..., 20:] - truth[..., 20:]).abs().max(), 0.005)
        whole = anchor(darker, region=0.0)["samples"].unbind()[0]
        self.assertGreater((whole[..., :2] - truth[..., :2]).abs().max(), 0.01)
        # no mask and a root pass through; a head of fewer than 5 held rows and a picture that does not hold still pass through
        # with a record of why
        for latent in ({"samples": take["samples"]}, dict(take, noise_mask=NestedTensor((torch.ones(1, 1, 57, 16, 24), torch.ones(1, 1, 2, 320))))):
            self.assertIs(anchor(latent), latent)
        short = held.clone()
        short[:, :, 4:] = 1.0
        moved = drifted.clone()
        moved[:, 9, 12:] += 1.0  # a lighting change
        for latent, share in ((dict(take, noise_mask=NestedTensor((short, torch.ones(1, 1, 2, 320)))), None), (dict(take, samples=NestedTensor((moved, audio))), 0.0)):
            out = anchor(latent)
            self.assertTrue(out["samples"] is latent["samples"] and not out["aex_drift_anchor"]["graded"] and out["aex_drift_anchor"]["share"] == share)
        # a dimming of the whole picture by about 7 codes is drift once half the tolerance of it is taken out, and is graded away;
        # one of about 11 codes counts as still only under a wider tolerance
        dim = lambda codes: dict(take, samples=NestedTensor((drifted + torch.eye(24)[9].view(1, -1, 1, 1, 1) * (torch.arange(57) >= 12).view(1, 1, -1, 1, 1)
                                                             * codes / (luma[9].item() * 127.5), audio)))
        self.assertTrue(torch.allclose(anchor(dim(7))["samples"].unbind()[0], truth, atol=1e-4))
        dimmer = dim(11)
        self.assertIs(anchor(dimmer)["samples"], dimmer["samples"])
        graded = anchor(dimmer, still_tolerance=10.0)
        self.assertTrue(graded["aex_drift_anchor"]["graded"] and torch.allclose(graded["samples"].unbind()[0], truth, atol=1e-4))
        # where too little holds still, as on a moving subject, the grade fades out instead of borrowing the background's
        torch.manual_seed(1)
        busy = drifted.clone()
        busy[:, :, 12:, :, :12] += torch.randn(1, 24, 45, 16, 12) * 0.5
        busy = dict(take, samples=NestedTensor((busy, audio)))
        graded = anchor(busy, detail=0.0)
        video, before = graded["samples"].unbind()[0], busy["samples"].unbind()[0]
        self.assertTrue(0.4 < graded["aex_drift_anchor"]["share"] < 0.6 and torch.equal(video[..., :2], before[..., :2]))
        self.assertLess((video[..., 20:] - truth[..., 20:]).abs().mean(), (before[..., 20:] - truth[..., 20:]).abs().mean() / 3)

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
