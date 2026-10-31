# AEX Drift

ComfyUI nodes that hold a MiniMax H3 extend to the tone and texture of the clip it continues, and show how far a take still moved.

An extend's new frames drift from the frames it holds: shadows darken and spread, mids drop a code or two, contrast grows, and textures roughen (blotchier skin, harder folds). The next extend continues from the drifted latent, so the drift adds up over a chain.

The held frames are the earlier clip's own latent, so they show what the new frames should look like. Where the picture holds still, Seam Anchor fits the decoded new frames back to them. It fits a gain and offset of luma and chroma in the low frequencies window by window, and a gain of the detail, one correction per latent row, then smooths both over time and encodes the change back into the latent. Handoff Match measures the first stage's look the same way at a two-stage run's handover, and First Stage Pins keeps the MP4 out of the first stage's starting point. Texture Hold softens the new frames' texture back to the held frames' level, moving subjects included. All of it happens before the decode and the save, so the next extend continues from the corrected latent. AEX Drift Check then measures what is left.

Caveats:

- It works only on extends that hold the earlier clip's latent under a noise mask. Roots and guide-only continuations pass through.
- It measures only the parts of the picture that hold still. A moving camera or a large lighting change makes the take pass through. Moving subjects keep their own tone: the grade fades out where little holds still, so the background's grade can't tint them, but it doesn't correct their drift either.
- A deliberate tone change of the whole picture under about 1.5 × `still_tolerance`, such as a slow fade, is treated as drift and pulled back.
- It corrects tone and texture contrast, not geometry or content, and it slows the drift rather than stopping it.
- Texture Hold only softens, never sharpens, and never below the held frames' level per material and window. A window whose texture jumped far past an extend's roughening (a hand moving onto a face, a turn of the head) counts as new content and is left alone.
- Handoff Match, Seam Anchor and Texture Hold each decode the take and re-encode the clips with new frames: about 2.7, 2.5 and 3 minutes for a 243-frame take at 1600×1184 on an RTX 5090.
- It was tested on the obvpm Timeline workflow. Single-stage runs, and Handoff Match outside obvpm, are untested.

## Requirements

- ComfyUI with its native MiniMax H3 support. No other dependencies.
- An extend that writes the earlier clip's latent into the head of the new latent under a noise mask, such as the obvpm Timeline's H3 MCtx Apply Pins with masked pins. Core's Add Guide for MiniMax H3 puts the frames in the conditioning instead, and those runs pass through unchanged.
- First Stage Pins also needs the obvpm packs and the clip's `.mctx.safetensors` sidecar.

## Install

Clone this repository into `ComfyUI/custom_nodes/` and restart ComfyUI. The nodes are in the `aex-drift` category.

## Nodes

| Node | What it intended to solve | Wiring | What it does |
|---|---|---|---|
| AEX Drift First Stage Pins | In a two-stage extend, obvpm pins the small first stage from the parent's MP4: 8-bit, read back about a code darker with more pixels clipped to black, and carrying any grade the save applied. The first stage then starts from a darker, re-graded copy of frames the second stage holds exactly. | Between the first stage's Apply Pins and its sampler. `pin_specs` from the Timeline, `vae` the H3 video VAE. | Re-encodes the first stage's pin from the clip's saved latent instead of its MP4. |
| AEX Drift Handoff Match | The step at the join of a two-stage extend. The second stage's Apply Pins swaps the parent's exact latent into the held rows, but the new rows keep the first stage's look, made small and upscaled. The second stage then grows new frames next to held ones they don't match, with darker shadows and more contrast from the join on. | Between the second stage's Apply Pins and its sampler. `unpinned` is the same latent before Apply Pins, `vae` the H3 video VAE. | Decodes both copies of the held frames, measures the first stage's look between them (a gain and offset per colour channel of the low frequencies) and takes it out of the new rows before the second stage samples them. In one scene it also held the geometry: without it the background came out 1–2 px squished vertically. |
| AEX Drift Seam Anchor | H3's own drift as it samples an extend: over the first few dozen new frames the low mids darken a code or two, shadows spread, and contrast and texture contrast grow. The next extend continues from the drifted latent, so the drift adds up over a chain. | After the last sampler, feeding both the decode and the saved latent. `vae` the H3 video VAE. | Grades the decoded new rows back to the held rows where the picture holds still, region by region, with one correction for all the frames of a latent row, and encodes the change back. Moving subjects are left as sampled. |
| AEX Drift Texture Hold | Each extend comes out a little rougher than the frames it holds, mostly at 3–20 px: blotchier skin, harder folds, more pronounced lines. Seam Anchor holds texture only where the picture is still, so faces and clothes keep roughening, extend after extend. Resampling the last steps, with the base model or a turbo LoRA, didn't soften it in tests. | After Seam Anchor, feeding both the decode and the saved latent. `vae` the H3 video VAE. | Splits the decoded frames into bands and the picture into colour materials, and measures each material on its own interior, window by window. Where a material came out rougher than in the held frames, it scales the 3–8 px and 8–20 px bands back to the held level, never further than the material's energy over the whole picture rose. The 1–3 px band (pores, grain) is left as sampled. The change goes back into the latent. |
| AEX Drift Check | Drift of a code or two is hard to judge by eye, and the 8-bit files can't show less than a code. Testers on different setups need numbers for each take and a hint of which knob to try next. | `images` from VAE Decode, untrimmed; `latent` from Seam Anchor. | Shows on the node how far the new frames moved off the held ones, and suggests a knob to try. |

Notes:

- Handoff Match compares two clean latents. If your graph adds the stage-2 noise in a separate pass before the pins are written, move the noise into the final sampler first.
- Seam Anchor needs at least 7 held latent steps and Texture Hold 5. The nodes were tested with 39-frame holds (12 steps).
- Roots, single-stage runs (for First Stage Pins and Handoff Match), and takes with too little still picture pass through.

## Check

AEX Drift Check compares one held frame with frames 4, 8, 16, 32, 64 and 120 after the hold, and the last frame. Both sides come from the same decode, so the numbers resolve tenths of a code. It measures only the part of the picture that holds still. For each frame it shows:

- the shift of seven luma percentiles, from p2 (near blacks) to p98 (highlights), in 8-bit BT.709 TV-range codes;
- the change in the share of pixels at or below Y 18, which shows blacks spreading where p2 is clipped at Y 16;
- the change in contrast, texture contrast (detail finer than about 80 px, which `detail` grades) and saturation.

Under the table it suggests which knob to try next, judged from frames +32 on, once the drift has settled:

- Seam Anchor leaves a record on its latent of which way it graded. A shift that still runs against that grade is drift left in: raise `strength`, or `detail` for texture. A shift that runs with it is an overshoot: lower them.
- A patch of the frame that moves off the whole picture's tone line asks for a smaller `region`, down to 0.12, the smallest width tested.
- A take Seam Anchor left alone asks for a lower `min_still` or a higher `still_tolerance` if the camera is locked off.
- Notices flag a held frame sitting on the Y 16 floor or the Y 235 ceiling, knobs outside the tested ranges, saturation (no knob grades it on its own), and a missing Seam Anchor record (bypassed, or not wired before the check).

A shift counts at 0.5 codes of p10–p75, 1 point of the <=18 share, 0.5% contrast, 2% texture, 3% saturation, or 1 code for a patch. These are starting points from our tests: with Seam Anchor, takes stayed within about half a code, while without it the mids dropped one to two codes per extend. Each suggestion is one step (`strength` ±0.2, `detail` ±0.25, `region` ×0.75), so confirm it with an A/B. The check doesn't measure geometry, so it has no suggestion for Handoff Match.

The `check` switch turns it off. The report also goes to the console log. With `save` on, it is also written to a `.txt` file in the output folder, named by `filename_prefix` (default `aex-drift/check`) and numbered like the other save nodes' files.

## Knobs

- Handoff Match `strength` (0.9): 0 passes the latent through.
- Seam Anchor:
  - `strength` (1.0): 0 passes through; above 1 overshoots.
  - `region` (0.2): the width of each fitting window, as a share of the frame width. 0 fits one grade to the whole picture.
  - `detail` (1.0): how much of the fine-contrast correction to apply.
  - `still_tolerance` (6): how far a 16 px block may change, in 8-bit codes of luma, and still count as still. A drift of the whole picture by up to half of it doesn't count as change.
  - `min_still` (0.1): below this share of still picture, the take passes through.
- Texture Hold:
  - `strength` (1.0): how much of the measured softening to apply. 0 passes through.
  - `window` (0.08): the width of the windows the texture is measured over, as a share of the frame width (128 px at 1600 wide).
  - `floor` (0.6): the most a band may be scaled down in one take.

## Results

Three chains from one root with the same seed and settings. The share of a dark door's pixels at or below Y 18:

| | Root → 1st → 2nd extend |
|---|---|
| Without Seam Anchor | 7% → 16% → 19% |
| Whole-picture grade (`region` 0, `detail` 0) | 7% → 10% → 13% |
| Default | 7% → 7% → 8% |

The geometry was unchanged. A dark shirt on the moving subject still came out one code lighter.

A later chain of two animated people showed two faults, both fixed since:

- One extend drifted by about 3 codes, enough that only 6% of its picture counted as still, so Seam Anchor left it alone and its back wall darkened by 4–5 codes. The still test now takes out up to half the tolerance of whole-picture drift; the same latent counts 13% as still, and the wall stays within about a code of the held frame at five of six sampled frames (2.7 codes down at the sixth).
- Where a window had little still picture, Seam Anchor used to apply the whole picture's grade, fitted on the background. On a moving face that grade lowered G − B by 2–4 codes, a pink cast. The grade now fades out there, and the face keeps the color it had without the anchor.

Texture Hold, on two-stage extends of one scene (two women on a couch, 1600×1184, 39-frame holds). Each pair comes from one sampling run, forked after Seam Anchor. The figures are the rise of luma band texture from the held frames to the new frames (+32 on), on the left woman's face and on her hoodie (3–8 px / 8–20 px):

| | Face, current | Face, Texture Hold | Hoodie, current | Hoodie, Texture Hold | Face 1–3 px vs held, Texture Hold |
|---|---|---|---|---|---|
| Extend of a take made without the hold | +7% / +14% | 0% / +5% | +2% / 0% | −2% / −5% | 98% |
| Another extend of a take made without the hold | +10% / +9% | +1% / 0% | +8% / +6% | +2% / −2% | 103% |
| The next extend, from that take's Texture Hold version | +7% / +2% | +2% / −3% | +3% / +2% | −1% / −4% | 102% |

Over those two extends the face gained +1% / −3% with the hold against about +19% / +13% without it. Two other scenes showed no rise to take out: a cat-headed man drinking coffee, where the subjects turn and lean between the held and the new frames, and a single-stage extend of a woman at a table. Texture Hold left both nearly as they were (pixels moved 0.1–0.2 code on average). AEX Drift Check showed no tone change from it in any of these.

## Known issues

**Neon and blown-out patches on faces.** A two-stage extend can come out with a flat neon, orange or white patch on a face, often a forehead under a raised hand, fixed in place while the subject moves. It forms in the second stage's first two steps.

- Before this version, Handoff Match's latent offset caused many of them: on one extend at three seeds, the old node patched all three, this version none at two of them.
- The rest start in the first stage: where its draft clips a highlight to white on something the held frames don't show, the second stage fills the flat area with an invented shape. That happens with Handoff Match bypassed too. If a take shows one, render it again with another seed.

**Flicker in still areas (fixed).** Before this version, Seam Anchor graded the latent channel by channel and window by window. The decoder draws some latent channels as change between the 4 frames of a latent row, so still areas flickered a code dark on every row's third frame, at about 6 Hz. Seam Anchor now grades the decoded frames with one correction per latent row.

## Tests

```
<ComfyUI>/.venv/bin/python -m unittest discover -s tests
```

## License

MIT. See [LICENSE](LICENSE).
