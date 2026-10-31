# AEX Drift

ComfyUI nodes that keep a MiniMax H3 extend close to the tone and texture of the clip it continues, and report how far a take still drifted.

When H3 extends a clip, the new frames drift from the held ones: shadows darken and spread, mids drop a code or two, contrast rises and textures get rougher. Each extend continues from the drifted latent, so the drift adds up over a chain.

The held frames are the earlier clip's own latent, so they show what the new frames should look like. The nodes correct the new frames toward them before the decode and the save, so the next extend starts from the corrected latent:

- First Stage Pins keeps the MP4 out of a two-stage run's first stage.
- Handoff Match takes the first stage's look out of the new frames before the second stage.
- Seam Anchor grades the new frames back to the held ones where the picture holds still.
- Texture Hold softens the new frames' texture back to the held level, moving subjects included.
- Check measures what is left and suggests a knob to try.

Handoff Match and Seam Anchor grade the latent by default. With `decode` on, they grade the decoded frames and encode the change back. That avoids the artifacts under Known issues, but takes a few minutes per take.

## Caveats

- It works only on extends that hold the earlier clip's latent under a noise mask. Roots and guide-only continuations pass through.
- It measures only the parts of the picture that hold still. A moving camera or a big lighting change passes through. Moving subjects keep their own tone: the grade fades out where little holds still, so it can't tint them, but it doesn't fix their drift either.
- A deliberate tone change of the whole picture under about 1.5 × `still_tolerance`, such as a slow fade, is treated as drift and pulled back.
- It corrects tone and texture contrast, not geometry or content, and it slows the drift rather than stopping it.
- Texture Hold only softens, never sharpens, and never below the held level. Texture that jumped far past normal roughening (a hand moving onto a face, a head turning) counts as new content and is left alone.
- Texture Hold, and Handoff Match and Seam Anchor with `decode` on, decode the take and re-encode it: about 3, 2.7 and 2.5 minutes for a 243-frame take at 1600×1184 on an RTX 5090.
- Tested on the obvpm Timeline workflow. Single-stage runs, and Handoff Match outside obvpm, are untested.

## Requirements

- ComfyUI with native MiniMax H3 support. No other dependencies.
- An extend that writes the earlier clip's latent into the head of the new latent under a noise mask, such as the obvpm Timeline's H3 MCtx Apply Pins with masked pins. Core's Add Guide for MiniMax H3 puts the frames in the conditioning instead; those runs pass through unchanged.
- First Stage Pins also needs the obvpm packs and the clip's `.mctx.safetensors` sidecar.

## Install

Clone this repository into `ComfyUI/custom_nodes/` and restart ComfyUI. The nodes are in the `aex-drift` category.

## Nodes

| Node | Problem | Wiring | What it does |
|---|---|---|---|
| AEX Drift First Stage Pins | A two-stage extend pins its small first stage from the parent's MP4: 8-bit, read back about a code darker with more clipped blacks, and with any grade the save applied. | Between the first stage's Apply Pins and its sampler. `pin_specs` from the Timeline, `vae` the H3 video VAE. | Re-encodes the first stage's pin from the clip's saved latent instead. |
| AEX Drift Handoff Match | The second stage's Apply Pins puts the parent's exact latent in the held rows, but the new rows keep the first stage's look. The second stage then grows new frames that don't match the held ones, with darker shadows and more contrast from the join on. | Between the second stage's Apply Pins and its sampler. `unpinned` is the same latent before Apply Pins; `vae` the H3 video VAE (for `decode`). | Measures the first stage's look by comparing the two copies of the held frames (a gain and offset per channel of the low frequencies) and takes it out of the new rows. In one test scene it also kept the background from coming out 1–2 px squished. |
| AEX Drift Seam Anchor | H3's own drift while it samples an extend: over the first few dozen new frames the low mids darken, shadows spread, and contrast and texture contrast rise. | After the last sampler, feeding the decode and the saved latent. `vae` the H3 video VAE (for `decode`). | Grades the new rows back to the held rows where the picture holds still, window by window: a gain and offset per channel of the low frequencies and a gain of the detail, smoothed over time. Moving subjects are left as sampled. |
| AEX Drift Texture Hold | Each extend comes out a little rougher, mostly at 3–20 px: blotchier skin, harder folds, sharper lines. Seam Anchor holds texture only where the picture is still, and resampling the last steps didn't soften it in tests. | After Seam Anchor, feeding the decode and the saved latent. `vae` the H3 video VAE. | Splits the decoded frames into bands and the picture into colour materials, measures each material window by window, and scales the 3–8 px and 8–20 px bands back to the held level where they came out rougher. The 1–3 px band (pores, grain) is left alone. The change goes back into the latent. |
| AEX Drift Check | Drift of a code or two is hard to see, and 8-bit files can't show less than a code. | `images` from VAE Decode, untrimmed; `latent` from Seam Anchor. | Shows on the node how far the new frames moved from the held ones, and suggests a knob to try. |

Notes:

- Handoff Match compares two clean latents. If your graph adds the stage-2 noise in a separate pass before the pins are written, move the noise into the final sampler first.
- Seam Anchor needs at least 5 held latent steps (7 with `decode` on), Texture Hold 5. The nodes were tested with 39-frame holds (12 steps).
- Roots, single-stage runs (for First Stage Pins and Handoff Match) and takes with too little still picture pass through.

## Check

AEX Drift Check compares one held frame with frames 4, 8, 16, 32, 64 and 120 after the hold, and the last frame, over the still part of the picture. Both come from the same decode, so it resolves tenths of a code. For each frame it shows:

- the shift of seven luma percentiles, from p2 (near blacks) to p98 (highlights), in 8-bit BT.709 TV-range codes;
- the change in the share of pixels at or below Y 18, which shows blacks spreading where p2 sits on the Y 16 floor;
- the change in contrast, texture contrast (detail finer than about 80 px) and saturation.

Below the table it suggests a knob to try, judged from frame +32 on:

- Seam Anchor leaves a record of which way it graded. A shift still running against that grade is drift left in: raise `strength`, or `detail` for texture. A shift running with it is an overshoot: lower it.
- A region whose tone moved off the rest of the picture asks for a smaller `region`, down to 0.12.
- A take Seam Anchor left alone asks for a lower `min_still`, or a higher `still_tolerance` if the camera is locked off.
- It also flags a held frame on the Y 16 floor or the Y 235 ceiling, knobs outside the tested ranges, saturation (no knob grades it on its own) and a missing Seam Anchor record.

A shift counts at 0.5 code for p10–p75, 1 point for the <=18 share, 0.5% contrast, 2% texture, 3% saturation, or 1 code for a region. These thresholds come from our tests. Each suggestion is one step (`strength` ±0.2, `detail` ±0.25, `region` ×0.75), so confirm it with an A/B. The check doesn't measure geometry, so it never suggests anything for Handoff Match.

`check` turns it off. The report also goes to the console. With `save` on, it is written to a `.txt` file in the output folder, named by `filename_prefix` (default `aex-drift/check`) and numbered like other save nodes' files.

## Knobs

- Handoff Match:
  - `strength` (0.9): 0 passes the latent through.
  - `decode` (off): grade decoded frames instead of the latent. Needs `vae`.
- Seam Anchor:
  - `strength` (1.0): 0 passes through; above 1 overshoots.
  - `region` (0.2): the width of each fitting window, as a share of the frame width. 0 fits one grade to the whole picture.
  - `detail` (1.0): how much of the fine-contrast correction to apply.
  - `still_tolerance` (6): how far a 16 px block may change, in 8-bit luma codes, and still count as still. A drift of the whole picture by up to half of it doesn't count.
  - `min_still` (0.1): below this share of still picture, the take passes through.
  - `decode` (off): grade decoded frames, one correction per latent row, instead of the latent. Needs `vae`.
- Texture Hold:
  - `strength` (1.0): how much of the measured softening to apply. 0 passes through.
  - `window` (0.08): the width of the measuring windows, as a share of the frame width (128 px at 1600 wide).
  - `floor` (0.6): the most a band may be scaled down in one take.

## Results

Typical results from our tests:

- **Seam Anchor:** without it, the mids dropped one to two codes per extend; with it, takes stayed within about half a code of the held frames. A dark door went from 7% to 19% of its pixels at or below Y 18 over two extends without it, and stayed at 7–8% with it. One grade for the whole picture (`region` 0) only halved that.
- **Texture Hold:** the per-extend rise in face texture (+7% to +14% at 3–20 px) dropped to 0–5%. Over two extends it came to about +1% with the hold, against about +19% without it. The 1–3 px band stayed within 3%, and scenes without roughening were left nearly as they were. It caused no tone change.
- **`decode` on:** the flicker and the Handoff Match patches under Known issues went away, with the same tone hold.

## Known issues

**Patches on faces.** A two-stage extend can come out with a flat neon, orange or white patch on a face, fixed in place while the subject moves. It forms in the second stage's first steps.

- Handoff Match's latent grade causes many of them: its offset also carries detail the soft upscaled copy lacks. In one test at three seeds, the latent grade patched all three takes; with `decode` on, only one did.
- The rest come from the first stage: where it clips a highlight to white on something the held frames don't show, the second stage fills the flat area with an invented shape, with or without Handoff Match. Render again with another seed.

**Flicker in still areas with Seam Anchor's latent grade.** The decoder draws some latent channels as change between the 4 frames of a latent row. A grade that varies window by window moves them, so still areas can flicker a code darker on every row's third frame (about 6 Hz). With `decode` on, the flicker was gone in tests and the tone hold was the same, but on some takes texture contrast came out up to 2% softer and highlights up to 1.5 codes darker.

## Tests

```
<ComfyUI>/.venv/bin/python -m unittest discover -s tests
```

## License

MIT. See [LICENSE](LICENSE).
