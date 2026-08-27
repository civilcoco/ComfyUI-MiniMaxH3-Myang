# Findings and changes

## Evidence

The five cached prompts all carried the same character image reference, so the
drift was not caused by a missing reference upload. The stronger prompt failure
was in segment 3: the screenplay's blushing human ear tip was rewritten as
`red-tipped ears`. The output grows animal ears in that segment and retains them
later. The reference sheet has no animal ears.

All five prompts also mislabeled the reusable character sheet as a concrete
opening picture with `at 0.00 seconds ... is fully referenced`. MiniMax's current
full-reference guide distinguishes a reusable character `Subject` from a
concrete `Picture` keyframe and provides `fully_preserved` retention semantics.

MotionContext preserves the previous tail at the next clip's head. It does not
provide a global identity memory after the pinned window; its own prompt guide
says the next clip's behavior is controlled by the prompt and recommends an
airlock before changing the setup. Myang's 22-frame latent path is therefore
working for motion and scene continuity but cannot correct an identity-changing
prompt later in the clip.

The active Myang detail pass previously rebuilt conditioning independently for
every segment. The displayed 768P segment was a second-pass result, while the
next segment inherited only the previous 540P first-pass latent. Existing aligned
mid-frame previews show that the detail pass usually changes the image modestly
(face-crop CLIP cosine about 0.92–0.96 for the reliable close/medium samples),
and the animal ears already exist before refinement. This ranks the prompt error
above detail refinement as the cause of the visible example, while the missing
detail chain remains a real boundary inconsistency.

The private baseline probe detected 92 face samples. Adjacent segment
face-centroid CLIP cosines were 0.904, 0.927, 0.616, and 0.598. The last two are
partly confounded by much smaller/profile faces and are used as regression
indicators rather than absolute identity scores. The source media, generated
contact sheets and path-bearing local report are not redistributed.

## Implemented changes

- Every character-reference segment receives a conditional continuity contract:
  unrequested face, hair, ear, costume, accessory, and proportion drift is
  discouraged, while explicit transformation, aging, injury, disguise,
  hairstyle, wardrobe, species, facial-feature, and accessory changes remain
  authoritative.
- Reusable character references no longer retain the false exact-first-frame
  sentence.
- Ambiguous `red-tipped ears` is rewritten as flushed human ear skin unless the
  authoritative global style explicitly contains animal ears.
- Splitter instructions forbid invented identity accessories and the cache key
  was versioned so old prompts are not silently reused.
- The second pass now chains the previous segment's final high-resolution detail
  latent into the next segment's second-pass conditioning. The original low-
  resolution MotionContext chain remains unchanged, preserving current motion
  and scene behavior.

## Remaining limit

No diffusion model can guarantee pixel-identical faces through arbitrary pose,
scale, occlusion, and shot changes from a single reference sheet. These changes
remove two avoidable sources of drift without freezing expression or camera.
The post-change visual A/B render requires a ComfyUI restart so the running
process loads the updated Python modules.
