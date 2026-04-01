---
title: Videos
permalink: /videos/
---

**Site Navigation:** [Home]({{ '/' | relative_url }}) | [Overview]({{ '/overview/' | relative_url }}) | [Version One]({{ '/version-one/' | relative_url }}) | [Version Two]({{ '/version-two/' | relative_url }}) | [Timeline]({{ '/timeline/' | relative_url }}) | [Videos]({{ '/videos/' | relative_url }}) | [Experiments]({{ '/experiments/' | relative_url }}) | [Results]({{ '/results/' | relative_url }})

## Media Gallery

Use this page to track visual progress by milestone. Keep local clips short and link out for larger files.

## Existing Local GIF

Current Cold Clear recording (copied from `Recordings/ColdClear.gif`):

![Cold Clear Survival Demo]({{ '/assets/gifs/ColdClear.gif' | relative_url }})

## Local MP4 Embeds

### Version Two Placeholder

Drop a clip at `docs/assets/videos/version-two/pygame_autoplay_round15.mp4` and it will render here:

<video controls muted playsinline width="720">
  <source src="{{ '/assets/videos/version-two/pygame_autoplay_round15.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Version One Placeholder

Drop a clip at `docs/assets/videos/version-one/ppo_modern_eval.mp4`:

<video controls muted playsinline width="720">
  <source src="{{ '/assets/videos/version-one/ppo_modern_eval.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Reusable Embed Snippet

```html
<video controls muted playsinline width="720">
  <source src="{{ '/assets/videos/version-two/<file>.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

## External Long-Form Videos

Use this table for YouTube/Drive links when files are too large for repository storage.

| Date | Topic | Link | Notes |
|---|---|---|---|
| TBD | Version Two autoplay deep dive | `TBD` | Add narration around decision errors and recoveries. |
| TBD | BC vs DAgger comparison | `TBD` | Show side-by-side gameplay and metrics. |
| TBD | Version One PPO training recap | `TBD` | Include reward-shaping ablations. |

