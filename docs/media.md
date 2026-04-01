---
title: Media
permalink: /media/
---

**Navigation:** [Home]({{ '/' | relative_url }}) | [System / Implementation]({{ '/system/' | relative_url }}) | [Algorithms]({{ '/algorithms/' | relative_url }}) | [Results]({{ '/results/' | relative_url }}) | [Media]({{ '/media/' | relative_url }}) | [Timeline]({{ '/timeline/' | relative_url }})

## Media Gallery

This page groups visual artifacts by track and milestone.

## Version One Media

<video controls muted playsinline width="760">
  <source src="{{ '/assets/videos/version-one/ppo_modern_eval.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>

![Version One Screenshot Placeholder]({{ '/assets/screenshots/version_one_eval.png' | relative_url }})

## Version Two Media

![Cold Clear Survival Demo]({{ '/assets/gifs/ColdClear.gif' | relative_url }})

<video controls muted playsinline width="760">
  <source src="{{ '/assets/videos/version-two/pygame_autoplay_round15.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Comparisons

<video controls muted playsinline width="760">
  <source src="{{ '/assets/videos/version-two/bc_vs_dagger_side_by_side.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Milestones

| Date | Milestone | Media |
|---|---|---|
| 2025-11-24 | Initial Cold Clear recording | `docs/assets/gifs/ColdClear.gif` |
| 2026-03-31 | BC/DAgger milestone | `TBD` |
| 2026-04-01+ | Random-board DAgger progression | `TBD` |

## External Long-Form Recordings

| Topic | Link | Notes |
|---|---|---|
| Version Two autoplay deep dive | `TBD` | Long-form walkthrough |
| BC vs DAgger analysis | `TBD` | Comparative behavior discussion |
| Version One PPO training recap | `TBD` | Training and evaluation narrative |

## Reusable Embed Snippets

```html
<video controls muted playsinline width="760">
  <source src="{{ '/assets/videos/version-two/<file>.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

```markdown
![Screenshot Caption]({{ '/assets/screenshots/<file>.png' | relative_url }})
```
