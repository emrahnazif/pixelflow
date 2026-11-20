# AGENTS.md: guide for AI agents working on PixelFlow

PixelFlow is a Python library that turns the output of any detection, segmentation or
classification model into one format you can filter, track, count and draw. Its users are
developers who install it from PyPI. Every decision here is judged by what is best for them.

## Ownership

One agent session maintains each project, and only that session changes its repo.
You are the owner and long-term maintainer of this codebase, and its design is up to you.

- You may read other repos, for example to see how they use PixelFlow. Never edit them,
  commit in them, or run git commands that write there.
- Find another project's session with `ListAgents`.

## Requests between projects

When another project needs something from PixelFlow, its session sends a request with
`SendMessage`. When PixelFlow needs something from another project, send one the same way.
Never work around a missing feature by editing the other repo.

A request says:

1. **What** is needed, as the caller would write it (`video.frame_time`, not "timing info").
2. **Why**: the user-facing problem it solves, not only the requester's own workflow.
3. **Proposed API**, if any, and what the requester does today without it.

When PixelFlow receives a request:

- **Judge it only by PixelFlow's users.** Would a typical PixelFlow user want this? Accept,
  propose a more general version, or refuse with a reason. A need only one project has stays
  in that project.
- **Prefer the general fix.** Example: a project needed the frame numbers that
  `VideoReader(rate=)` yields; the general need is the number of the frame you are holding,
  which serves every user.
- **New or changed public API needs the user's approval** before it is committed. Removing
  public API is a breaking change.
- **Reply** with the decision, and later with the version that ships it. Requesters depend on
  released versions from PyPI, never on this working tree.

## Rules

- Images are NumPy arrays, RGB, `uint8`, HxWx3. RGBA (HxWx4) is accepted only by
  `save_image` and `encode_image`. Colour order is converted only at the codec boundary in
  `pixelflow/media.py`: never add a second encoder or decoder.
- Never commit, push, tag or release without the user asking. No `Co-Authored-By` lines.
- Changes go under `## [Unreleased]` in `CHANGELOG.md`, in plain language, one bold first
  sentence per entry.
- Public API changes update the docs page in `docs/` that covers them. Docs follow
  `DOCS-GUIDE.md`.
- Keep the public API small: adding API later is easy, removing it breaks users.

## Before you finish

```bash
python -m pytest -q --no-cov
```

All tests must pass. Show the result; don't just say it passed.

## Releasing (only when the user asks)

1. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a short
   summary paragraph, and add a new empty `## [Unreleased]` above it.
2. Set `__version__` in `pixelflow/__init__.py`.
3. Run the tests, and `python scripts/release_notes.py X.Y.Z`.
4. Commit "Release X.Y.Z: ...", then tag: `git tag -a vX.Y.Z -m "Release X.Y.Z"`.
5. Pushing the tag publishes to PyPI and cannot be undone: the user decides when.

## Stop and ask the user

- New, changed or removed public API.
- Releases, tags, pushes, version numbers.
- New dependencies or licence changes.
- Deleting or rewriting files or commits you did not make.
