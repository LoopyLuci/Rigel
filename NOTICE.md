# Rigel

Rigel is ModelMistress's own inference engine for Windows and AMD GPUs (HIP), with paged mixture-of-experts. It is a
standalone snapshot, not a fork: it has its own history and no link to its source.

It was made from Betelgeuse (https://github.com/Betelgeuse-AI/Betelgeuse, commit a6b46ade3), which is itself derived from
llama.cpp and ggml (MIT, Copyright (c) 2023-2026 The ggml authors; see LICENSE). The Windows and HIP port, the program
names `rigel-server` and `rigel-cli`, and everything added after this commit are ModelMistress's.

Built by ModelMistress's `src/build.rs` (`cargo build --features rigel-engine`; see docs/rigel.md in that repository).
