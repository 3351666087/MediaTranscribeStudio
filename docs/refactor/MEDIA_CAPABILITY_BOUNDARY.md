# Media capability boundary

MediaTranscribeStudio does not define “supported media” with a fixed extension
allowlist. File-picker extensions are convenience hints only. The trusted
admission decision is made from the content reported by the installed local
FFprobe build and, in production mode, a bounded FFmpeg first-frame decode
smoke test.

## Admission sequence

1. Require an absolute local path.
2. Reject URLs, UNC/device paths, missing paths and non-regular files.
3. Canonicalize the source path without modifying the source.
4. Run FFprobe without a shell:
   - `-print_format json`
   - `-show_format`
   - `-show_streams`
   - `-show_programs`
   - `-show_chapters`
5. Enforce a process timeout and bounded stdout/stderr.
6. Reject malformed JSON, failed probing, missing audio/video streams, unknown
   media codecs and encryption/DRM evidence.
7. Record container aliases, every stream, audio/video/subtitle indexes,
   existing subtitle codecs, duration, dimensions, sample layout, rotation and
   HDR/color metadata.
8. Record exact FFprobe/FFmpeg version and configuration fingerprints.
9. Decode one frame from the first audio and/or non-attached video stream into
   FFmpeg’s null muxer. This proves that the local build can decode the content
   before expensive transcription work starts.

This policy admits extensionless files and files with uncommon or misleading
extensions when their content is supported. It rejects a familiar extension
when the content is malformed, encrypted, unsupported or not media.

## Source safety

The probe is read-only. Subtitle sidecars, soft-muxed copies and burned-in
copies are separate output artifacts governed by the subtitle output contract.
No command produced by this boundary grants permission to overwrite source
media.

## Evidence

`contracts/media-probe.schema.json` defines the portable evidence document.
`probeFingerprintSha256` binds canonical stream metadata and local toolchain
fingerprints so later subtitle mux/burn QA can prove that it acted on the media
that passed admission.
