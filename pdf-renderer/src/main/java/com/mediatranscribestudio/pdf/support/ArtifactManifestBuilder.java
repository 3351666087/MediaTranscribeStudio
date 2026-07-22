package com.mediatranscribestudio.pdf.support;

import com.mediatranscribestudio.pdf.contract.ArtifactManifest;
import com.mediatranscribestudio.pdf.render.PageImageRenderer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public final class ArtifactManifestBuilder {
    public ArtifactManifest build(
            Path outputRoot,
            String jobId,
            String documentId,
            String rendererVersion,
            String generatedAt,
            boolean published,
            Path reportDocument,
            Path xhtml,
            Path pdf,
            List<PageImageRenderer.PageEvidence> pages,
            Path contactSheet,
            Path qualityReport,
            Path repairQueue
    ) throws IOException {
        ArtifactManifest manifest = new ArtifactManifest();
        manifest.jobId = jobId;
        manifest.documentId = documentId;
        manifest.rendererVersion = rendererVersion;
        manifest.createdAt = generatedAt;
        add(manifest, outputRoot, "report-document", "report-document", reportDocument,
                "application/json", null, List.of());
        if (published) {
            add(manifest, outputRoot, "canonical-xhtml", "canonical-xhtml", xhtml,
                    "application/xhtml+xml", null, List.of("report-document"));
            add(manifest, outputRoot, "pdf", "pdf", pdf,
                    "application/pdf", null, List.of("canonical-xhtml"));
        }
        for (PageImageRenderer.PageEvidence page : pages) {
            add(manifest, outputRoot, String.format("page-image-%03d", page.pageNumber()),
                    "page-image", page.path(), "image/png", page.pageNumber(),
                    published ? List.of("pdf") : List.of());
        }
        add(manifest, outputRoot, "contact-sheet", "contact-sheet", contactSheet,
                "image/png", null,
                pages.stream().map(page -> String.format("page-image-%03d", page.pageNumber())).toList());
        add(manifest, outputRoot, "quality-report", "quality-report", qualityReport,
                "application/json", null,
                published ? List.of("pdf", "contact-sheet") : List.of("contact-sheet"));
        add(manifest, outputRoot, "repair-queue", "repair-queue", repairQueue,
                "application/json", null, List.of("quality-report"));
        return manifest;
    }

    private static void add(
            ArtifactManifest manifest,
            Path root,
            String artifactId,
            String type,
            Path path,
            String mimeType,
            Integer pageNumber,
            List<String> sourceIds
    ) throws IOException {
        if (!Files.isRegularFile(path)) {
            throw new IOException("manifest artifact does not exist: " + path);
        }
        ArtifactManifest.Artifact artifact = new ArtifactManifest.Artifact();
        artifact.artifactId = artifactId;
        artifact.type = type;
        artifact.relativePath = relative(root, path);
        artifact.mimeType = mimeType;
        artifact.sha256 = Hashing.sha256(path);
        artifact.bytes = Files.size(path);
        artifact.verified = true;
        artifact.pageNumber = pageNumber;
        artifact.producedBy = "media-transcribe-studio-pdf-renderer";
        artifact.sourceArtifactIds = sourceIds.isEmpty() ? null : List.copyOf(sourceIds);
        manifest.artifacts.add(artifact);
    }

    private static String relative(Path root, Path path) {
        Path normalizedRoot = root.toAbsolutePath().normalize();
        Path normalizedPath = path.toAbsolutePath().normalize();
        if (!normalizedPath.startsWith(normalizedRoot)) {
            throw new IllegalArgumentException("artifact is outside output root: " + normalizedPath);
        }
        return normalizedRoot.relativize(normalizedPath).toString().replace('\\', '/');
    }
}
