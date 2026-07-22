package com.mediatranscribestudio.pdf.validation;

import com.mediatranscribestudio.pdf.contract.RenderRequest;
import com.mediatranscribestudio.pdf.qa.AestheticFacetId;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Objects;
import java.util.regex.Pattern;

public final class RenderRequestValidator {
    private static final Pattern URL =
            Pattern.compile("^[A-Za-z][A-Za-z0-9+.-]*://.*");

    private RenderRequestValidator() {
    }

    public static Paths validate(RenderRequest request) {
        require(request != null, "request is required");
        require("1.0.0".equals(request.schemaVersion), "schemaVersion must be 1.0.0");
        requireText(request.requestId, "requestId", 160);
        requireText(request.jobId, "jobId", 160);
        requireText(request.reportDocumentPath, "reportDocumentPath", 1000);
        requireText(request.outputDirectory, "outputDirectory", 1000);
        require(request.renderer != null, "renderer is required");
        require("java-openhtmltopdf".equals(request.renderer.provider),
                "renderer.provider must be java-openhtmltopdf");
        require(Boolean.TRUE.equals(request.renderer.offline), "renderer.offline must be true");
        require(request.renderer.page != null, "renderer.page is required");
        require("A4".equals(request.renderer.page.size), "renderer.page.size must be A4");
        require(request.renderer.page.marginMm != null
                        && request.renderer.page.marginMm >= 8
                        && request.renderer.page.marginMm <= 30,
                "renderer.page.marginMm must be in [8,30]");
        require(request.renderer.fontPolicy != null, "renderer.fontPolicy is required");
        require(Boolean.TRUE.equals(request.renderer.fontPolicy.requireEmbeddedCjk),
                "renderer.fontPolicy.requireEmbeddedCjk must be true");
        require(Boolean.FALSE.equals(request.renderer.fontPolicy.allowSystemFallback),
                "renderer.fontPolicy.allowSystemFallback must be false");
        if (request.renderer.fontPolicy.preferredFont != null) {
            require("LXGW WenKai".equals(request.renderer.fontPolicy.preferredFont),
                    "only the bundled LXGW WenKai font is supported");
        }
        require(request.qualityPolicy != null, "qualityPolicy is required");
        require(Boolean.TRUE.equals(request.qualityPolicy.requireHardGates),
                "qualityPolicy.requireHardGates must be true");
        require(request.qualityPolicy.minimumScore != null
                        && request.qualityPolicy.minimumScore >= 85
                        && request.qualityPolicy.minimumScore <= 100,
                "qualityPolicy.minimumScore must be in [85,100]");
        require(request.qualityPolicy.maxRounds != null
                        && request.qualityPolicy.maxRounds >= 1
                        && request.qualityPolicy.maxRounds <= 5,
                "qualityPolicy.maxRounds must be in [1,5]");
        require(request.qualityPolicy.captureDpi != null
                        && request.qualityPolicy.captureDpi >= 96
                        && request.qualityPolicy.captureDpi <= 300,
                "qualityPolicy.captureDpi must be in [96,300]");
        require(Objects.equals(AestheticFacetId.ids(), request.qualityPolicy.facetIds),
                "qualityPolicy.facetIds must exactly match the canonical 14-item order");

        rejectRemotePathSyntax(request.reportDocumentPath, "reportDocumentPath");
        rejectRemotePathSyntax(request.outputDirectory, "outputDirectory");
        Path report = Path.of(request.reportDocumentPath).toAbsolutePath().normalize();
        Path output = Path.of(request.outputDirectory).toAbsolutePath().normalize();
        require(Files.isRegularFile(report) && Files.isReadable(report),
                "reportDocumentPath must be a readable regular file");
        require(report.startsWith(output),
                "reportDocumentPath must be inside outputDirectory");
        rejectLinkedPath(report);
        rejectLinkedPath(output);
        return new Paths(report, output);
    }

    private static void rejectRemotePathSyntax(String value, String field) {
        require(!URL.matcher(value).matches(), field + " must not be a URL");
        require(!value.startsWith("\\\\") && !value.startsWith("//"),
                field + " must not be a UNC share");
    }

    private static void rejectLinkedPath(Path candidate) {
        Path current = candidate;
        while (current != null) {
            if (Files.isSymbolicLink(current)) {
                throw new IllegalArgumentException("symbolic links are forbidden: " + current);
            }
            if (Files.exists(current, LinkOption.NOFOLLOW_LINKS)) {
                try {
                    BasicFileAttributes attributes = Files.readAttributes(
                            current, BasicFileAttributes.class, LinkOption.NOFOLLOW_LINKS);
                    if (attributes.isOther()) {
                        throw new IllegalArgumentException(
                                "junctions and reparse points are forbidden: " + current);
                    }
                } catch (IOException exception) {
                    throw new IllegalArgumentException(
                            "unable to validate local path: " + current, exception);
                }
            }
            current = current.getParent();
        }
    }

    private static void requireText(String value, String field, int max) {
        require(value != null && !value.isBlank() && value.length() <= max,
                field + " must contain 1-" + max + " characters");
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException(message);
        }
    }

    public record Paths(Path reportDocument, Path outputDirectory) {
    }
}
