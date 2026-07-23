package com.mediatranscribestudio.pdf.validation;

import com.fasterxml.jackson.databind.JsonNode;
import com.mediatranscribestudio.pdf.contract.ReportDocument;

import java.time.OffsetDateTime;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.IllformedLocaleException;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

public final class ReportDocumentValidator {
    private static final int MAX_LANGUAGE_TAG_LENGTH = 255;
    private static final String UNKNOWN_LANGUAGE_TAG = "und";
    private static final Pattern PRIMARY_LANGUAGE = Pattern.compile("^[A-Za-z]{2,8}$");
    private static final Pattern EXTLANG = Pattern.compile("^[A-Za-z]{3}$");
    private static final Pattern SCRIPT = Pattern.compile("^[A-Za-z]{4}$");
    private static final Pattern REGION = Pattern.compile("^(?:[A-Za-z]{2}|[0-9]{3})$");
    private static final Pattern VARIANT =
            Pattern.compile("^(?:[A-Za-z0-9]{5,8}|[0-9][A-Za-z0-9]{3})$");
    private static final Pattern EXTENSION_SINGLETON =
            Pattern.compile("^[0-9A-WY-Za-wy-z]$");
    private static final Pattern EXTENSION_SUBTAG = Pattern.compile("^[A-Za-z0-9]{2,8}$");
    private static final Pattern PRIVATE_USE_SUBTAG = Pattern.compile("^[A-Za-z0-9]{1,8}$");
    private static final Set<String> GRANDFATHERED_LANGUAGE_TAGS = Set.of(
            "art-lojban",
            "cel-gaulish",
            "en-gb-oed",
            "i-ami",
            "i-bnn",
            "i-default",
            "i-enochian",
            "i-hak",
            "i-klingon",
            "i-lux",
            "i-mingo",
            "i-navajo",
            "i-pwn",
            "i-tao",
            "i-tay",
            "i-tsu",
            "no-bok",
            "no-nyn",
            "sgn-be-fr",
            "sgn-be-nl",
            "sgn-ch-de",
            "zh-guoyu",
            "zh-hakka",
            "zh-min",
            "zh-min-nan",
            "zh-xiang"
    );
    private static final Set<String> RTL_LANGUAGES = Set.of(
            "ar", "arc", "ckb", "dv", "fa", "he", "khw", "ks", "nqo",
            "ps", "sd", "syr", "ug", "ur", "yi"
    );
    private static final Set<String> RTL_SCRIPTS = Set.of(
            "Adlm", "Arab", "Hebr", "Nkoo", "Rohg", "Samr", "Syrc", "Thaa", "Yezi"
    );
    private static final Set<String> MODES = Set.of("auto", "manual", "hybrid");
    private static final Set<String> REVISION_TYPES =
            Set.of("text", "speaker", "boundary", "split", "merge");
    private static final Set<String> REVISION_SOURCES =
            Set.of("acoustic", "deterministic", "llm", "manual");
    private static final Set<String> FORBIDDEN_REVISION_TYPES =
            Set.of("translation", "summary", "analysis");

    private ReportDocumentValidator() {
    }

    public static void validate(ReportDocument document) {
        require(document != null, "report document is required");
        require("1.0.0".equals(document.schemaVersion), "schemaVersion must be 1.0.0");
        requireText(document.documentId, "documentId");
        try {
            OffsetDateTime.parse(document.generatedAt);
        } catch (DateTimeParseException | NullPointerException exception) {
            throw new IllegalArgumentException("generatedAt must be RFC 3339", exception);
        }
        document.language = requireLanguageTag(document.language, "language");
        if (document.reportLocale != null) {
            document.reportLocale = requireLanguageTag(document.reportLocale, "reportLocale");
        }
        require(document.source != null, "source is required");
        requireText(document.source.fileName, "source.fileName");
        requireText(document.source.mediaType, "source.mediaType");
        require(document.source.durationMs != null && document.source.durationMs >= 1,
                "source.durationMs must be >= 1");
        validatePolicy(document.speakerPolicy);

        int count = document.speakerPolicy.resolvedCount;
        List<String> canonical = canonicalSpeakerIds(count);
        require(document.speakers != null && document.speakers.size() == count,
                "speakers.size must equal resolvedCount");
        require(canonical.equals(document.speakerPolicy.speakerIds),
                "speakerPolicy.speakerIds must be speaker-1..speaker-N");
        Set<String> speakerIds = new HashSet<>();
        for (int index = 0; index < count; index++) {
            ReportDocument.Speaker speaker = document.speakers.get(index);
            require(speaker != null, "speaker is required");
            require(canonical.get(index).equals(speaker.id), "speaker IDs must be canonical and ordered");
            require(Integer.valueOf(index + 1).equals(speaker.order), "speaker.order must be contiguous 1..N");
            require(("speaker." + (index + 1)).equals(speaker.colorToken),
                    "speaker.colorToken must match speaker.order");
            requireText(speaker.displayName, "speaker.displayName");
            requireText(speaker.shortLabel, "speaker.shortLabel");
            require(speakerIds.add(speaker.id), "duplicate speaker ID");
        }

        require(document.segments != null && !document.segments.isEmpty(), "segments must not be empty");
        Set<String> segmentIds = new HashSet<>();
        Set<String> revisionIds = new HashSet<>();
        Set<String> usedSpeakers = new HashSet<>();
        long previousStart = -1;
        for (ReportDocument.Segment segment : document.segments) {
            require(segment != null, "segment is required");
            requireText(segment.id, "segment.id");
            require(segmentIds.add(segment.id), "segment IDs must be unique");
            require(segment.startMs != null && segment.startMs >= 0, "segment.startMs must be >= 0");
            require(segment.endMs != null && segment.endMs > segment.startMs,
                    "segment.endMs must be greater than startMs");
            require(segment.endMs <= document.source.durationMs,
                    "segment.endMs exceeds source duration");
            require(segment.startMs >= previousStart, "segments must be ordered by startMs");
            previousStart = segment.startMs;
            require(speakerIds.contains(segment.speakerId), "segment has unknown speakerId");
            usedSpeakers.add(segment.speakerId);
            requireText(segment.rawText, "segment.rawText");
            requireText(segment.normalizedText, "segment.normalizedText");
            requireText(segment.displayText, "segment.displayText");
            segment.language = segment.language == null
                    ? UNKNOWN_LANGUAGE_TAG
                    : requireLanguageTag(segment.language, "segment.language");
            require(segment.confidence != null && unit(segment.confidence),
                    "segment.confidence must be in [0,1]");
            validateEvidence(segment, canonical);
            validateTextAudit(segment, revisionIds);
        }
        require(usedSpeakers.equals(speakerIds), "every declared speaker must occur in the transcript");
        require(document.provenance != null, "provenance is required");
        requireText(document.provenance.pipelineVersion, "provenance.pipelineVersion");
        require(Boolean.TRUE.equals(document.provenance.offline), "provenance.offline must be true");
        require(document.provenance.models != null && !document.provenance.models.isEmpty(),
                "provenance.models must not be empty");
    }

    public static List<String> canonicalSpeakerIds(int count) {
        require(count >= 1, "speaker count must be positive");
        return IntStream.rangeClosed(1, count).mapToObj(i -> "speaker-" + i).toList();
    }

    public static String canonicalizeLanguageTag(String value) {
        if (value == null
                || value.isEmpty()
                || !value.equals(value.trim())
                || value.length() > MAX_LANGUAGE_TAG_LENGTH
                || value.chars().anyMatch(codePoint -> codePoint > 0x7f)) {
            throw invalidLanguageTag();
        }

        String normalized = value.replace('_', '-');
        String normalizedLowerCase = normalized.toLowerCase(Locale.ROOT);
        if (normalized.startsWith("-")
                || normalized.endsWith("-")
                || normalized.contains("--")
                || "auto".equals(normalizedLowerCase)) {
            throw invalidLanguageTag();
        }

        if (GRANDFATHERED_LANGUAGE_TAGS.contains(normalizedLowerCase)) {
            return canonicalizeWithLocaleBuilder(normalized);
        }

        String[] subtags = normalized.split("-", -1);
        List<String> output = new ArrayList<>(subtags.length);
        int index = 0;

        if ("x".equalsIgnoreCase(subtags[0])) {
            if (subtags.length == 1) {
                throw invalidLanguageTag();
            }
            output.add("x");
            for (index = 1; index < subtags.length; index++) {
                if (!PRIVATE_USE_SUBTAG.matcher(subtags[index]).matches()) {
                    throw invalidLanguageTag();
                }
                output.add(subtags[index].toLowerCase(Locale.ROOT));
            }
            return canonicalizeWithLocaleBuilder(String.join("-", output));
        }

        String primary = subtags[index];
        if (!PRIMARY_LANGUAGE.matcher(primary).matches()) {
            throw invalidLanguageTag();
        }
        output.add(primary.toLowerCase(Locale.ROOT));
        index++;

        if (primary.length() <= 3) {
            int extlangCount = 0;
            while (index < subtags.length
                    && extlangCount < 3
                    && EXTLANG.matcher(subtags[index]).matches()) {
                output.add(subtags[index].toLowerCase(Locale.ROOT));
                index++;
                extlangCount++;
            }
        }

        if (index < subtags.length && SCRIPT.matcher(subtags[index]).matches()) {
            String script = subtags[index].toLowerCase(Locale.ROOT);
            output.add(script.substring(0, 1).toUpperCase(Locale.ROOT) + script.substring(1));
            index++;
        }

        if (index < subtags.length && REGION.matcher(subtags[index]).matches()) {
            String region = subtags[index];
            output.add(region.chars().allMatch(Character::isLetter)
                    ? region.toUpperCase(Locale.ROOT)
                    : region);
            index++;
        }

        Set<String> variants = new HashSet<>();
        while (index < subtags.length && VARIANT.matcher(subtags[index]).matches()) {
            String variant = subtags[index].toLowerCase(Locale.ROOT);
            if (!variants.add(variant)) {
                throw invalidLanguageTag();
            }
            output.add(variant);
            index++;
        }

        Set<String> extensionSingletons = new HashSet<>();
        while (index < subtags.length
                && EXTENSION_SINGLETON.matcher(subtags[index]).matches()) {
            String singleton = subtags[index].toLowerCase(Locale.ROOT);
            if (!extensionSingletons.add(singleton)) {
                throw invalidLanguageTag();
            }
            output.add(singleton);
            index++;

            int extensionStart = index;
            while (index < subtags.length
                    && EXTENSION_SUBTAG.matcher(subtags[index]).matches()) {
                output.add(subtags[index].toLowerCase(Locale.ROOT));
                index++;
            }
            if (index == extensionStart) {
                throw invalidLanguageTag();
            }
        }

        if (index < subtags.length && "x".equalsIgnoreCase(subtags[index])) {
            output.add("x");
            index++;
            int privateUseStart = index;
            while (index < subtags.length
                    && PRIVATE_USE_SUBTAG.matcher(subtags[index]).matches()) {
                output.add(subtags[index].toLowerCase(Locale.ROOT));
                index++;
            }
            if (index == privateUseStart) {
                throw invalidLanguageTag();
            }
        }

        if (index != subtags.length) {
            throw invalidLanguageTag();
        }
        return canonicalizeWithLocaleBuilder(String.join("-", output));
    }

    public static boolean isRightToLeftLanguage(String value) {
        String language = canonicalizeLanguageTag(value);
        String[] subtags = language.split("-");
        if ("x".equals(subtags[0])) {
            return false;
        }

        int index = 1;
        if (subtags[0].length() <= 3) {
            int extlangCount = 0;
            while (index < subtags.length
                    && extlangCount < 3
                    && EXTLANG.matcher(subtags[index]).matches()) {
                index++;
                extlangCount++;
            }
        }
        if (index < subtags.length && SCRIPT.matcher(subtags[index]).matches()) {
            return RTL_SCRIPTS.contains(subtags[index]);
        }
        return RTL_LANGUAGES.contains(subtags[0]);
    }

    private static void validatePolicy(ReportDocument.SpeakerPolicy policy) {
        require(policy != null, "speakerPolicy is required");
        require(MODES.contains(policy.mode), "speakerPolicy.mode must be auto/manual/hybrid");
        require(policy.resolvedCount != null && policy.resolvedCount >= 1,
                "speakerPolicy.resolvedCount must be positive");
        require(Boolean.TRUE.equals(policy.requireExactSet), "speakerPolicy.requireExactSet must be true");
        require(Boolean.FALSE.equals(policy.unknownSpeakerAllowed),
                "speakerPolicy.unknownSpeakerAllowed must be false");
        require(Boolean.TRUE.equals(policy.speakerChangeRequiresEvidence),
                "speakerPolicy.speakerChangeRequiresEvidence must be true");
        if ("manual".equals(policy.mode) || "hybrid".equals(policy.mode)) {
            require(policy.resolvedCount.equals(policy.requestedCount),
                    "manual/hybrid requestedCount must equal resolvedCount");
        }
        if ("auto".equals(policy.mode) || "hybrid".equals(policy.mode)) {
            require(policy.detection != null
                            && policy.resolvedCount.equals(policy.detection.estimatedCount),
                    "auto/hybrid detection must resolve the declared count");
            requireText(policy.detection.provider, "speakerPolicy.detection.provider");
            if (policy.detection.confidence != null) {
                require(unit(policy.detection.confidence),
                        "speakerPolicy.detection.confidence must be in [0,1]");
            }
            if (policy.detection.candidates != null) {
                Set<Integer> candidateCounts = new HashSet<>();
                for (ReportDocument.CountCandidate candidate : policy.detection.candidates) {
                    require(candidate != null, "speaker count candidate is required");
                    require(candidate.count != null && candidate.count >= 1,
                            "speaker count candidate.count must be positive");
                    require(candidateCounts.add(candidate.count),
                            "speaker count candidate counts must be unique");
                    require(candidate.confidence != null && unit(candidate.confidence),
                            "speaker count candidate.confidence must be in [0,1]");
                }
            }
        }
        if (policy.minimumCount != null) {
            require(policy.minimumCount >= 1 && policy.minimumCount <= policy.resolvedCount,
                    "minimumCount exceeds resolvedCount");
        }
        if (policy.maximumCount != null) {
            require(policy.maximumCount >= policy.resolvedCount,
                    "maximumCount is below resolvedCount");
        }
    }

    private static void validateEvidence(ReportDocument.Segment segment, List<String> canonical) {
        require(segment.evidence != null, "segment.evidence is required");
        require(segment.evidence.asr != null, "evidence.asr is required");
        requireText(segment.evidence.asr.provider, "evidence.asr.provider");
        requireText(segment.evidence.asr.model, "evidence.asr.model");
        require(segment.evidence.asr.confidence != null && unit(segment.evidence.asr.confidence),
                "evidence.asr.confidence must be in [0,1]");
        require(segment.evidence.boundary != null, "evidence.boundary is required");
        requireText(segment.evidence.boundary.provider, "evidence.boundary.provider");
        require(segment.evidence.boundary.confidence != null
                        && unit(segment.evidence.boundary.confidence),
                "evidence.boundary.confidence must be in [0,1]");
        require(segment.evidence.speaker != null, "evidence.speaker is required");
        requireText(segment.evidence.speaker.provider, "evidence.speaker.provider");
        require(segment.speakerId.equals(segment.evidence.speaker.assignment),
                "speaker assignment must equal segment.speakerId");
        require(segment.evidence.speaker.locked != null, "evidence.speaker.locked is required");
        if (segment.evidence.speaker.margin != null) {
            require(segment.evidence.speaker.margin >= -2
                            && segment.evidence.speaker.margin <= 2
                            && Double.isFinite(segment.evidence.speaker.margin),
                    "speaker margin must be finite and in [-2,2]");
        }
        require(segment.evidence.speaker.scores != null
                        && segment.evidence.speaker.scores.size() == canonical.size(),
                "speaker score vector must cover all speakers");
        for (int index = 0; index < canonical.size(); index++) {
            ReportDocument.SpeakerScore score = segment.evidence.speaker.scores.get(index);
            require(score != null && canonical.get(index).equals(score.speakerId),
                    "speaker score vector must preserve canonical order");
            require(score.score != null && score.score >= -2 && score.score <= 2
                            && Double.isFinite(score.score),
                    "speaker score must be finite and in [-2,2]");
        }
        if (segment.evidence.semantic != null) {
            require(!Boolean.TRUE.equals(segment.evidence.semantic.autoApplied),
                    "semantic.autoApplied must not be true");
        }
    }

    private static void validateTextAudit(ReportDocument.Segment segment, Set<String> revisionIds) {
        require(segment.revisions != null, "segment.revisions is required");
        if (segment.revisions.isEmpty()) {
            require(segment.rawText.equals(segment.normalizedText)
                            && segment.rawText.equals(segment.displayText),
                    "segments without revisions must preserve rawText as normalizedText and displayText");
            return;
        }

        String expectedBefore = segment.rawText;
        boolean hasManualRevision = false;
        for (ReportDocument.Revision revision : segment.revisions) {
            require(revision != null, "revision is required");
            requireText(revision.revisionId, "revision.revisionId");
            require(revisionIds.add(revision.revisionId), "revision IDs must be globally unique");
            requireText(revision.type, "revision.type");
            require(!FORBIDDEN_REVISION_TYPES.contains(revision.type),
                    "translation, summary, and analysis revisions are forbidden");
            require(REVISION_TYPES.contains(revision.type), "revision.type is unsupported");
            requireText(revision.source, "revision.source");
            require(REVISION_SOURCES.contains(revision.source), "revision.source is unsupported");
            requireText(revision.reasonCode, "revision.reasonCode");
            requireText(revision.actor, "revision.actor");
            try {
                OffsetDateTime.parse(revision.occurredAt);
            } catch (DateTimeParseException | NullPointerException exception) {
                throw new IllegalArgumentException("revision.occurredAt must be RFC 3339", exception);
            }
            String before = textNode(revision.before, "revision.before");
            String after = textNode(revision.after, "revision.after");
            require(expectedBefore.equals(before), "revision chain must start at rawText and be continuous");
            expectedBefore = after;
            hasManualRevision |= "manual".equals(revision.source);
        }
        require(hasManualRevision, "edited text requires at least one manual revision");
        require(expectedBefore.equals(segment.normalizedText)
                        && expectedBefore.equals(segment.displayText),
                "final revision must equal normalizedText and displayText");
    }

    private static String textNode(JsonNode node, String field) {
        require(node != null && node.isTextual() && !node.textValue().isBlank(),
                field + " must be a non-empty text node");
        return node.textValue();
    }

    private static boolean unit(double value) {
        return value >= 0 && value <= 1 && Double.isFinite(value);
    }

    private static void requireText(String value, String field) {
        require(value != null && !value.isBlank(), field + " is required");
    }

    private static String requireLanguageTag(String value, String field) {
        try {
            return canonicalizeLanguageTag(value);
        } catch (IllegalArgumentException exception) {
            throw new IllegalArgumentException(
                    field + " must be a concrete persisted BCP-47 language tag, not auto",
                    exception
            );
        }
    }

    private static IllegalArgumentException invalidLanguageTag() {
        return new IllegalArgumentException(
                "language must be a concrete persisted BCP-47 language tag, not auto"
        );
    }

    private static String canonicalizeWithLocaleBuilder(String value) {
        try {
            String canonical = new Locale.Builder()
                    .setLanguageTag(value)
                    .build()
                    .toLanguageTag();
            if (canonical.isEmpty()) {
                throw invalidLanguageTag();
            }
            return canonical;
        } catch (IllformedLocaleException exception) {
            throw invalidLanguageTag();
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException(message);
        }
    }
}
