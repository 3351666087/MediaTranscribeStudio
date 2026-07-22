package com.mediatranscribestudio.pdf.support;

import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class OfflinePolicy {
    private static final Pattern RESOURCE_ATTRIBUTE = Pattern.compile(
            "(?is)\\b(src|srcset|href|xlink:href|poster|action|formaction|background)"
                    + "\\s*=\\s*([\"'])(.*?)\\2");
    private static final Pattern UNQUOTED_RESOURCE_ATTRIBUTE = Pattern.compile(
            "(?is)\\b(?:src|srcset|href|xlink:href|poster|action|formaction|background)"
                    + "\\s*=\\s*(?![\"'])");
    private static final Pattern ACTIVE_HANDLER = Pattern.compile(
            "(?is)\\bon[a-z0-9_-]+\\s*=");
    private static final Pattern META_REFRESH = Pattern.compile(
            "(?is)<meta\\b[^>]*http-equiv\\s*=\\s*[\"']?\\s*refresh\\b");
    private static final Pattern CSS_RESOURCE = Pattern.compile(
            "(?is)(?:@import\\b|url\\s*\\()");

    private OfflinePolicy() {
    }

    public static void requireOfflineXhtml(String xhtml) {
        if (xhtml == null || xhtml.isBlank()) {
            throw new IllegalArgumentException("XHTML is required");
        }
        String lower = xhtml.toLowerCase(Locale.ROOT);
        if (lower.contains("<script")
                || lower.contains("<iframe")
                || lower.contains("<object")
                || lower.contains("<embed")
                || lower.contains("<base")
                || lower.contains("<link")
                || META_REFRESH.matcher(xhtml).find()
                || ACTIVE_HANDLER.matcher(xhtml).find()
                || UNQUOTED_RESOURCE_ATTRIBUTE.matcher(xhtml).find()
                || CSS_RESOURCE.matcher(xhtml).find()) {
            throw new IllegalArgumentException("XHTML contains a forbidden active or non-local resource");
        }
        Matcher attributes = RESOURCE_ATTRIBUTE.matcher(xhtml);
        while (attributes.find()) {
            String name = attributes.group(1).toLowerCase(Locale.ROOT);
            String value = attributes.group(3).trim();
            boolean internalFragment = ("href".equals(name) || "xlink:href".equals(name))
                    && value.startsWith("#")
                    && value.length() > 1;
            if (!internalFragment) {
                throw new IllegalArgumentException(
                        "XHTML resource attribute is not an internal fragment: " + name);
            }
        }
    }
}
