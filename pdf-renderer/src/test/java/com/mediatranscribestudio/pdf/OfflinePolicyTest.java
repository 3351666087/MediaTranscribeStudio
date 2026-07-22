package com.mediatranscribestudio.pdf;

import com.mediatranscribestudio.pdf.support.OfflinePolicy;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

final class OfflinePolicyTest {
    @ParameterizedTest
    @ValueSource(strings = {
            "<html><body><script src=\"https://example.test/a.js\"></script></body></html>",
            "<html><body><iframe src='//example.test/'></iframe></body></html>",
            "<html><body><object data=\"https://example.test/a\"></object></body></html>",
            "<html><body><embed src=\"data:text/plain,x\" /></body></html>",
            "<html><head><base href=\"https://example.test/\" /></head><body /></html>",
            "<html><head><link rel=\"stylesheet\" href=\"styles.css\" /></head><body /></html>",
            "<html><body><img src=\"https://example.test/a.png\" /></body></html>",
            "<html><body><img src='file:///tmp/a.png' /></body></html>",
            "<html><body><img src=\"data:image/png;base64,AA==\" /></body></html>",
            "<html><body><img srcset=\"a.png 1x, b.png 2x\" /></body></html>",
            "<html><body><a href=\"//example.test/a\">x</a></body></html>",
            "<html><body><form action=\"https://example.test/\"></form></body></html>",
            "<html><body><div onclick=\"location.href='https://example.test/'\">x</div></body></html>",
            "<html><head><meta http-equiv=\"refresh\" content=\"0;url=https://example.test/\" /></head></html>",
            "<html><head><style>@import 'https://example.test/a.css';</style></head></html>",
            "<html><head><style>.x{background:url(https://example.test/a.png)}</style></head></html>",
            "<html><body><img src=https://example.test/a.png /></body></html>"
    })
    void rejectsActiveOrNonLocalResources(String xhtml) {
        assertThrows(IllegalArgumentException.class,
                () -> OfflinePolicy.requireOfflineXhtml(xhtml));
    }

    @Test
    void acceptsOnlySelfContainedMarkupAndInternalFragments() {
        assertDoesNotThrow(() -> OfflinePolicy.requireOfflineXhtml(
                "<html><head><style>.x{color:#123456}</style></head>"
                        + "<body><a href=\"#section\">跳转</a><div id=\"section\">内容</div></body></html>"
        ));
    }

    @Test
    void rejectsMissingMarkup() {
        assertThrows(IllegalArgumentException.class,
                () -> OfflinePolicy.requireOfflineXhtml(null));
        assertThrows(IllegalArgumentException.class,
                () -> OfflinePolicy.requireOfflineXhtml(" \n\t "));
    }
}
