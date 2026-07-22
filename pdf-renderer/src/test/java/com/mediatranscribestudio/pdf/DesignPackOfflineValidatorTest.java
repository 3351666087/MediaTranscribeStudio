package com.mediatranscribestudio.pdf;

import com.mediatranscribestudio.pdf.support.DesignPackOfflineValidator;
import com.mediatranscribestudio.pdf.support.OfflineValidationResult;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class DesignPackOfflineValidatorTest {
    private final DesignPackOfflineValidator validator = new DesignPackOfflineValidator();

    @Test
    void selfContainedXhtmlSatisfiesEmbeddedContract() throws Exception {
        OfflineValidationResult result = validator.verify(
                "<html xmlns=\"http://www.w3.org/1999/xhtml\"><body>离线</body></html>");
        assertTrue(result.verified());
        assertEquals("codex-offline-frontend-design-runtime/v1", result.contractVersion());
    }

    @Test
    void activeAndRemoteResourcesFailClosed() {
        for (String xhtml : new String[]{
                "<html><body><script>bad()</script></body></html>",
                "<html><body><img src=\"https://example.com/a.png\" /></body></html>",
                "<html><body><img src=\"data:image/png;base64,AA==\" /></body></html>"
        }) {
            assertThrows(IllegalArgumentException.class, () -> validator.verify(xhtml));
        }
    }
}
