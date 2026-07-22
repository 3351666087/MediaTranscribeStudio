package com.mediatranscribestudio.pdf;

import com.fasterxml.jackson.databind.JsonNode;
import com.mediatranscribestudio.pdf.contract.RenderRequest;
import com.mediatranscribestudio.pdf.support.JsonSupport;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class MainProtocolTest {
    @TempDir
    Path temporaryDirectory;

    @Test
    void validRequestProducesExactlyOneJsonDocument() throws Exception {
        Path output = temporaryDirectory.resolve("valid");
        RenderRequest request = TestFixtures.request(output, 2);
        Path requestPath = output.resolve(".render/request.json");
        JsonSupport.write(requestPath, request);
        Protocol protocol = invoke("--request", requestPath.toString());
        assertEquals(0, protocol.exitCode);
        JsonNode result = JsonSupport.mapper().readTree(protocol.stdout.trim());
        assertEquals("passed", result.path("status").asText());
        assertEquals(1, protocol.stdout.lines().count());
    }

    @Test
    void unknownFieldsTrailingJsonAndMalformedInputFailClosed() throws Exception {
        String valid = JsonSupport.compact(TestFixtures.request(
                temporaryDirectory.resolve("negative-base"), 1));
        for (String payload : new String[]{
                valid.substring(0, valid.length() - 1) + ",\"unknown\":true}",
                valid + " {}",
                "{\"schemaVersion\":"
        }) {
            Path requestPath = temporaryDirectory.resolve(
                    "bad-" + Math.abs(payload.hashCode()) + ".json");
            Files.writeString(requestPath, payload, StandardCharsets.UTF_8);
            Protocol protocol = invoke("--request", requestPath.toString());
            assertNotEquals(0, protocol.exitCode);
            JsonNode result = JsonSupport.mapper().readTree(protocol.stdout.trim());
            assertEquals("failed", result.path("status").asText());
            assertEquals(1, protocol.stdout.lines().count());
            assertTrue(protocol.stderr.contains("PDF renderer failed"));
        }
    }

    @Test
    void invalidArgumentsStillEmitOneFailureJsonAndNonzeroExit() throws Exception {
        Protocol protocol = invoke("--wrong");
        assertNotEquals(0, protocol.exitCode);
        assertEquals("failed",
                JsonSupport.mapper().readTree(protocol.stdout.trim()).path("status").asText());
        assertEquals(1, protocol.stdout.lines().count());
    }

    private static Protocol invoke(String... args) {
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        ByteArrayOutputStream stderr = new ByteArrayOutputStream();
        int exit = Main.run(
                args,
                new PrintStream(stdout, true, StandardCharsets.UTF_8),
                new PrintStream(stderr, true, StandardCharsets.UTF_8)
        );
        return new Protocol(
                exit,
                stdout.toString(StandardCharsets.UTF_8),
                stderr.toString(StandardCharsets.UTF_8)
        );
    }

    private record Protocol(int exitCode, String stdout, String stderr) {
    }
}
