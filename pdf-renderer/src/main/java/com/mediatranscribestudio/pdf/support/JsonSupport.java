package com.mediatranscribestudio.pdf.support;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class JsonSupport {
    private static final ObjectMapper MAPPER = new ObjectMapper()
            .setSerializationInclusion(JsonInclude.Include.NON_NULL)
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS)
            .enable(DeserializationFeature.FAIL_ON_NULL_FOR_PRIMITIVES)
            .enable(SerializationFeature.INDENT_OUTPUT)
            .enable(MapperFeature.BLOCK_UNSAFE_POLYMORPHIC_BASE_TYPES)
            .disable(JsonGenerator.Feature.AUTO_CLOSE_TARGET)
            .disable(JsonParser.Feature.AUTO_CLOSE_SOURCE);

    private JsonSupport() {
    }

    public static ObjectMapper mapper() {
        return MAPPER;
    }

    public static <T> T read(Path path, Class<T> type) throws IOException {
        return MAPPER.readValue(Files.readString(path, StandardCharsets.UTF_8), type);
    }

    public static void write(Path path, Object value) throws IOException {
        AtomicFiles.write(path, MAPPER.writerWithDefaultPrettyPrinter().writeValueAsBytes(value));
    }

    public static String compact(Object value) throws IOException {
        return MAPPER.writer().without(SerializationFeature.INDENT_OUTPUT)
                .writeValueAsString(value);
    }
}
