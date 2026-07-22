package com.mediatranscribestudio.pdf.support;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public final class DesignPackOfflineValidator {
    private static final String CONTRACT_RESOURCE =
            "/design-validation/frontend-design-pack-contract.json";
    private static final String EXPECTED_CONTRACT =
            "codex-offline-frontend-design-runtime/v1";
    private static final String EXPECTED_POLICY =
            "frontend-design-pack-global-offline-2026-07-22";

    public OfflineValidationResult verify(String xhtml) throws IOException {
        Contract contract = loadContract();
        require("1.0.0".equals(contract.schemaVersion), "unsupported design contract schema");
        require(EXPECTED_POLICY.equals(contract.policyVersion), "design policy version drift");
        require(EXPECTED_CONTRACT.equals(contract.contract), "design contract version drift");
        require(!contract.networkRequired, "design contract unexpectedly requires network access");
        require(contract.deterministic, "design contract must be deterministic");
        require(contract.runtimeOfflineReady, "design runtime is not offline-ready");
        require(!contract.implementationReady,
                "external Design Pack readiness flag changed; refresh the audited snapshot");
        require(contract.quality != null
                        && contract.quality.minimumScore == 85
                        && contract.quality.maximumPasses == 5,
                "design quality contract drift");
        require(contract.paths != null
                        && contract.paths.localPathsOnly
                        && contract.paths.rejectUrls
                        && contract.paths.rejectUncShares
                        && contract.paths.rejectSymlinksAndJunctions
                        && contract.paths.sessionMustBeInsideProjectRoot
                        && !contract.paths.sessionMayEqualProjectRoot,
                "design path contract drift");
        OfflinePolicy.requireOfflineXhtml(xhtml);
        return new OfflineValidationResult(
                true,
                contract.policyVersion,
                contract.contract,
                Hashing.sha256(xhtml),
                List.of(
                        "embedded-contract-verified",
                        "network-disabled",
                        "self-contained-xhtml",
                        "deterministic-policy",
                        "local-path-policy"
                ),
                null
        );
    }

    private static Contract loadContract() throws IOException {
        try (InputStream input =
                     DesignPackOfflineValidator.class.getResourceAsStream(CONTRACT_RESOURCE)) {
            if (input == null) {
                throw new IOException("embedded frontend design validation contract is missing");
            }
            return JsonSupport.mapper().readValue(input, Contract.class);
        } catch (RuntimeException exception) {
            throw new IOException("embedded frontend design validation contract is invalid",
                    exception);
        }
    }

    private static void require(boolean condition, String message) throws IOException {
        if (!condition) {
            throw new IOException(message);
        }
    }

    public static final class Contract {
        public String schemaVersion;
        public String policyVersion;
        public String contract;
        public boolean networkRequired;
        public boolean deterministic;
        public boolean runtimeOfflineReady;
        public boolean implementationReady;
        public Quality quality;
        public Paths paths;
    }

    public static final class Quality {
        public int minimumScore;
        public int maximumPasses;
    }

    public static final class Paths {
        public boolean localPathsOnly;
        public boolean rejectUrls;
        public boolean rejectUncShares;
        public boolean rejectSymlinksAndJunctions;
        public boolean sessionMustBeInsideProjectRoot;
        public boolean sessionMayEqualProjectRoot;
    }
}
