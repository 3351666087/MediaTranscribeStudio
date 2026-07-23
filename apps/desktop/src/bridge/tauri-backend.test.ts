import type {
  CreateJobRequest,
  ReviewDecision,
  UpdateSpeakerRequest,
} from "../contracts/studio";
import { ContractValidationError } from "../contracts/runtime-validation";
import { studioFixture } from "../mocks/studio-fixture";
import { TauriDesktopBackend } from "./tauri-backend";

type InvokeCommand = <T>(
  command: string,
  args?: Record<string, unknown>,
) => Promise<T>;

function validCreateRequest(): CreateJobRequest {
  return {
    title: "IPC 动态人数测试",
    mediaPath: "D:\\media\\meeting.mov",
    outputDirectory: "D:\\output",
    strategyId: "balanced",
    language: "sr-Latn-RS",
    localLlmMode: "business",
    localLlmModel: "qwen3.5:4b",
    localLlmEndpoint: "http://127.0.0.1:11434",
    localLlmEndpointPolicy: "loopback-only",
    localLlmAutoApply: false,
    translationTargets: ["en-US", "de-DE"],
    polish: true,
    summary: true,
    outputLocale: "en-US",
    businessPromptVersion: "business-v1",
    speakerPolicy: {
      mode: "hybrid",
      minSpeakers: 2,
      priorCount: 7,
      maxSpeakers: 12,
    },
    speakerLabels: Array.from(
      { length: 7 },
      (_, index) => `角色 ${index + 1}`,
    ),
  };
}

function createBackend(
  implementation: (
    command: string,
    args?: Record<string, unknown>,
  ) => Promise<unknown>,
) {
  const invokeMock = vi.fn(implementation);
  return {
    backend: new TauriDesktopBackend(invokeMock as InvokeCommand),
    invokeMock,
  };
}

describe("TauriDesktopBackend strict IPC boundary", () => {
  it("uses fixed command names with exact typed argument envelopes", async () => {
    const request = validCreateRequest();
    const speakerRequest: UpdateSpeakerRequest = {
      speakerId: "speaker-2",
      label: "产品负责人",
      locked: true,
      reviewStatus: "confirmed",
    };
    const reviewDecision: ReviewDecision = {
      reviewId: "review-181",
      speakerId: "speaker-2",
      normalizedText: "这是经过人工复核的中文原文。",
      reason: "人工判断该片段应归属产品负责人。",
      evidence: "已本地复听并对比前后片段声纹与问答关系。",
      confidence: 0.94,
    };
    const originalReview = structuredClone(studioFixture.reviews[0]);
    const responses: Record<string, unknown> = {
      get_snapshot: structuredClone(studioFixture),
      create_job: {
        accepted: true,
        jobId: "job-ipc-001",
        message: "已创建。",
      },
      cancel_job: null,
      update_speaker: {
        ...structuredClone(studioFixture.speakers[1]),
        label: speakerRequest.label,
        locked: true,
        reviewStatus: "confirmed",
      },
      apply_review_decision: {
        ...originalReview,
        currentSpeakerId: "speaker-2",
        normalizedText: reviewDecision.normalizedText,
        auditTrail: [
          {
            id: "review-181:human:1:1784678400000",
            sequence: 1,
            recordedAtUnixMs: 1784678400000,
            actor: "human",
            reason: reviewDecision.reason,
            evidence: reviewDecision.evidence,
            confidence: reviewDecision.confidence,
            previousSpeakerId: originalReview.currentSpeakerId,
            speakerId: reviewDecision.speakerId,
            previousNormalizedText: originalReview.normalizedText,
            normalizedText: reviewDecision.normalizedText,
          },
        ],
        locked: true,
        reviewed: true,
      },
      open_artifact: {
        artifactId: "artifact-pdf",
        canonicalPath: "D:\\output\\pdf\\中文逐字稿.pdf",
        opened: false,
        message: "路径已通过边界校验。",
      },
    };
    const { backend, invokeMock } = createBackend(async (command) => {
      await Promise.resolve();
      return structuredClone(responses[command]);
    });

    await backend.getSnapshot();
    await backend.createJob(request);
    await backend.cancelJob("job-ipc-001");
    await backend.updateSpeaker(speakerRequest);
    await backend.applyReviewDecision(reviewDecision);
    await backend.openArtifact("artifact-pdf");

    expect(invokeMock.mock.calls).toEqual([
      ["get_snapshot", undefined],
      ["create_job", { request }],
      ["cancel_job", { jobId: "job-ipc-001" }],
      ["update_speaker", { request: speakerRequest }],
      ["apply_review_decision", { decision: reviewDecision }],
      ["open_artifact", { artifactId: "artifact-pdf" }],
    ]);
  });

  it("rejects malformed requests before invoke is reached", async () => {
    const { backend, invokeMock } = createBackend(async () => {
      await Promise.resolve();
      throw new Error("不应调用 IPC");
    });
    const request = validCreateRequest();
    request.speakerLabels = request.speakerLabels.slice(0, -1);

    await expect(backend.createJob(request)).rejects.toBeInstanceOf(
      ContractValidationError,
    );
    await expect(
      backend.updateSpeaker({
        speakerId: "speaker-0",
        label: "非法角色",
        locked: false,
        reviewStatus: "pending",
      }),
    ).rejects.toBeInstanceOf(ContractValidationError);
    await expect(backend.cancelJob("")).rejects.toThrow(/jobId/u);
    await expect(backend.openArtifact("x".repeat(129))).rejects.toThrow(
      /artifactId/u,
    );
    await expect(
      backend.applyReviewDecision({
        reviewId: "review-181",
        speakerId: "speaker-1",
        normalizedText: "人工校对正文。",
        reason: "",
        evidence: "本地复听证据。",
        confidence: 0.9,
      }),
    ).rejects.toBeInstanceOf(ContractValidationError);

    expect(invokeMock).not.toHaveBeenCalled();
  });

  it("rejects malformed or extended backend responses instead of trusting IPC data", async () => {
    const malformedSnapshot = {
      ...structuredClone(studioFixture),
      fixedSpeakerLimit: 5,
    };
    const { backend } = createBackend(async () => {
      await Promise.resolve();
      return malformedSnapshot;
    });

    await expect(backend.getSnapshot()).rejects.toThrow(/unknown fields/u);
  });

  it("accepts inferenceWorker and rejects the legacy pythonWorker field", async () => {
    const validSnapshot = structuredClone(studioFixture);
    const valid = createBackend(async () => {
      await Promise.resolve();
      return validSnapshot;
    });

    await expect(valid.backend.getSnapshot()).resolves.toMatchObject({
      system: {
        inferenceWorker: "ready",
      },
    });

    const legacySnapshot = structuredClone(studioFixture) as unknown as {
      system: Record<string, unknown>;
    };
    delete legacySnapshot.system.inferenceWorker;
    legacySnapshot.system.pythonWorker = "ready";
    const legacy = createBackend(async () => {
      await Promise.resolve();
      return legacySnapshot;
    });

    await expect(legacy.backend.getSnapshot()).rejects.toThrow(/unknown fields/u);
  });

  it("parses a structured IPC error and never falls back to mock state", async () => {
    const ipcError = new Error("产物路径越过输出目录边界。") as Error & {
      code: string;
    };
    ipcError.code = "path_boundary_violation";
    const { backend, invokeMock } = createBackend(async () => {
      await Promise.resolve();
      throw ipcError;
    });

    await expect(backend.openArtifact("artifact-pdf")).rejects.toThrow(
      "[path_boundary_violation] 产物路径越过输出目录边界。",
    );
    expect(invokeMock).toHaveBeenCalledTimes(1);
    expect(invokeMock).toHaveBeenCalledWith("open_artifact", {
      artifactId: "artifact-pdf",
    });
  });

  it("requires cancel_job to return an empty value", async () => {
    const { backend } = createBackend(async () => {
      await Promise.resolve();
      return { cancelled: true };
    });

    await expect(backend.cancelJob("job-ipc-001")).rejects.toThrow(
      /cancel_job must return no value/u,
    );
  });
});
