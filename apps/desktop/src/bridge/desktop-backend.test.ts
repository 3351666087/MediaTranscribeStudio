import type {
  CreateJobRequest,
  ReviewDecision,
  SpeakerCountPolicy,
} from "../contracts/studio";
import { MockDesktopBackend } from "./desktop-backend";

const dynamicCounts = [1, 2, 5, 8, 13, 21, 64, 129] as const;

function createRequest(
  speakerPolicy: SpeakerCountPolicy,
  speakerLabels: string[] = [],
): CreateJobRequest {
  return {
    title: "动态人数后端测试",
    mediaPath: "D:\\media\\meeting.mov",
    outputDirectory: "D:\\output",
    strategyId: "balanced",
    language: "auto",
    localLlmMode: "disabled",
    localLlmModel: "qwen3.5:4b",
    localLlmEndpoint: "http://127.0.0.1:11434",
    localLlmEndpointPolicy: "loopback-only",
    localLlmAutoApply: false,
    translationTargets: [],
    polish: false,
    summary: false,
    outputLocale: "en-US",
    businessPromptVersion: "business-v1",
    speakerPolicy,
    speakerLabels,
  };
}

function reviewDecision(
  overrides: Partial<ReviewDecision> = {},
): ReviewDecision {
  return {
    reviewId: "review-181",
    speakerId: "speaker-2",
    normalizedText:
      "那这个边界我觉得可以先锁。然后我补充一下，不是这个意思。",
    reason: "人工判断该片段应归属第二位说话人。",
    evidence: "本地复听确认声纹、边界与局部音频一致。",
    confidence: 0.96,
    ...overrides,
  };
}

async function advance<T>(promise: Promise<T>, milliseconds = 400): Promise<T> {
  const observed = promise.then(
    (value) => ({ status: "fulfilled" as const, value }),
    (reason: unknown) => ({ status: "rejected" as const, reason }),
  );
  await vi.advanceTimersByTimeAsync(milliseconds);
  const result = await observed;
  if (result.status === "rejected") {
    throw result.reason;
  }
  return result.value;
}

describe("MockDesktopBackend dynamic speaker behavior", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it.each(dynamicCounts)(
    "creates a manual %i-speaker job with contiguous ids and complete custom labels",
    async (count) => {
      const backend = new MockDesktopBackend();
      const labels = Array.from(
        { length: count },
        (_, index) => `角色 ${index + 1}`,
      );

      const result = await advance(
        backend.createJob(
          createRequest({ mode: "manual", count }, labels),
        ),
      );
      const snapshot = await advance(backend.getSnapshot(), 100);

      expect(result.accepted).toBe(true);
      expect(snapshot.job.speakerPolicy).toEqual({ mode: "manual", count });
      expect(snapshot.job.speakerCount).toBe(count);
      expect(snapshot.job.speakerDetection).toBeNull();
      expect(snapshot.speakers).toHaveLength(count);
      expect(snapshot.speakers.map(({ id }) => id)).toEqual(
        Array.from({ length: count }, (_, index) => `speaker-${index + 1}`),
      );
      expect(snapshot.speakers.map(({ label }) => label)).toEqual(labels);
    },
  );

  it("exposes deterministic browser-only evidence fixtures for 13, 64, and 129 speakers", async () => {
    for (const count of [13, 64, 129]) {
      const backend = new MockDesktopBackend(
        `?evidence=speakers&speakers=${count}`,
      );
      const snapshot = await advance(backend.getSnapshot(), 100);

      expect(snapshot.speakers).toHaveLength(count);
      expect(snapshot.job.speakerCount).toBe(count);
      expect(snapshot.speakers.at(-1)?.id).toBe(`speaker-${count}`);
    }

    const fallback = new MockDesktopBackend(
      "?evidence=speakers&speakers=999999",
    );
    expect((await advance(fallback.getSnapshot(), 100)).speakers).toHaveLength(8);
  });

  it("exposes a stable browser-only load-failure evidence state", async () => {
    const backend = new MockDesktopBackend("?evidence=failure");

    await expect(advance(backend.getSnapshot(), 100)).rejects.toThrow(
      "Visual evidence mode: the local backend returned no workspace snapshot.",
    );
  });

  it("generates default labels when manual mode submits an empty label list", async () => {
    const backend = new MockDesktopBackend();

    await advance(
      backend.createJob(createRequest({ mode: "manual", count: 4 })),
    );
    const snapshot = await advance(backend.getSnapshot(), 100);

    expect(snapshot.speakers).toHaveLength(4);
    expect(snapshot.speakers.map(({ label }) => label)).toEqual(
      Array.from({ length: 4 }, (_, index) => `Speaker ${index + 1}`),
    );
  });

  it.each([
    { mode: "auto" } as const,
    {
      mode: "hybrid",
      minSpeakers: 2,
      priorCount: 6,
      maxSpeakers: 9,
    } as const,
  ])(
    "keeps $mode speaker evidence unresolved until media analysis",
    async (policy) => {
      const backend = new MockDesktopBackend();

      await advance(backend.createJob(createRequest(policy)));
      const snapshot = await advance(backend.getSnapshot(), 100);

      expect(snapshot.job.speakerCount).toBeNull();
      expect(snapshot.job.speakerDetection).toBeNull();
      expect(snapshot.speakers).toEqual([]);
      expect(snapshot.events.at(-1)?.detail).toContain(
        "Awaiting media analysis",
      );
    },
  );

  it("rejects a partial label list before scheduling backend work", async () => {
    const backend = new MockDesktopBackend();

    await expect(
      backend.createJob(
        createRequest(
          { mode: "manual", count: 3 },
          ["主持", "产品"],
        ),
      ),
    ).rejects.toThrow(
      /accepts either an empty label list or exactly 3 speaker labels/u,
    );
    expect(vi.getTimerCount()).toBe(0);
  });

  it("requires unlocking a speaker before renaming and preserves review status changes", async () => {
    const backend = new MockDesktopBackend();

    await expect(
      advance(
        backend.updateSpeaker({
          speakerId: "speaker-1",
          label: "新的主持人",
          locked: true,
          reviewStatus: "confirmed",
        }),
        200,
      ),
    ).rejects.toThrow(/Unlock the speaker before changing the name/u);

    const unlocked = await advance(
      backend.updateSpeaker({
        speakerId: "speaker-1",
        label: "Speaker 1 · Host",
        locked: false,
        reviewStatus: "needs_review",
      }),
      200,
    );
    expect(unlocked.locked).toBe(false);
    expect(unlocked.reviewStatus).toBe("needs_review");

    const renamed = await advance(
      backend.updateSpeaker({
        speakerId: "speaker-1",
        label: "新的主持人",
        locked: true,
        reviewStatus: "confirmed",
      }),
      200,
    );
    expect(renamed).toMatchObject({
      id: "speaker-1",
      label: "新的主持人",
      locked: true,
      reviewStatus: "confirmed",
    });
  });

  it("rejects review decisions that reference a speaker outside the current task", async () => {
    const backend = new MockDesktopBackend();

    await expect(
      advance(
        backend.applyReviewDecision({
          reviewId: "review-181",
          speakerId: "speaker-999",
          normalizedText: "保持中文原文。",
          reason: "测试未知角色必须被拒绝。",
          evidence:
            "Intentionally reference speaker-999 outside the current task.",
          confidence: 0.9,
        }),
        300,
      ),
    ).rejects.toThrow(/outside the current task/u);
  });

  it("resets quality and performance observability for every new task", async () => {
    const backend = new MockDesktopBackend();

    await advance(
      backend.createJob(
        createRequest({
          mode: "hybrid",
          minSpeakers: 2,
          priorCount: 8,
          maxSpeakers: 13,
        }),
      ),
    );
    const snapshot = await advance(backend.getSnapshot(), 100);

    expect(snapshot.diarizationQuality.der.status).toBe("unavailable");
    expect(snapshot.diarizationQuality.jer.status).toBe("unavailable");
    expect(snapshot.diarizationQuality.confusion.status).toBe("unavailable");
    expect(snapshot.diarizationQuality.overlapF1.status).toBe("unavailable");
    expect(snapshot.diarizationQuality).toEqual({
      der: snapshot.diarizationQuality.der,
      jer: snapshot.diarizationQuality.jer,
      confusion: snapshot.diarizationQuality.confusion,
      overlapF1: snapshot.diarizationQuality.overlapF1,
      reviewRate: {
        status: "available",
        value: 0,
        unit: "percent",
        source: "Pending review segments / generated valid speech segments",
      },
    });
    expect(snapshot.performance).toEqual({
      status: "unavailable",
      reason:
        "The task has not run, so model-stage and resource-sampling data are unavailable.",
    });
  });

  it("keeps the measured review-rate denominator stable after a review decision", async () => {
    const backend = new MockDesktopBackend();
    const before = await advance(backend.getSnapshot(), 100);
    const reviewRate = structuredClone(before.diarizationQuality.reviewRate);

    await advance(
      backend.applyReviewDecision(reviewDecision()),
      300,
    );
    const after = await advance(backend.getSnapshot(), 100);

    expect(after.job.reviewOpenCount).toBe(before.job.reviewOpenCount - 1);
    expect(after.diarizationQuality.reviewRate).toEqual(reviewRate);
    expect(after.diarizationQuality.reviewRate).not.toMatchObject({
      value: 75,
    });
  });

  it("rejects a malicious locked-speaker rewrite without any state mutation", async () => {
    const backend = new MockDesktopBackend();
    const before = await advance(backend.getSnapshot(), 100);
    const lockedBefore = before.reviews.find(({ id }) => id === "review-244");
    expect(lockedBefore).toBeDefined();

    await expect(
      advance(
        backend.applyReviewDecision(
          reviewDecision({
            reviewId: "review-244",
            speakerId: "speaker-4",
            normalizedText: "恶意绕过 UI 改写角色。",
          }),
        ),
        300,
      ),
    ).rejects.toThrow(
      /A human-locked segment cannot be reassigned to another speaker/u,
    );

    const after = await advance(backend.getSnapshot(), 100);
    expect(after).toEqual(before);
    expect(
      after.reviews.find(({ id }) => id === "review-244")?.rawText,
    ).toBe(lockedBefore?.rawText);
    expect(
      after.reviews.find(({ id }) => id === "review-244")?.auditTrail,
    ).toEqual([]);
  });

  it("allows normalizedText review on a locked speaker while preserving rawText", async () => {
    const backend = new MockDesktopBackend();
    const before = await advance(backend.getSnapshot(), 100);
    const previous = before.reviews.find(({ id }) => id === "review-244");
    if (!previous) {
      throw new Error("fixture 缺少 review-244。");
    }

    const updated = await advance(
      backend.applyReviewDecision(
        reviewDecision({
          reviewId: "review-244",
          speakerId: previous.currentSpeakerId,
          normalizedText: "这个分数低于 85 就不能说通过，这个规则必须写死。",
          reason: "角色已锁定，仅校对人工逐字正文。",
          evidence: "本地复听 00:34:40.320–00:34:46.710 后逐字确认。",
          confidence: 0.99,
        }),
      ),
      300,
    );

    expect(updated.currentSpeakerId).toBe(previous.currentSpeakerId);
    expect(updated.normalizedText).toBe(
      "这个分数低于 85 就不能说通过，这个规则必须写死。",
    );
    expect(updated.rawText).toBe(previous.rawText);
    expect(updated.auditTrail).toHaveLength(1);
    expect(updated.auditTrail[0]).toMatchObject({
      sequence: 1,
      actor: "human",
      previousSpeakerId: previous.currentSpeakerId,
      speakerId: previous.currentSpeakerId,
      previousNormalizedText: previous.normalizedText,
      normalizedText: "这个分数低于 85 就不能说通过，这个规则必须写死。",
    });
  });

  it("appends immutable audit events instead of overwriting prior decisions", async () => {
    const backend = new MockDesktopBackend();
    const before = await advance(backend.getSnapshot(), 100);
    const rawText = before.reviews.find(({ id }) => id === "review-181")?.rawText;

    const first = await advance(
      backend.applyReviewDecision(reviewDecision()),
      300,
    );
    const firstEvent = structuredClone(first.auditTrail[0]);
    const second = await advance(
      backend.applyReviewDecision(
        reviewDecision({
          normalizedText:
            "那这个边界我觉得可以先锁。然后我补充一下：不是这个意思。",
          reason: "二次人工复听后只修正标点。",
          evidence: "重复复听同一区间，声纹与角色判断保持不变。",
          confidence: 0.98,
        }),
      ),
      300,
    );

    expect(second.rawText).toBe(rawText);
    expect(second.auditTrail).toHaveLength(2);
    expect(second.auditTrail[0]).toEqual(firstEvent);
    expect(second.auditTrail.map(({ sequence }) => sequence)).toEqual([1, 2]);
    expect(second.auditTrail[1]).toMatchObject({
      previousSpeakerId: "speaker-2",
      speakerId: "speaker-2",
      previousNormalizedText: first.normalizedText,
      normalizedText:
        "那这个边界我觉得可以先锁。然后我补充一下：不是这个意思。",
    });
    expect(second.auditTrail[0].id).not.toBe(second.auditTrail[1].id);
  });

  it.each([
    {
      label: "empty reason",
      decision: reviewDecision({ reason: "" }),
    },
    {
      label: "blank evidence",
      decision: reviewDecision({ evidence: "   " }),
    },
    {
      label: "confidence below zero",
      decision: reviewDecision({ confidence: -0.01 }),
    },
    {
      label: "confidence above one",
      decision: reviewDecision({ confidence: 1.01 }),
    },
    {
      label: "NaN confidence",
      decision: reviewDecision({ confidence: Number.NaN }),
    },
    {
      label: "infinite confidence",
      decision: reviewDecision({ confidence: Number.POSITIVE_INFINITY }),
    },
  ])("rejects $label before mutation", async ({ decision }) => {
    const backend = new MockDesktopBackend();
    const before = await advance(backend.getSnapshot(), 100);

    await expect(backend.applyReviewDecision(decision)).rejects.toThrow();
    expect(vi.getTimerCount()).toBe(0);

    const after = await advance(backend.getSnapshot(), 100);
    expect(after).toEqual(before);
  });
});
