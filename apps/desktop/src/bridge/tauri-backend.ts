import { invoke } from "@tauri-apps/api/core";
import type {
  ArtifactOpenResult,
  CreateJobRequest,
  CreateJobResult,
  DesktopBackend,
  JobRuntimeStatus,
  ReviewDecision,
  ReviewSegment,
  SpeakerProfile,
  StudioSnapshot,
  UpdateSpeakerRequest,
} from "../contracts/studio";
import {
  assertCreateJobRequest,
  assertReviewDecision,
  assertUpdateSpeakerRequest,
  assertVoidResult,
  parseArtifactOpenResult,
  parseCreateJobResult,
  parseIpcError,
  parseJobRuntimeStatuses,
  parseReviewSegment,
  parseSpeakerProfile,
  parseStudioSnapshot,
} from "../contracts/runtime-validation";

type Invoke = <T>(command: string, args?: Record<string, unknown>) => Promise<T>;

export class TauriDesktopBackend implements DesktopBackend {
  constructor(private readonly invokeCommand: Invoke = invoke) {}

  private assertJobId(jobId: string): void {
    if (
      typeof jobId !== "string" ||
      jobId.length === 0 ||
      jobId.length > 128
    ) {
      throw new Error(
        "jobId must be a non-empty string of 1–128 characters.",
      );
    }
  }

  private async call(command: string, args?: Record<string, unknown>): Promise<unknown> {
    try {
      return await this.invokeCommand<unknown>(command, args);
    } catch (error) {
      throw parseIpcError(error);
    }
  }

  async getSnapshot(): Promise<StudioSnapshot> {
    return parseStudioSnapshot(await this.call("get_snapshot"));
  }

  async listJobs(): Promise<JobRuntimeStatus[]> {
    return parseJobRuntimeStatuses(await this.call("list_jobs"));
  }

  async selectJob(jobId: string): Promise<StudioSnapshot> {
    this.assertJobId(jobId);
    return parseStudioSnapshot(
      await this.call("select_job", { jobId }),
    );
  }

  async createJob(request: CreateJobRequest): Promise<CreateJobResult> {
    assertCreateJobRequest(request);
    return parseCreateJobResult(
      await this.call("create_job", {
        request,
      }),
    );
  }

  async cancelJob(jobId: string): Promise<void> {
    this.assertJobId(jobId);
    assertVoidResult(await this.call("cancel_job", { jobId }), "cancel_job");
  }

  async updateSpeaker(request: UpdateSpeakerRequest): Promise<SpeakerProfile> {
    assertUpdateSpeakerRequest(request);
    return parseSpeakerProfile(await this.call("update_speaker", { request }));
  }

  async applyReviewDecision(decision: ReviewDecision): Promise<ReviewSegment> {
    assertReviewDecision(decision);
    return parseReviewSegment(await this.call("apply_review_decision", { decision }));
  }

  async openArtifact(artifactId: string): Promise<ArtifactOpenResult> {
    if (typeof artifactId !== "string" || artifactId.length === 0 || artifactId.length > 128) {
      throw new Error(
        "artifactId must be a non-empty string of 1–128 characters.",
      );
    }
    return parseArtifactOpenResult(await this.call("open_artifact", { artifactId }));
  }
}
