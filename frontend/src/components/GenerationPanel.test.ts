import { describe, it, expect } from "vitest";
import * as fc from "fast-check";
import {
  GENERATION_PARAM_HELP,
  SAMPLE_PROMPTS,
} from "./GenerationPanel";
import type { GenerationParams } from "@/types";

/**
 * Unit tests for GenerationPanel exports.
 *
 * Validates that parameter help text and sample prompts are correctly defined.
 */

type ParamKey = keyof Omit<GenerationParams, "prompt">;

const paramKeys: ParamKey[] = ["temperature", "topK", "topP", "maxLength"];

describe("GenerationPanel — parameter tooltips", () => {
  it("every generation parameter has non-empty help text", () => {
    fc.assert(
      fc.property(fc.constantFrom(...paramKeys), (key) => {
        const help = GENERATION_PARAM_HELP[key];
        expect(help).toBeDefined();
        expect(typeof help).toBe("string");
        expect(help.trim().length).toBeGreaterThan(0);
      }),
      { numRuns: 100 },
    );
  });

  it("GENERATION_PARAM_HELP covers all parameter keys", () => {
    fc.assert(
      fc.property(fc.constantFrom(...paramKeys), (key) => {
        expect(key in GENERATION_PARAM_HELP).toBe(true);
      }),
      { numRuns: 100 },
    );
  });
});

describe("GenerationPanel — sample prompt library", () => {
  it("has at least one sample prompt", () => {
    expect(SAMPLE_PROMPTS.length).toBeGreaterThan(0);
  });

  it("every sample prompt has a non-empty label and text", () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: SAMPLE_PROMPTS.length - 1 }),
        (idx) => {
          const sp = SAMPLE_PROMPTS[idx];
          expect(sp.label.trim().length).toBeGreaterThan(0);
          expect(sp.text.trim().length).toBeGreaterThan(0);
        },
      ),
      { numRuns: 100 },
    );
  });
});
