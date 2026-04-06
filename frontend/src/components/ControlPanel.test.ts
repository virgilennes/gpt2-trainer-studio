import { describe, it, expect } from "vitest";
import * as fc from "fast-check";
import { CONFIG_FIELD_HELP } from "./ControlPanel";
import type { TrainingConfig } from "@/types";

/**
 * Property 14: Config fields have help text
 *
 * **Validates: Requirements 10.5**
 *
 * For any configuration field in TrainingConfig, the associated tooltip/help
 * text must be a non-empty string.
 */
describe("Property 14: Config fields have help text", () => {
  const configFieldNames: (keyof TrainingConfig)[] = [
    "learningRate",
    "batchSize",
    "numEpochs",
    "warmupSteps",
    "weightDecay",
  ];

  it("every config field has non-empty help text", () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...configFieldNames),
        (fieldName) => {
          const helpText = CONFIG_FIELD_HELP[fieldName];
          // Help text must exist and be a non-empty string
          expect(helpText).toBeDefined();
          expect(typeof helpText).toBe("string");
          expect(helpText.trim().length).toBeGreaterThan(0);
        },
      ),
      { numRuns: 100 },
    );
  });

  it("CONFIG_FIELD_HELP covers all TrainingConfig keys", () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...configFieldNames),
        (fieldName) => {
          // The help map must have an entry for every config field
          expect(fieldName in CONFIG_FIELD_HELP).toBe(true);
        },
      ),
      { numRuns: 100 },
    );
  });
});
