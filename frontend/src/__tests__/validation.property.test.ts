import { describe, it, expect } from "vitest";
import * as fc from "fast-check";
import { TrainingConfigSchema } from "@/lib/validation";

/**
 * Feature: gpt-text-generator
 * Property 2: Training config validation rejects out-of-range values
 *
 * For any hyperparameter value outside its acceptable range, the Zod schema
 * rejects the input and returns an error for the specific field.
 *
 * **Validates: Requirements 3.2, 3.6**
 */
describe("Property 2: Training config validation rejects out-of-range values", () => {
  // Valid defaults used when testing a single field out of range
  const validBase = {
    learningRate: 5e-5,
    batchSize: 8,
    numEpochs: 3,
    warmupSteps: 0,
    weightDecay: 0.0,
  };

  it("rejects learningRate below 1e-6", () => {
    fc.assert(
      fc.property(
        fc.double({ max: 1e-6 - Number.EPSILON, noNaN: true, noDefaultInfinity: true, min: -1e10 }),
        (lr) => {
          if (lr >= 1e-6) return; // skip edge values that round to boundary
          const result = TrainingConfigSchema.safeParse({ ...validBase, learningRate: lr });
          expect(result.success).toBe(false);
        }
      ),
      { numRuns: 100 }
    );
  });

  it("rejects learningRate above 1e-2", () => {
    fc.assert(
      fc.property(
        fc.double({ min: 1e-2 + Number.EPSILON, noNaN: true, noDefaultInfinity: true, max: 1e10 }),
        (lr) => {
          if (lr <= 1e-2) return; // skip edge values that round to boundary
          const result = TrainingConfigSchema.safeParse({ ...validBase, learningRate: lr });
          expect(result.success).toBe(false);
        }
      ),
      { numRuns: 100 }
    );
  });

  it("rejects batchSize below 1", () => {
    fc.assert(
      fc.property(
        fc.integer({ max: 0 }),
        (bs) => {
          const result = TrainingConfigSchema.safeParse({ ...validBase, batchSize: bs });
          expect(result.success).toBe(false);
        }
      ),
      { numRuns: 100 }
    );
  });

  it("rejects batchSize above 64", () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 65 }),
        (bs) => {
          const result = TrainingConfigSchema.safeParse({ ...validBase, batchSize: bs });
          expect(result.success).toBe(false);
        }
      ),
      { numRuns: 100 }
    );
  });

  it("rejects numEpochs below 1", () => {
    fc.assert(
      fc.property(
        fc.integer({ max: 0 }),
        (epochs) => {
          const result = TrainingConfigSchema.safeParse({ ...validBase, numEpochs: epochs });
          expect(result.success).toBe(false);
        }
      ),
      { numRuns: 100 }
    );
  });

  it("rejects numEpochs above 20", () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 21 }),
        (epochs) => {
          const result = TrainingConfigSchema.safeParse({ ...validBase, numEpochs: epochs });
          expect(result.success).toBe(false);
        }
      ),
      { numRuns: 100 }
    );
  });

  it("rejects warmupSteps below 0", () => {
    fc.assert(
      fc.property(
        fc.integer({ max: -1 }),
        (ws) => {
          const result = TrainingConfigSchema.safeParse({ ...validBase, warmupSteps: ws });
          expect(result.success).toBe(false);
        }
      ),
      { numRuns: 100 }
    );
  });

  it("rejects warmupSteps above 1000", () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1001 }),
        (ws) => {
          const result = TrainingConfigSchema.safeParse({ ...validBase, warmupSteps: ws });
          expect(result.success).toBe(false);
        }
      ),
      { numRuns: 100 }
    );
  });

  it("rejects weightDecay below 0", () => {
    fc.assert(
      fc.property(
        fc.double({ max: -Number.EPSILON, noNaN: true, noDefaultInfinity: true, min: -1e10 }),
        (wd) => {
          if (wd >= 0) return;
          const result = TrainingConfigSchema.safeParse({ ...validBase, weightDecay: wd });
          expect(result.success).toBe(false);
        }
      ),
      { numRuns: 100 }
    );
  });

  it("rejects weightDecay above 1", () => {
    fc.assert(
      fc.property(
        fc.double({ min: 1 + Number.EPSILON, noNaN: true, noDefaultInfinity: true, max: 1e10 }),
        (wd) => {
          if (wd <= 1) return;
          const result = TrainingConfigSchema.safeParse({ ...validBase, weightDecay: wd });
          expect(result.success).toBe(false);
        }
      ),
      { numRuns: 100 }
    );
  });

  it("accepts any valid training config within all ranges", () => {
    fc.assert(
      fc.property(
        fc.record({
          learningRate: fc.double({ min: 1e-6, max: 1e-2, noNaN: true, noDefaultInfinity: true }),
          batchSize: fc.integer({ min: 1, max: 64 }),
          numEpochs: fc.integer({ min: 1, max: 20 }),
          warmupSteps: fc.integer({ min: 0, max: 1000 }),
          weightDecay: fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
        }),
        (config) => {
          const result = TrainingConfigSchema.safeParse(config);
          expect(result.success).toBe(true);
        }
      ),
      { numRuns: 100 }
    );
  });
});
