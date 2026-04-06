import { z } from "zod";

export const TrainingConfigSchema = z.object({
  learningRate: z
    .number()
    .min(1e-6, "Learning rate must be at least 1e-6")
    .max(1e-2, "Learning rate must be at most 1e-2"),
  batchSize: z
    .number()
    .int("Batch size must be an integer")
    .min(1, "Batch size must be at least 1")
    .max(64, "Batch size must be at most 64"),
  numEpochs: z
    .number()
    .int("Number of epochs must be an integer")
    .min(1, "Number of epochs must be at least 1")
    .max(20, "Number of epochs must be at most 20"),
  warmupSteps: z
    .number()
    .int("Warmup steps must be an integer")
    .min(0, "Warmup steps must be at least 0")
    .max(1000, "Warmup steps must be at most 1000"),
  weightDecay: z
    .number()
    .min(0, "Weight decay must be at least 0")
    .max(1, "Weight decay must be at most 1"),
});

export const GenerationParamsSchema = z.object({
  prompt: z.string().min(1, "Prompt must not be empty"),
  temperature: z
    .number()
    .min(0.1, "Temperature must be at least 0.1")
    .max(2.0, "Temperature must be at most 2.0"),
  topK: z
    .number()
    .int("Top-K must be an integer")
    .min(1, "Top-K must be at least 1")
    .max(100, "Top-K must be at most 100"),
  topP: z
    .number()
    .min(0.1, "Top-P must be at least 0.1")
    .max(1.0, "Top-P must be at most 1.0"),
  maxLength: z
    .number()
    .int("Max length must be an integer")
    .min(10, "Max length must be at least 10")
    .max(500, "Max length must be at most 500"),
});

export type TrainingConfigInput = z.infer<typeof TrainingConfigSchema>;
export type GenerationParamsInput = z.infer<typeof GenerationParamsSchema>;
