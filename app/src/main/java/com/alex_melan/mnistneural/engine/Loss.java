package com.alex_melan.mnistneural.engine;

/**
 * Loss functions. Each {@link #forward} writes the <b>per-sample</b> gradient dL/d(output) into
 * {@code gradOut} (not divided by batch — {@link Network} averages) and returns the <b>summed</b>
 * loss over the batch.
 */
public enum Loss {

    /**
     * Softmax + cross-entropy fused over raw logits. The network's last layer must output logits
     * ({@link Activation#IDENTITY}). Gradient w.r.t. logits is the well-known {@code softmax(z) - y};
     * loss is computed via log-sum-exp for numerical stability.
     */
    SOFTMAX_CROSS_ENTROPY {
        @Override
        public double forward(float[][] output, int[] labels, float[][] gradOut) {
            double total = 0.0;
            final int classes = output[0].length;
            for (int b = 0; b < output.length; b++) {
                final float[] z = output[b];
                float max = z[0];
                for (int j = 1; j < classes; j++) if (z[j] > max) max = z[j];
                double sumExp = 0.0;
                for (int j = 0; j < classes; j++) sumExp += Math.exp(z[j] - max);
                double logSumExp = max + Math.log(sumExp);
                final int y = labels[b];
                total += logSumExp - z[y]; // -log p[y]
                final float[] g = gradOut[b];
                for (int j = 0; j < classes; j++) {
                    float p = (float) (Math.exp(z[j] - max) / sumExp);
                    g[j] = p - (j == y ? 1f : 0f);
                }
            }
            return total;
        }

        @Override
        public float[] toProbabilities(float[] logits) {
            return softmax(logits);
        }
    },

    /** Mean-squared error against a one-hot target, averaged over classes. */
    MSE {
        @Override
        public double forward(float[][] output, int[] labels, float[][] gradOut) {
            double total = 0.0;
            final int classes = output[0].length;
            final float invC = 1f / classes;
            for (int b = 0; b < output.length; b++) {
                final float[] o = output[b];
                final int y = labels[b];
                final float[] g = gradOut[b];
                for (int j = 0; j < classes; j++) {
                    float t = (j == y) ? 1f : 0f;
                    float d = o[j] - t;
                    total += (double) d * d * invC;
                    g[j] = 2f * d * invC;
                }
            }
            return total;
        }

        @Override
        public float[] toProbabilities(float[] output) {
            // Outputs are already in a comparable range (e.g. via sigmoid); normalize to sum 1.
            float sum = 0f;
            for (float v : output) sum += Math.max(0f, v);
            if (sum <= 0f) return softmax(output);
            float[] p = new float[output.length];
            for (int j = 0; j < output.length; j++) p[j] = Math.max(0f, output[j]) / sum;
            return p;
        }
    };

    public abstract double forward(float[][] output, int[] labels, float[][] gradOut);

    /** Converts a single network output vector into a probability distribution for display. */
    public abstract float[] toProbabilities(float[] output);

    public static float[] softmax(float[] z) {
        float max = z[0];
        for (int j = 1; j < z.length; j++) if (z[j] > max) max = z[j];
        double sum = 0.0;
        float[] p = new float[z.length];
        for (int j = 0; j < z.length; j++) {
            p[j] = (float) Math.exp(z[j] - max);
            sum += p[j];
        }
        float inv = (float) (1.0 / sum);
        for (int j = 0; j < z.length; j++) p[j] *= inv;
        return p;
    }
}
