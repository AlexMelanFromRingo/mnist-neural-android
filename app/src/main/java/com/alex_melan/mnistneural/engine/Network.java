package com.alex_melan.mnistneural.engine;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

/**
 * An ordered stack of {@link Layer}s plus a {@link Loss}. Pure compute — it knows nothing about
 * Android or the dataset, which keeps it unit-testable on the JVM and cross-checkable against
 * reference frameworks.
 */
public final class Network {

    private static final int SAVE_MAGIC = 0x4D4E4E57; // 'MNNW'

    private final Shape inputShape;
    private final List<Layer> layers;
    private final Loss loss;

    public Network(Shape inputShape, List<Layer> layers, Loss loss) {
        if (layers.isEmpty()) throw new IllegalArgumentException("network has no layers");
        this.inputShape = inputShape;
        this.layers = layers;
        this.loss = loss;
    }

    public Shape inputShape() { return inputShape; }

    public Loss loss() { return loss; }

    public List<Layer> layers() { return Collections.unmodifiableList(layers); }

    public int paramCount() {
        int sum = 0;
        for (Layer l : layers) sum += l.paramCount();
        return sum;
    }

    /** Runs all layers. {@code training=true} caches activations for {@link #trainBatch}. */
    public float[][] forward(float[][] x, boolean training) {
        float[][] cur = x;
        for (Layer l : layers) cur = l.forward(cur, training);
        return cur;
    }

    /**
     * One mini-batch SGD update. {@code x} is [B][inputSize], {@code labels} the class indices.
     * Returns metrics (summed loss + correct count) measured on this batch <i>before</i> the update.
     */
    public Metrics trainBatch(float[][] x, int[] labels, float lr, float l2, float momentum) {
        final int batch = x.length;
        final float[][] out = forward(x, true);
        final int classes = out[0].length;
        final float[][] grad = new float[batch][classes];
        final double sumLoss = loss.forward(out, labels, grad);

        int correct = 0;
        for (int b = 0; b < batch; b++) {
            if (argMax(out[b]) == labels[b]) correct++;
        }

        // Average over the batch at the source; backward is linear so all param grads come out averaged.
        final float inv = 1f / batch;
        for (float[] row : grad) {
            for (int j = 0; j < classes; j++) row[j] *= inv;
        }

        float[][] g = grad;
        for (int li = layers.size() - 1; li >= 0; li--) {
            g = layers.get(li).backward(g);
        }
        for (Layer l : layers) l.sgdStep(lr, l2, momentum);
        return new Metrics(sumLoss, correct, batch);
    }

    /** Evaluates a single batch without updating weights. */
    public Metrics evaluateBatch(float[][] x, int[] labels) {
        final float[][] out = forward(x, false);
        final int classes = out[0].length;
        final float[][] grad = new float[out.length][classes];
        final double sumLoss = loss.forward(out, labels, grad);
        int correct = 0;
        for (int b = 0; b < out.length; b++) {
            if (argMax(out[b]) == labels[b]) correct++;
        }
        return new Metrics(sumLoss, correct, out.length);
    }

    /** Probability distribution over classes for one input image. */
    public float[] predictProbs(float[] image) {
        float[][] out = forward(new float[][]{image}, false);
        return loss.toProbabilities(out[0]);
    }

    public static int argMax(float[] v) {
        int idx = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[idx]) idx = i;
        return idx;
    }

    public void writeTo(DataOutputStream out) throws IOException {
        out.writeInt(SAVE_MAGIC);
        out.writeInt(layers.size());
        for (Layer l : layers) l.writeParams(out);
    }

    public void readFrom(DataInputStream in) throws IOException {
        int magic = in.readInt();
        if (magic != SAVE_MAGIC) throw new IOException("bad network magic");
        int n = in.readInt();
        if (n != layers.size()) throw new IOException("layer count mismatch: " + n + " != " + layers.size());
        for (Layer l : layers) l.readParams(in);
    }

    /** Accumulable evaluation metrics over one or more batches. */
    public static final class Metrics {
        public final double sumLoss;
        public final int correct;
        public final int count;

        public Metrics(double sumLoss, int correct, int count) {
            this.sumLoss = sumLoss;
            this.correct = correct;
            this.count = count;
        }

        public Metrics plus(Metrics o) {
            return new Metrics(sumLoss + o.sumLoss, correct + o.correct, count + o.count);
        }

        public double meanLoss() { return count == 0 ? 0.0 : sumLoss / count; }

        public double accuracy() { return count == 0 ? 0.0 : (double) correct / count; }
    }
}
