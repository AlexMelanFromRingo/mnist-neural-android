package com.alex_melan.mnistneural.engine;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;

/**
 * Fully-connected layer: {@code a = activation(W·x + b)}.
 *
 * <p>Weights are stored row-major in a flat array: {@code W[o*nIn + i]} is the weight from input
 * {@code i} to output neuron {@code o}.
 *
 * <p><b>Connection masks.</b> An optional {@code boolean[nOut*nIn]} disables individual edges (the
 * per-neuron connection editor). A disabled weight is held at exactly 0 and never receives a
 * gradient, so it contributes nothing to forward or backward. {@code null} mask = fully connected.
 */
public final class DenseLayer implements Layer {

    private final int nIn;
    private final int nOut;
    private final Activation act;

    private final float[] w;   // [nOut*nIn]
    private final float[] b;   // [nOut]
    private boolean[] mask;     // [nOut*nIn] or null (true = active edge)

    // Gradient + momentum buffers.
    private final float[] gradW;
    private final float[] gradB;
    private final float[] velW;
    private final float[] velB;

    // Forward caches (valid after a training forward pass).
    private float[][] lastIn;
    private float[][] lastOut;

    public DenseLayer(int nIn, int nOut, Activation act, long seed) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.act = act;
        this.w = new float[nOut * nIn];
        this.b = new float[nOut];
        this.gradW = new float[nOut * nIn];
        this.gradB = new float[nOut];
        this.velW = new float[nOut * nIn];
        this.velB = new float[nOut];
        initWeights(seed);
    }

    private void initWeights(long seed) {
        Random rnd = new Random(seed);
        if (act.prefersHeInit()) {
            float std = (float) Math.sqrt(2.0 / nIn);
            for (int k = 0; k < w.length; k++) w[k] = (float) rnd.nextGaussian() * std;
        } else {
            float limit = (float) Math.sqrt(6.0 / (nIn + nOut));
            for (int k = 0; k < w.length; k++) w[k] = (rnd.nextFloat() * 2f - 1f) * limit;
        }
        // biases already 0
    }

    /** Sets the connection mask (length nOut*nIn). Disabled weights are zeroed immediately. */
    public void setMask(boolean[] mask) {
        if (mask != null && mask.length != nOut * nIn) {
            throw new IllegalArgumentException("mask length " + mask.length + " != " + (nOut * nIn));
        }
        this.mask = mask;
        if (mask != null) {
            for (int k = 0; k < w.length; k++) {
                if (!mask[k]) {
                    w[k] = 0f;
                    velW[k] = 0f;
                }
            }
        }
    }

    public boolean[] mask() {
        return mask;
    }

    @Override public Shape inputShape() { return Shape.vector(nIn); }

    @Override public Shape outputShape() { return Shape.vector(nOut); }

    @Override
    public int paramCount() {
        int weights;
        if (mask == null) {
            weights = nOut * nIn;
        } else {
            weights = 0;
            for (boolean m : mask) if (m) weights++;
        }
        return weights + nOut; // + biases
    }

    @Override public String typeName() { return "Dense"; }

    @Override
    public String describe() {
        return "Dense(" + nIn + "→" + nOut + ", " + act + ")";
    }

    @Override
    public float[][] forward(float[][] x, boolean training) {
        final int batch = x.length;
        final float[][] out = new float[batch][nOut];
        final boolean[] m = mask;
        Parallel.forRange(0, batch, 8, (s, e) -> {
            for (int bi = s; bi < e; bi++) {
                final float[] xb = x[bi];
                final float[] ob = out[bi];
                for (int o = 0; o < nOut; o++) {
                    final int base = o * nIn;
                    float z = b[o];
                    if (m == null) {
                        for (int i = 0; i < nIn; i++) z += w[base + i] * xb[i];
                    } else {
                        for (int i = 0; i < nIn; i++) if (m[base + i]) z += w[base + i] * xb[i];
                    }
                    ob[o] = act.apply(z);
                }
            }
        });
        if (training) {
            lastIn = x;
            lastOut = out;
        }
        return out;
    }

    @Override
    public float[][] backward(float[][] gradOut) {
        final int batch = gradOut.length;
        final float[][] x = lastIn;
        final float[][] a = lastOut;
        final boolean[] m = mask;

        // gradZ[b][o] = gradOut[b][o] * act'(a[b][o])
        final float[][] gradZ = new float[batch][nOut];
        for (int bi = 0; bi < batch; bi++) {
            final float[] go = gradOut[bi];
            final float[] ab = a[bi];
            final float[] gz = gradZ[bi];
            for (int o = 0; o < nOut; o++) {
                gz[o] = go[o] * act.derivativeFromOutput(ab[o]);
            }
        }

        // Parameter gradients: parallelize over output neurons (disjoint rows of gradW + gradB[o]).
        Parallel.forRange(0, nOut, 4, (s, e) -> {
            for (int o = s; o < e; o++) {
                final int base = o * nIn;
                float gb = 0f;
                for (int bi = 0; bi < batch; bi++) gb += gradZ[bi][o];
                gradB[o] = gb;
                if (m == null) {
                    for (int i = 0; i < nIn; i++) {
                        float g = 0f;
                        for (int bi = 0; bi < batch; bi++) g += gradZ[bi][o] * x[bi][i];
                        gradW[base + i] = g;
                    }
                } else {
                    for (int i = 0; i < nIn; i++) {
                        if (!m[base + i]) { gradW[base + i] = 0f; continue; }
                        float g = 0f;
                        for (int bi = 0; bi < batch; bi++) g += gradZ[bi][o] * x[bi][i];
                        gradW[base + i] = g;
                    }
                }
            }
        });

        // Input gradient: gradIn[b][i] = sum_o gradZ[b][o] * W[o,i]; parallelize over batch.
        final float[][] gradIn = new float[batch][nIn];
        Parallel.forRange(0, batch, 8, (s, e) -> {
            for (int bi = s; bi < e; bi++) {
                final float[] gz = gradZ[bi];
                final float[] gi = gradIn[bi];
                for (int o = 0; o < nOut; o++) {
                    final float g = gz[o];
                    if (g == 0f) continue;
                    final int base = o * nIn;
                    for (int i = 0; i < nIn; i++) gi[i] += g * w[base + i];
                }
            }
        });
        return gradIn;
    }

    @Override
    public void sgdStep(float lr, float l2, float momentum) {
        final boolean[] m = mask;
        for (int k = 0; k < w.length; k++) {
            if (m != null && !m[k]) continue; // frozen edge stays 0
            float g = gradW[k] + l2 * w[k];
            float v = momentum * velW[k] - lr * g;
            velW[k] = v;
            w[k] += v;
        }
        for (int o = 0; o < nOut; o++) {
            float g = gradB[o]; // biases are not L2-regularized
            float v = momentum * velB[o] - lr * g;
            velB[o] = v;
            b[o] += v;
        }
    }

    @Override
    public void writeParams(DataOutputStream out) throws IOException {
        for (float v : w) out.writeFloat(v);
        for (float v : b) out.writeFloat(v);
        out.writeBoolean(mask != null);
        if (mask != null) {
            for (boolean mm : mask) out.writeBoolean(mm);
        }
    }

    @Override
    public void readParams(DataInputStream in) throws IOException {
        for (int k = 0; k < w.length; k++) w[k] = in.readFloat();
        for (int o = 0; o < nOut; o++) b[o] = in.readFloat();
        boolean hasMask = in.readBoolean();
        if (hasMask) {
            boolean[] mm = new boolean[nOut * nIn];
            for (int k = 0; k < mm.length; k++) mm[k] = in.readBoolean();
            this.mask = mm;
        } else {
            this.mask = null;
        }
    }

    // --- accessors for tests / gradient checking ---
    public float[] weights() { return w; }
    public float[] biases() { return b; }
    public float[] gradWeights() { return gradW; }
    public float[] gradBiases() { return gradB; }
    public int nIn() { return nIn; }
    public int nOut() { return nOut; }
}
