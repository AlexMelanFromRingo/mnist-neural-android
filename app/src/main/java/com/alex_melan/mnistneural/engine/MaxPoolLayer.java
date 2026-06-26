package com.alex_melan.mnistneural.engine;

import java.io.DataInputStream;
import java.io.DataOutputStream;

/**
 * 2-D max pooling with square kernel and stride, no padding. Parameter-free. For backward it routes
 * the incoming gradient to the argmax position recorded during the forward pass (ties go to the
 * first maximum, matching the forward scan order).
 */
public final class MaxPoolLayer implements Layer {

    private final int c, h, w;
    private final int k, stride;
    private final int hOut, wOut;

    private int[][] argMax; // [batch][c*hOut*wOut] -> flat input index

    public MaxPoolLayer(Shape in, int kernel, int stride) {
        this.c = in.channels;
        this.h = in.height;
        this.w = in.width;
        this.k = kernel;
        this.stride = stride;
        this.hOut = (h - k) / stride + 1;
        this.wOut = (w - k) / stride + 1;
        if (hOut <= 0 || wOut <= 0) {
            throw new IllegalArgumentException("MaxPool produces non-positive output: " + hOut + "x" + wOut);
        }
    }

    @Override public Shape inputShape() { return new Shape(c, h, w); }

    @Override public Shape outputShape() { return new Shape(c, hOut, wOut); }

    @Override public int paramCount() { return 0; }

    @Override public String typeName() { return "MaxPool2D"; }

    @Override public String describe() { return "MaxPool2D(k" + k + ", s" + stride + ")"; }

    @Override
    public float[][] forward(float[][] x, boolean training) {
        final int batch = x.length;
        final int outSize = c * hOut * wOut;
        final float[][] out = new float[batch][outSize];
        final int[][] arg = new int[batch][outSize];
        Parallel.forRange(0, batch, 1, (s, e) -> {
            for (int bi = s; bi < e; bi++) {
                final float[] xb = x[bi];
                final float[] ob = out[bi];
                final int[] ag = arg[bi];
                for (int ch = 0; ch < c; ch++) {
                    final int xc = ch * h * w;
                    final int oc = ch * hOut * wOut;
                    for (int oh = 0; oh < hOut; oh++) {
                        final int ihBase = oh * stride;
                        for (int ow = 0; ow < wOut; ow++) {
                            final int iwBase = ow * stride;
                            float best = Float.NEGATIVE_INFINITY;
                            int bestIdx = -1;
                            for (int ki = 0; ki < k; ki++) {
                                final int xrow = xc + (ihBase + ki) * w;
                                for (int kj = 0; kj < k; kj++) {
                                    final int idx = xrow + iwBase + kj;
                                    final float v = xb[idx];
                                    if (v > best) {
                                        best = v;
                                        bestIdx = idx;
                                    }
                                }
                            }
                            final int o = oc + oh * wOut + ow;
                            ob[o] = best;
                            ag[o] = bestIdx;
                        }
                    }
                }
            }
        });
        if (training) argMax = arg;
        return out;
    }

    @Override
    public float[][] backward(float[][] gradOut) {
        final int batch = gradOut.length;
        final float[][] gradIn = new float[batch][c * h * w];
        Parallel.forRange(0, batch, 1, (s, e) -> {
            for (int bi = s; bi < e; bi++) {
                final float[] go = gradOut[bi];
                final float[] gi = gradIn[bi];
                final int[] ag = argMax[bi];
                for (int o = 0; o < go.length; o++) {
                    gi[ag[o]] += go[o];
                }
            }
        });
        return gradIn;
    }

    @Override public void sgdStep(float lr, float l2, float momentum) { /* no params */ }

    @Override public void writeParams(DataOutputStream out) { /* no params */ }

    @Override public void readParams(DataInputStream in) { /* no params */ }
}
