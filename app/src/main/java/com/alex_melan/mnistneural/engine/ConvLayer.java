package com.alex_melan.mnistneural.engine;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;

/**
 * 2-D convolution (cross-correlation, matching {@code torch.nn.Conv2d}) with square stride and
 * zero padding, followed by an element-wise activation.
 *
 * <p>Weights are laid out as {@code w[f, c, ki, kj]} flattened to
 * {@code f*(Cin*kH*kW) + c*(kH*kW) + ki*kW + kj}. Inputs/outputs are channel-major then row-major,
 * consistent with {@link Shape}.
 */
public final class ConvLayer implements Layer {

    private final int cin, h, w;            // input dims
    private final int f, kh, kw;            // filters, kernel
    private final int stride, pad;
    private final int hOut, wOut;
    private final Activation act;

    private final float[] weights;          // [f*cin*kh*kw]
    private final float[] bias;             // [f]
    private final float[] gradW;
    private final float[] gradB;
    private final float[] velW;
    private final float[] velB;

    private float[][] lastIn;
    private float[][] lastOut;

    public ConvLayer(Shape in, int filters, int kernel, int stride, int pad, Activation act, long seed) {
        this.cin = in.channels;
        this.h = in.height;
        this.w = in.width;
        this.f = filters;
        this.kh = kernel;
        this.kw = kernel;
        this.stride = stride;
        this.pad = pad;
        this.act = act;
        this.hOut = (h + 2 * pad - kh) / stride + 1;
        this.wOut = (w + 2 * pad - kw) / stride + 1;
        if (hOut <= 0 || wOut <= 0) {
            throw new IllegalArgumentException("Conv produces non-positive output: " + hOut + "x" + wOut);
        }
        int wlen = f * cin * kh * kw;
        this.weights = new float[wlen];
        this.bias = new float[f];
        this.gradW = new float[wlen];
        this.gradB = new float[f];
        this.velW = new float[wlen];
        this.velB = new float[f];
        initWeights(seed);
    }

    private void initWeights(long seed) {
        Random rnd = new Random(seed);
        int fanIn = cin * kh * kw;
        if (act.prefersHeInit()) {
            float std = (float) Math.sqrt(2.0 / fanIn);
            for (int k = 0; k < weights.length; k++) weights[k] = (float) rnd.nextGaussian() * std;
        } else {
            float limit = (float) Math.sqrt(6.0 / (fanIn + f * kh * kw));
            for (int k = 0; k < weights.length; k++) weights[k] = (rnd.nextFloat() * 2f - 1f) * limit;
        }
    }

    @Override public Shape inputShape() { return new Shape(cin, h, w); }

    @Override public Shape outputShape() { return new Shape(f, hOut, wOut); }

    @Override public int paramCount() { return weights.length + f; }

    @Override public String typeName() { return "Conv2D"; }

    @Override
    public String describe() {
        return "Conv2D(" + cin + "→" + f + ", k" + kh + ", s" + stride + ", p" + pad + ", " + act + ")";
    }

    @Override
    public float[][] forward(float[][] x, boolean training) {
        final int batch = x.length;
        final int outSize = f * hOut * wOut;
        final float[][] out = new float[batch][outSize];
        final int khw = kh * kw;
        final int chw = cin * khw;
        Parallel.forRange(0, batch, 1, (s, e) -> {
            for (int bi = s; bi < e; bi++) {
                final float[] xb = x[bi];
                final float[] ob = out[bi];
                for (int fi = 0; fi < f; fi++) {
                    final int wf = fi * chw;
                    final float bias0 = bias[fi];
                    final int outF = fi * hOut * wOut;
                    for (int oh = 0; oh < hOut; oh++) {
                        final int ihBase = oh * stride - pad;
                        for (int ow = 0; ow < wOut; ow++) {
                            final int iwBase = ow * stride - pad;
                            float z = bias0;
                            for (int c = 0; c < cin; c++) {
                                final int xc = c * h * w;
                                final int wfc = wf + c * khw;
                                for (int ki = 0; ki < kh; ki++) {
                                    final int ih = ihBase + ki;
                                    if (ih < 0 || ih >= h) continue;
                                    final int xrow = xc + ih * w;
                                    final int wrow = wfc + ki * kw;
                                    for (int kj = 0; kj < kw; kj++) {
                                        final int iw = iwBase + kj;
                                        if (iw < 0 || iw >= w) continue;
                                        z += weights[wrow + kj] * xb[xrow + iw];
                                    }
                                }
                            }
                            ob[outF + oh * wOut + ow] = act.apply(z);
                        }
                    }
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
        final int khw = kh * kw;
        final int chw = cin * khw;
        final int outSize = f * hOut * wOut;

        // gradZ = gradOut * act'(a)
        final float[][] gradZ = new float[batch][outSize];
        for (int bi = 0; bi < batch; bi++) {
            final float[] go = gradOut[bi];
            final float[] ab = a[bi];
            final float[] gz = gradZ[bi];
            for (int o = 0; o < outSize; o++) gz[o] = go[o] * act.derivativeFromOutput(ab[o]);
        }

        // gradW / gradBias: parallelize over filters (disjoint weight blocks).
        Parallel.forRange(0, f, 1, (fs, fe) -> {
            for (int fi = fs; fi < fe; fi++) {
                final int wf = fi * chw;
                for (int k = 0; k < chw; k++) gradW[wf + k] = 0f;
                float gb = 0f;
                final int outF = fi * hOut * wOut;
                for (int bi = 0; bi < batch; bi++) {
                    final float[] xb = x[bi];
                    final float[] gz = gradZ[bi];
                    for (int oh = 0; oh < hOut; oh++) {
                        final int ihBase = oh * stride - pad;
                        for (int ow = 0; ow < wOut; ow++) {
                            final float g = gz[outF + oh * wOut + ow];
                            gb += g;
                            if (g == 0f) continue;
                            final int iwBase = ow * stride - pad;
                            for (int c = 0; c < cin; c++) {
                                final int xc = c * h * w;
                                final int wfc = wf + c * khw;
                                for (int ki = 0; ki < kh; ki++) {
                                    final int ih = ihBase + ki;
                                    if (ih < 0 || ih >= h) continue;
                                    final int xrow = xc + ih * w;
                                    final int wrow = wfc + ki * kw;
                                    for (int kj = 0; kj < kw; kj++) {
                                        final int iw = iwBase + kj;
                                        if (iw < 0 || iw >= w) continue;
                                        gradW[wrow + kj] += g * xb[xrow + iw];
                                    }
                                }
                            }
                        }
                    }
                }
                gradB[fi] = gb;
            }
        });

        // gradInput: parallelize over batch (disjoint gradIn rows).
        final float[][] gradIn = new float[batch][cin * h * w];
        Parallel.forRange(0, batch, 1, (bs, be) -> {
            for (int bi = bs; bi < be; bi++) {
                final float[] gi = gradIn[bi];
                final float[] gz = gradZ[bi];
                for (int fi = 0; fi < f; fi++) {
                    final int wf = fi * chw;
                    final int outF = fi * hOut * wOut;
                    for (int oh = 0; oh < hOut; oh++) {
                        final int ihBase = oh * stride - pad;
                        for (int ow = 0; ow < wOut; ow++) {
                            final float g = gz[outF + oh * wOut + ow];
                            if (g == 0f) continue;
                            final int iwBase = ow * stride - pad;
                            for (int c = 0; c < cin; c++) {
                                final int xc = c * h * w;
                                final int wfc = wf + c * khw;
                                for (int ki = 0; ki < kh; ki++) {
                                    final int ih = ihBase + ki;
                                    if (ih < 0 || ih >= h) continue;
                                    final int xrow = xc + ih * w;
                                    final int wrow = wfc + ki * kw;
                                    for (int kj = 0; kj < kw; kj++) {
                                        final int iw = iwBase + kj;
                                        if (iw < 0 || iw >= w) continue;
                                        gi[xrow + iw] += g * weights[wrow + kj];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        return gradIn;
    }

    @Override
    public void sgdStep(float lr, float l2, float momentum) {
        for (int k = 0; k < weights.length; k++) {
            float g = gradW[k] + l2 * weights[k];
            float v = momentum * velW[k] - lr * g;
            velW[k] = v;
            weights[k] += v;
        }
        for (int fi = 0; fi < f; fi++) {
            float v = momentum * velB[fi] - lr * gradB[fi];
            velB[fi] = v;
            bias[fi] += v;
        }
    }

    @Override
    public void writeParams(DataOutputStream out) throws IOException {
        for (float v : weights) out.writeFloat(v);
        for (float v : bias) out.writeFloat(v);
    }

    @Override
    public void readParams(DataInputStream in) throws IOException {
        for (int k = 0; k < weights.length; k++) weights[k] = in.readFloat();
        for (int k = 0; k < bias.length; k++) bias[k] = in.readFloat();
    }

    // --- test accessors ---
    public float[] weights() { return weights; }
    public float[] bias() { return bias; }
    public float[] gradWeights() { return gradW; }
    public float[] gradBias() { return gradB; }
}
