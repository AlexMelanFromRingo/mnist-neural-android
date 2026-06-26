package com.alex_melan.mnistneural.engine;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Framework-independent checks: central-difference gradient verification (a second, independent
 * confirmation of the formulas on top of the PyTorch cross-check), connection-mask freezing,
 * weight serialization, and end-to-end convergence on separable data.
 */
public class EngineGradientTest {

    /** Central-difference check of analytic gradients against the numeric derivative of the mean loss. */
    private static void checkGrad(Network net, float[][] x, int[] labels,
                                  float[] params, float[] analytic, double eps, String tag) {
        for (int p = 0; p < params.length; p++) {
            float orig = params[p];
            params[p] = (float) (orig + eps);
            double lp = net.evaluateBatch(x, labels).meanLoss();
            params[p] = (float) (orig - eps);
            double lm = net.evaluateBatch(x, labels).meanLoss();
            params[p] = orig;
            double num = (lp - lm) / (2 * eps);
            double diff = Math.abs(num - analytic[p]);
            double tol = 2e-2 + 2e-2 * Math.abs(num);
            assertTrue(tag + " grad[" + p + "] numeric=" + num + " analytic=" + analytic[p]
                    + " diff=" + diff + " tol=" + tol, diff <= tol);
        }
    }

    private static float[][] randMatrix(Random r, int rows, int cols, double scale) {
        float[][] m = new float[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) m[i][j] = (float) (r.nextGaussian() * scale);
        return m;
    }

    @Test
    public void denseGradientCheck() {
        Random r = new Random(11);
        int b = 4, nIn = 5, h = 4, nOut = 3;
        float[][] x = randMatrix(r, b, nIn, 1.0);
        int[] labels = {0, 2, 1, 0};

        DenseLayer l1 = new DenseLayer(nIn, h, Activation.TANH, 21L);
        DenseLayer l2 = new DenseLayer(h, nOut, Activation.IDENTITY, 22L);
        Network net = new Network(Shape.vector(nIn), Arrays.asList(l1, l2), Loss.SOFTMAX_CROSS_ENTROPY);

        net.trainBatch(x, labels, 0f, 0f, 0f); // populate analytic grads, weights unchanged
        float[] gW1 = l1.gradWeights().clone();
        float[] gb1 = l1.gradBiases().clone();
        float[] gW2 = l2.gradWeights().clone();

        checkGrad(net, x, labels, l1.weights(), gW1, 1e-3, "dense W1");
        checkGrad(net, x, labels, l1.biases(), gb1, 1e-3, "dense b1");
        checkGrad(net, x, labels, l2.weights(), gW2, 1e-3, "dense W2");
    }

    @Test
    public void convGradientCheck() {
        Random r = new Random(13);
        int b = 3, cin = 1, hh = 4, ww = 4;
        float[][] x = randMatrix(r, b, cin * hh * ww, 1.0);
        int[] labels = {0, 1, 2};

        ConvLayer conv = new ConvLayer(new Shape(cin, hh, ww), 2, 3, 1, 1, Activation.TANH, 31L);
        FlattenLayer flat = new FlattenLayer(conv.outputShape());
        DenseLayer out = new DenseLayer(flat.outputShape().channels, 3, Activation.IDENTITY, 32L);
        Network net = new Network(new Shape(cin, hh, ww), Arrays.asList(conv, flat, out),
                Loss.SOFTMAX_CROSS_ENTROPY);

        net.trainBatch(x, labels, 0f, 0f, 0f);
        float[] gConv = conv.gradWeights().clone();
        float[] gOut = out.gradWeights().clone();

        checkGrad(net, x, labels, conv.weights(), gConv, 1e-3, "conv W");
        checkGrad(net, x, labels, out.weights(), gOut, 1e-3, "conv out W");
    }

    @Test
    public void connectionMaskFreezesWeights() {
        DenseLayer l = new DenseLayer(4, 3, Activation.IDENTITY, 41L);
        boolean[] mask = new boolean[3 * 4];
        Arrays.fill(mask, true);
        for (int i = 0; i < 4; i++) mask[i] = false; // disable all edges into neuron 0
        l.setMask(mask);

        // Masked weights are zero right away.
        for (int i = 0; i < 4; i++) assertEquals(0f, l.weights()[i], 0f);
        // paramCount: 8 active weights + 3 biases.
        assertEquals(11, l.paramCount());

        Network net = new Network(Shape.vector(4),
                new ArrayList<>(List.of(l)), Loss.SOFTMAX_CROSS_ENTROPY);
        float[][] x = {{0.5f, -0.3f, 0.8f, 0.1f}, {-0.2f, 0.4f, -0.6f, 0.9f}};
        int[] y = {1, 2};

        float[] before = l.weights().clone();
        net.trainBatch(x, y, 0.5f, 0f, 0f);

        // Masked edges stay exactly zero with zero gradient...
        for (int i = 0; i < 4; i++) {
            assertEquals("masked weight " + i, 0f, l.weights()[i], 0f);
            assertEquals("masked grad " + i, 0f, l.gradWeights()[i], 0f);
        }
        // ...while at least one active edge actually moved.
        boolean anyActiveChanged = false;
        for (int i = 4; i < 12; i++) if (l.weights()[i] != before[i]) anyActiveChanged = true;
        assertTrue("an active weight should have updated", anyActiveChanged);
    }

    @Test
    public void serializationRoundTrip() throws Exception {
        List<LayerSpec> specs = new ArrayList<>();
        specs.add(LayerSpec.dense(16, Activation.RELU));
        Shape in = new Shape(1, 28, 28);

        Network a = NetworkBuilder.build(in, specs, 10, Loss.SOFTMAX_CROSS_ENTROPY, 1L);
        // Train a couple of steps so weights are non-trivial.
        Random r = new Random(5);
        float[][] x = randMatrix(r, 8, 784, 0.5);
        int[] y = {0, 1, 2, 3, 4, 5, 6, 7};
        for (int i = 0; i < 3; i++) a.trainBatch(x, y, 0.1f, 0f, 0.9f);

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        try (DataOutputStream dos = new DataOutputStream(bos)) {
            a.writeTo(dos);
        }

        Network b = NetworkBuilder.build(in, specs, 10, Loss.SOFTMAX_CROSS_ENTROPY, 999L);
        try (DataInputStream dis = new DataInputStream(new ByteArrayInputStream(bos.toByteArray()))) {
            b.readFrom(dis);
        }

        float[] pa = a.predictProbs(x[0]);
        float[] pb = b.predictProbs(x[0]);
        for (int i = 0; i < pa.length; i++) {
            assertEquals("prob " + i, pa[i], pb[i], 1e-6f);
        }
    }

    @Test
    public void convergesOnSeparableData() {
        Random r = new Random(7);
        int dim = 8, perClass = 50, classes = 3;
        int n = perClass * classes;
        float[][] x = new float[n][dim];
        int[] y = new int[n];
        int idx = 0;
        for (int c = 0; c < classes; c++) {
            for (int s = 0; s < perClass; s++) {
                for (int d = 0; d < dim; d++) x[idx][d] = (float) (r.nextGaussian() * 0.5);
                x[idx][c] += 3.0f; // push class c along its own axis -> linearly separable
                y[idx] = c;
                idx++;
            }
        }

        DenseLayer l1 = new DenseLayer(dim, 16, Activation.RELU, 51L);
        DenseLayer l2 = new DenseLayer(16, classes, Activation.IDENTITY, 52L);
        Network net = new Network(Shape.vector(dim), Arrays.asList(l1, l2), Loss.SOFTMAX_CROSS_ENTROPY);

        double firstLoss = net.evaluateBatch(x, y).meanLoss();
        for (int it = 0; it < 400; it++) net.trainBatch(x, y, 0.05f, 0f, 0.9f);
        Network.Metrics m = net.evaluateBatch(x, y);

        assertTrue("loss should decrease: " + firstLoss + " -> " + m.meanLoss(), m.meanLoss() < firstLoss);
        assertTrue("accuracy should exceed 0.95 but was " + m.accuracy(), m.accuracy() > 0.95);
    }
}
