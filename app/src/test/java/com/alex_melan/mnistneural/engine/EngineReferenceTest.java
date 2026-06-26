package com.alex_melan.mnistneural.engine;

import static com.alex_melan.mnistneural.engine.TestSupport.assertClose;
import static com.alex_melan.mnistneural.engine.TestSupport.flatten;
import static com.alex_melan.mnistneural.engine.TestSupport.reshape;
import static com.alex_melan.mnistneural.engine.TestSupport.setInto;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

/**
 * Cross-checks the engine against golden values produced by PyTorch (see scratchpad/gen_refs.py).
 * Tolerances account for float32 summation-order differences between the two implementations.
 */
public class EngineReferenceTest {

    private static final double ATOL = 1e-3;
    private static final double RTOL = 1e-3;

    @Test
    public void denseMlpMatchesPyTorch() {
        TestSupport d = TestSupport.load("refs/dense_mlp.txt");
        int[] dims = d.ints("dims");
        int b = dims[0], nIn = dims[1], h = dims[2], nOut = dims[3];

        float[][] x = reshape(d.arr("x"), b, nIn);
        int[] labels = d.ints("labels");

        DenseLayer l1 = new DenseLayer(nIn, h, Activation.RELU, 1L);
        setInto(l1.weights(), d.arr("W1"));
        setInto(l1.biases(), d.arr("b1"));
        DenseLayer l2 = new DenseLayer(h, nOut, Activation.IDENTITY, 2L);
        setInto(l2.weights(), d.arr("W2"));
        setInto(l2.biases(), d.arr("b2"));

        List<Layer> layers = Arrays.asList(l1, l2);
        Network net = new Network(Shape.vector(nIn), layers, Loss.SOFTMAX_CROSS_ENTROPY);

        // Forward logits.
        float[][] logits = net.forward(x, false);
        assertClose("logits", d.arr("logits"), flatten(logits), ATOL, RTOL);

        // Loss + gradients: lr=0 leaves weights untouched but populates grad buffers (mean reduction).
        Network.Metrics m = net.trainBatch(x, labels, 0f, 0f, 0f);
        assertClose("loss", d.scalar("loss"), m.meanLoss(), 1e-4, 1e-4);
        assertClose("gW1", d.arr("gW1"), l1.gradWeights(), ATOL, RTOL);
        assertClose("gb1", d.arr("gb1"), l1.gradBiases(), ATOL, RTOL);
        assertClose("gW2", d.arr("gW2"), l2.gradWeights(), ATOL, RTOL);
        assertClose("gb2", d.arr("gb2"), l2.gradBiases(), ATOL, RTOL);
    }

    @Test
    public void convMatchesPyTorch() {
        TestSupport d = TestSupport.load("refs/conv.txt");
        int[] dims = d.ints("dims");
        int b = dims[0], cin = dims[1], hh = dims[2], ww = dims[3];
        int fo = dims[4], k = dims[5], s = dims[6], p = dims[7];
        int hOut = (hh + 2 * p - k) / s + 1;
        int wOut = (ww + 2 * p - k) / s + 1;

        ConvLayer conv = new ConvLayer(new Shape(cin, hh, ww), fo, k, s, p, Activation.RELU, 7L);
        setInto(conv.weights(), d.arr("weight"));
        setInto(conv.bias(), d.arr("bias"));

        float[][] x = reshape(d.arr("x"), b, cin * hh * ww);
        float[][] a = conv.forward(x, true);
        assertClose("conv activation", d.arr("a"), flatten(a), ATOL, RTOL);

        float[][] gradOut = reshape(d.arr("gradOut"), b, fo * hOut * wOut);
        float[][] gradIn = conv.backward(gradOut);
        assertClose("conv gradWeight", d.arr("gradWeight"), conv.gradWeights(), ATOL, RTOL);
        assertClose("conv gradBias", d.arr("gradBias"), conv.gradBias(), ATOL, RTOL);
        assertClose("conv gradInput", d.arr("gradInput"), flatten(gradIn), ATOL, RTOL);
    }

    @Test
    public void maxPoolMatchesPyTorch() {
        TestSupport d = TestSupport.load("refs/maxpool.txt");
        int[] dims = d.ints("dims");
        int b = dims[0], c = dims[1], hh = dims[2], ww = dims[3], k = dims[4], s = dims[5];
        int hOut = (hh - k) / s + 1;
        int wOut = (ww - k) / s + 1;

        MaxPoolLayer pool = new MaxPoolLayer(new Shape(c, hh, ww), k, s);
        float[][] x = reshape(d.arr("x"), b, c * hh * ww);
        float[][] out = pool.forward(x, true);
        assertClose("pool out", d.arr("out"), flatten(out), ATOL, RTOL);

        float[][] gradOut = reshape(d.arr("gradOut"), b, c * hOut * wOut);
        float[][] gradIn = pool.backward(gradOut);
        assertClose("pool gradInput", d.arr("gradInput"), flatten(gradIn), ATOL, RTOL);
    }
}
