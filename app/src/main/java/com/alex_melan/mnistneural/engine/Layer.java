package com.alex_melan.mnistneural.engine;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * A feed-forward layer operating on mini-batches.
 *
 * <p>Data flows as {@code float[batch][flatSize]}. Gradients are <b>already batch-averaged</b> by
 * the time they reach a layer: {@link Network} scales the loss seed gradient by {@code 1/B}, and
 * since every backward op is linear in the incoming gradient, all parameter gradients come out
 * averaged. Therefore {@link #sgdStep} simply applies {@code w -= lr * (gradW + l2*w)} (with
 * optional momentum).
 */
public interface Layer {

    Shape inputShape();

    Shape outputShape();

    /** Number of trainable parameters (respects connection masks). */
    int paramCount();

    String typeName();

    String describe();

    /**
     * Forward pass. {@code x} is [B][inputShape.size()]. Returns [B][outputShape.size()].
     * When {@code training} is true the layer caches whatever it needs for {@link #backward}.
     */
    float[][] forward(float[][] x, boolean training);

    /**
     * Backward pass. {@code gradOut} is dL/d(output) of shape [B][outSize]. Returns dL/d(input) of
     * shape [B][inSize] and stores parameter gradients internally for the next {@link #sgdStep}.
     */
    float[][] backward(float[][] gradOut);

    /** Applies the accumulated gradients. No-op for parameter-free layers. */
    void sgdStep(float lr, float l2, float momentum);

    void writeParams(DataOutputStream out) throws IOException;

    void readParams(DataInputStream in) throws IOException;
}
