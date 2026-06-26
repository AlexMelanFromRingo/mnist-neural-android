package com.alex_melan.mnistneural.engine;

import java.io.DataInputStream;
import java.io.DataOutputStream;

/**
 * Reshapes a (C,H,W) feature map into a vector of length C*H*W. Because the engine already stores
 * activations in a flat channel-major layout, this is a pure metadata change — forward and backward
 * pass the same array straight through.
 */
public final class FlattenLayer implements Layer {

    private final Shape in;
    private final Shape out;

    public FlattenLayer(Shape in) {
        this.in = in;
        this.out = Shape.vector(in.size());
    }

    @Override public Shape inputShape() { return in; }

    @Override public Shape outputShape() { return out; }

    @Override public int paramCount() { return 0; }

    @Override public String typeName() { return "Flatten"; }

    @Override public String describe() { return "Flatten(" + in + "→" + out.channels + ")"; }

    @Override public float[][] forward(float[][] x, boolean training) { return x; }

    @Override public float[][] backward(float[][] gradOut) { return gradOut; }

    @Override public void sgdStep(float lr, float l2, float momentum) { }

    @Override public void writeParams(DataOutputStream o) { }

    @Override public void readParams(DataInputStream i) { }
}
