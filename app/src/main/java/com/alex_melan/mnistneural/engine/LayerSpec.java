package com.alex_melan.mnistneural.engine;

/**
 * A no-code description of one hidden layer, produced by the constructor UI and consumed by
 * {@link NetworkBuilder}. The input layer (1×28×28) and the output layer (10 logits) are implicit
 * and added automatically, so a spec list contains only the hidden stack.
 *
 * <p>Mutable on purpose: Compose state edits these fields directly via sliders / dropdowns.
 */
public final class LayerSpec {

    public enum Type { DENSE, CONV, POOL }

    public Type type;
    public Activation activation = Activation.RELU;

    // DENSE
    public int units = 64;
    /** Optional per-edge connection mask (length nOut*nIn), set by the connection editor. */
    public boolean[] mask;

    // CONV
    public int filters = 8;
    public int kernel = 3;
    public int stride = 1;
    public int pad = 1;

    // POOL
    public int poolKernel = 2;
    public int poolStride = 2;

    public LayerSpec(Type type) {
        this.type = type;
    }

    public static LayerSpec dense(int units, Activation act) {
        LayerSpec s = new LayerSpec(Type.DENSE);
        s.units = units;
        s.activation = act;
        return s;
    }

    public static LayerSpec conv(int filters, int kernel, int stride, int pad, Activation act) {
        LayerSpec s = new LayerSpec(Type.CONV);
        s.filters = filters;
        s.kernel = kernel;
        s.stride = stride;
        s.pad = pad;
        s.activation = act;
        return s;
    }

    public static LayerSpec pool(int kernel, int stride) {
        LayerSpec s = new LayerSpec(Type.POOL);
        s.poolKernel = kernel;
        s.poolStride = stride;
        return s;
    }

    public LayerSpec copy() {
        LayerSpec s = new LayerSpec(type);
        s.activation = activation;
        s.units = units;
        s.mask = (mask == null) ? null : mask.clone();
        s.filters = filters;
        s.kernel = kernel;
        s.stride = stride;
        s.pad = pad;
        s.poolKernel = poolKernel;
        s.poolStride = poolStride;
        return s;
    }
}
