package com.alex_melan.mnistneural.engine;

/**
 * Element-wise activation functions and their derivatives.
 *
 * <p>Note: <b>softmax is intentionally not here</b>. For classification it is fused with
 * cross-entropy in {@link Loss#SOFTMAX_CROSS_ENTROPY}, which yields the clean, numerically stable
 * gradient {@code softmax(z) - y}. A layer feeding that loss uses {@link #IDENTITY}.
 *
 * <p>{@link #derivativeFromOutput(float)} computes f'(z) from the activation output a = f(z), which
 * is cheaper and exact for all the functions below:
 * <ul>
 *   <li>identity: f'(z) = 1</li>
 *   <li>relu:     f'(z) = a &gt; 0 ? 1 : 0   (a = max(0,z), so a&gt;0 ⇔ z&gt;0)</li>
 *   <li>sigmoid:  f'(z) = a(1 - a)</li>
 *   <li>tanh:     f'(z) = 1 - a²</li>
 * </ul>
 */
public enum Activation {
    IDENTITY {
        @Override public float apply(float z) { return z; }
        @Override public float derivativeFromOutput(float a) { return 1f; }
    },
    RELU {
        @Override public float apply(float z) { return z > 0f ? z : 0f; }
        @Override public float derivativeFromOutput(float a) { return a > 0f ? 1f : 0f; }
    },
    SIGMOID {
        @Override public float apply(float z) {
            // Branchless-ish stable sigmoid.
            if (z >= 0f) {
                float e = (float) Math.exp(-z);
                return 1f / (1f + e);
            } else {
                float e = (float) Math.exp(z);
                return e / (1f + e);
            }
        }
        @Override public float derivativeFromOutput(float a) { return a * (1f - a); }
    },
    TANH {
        @Override public float apply(float z) { return (float) Math.tanh(z); }
        @Override public float derivativeFromOutput(float a) { return 1f - a * a; }
    };

    public abstract float apply(float z);

    /** Derivative f'(z) expressed in terms of the output a = f(z). */
    public abstract float derivativeFromOutput(float a);

    /** Whether this activation tends to need He init (true) vs Xavier/Glorot (false). */
    public boolean prefersHeInit() {
        return this == RELU;
    }
}
