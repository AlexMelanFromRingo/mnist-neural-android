package com.alex_melan.mnistneural.engine;

import java.util.ArrayList;
import java.util.List;

/**
 * Turns a no-code {@link LayerSpec} list into a runnable {@link Network}, inserting the structural
 * glue the user shouldn't have to think about:
 * <ul>
 *   <li>a {@link FlattenLayer} whenever a Dense layer follows a 2-D feature map;</li>
 *   <li>a final output {@link DenseLayer} of {@code numClasses} units — IDENTITY (logits) for
 *       {@link Loss#SOFTMAX_CROSS_ENTROPY}, SIGMOID for {@link Loss#MSE}.</li>
 * </ul>
 *
 * <p>{@link #plan} does the same shape arithmetic <i>without allocating weights</i>, so the UI can
 * show a live parameter count and catch invalid geometries (e.g. a kernel larger than the feature
 * map) as the user drags sliders.
 */
public final class NetworkBuilder {

    private NetworkBuilder() {}

    public static Network build(Shape input, List<LayerSpec> hidden, int numClasses, Loss loss, long seed) {
        List<Layer> layers = new ArrayList<>();
        Shape cur = input;
        long s = seed;
        for (LayerSpec spec : hidden) {
            switch (spec.type) {
                case CONV: {
                    ConvLayer c = new ConvLayer(cur, spec.filters, spec.kernel, spec.stride, spec.pad,
                            spec.activation, s++);
                    layers.add(c);
                    cur = c.outputShape();
                    break;
                }
                case POOL: {
                    MaxPoolLayer p = new MaxPoolLayer(cur, spec.poolKernel, spec.poolStride);
                    layers.add(p);
                    cur = p.outputShape();
                    break;
                }
                case DENSE: {
                    if (!cur.isVector()) {
                        layers.add(new FlattenLayer(cur));
                        cur = Shape.vector(cur.size());
                    }
                    DenseLayer d = new DenseLayer(cur.channels, spec.units, spec.activation, s++);
                    if (spec.mask != null && spec.mask.length == spec.units * cur.channels) {
                        d.setMask(spec.mask);
                    }
                    layers.add(d);
                    cur = Shape.vector(spec.units);
                    break;
                }
                default:
                    throw new IllegalArgumentException("Unknown layer type " + spec.type);
            }
        }
        if (!cur.isVector()) {
            layers.add(new FlattenLayer(cur));
            cur = Shape.vector(cur.size());
        }
        Activation outAct = (loss == Loss.SOFTMAX_CROSS_ENTROPY) ? Activation.IDENTITY : Activation.SIGMOID;
        layers.add(new DenseLayer(cur.channels, numClasses, outAct, s));
        return new Network(input, layers, loss);
    }

    /** Live, allocation-free shape + parameter analysis for the constructor UI. */
    public static Plan plan(Shape input, List<LayerSpec> hidden, int numClasses, Loss loss) {
        List<String> lines = new ArrayList<>();
        lines.add("Вход: " + input + " (" + input.size() + ")");
        Shape cur = input;
        long total = 0;
        try {
            for (LayerSpec spec : hidden) {
                switch (spec.type) {
                    case CONV: {
                        int hOut = (cur.height + 2 * spec.pad - spec.kernel) / spec.stride + 1;
                        int wOut = (cur.width + 2 * spec.pad - spec.kernel) / spec.stride + 1;
                        if (hOut <= 0 || wOut <= 0) {
                            return Plan.error("Conv-слой даёт некорректный размер "
                                    + hOut + "×" + wOut + " (ядро больше карты признаков?)");
                        }
                        int p = spec.filters * cur.channels * spec.kernel * spec.kernel + spec.filters;
                        total += p;
                        cur = new Shape(spec.filters, hOut, wOut);
                        lines.add("Conv2D " + spec.filters + "×k" + spec.kernel + " → " + cur
                                + "  (" + p + " пар.)");
                        break;
                    }
                    case POOL: {
                        int hOut = (cur.height - spec.poolKernel) / spec.poolStride + 1;
                        int wOut = (cur.width - spec.poolKernel) / spec.poolStride + 1;
                        if (hOut <= 0 || wOut <= 0) {
                            return Plan.error("Pool-слой даёт некорректный размер " + hOut + "×" + wOut);
                        }
                        cur = new Shape(cur.channels, hOut, wOut);
                        lines.add("MaxPool k" + spec.poolKernel + " → " + cur);
                        break;
                    }
                    case DENSE: {
                        int inN = cur.size();
                        if (spec.mask != null && spec.mask.length == spec.units * inN) {
                            int p = countActive(spec.mask) + spec.units;
                            total += p;
                            lines.add("Dense → " + spec.units + " (" + spec.activation + ", маска, "
                                    + p + " пар.)");
                        } else {
                            int p = inN * spec.units + spec.units;
                            total += p;
                            lines.add("Dense → " + spec.units + " (" + spec.activation + ", " + p + " пар.)");
                        }
                        cur = Shape.vector(spec.units);
                        break;
                    }
                    default:
                        return Plan.error("Неизвестный тип слоя");
                }
            }
            int inN = cur.size();
            int outP = inN * numClasses + numClasses;
            total += outP;
            lines.add("Выход: Dense → " + numClasses
                    + (loss == Loss.SOFTMAX_CROSS_ENTROPY ? " (Softmax)" : " (Sigmoid)")
                    + "  (" + outP + " пар.)");
        } catch (RuntimeException ex) {
            return Plan.error("Ошибка конфигурации: " + ex.getMessage());
        }
        return new Plan(true, null, total, lines);
    }

    private static int countActive(boolean[] mask) {
        int n = 0;
        for (boolean m : mask) if (m) n++;
        return n;
    }

    /**
     * Flat input size that will feed the hidden layer at {@code index} (i.e. the {@code nIn} a Dense
     * layer there would see after any implicit flatten). Used by the connection editor to size masks.
     * Assumes the preceding layers form a valid geometry.
     */
    public static int flatInputSizeAt(Shape input, List<LayerSpec> hidden, int index) {
        Shape cur = input;
        for (int i = 0; i < index && i < hidden.size(); i++) {
            LayerSpec s = hidden.get(i);
            switch (s.type) {
                case CONV: {
                    int hOut = (cur.height + 2 * s.pad - s.kernel) / s.stride + 1;
                    int wOut = (cur.width + 2 * s.pad - s.kernel) / s.stride + 1;
                    cur = new Shape(s.filters, Math.max(1, hOut), Math.max(1, wOut));
                    break;
                }
                case POOL: {
                    int hOut = (cur.height - s.poolKernel) / s.poolStride + 1;
                    int wOut = (cur.width - s.poolKernel) / s.poolStride + 1;
                    cur = new Shape(cur.channels, Math.max(1, hOut), Math.max(1, wOut));
                    break;
                }
                case DENSE:
                    cur = Shape.vector(s.units);
                    break;
                default:
                    break;
            }
        }
        return cur.size();
    }

    public static final class Plan {
        public final boolean valid;
        public final String error;
        public final long totalParams;
        public final List<String> layerLines;

        Plan(boolean valid, String error, long totalParams, List<String> layerLines) {
            this.valid = valid;
            this.error = error;
            this.totalParams = totalParams;
            this.layerLines = layerLines;
        }

        static Plan error(String msg) {
            return new Plan(false, msg, 0, new ArrayList<>());
        }
    }
}
