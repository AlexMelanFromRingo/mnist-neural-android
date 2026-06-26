package com.alex_melan.mnistneural.engine;

/**
 * Tensor shape for a single sample, as (channels, height, width). Dense activations are represented
 * as (units, 1, 1). The flat length is {@code channels * height * width}; layers exchange data as
 * {@code float[batch][size()]} laid out channel-major then row-major
 * ({@code index = c*H*W + h*W + w}).
 */
public final class Shape {
    public final int channels;
    public final int height;
    public final int width;

    public Shape(int channels, int height, int width) {
        if (channels <= 0 || height <= 0 || width <= 0) {
            throw new IllegalArgumentException("Shape dims must be positive: "
                    + channels + "x" + height + "x" + width);
        }
        this.channels = channels;
        this.height = height;
        this.width = width;
    }

    /** Vector shape (n units). */
    public static Shape vector(int n) {
        return new Shape(n, 1, 1);
    }

    public int size() {
        return channels * height * width;
    }

    public boolean isVector() {
        return height == 1 && width == 1;
    }

    @Override
    public String toString() {
        return isVector() ? (channels + "") : (channels + "×" + height + "×" + width);
    }
}
