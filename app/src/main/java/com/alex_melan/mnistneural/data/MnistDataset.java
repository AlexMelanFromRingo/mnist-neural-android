package com.alex_melan.mnistneural.data;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;
import java.util.zip.GZIPInputStream;

/**
 * In-memory MNIST loaded from the compact gzipped binary in assets ({@code mnist.bin.gz}).
 *
 * <p>Pixels are kept as raw unsigned bytes (0..255) in a single flat array — ~47&nbsp;MB for the
 * full 60k set — and normalized to [0,1] on the fly when a batch is materialized. This avoids the
 * ~376&nbsp;MB {@code List<double[]>} blow-up of the original loader.
 *
 * <p>The binary is pre-shuffled (fixed seed) by the offline converter, so a deterministic tail
 * slice is a clean, class-balanced validation split.
 */
public final class MnistDataset {

    public static final int ROWS = 28;
    public static final int COLS = 28;
    public static final int IMG_SIZE = ROWS * COLS; // 784
    public static final int CLASSES = 10;

    private static final int MAGIC = 0x4D4E4953; // 'MNIS'

    private final byte[] pixels; // count * IMG_SIZE, unsigned
    private final byte[] labels; // count
    private final int count;

    /** Shuffled index buffer; [0, trainCount) are training indices, the rest are validation. */
    private final int[] order;
    private int trainCount;

    private MnistDataset(byte[] pixels, byte[] labels, int count) {
        this.pixels = pixels;
        this.labels = labels;
        this.count = count;
        this.order = new int[count];
        for (int i = 0; i < count; i++) order[i] = i;
        this.trainCount = count;
    }

    /** Streams and decodes the bundled MNIST binary. Call off the UI thread. */
    public static MnistDataset loadFromAssets(Context ctx) throws IOException {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(openAsset(ctx.getAssets()), 1 << 16))) {

            int magic = in.readInt();
            if (magic != MAGIC) {
                throw new IOException("Bad MNIST magic: 0x" + Integer.toHexString(magic));
            }
            int count = in.readInt();
            int rows = in.readInt();
            int cols = in.readInt();
            if (rows != ROWS || cols != COLS) {
                throw new IOException("Unexpected image dims " + rows + "x" + cols);
            }
            byte[] pixels = new byte[count * IMG_SIZE];
            byte[] labels = new byte[count];
            for (int i = 0; i < count; i++) {
                labels[i] = in.readByte();
                in.readFully(pixels, i * IMG_SIZE, IMG_SIZE);
            }
            return new MnistDataset(pixels, labels, count);
        }
    }

    /**
     * AGP gunzips {@code .gz} assets during the merge step, so the packaged file is normally the
     * raw {@code mnist.bin}. We try that first and fall back to inflating {@code mnist.bin.gz}
     * ourselves, so the loader works regardless of how the asset ended up packaged.
     */
    private static InputStream openAsset(AssetManager assets) throws IOException {
        try {
            return assets.open("mnist.bin");
        } catch (FileNotFoundException notDecompressed) {
            return new GZIPInputStream(assets.open("mnist.bin.gz"));
        }
    }

    public int totalCount() {
        return count;
    }

    public int trainCount() {
        return trainCount;
    }

    public int valCount() {
        return count - trainCount;
    }

    /**
     * Carves the last {@code valCount} samples (in the current order) off as a validation set.
     * The binary is pre-shuffled, so this stays class-balanced.
     */
    public void splitValidation(int valCount) {
        valCount = Math.max(0, Math.min(valCount, count));
        this.trainCount = count - valCount;
    }

    /** Shuffles only the training portion in place; validation indices are untouched. */
    public void shuffleTrain(Random rnd) {
        for (int i = trainCount - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
    }

    /** Real sample index for the i-th training item (after shuffling). */
    public int trainIndex(int i) {
        return order[i];
    }

    /** Real sample index for the i-th validation item. */
    public int valIndex(int i) {
        return order[trainCount + i];
    }

    /** Writes the normalized [0,1] image at {@code sampleIndex} into {@code dst} (length 784). */
    public void copyImage(int sampleIndex, float[] dst) {
        int base = sampleIndex * IMG_SIZE;
        for (int j = 0; j < IMG_SIZE; j++) {
            dst[j] = (pixels[base + j] & 0xFF) / 255f;
        }
    }

    public int label(int sampleIndex) {
        return labels[sampleIndex] & 0xFF;
    }
}
