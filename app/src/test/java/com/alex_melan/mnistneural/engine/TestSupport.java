package com.alex_melan.mnistneural.engine;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

/** Loads the PyTorch golden reference files and provides float-array comparison helpers. */
final class TestSupport {

    private final Map<String, float[]> map = new HashMap<>();

    static TestSupport load(String resource) {
        TestSupport d = new TestSupport();
        try (InputStream is = TestSupport.class.getClassLoader().getResourceAsStream(resource)) {
            assertNotNull("missing test resource: " + resource, is);
            BufferedReader br = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8));
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                int colon = line.indexOf(':');
                String key = line.substring(0, colon).trim();
                String rest = line.substring(colon + 1).trim();
                if (rest.isEmpty()) {
                    d.map.put(key, new float[0]);
                    continue;
                }
                String[] parts = rest.split("\\s+");
                float[] arr = new float[parts.length];
                for (int i = 0; i < parts.length; i++) arr[i] = Float.parseFloat(parts[i]);
                d.map.put(key, arr);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return d;
    }

    float[] arr(String key) {
        float[] v = map.get(key);
        assertNotNull("no key " + key, v);
        return v;
    }

    double scalar(String key) {
        return arr(key)[0];
    }

    int[] ints(String key) {
        float[] v = arr(key);
        int[] r = new int[v.length];
        for (int i = 0; i < v.length; i++) r[i] = Math.round(v[i]);
        return r;
    }

    static float[][] reshape(float[] flat, int rows, int cols) {
        assertEquals("reshape size", rows * cols, flat.length);
        float[][] m = new float[rows][cols];
        for (int r = 0; r < rows; r++) System.arraycopy(flat, r * cols, m[r], 0, cols);
        return m;
    }

    static float[] flatten(float[][] m) {
        int cols = m[0].length;
        float[] out = new float[m.length * cols];
        for (int r = 0; r < m.length; r++) System.arraycopy(m[r], 0, out, r * cols, cols);
        return out;
    }

    /** Overwrites the contents of {@code dst} with {@code src} (lengths must match). */
    static void setInto(float[] dst, float[] src) {
        assertEquals("param length", dst.length, src.length);
        System.arraycopy(src, 0, dst, 0, dst.length);
    }

    static void assertClose(String msg, float[] expected, float[] actual, double atol, double rtol) {
        assertEquals(msg + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertClose(msg + " [" + i + "]", expected[i], actual[i], atol, rtol);
        }
    }

    static void assertClose(String msg, double expected, double actual, double atol, double rtol) {
        double diff = Math.abs(expected - actual);
        double tol = atol + rtol * Math.abs(expected);
        assertTrue(msg + ": expected " + expected + " got " + actual + " (diff " + diff + " > tol " + tol + ")",
                diff <= tol);
    }
}
