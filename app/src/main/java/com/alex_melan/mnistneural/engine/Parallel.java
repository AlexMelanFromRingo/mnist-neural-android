package com.alex_melan.mnistneural.engine;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Tiny fork-join helper for splitting an index range across CPU cores.
 *
 * <p>Engine layers parallelize over an axis whose entries write to <b>disjoint</b> output locations
 * (e.g. output neurons, filters, or batch rows). Because each chunk reduces over the contracted
 * axis in a fixed order, results are bit-for-bit deterministic regardless of how many threads run —
 * which keeps unit tests and framework cross-checks stable.
 */
public final class Parallel {

    private Parallel() {}

    private static final int CORES = Math.max(1, Runtime.getRuntime().availableProcessors());

    private static final ExecutorService POOL = Executors.newFixedThreadPool(CORES, r -> {
        Thread t = new Thread(r, "nn-worker");
        t.setDaemon(true);
        return t;
    });

    public interface RangeTask {
        void run(int start, int end);
    }

    public static int cores() {
        return CORES;
    }

    /**
     * Runs {@code task} over [from, to). Splits into at most {@code cores} chunks, but only if each
     * chunk would hold at least {@code grain} items — small ranges run inline to avoid thread
     * overhead dominating tiny workloads (e.g. unit tests).
     */
    public static void forRange(int from, int to, int grain, RangeTask task) {
        int n = to - from;
        if (n <= 0) return;
        int maxThreads = Math.min(CORES, Math.max(1, n / Math.max(1, grain)));
        if (maxThreads <= 1) {
            task.run(from, to);
            return;
        }
        int chunk = (n + maxThreads - 1) / maxThreads;
        List<Future<?>> futures = new ArrayList<>(maxThreads);
        for (int t = 0; t < maxThreads; t++) {
            final int s = from + t * chunk;
            final int e = Math.min(to, s + chunk);
            if (s >= e) break;
            futures.add(POOL.submit(() -> task.run(s, e)));
        }
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(ie);
            } catch (ExecutionException ee) {
                Throwable cause = ee.getCause();
                if (cause instanceof RuntimeException) throw (RuntimeException) cause;
                throw new RuntimeException(cause);
            }
        }
    }
}
