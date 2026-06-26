package com.alex_melan.mnistneural.engine;

import com.alex_melan.mnistneural.data.MnistDataset;

import java.util.Random;

/**
 * Drives mini-batch SGD over a {@link MnistDataset}. Stateless w.r.t. weights — it mutates the
 * passed-in {@link Network}, so calling {@link #train} again continues from the current weights.
 *
 * <p>Designed to run on a background thread: it reports progress through a {@link ProgressListener}
 * and stops promptly when {@link #cancel()} is called.
 */
public final class Trainer {

    public interface ProgressListener {
        void onProgress(Progress p);
    }

    private volatile boolean cancelled = false;

    public void cancel() {
        cancelled = true;
    }

    public boolean isCancelled() {
        return cancelled;
    }

    public Progress train(Network net, MnistDataset ds, TrainConfig cfg, ProgressListener listener) {
        cancelled = false;
        ds.splitValidation(cfg.valSamples);

        final int trainN = Math.min(cfg.trainSamples, ds.trainCount());
        final int bs = Math.max(1, cfg.batchSize);
        final int stepsPerEpoch = (trainN + bs - 1) / bs;
        final int reportEvery = Math.max(1, stepsPerEpoch / 100);
        final Random rnd = new Random(cfg.seed);

        Progress last = new Progress();
        last.totalEpochs = cfg.epochs;
        last.stepsPerEpoch = stepsPerEpoch;
        last.valAcc = -1;
        last.valLoss = -1;

        for (int epoch = 1; epoch <= cfg.epochs && !cancelled; epoch++) {
            ds.shuffleTrain(rnd);
            double runLoss = 0;
            int runCorrect = 0;
            int runSeen = 0;

            for (int step = 0; step < stepsPerEpoch && !cancelled; step++) {
                int base = step * bs;
                int count = Math.min(bs, trainN - base);
                float[][] xb = new float[count][MnistDataset.IMG_SIZE];
                int[] yb = new int[count];
                for (int k = 0; k < count; k++) {
                    int idx = ds.trainIndex(base + k);
                    ds.copyImage(idx, xb[k]);
                    yb[k] = ds.label(idx);
                }
                Network.Metrics m = net.trainBatch(xb, yb, cfg.learningRate, cfg.l2, cfg.momentum);
                runLoss += m.sumLoss;
                runCorrect += m.correct;
                runSeen += m.count;

                if (step % reportEvery == 0 || step == stepsPerEpoch - 1) {
                    last.epoch = epoch;
                    last.step = step + 1;
                    last.trainLoss = runLoss / runSeen;
                    last.trainAcc = (double) runCorrect / runSeen;
                    last.finished = false;
                    if (listener != null) listener.onProgress(last.snapshot());
                }
            }

            if (cancelled) break;

            // Validation pass in chunks.
            Network.Metrics val = evaluate(net, ds, ds.valCount());
            last.epoch = epoch;
            last.step = stepsPerEpoch;
            last.trainLoss = runLoss / Math.max(1, runSeen);
            last.trainAcc = (double) runCorrect / Math.max(1, runSeen);
            last.valLoss = val.meanLoss();
            last.valAcc = val.accuracy();
            last.finished = (epoch == cfg.epochs);
            if (listener != null) listener.onProgress(last.snapshot());
        }

        last.finished = true;
        last.cancelled = cancelled;
        if (listener != null) listener.onProgress(last.snapshot());
        return last;
    }

    /** Evaluates the validation split (first {@code valN} val samples) in chunks of 256. */
    private Network.Metrics evaluate(Network net, MnistDataset ds, int valN) {
        Network.Metrics total = new Network.Metrics(0, 0, 0);
        final int chunk = 256;
        for (int base = 0; base < valN && !cancelled; base += chunk) {
            int count = Math.min(chunk, valN - base);
            float[][] xb = new float[count][MnistDataset.IMG_SIZE];
            int[] yb = new int[count];
            for (int k = 0; k < count; k++) {
                int idx = ds.valIndex(base + k);
                ds.copyImage(idx, xb[k]);
                yb[k] = ds.label(idx);
            }
            total = total.plus(net.evaluateBatch(xb, yb));
        }
        return total;
    }

    /** Mutable progress accumulator; {@link #snapshot()} produces an immutable copy for the UI. */
    public static final class Progress {
        public int epoch;
        public int totalEpochs;
        public int step;
        public int stepsPerEpoch;
        public double trainLoss;
        public double trainAcc;
        public double valLoss = -1;
        public double valAcc = -1;
        public boolean finished;
        public boolean cancelled;

        public Progress snapshot() {
            Progress p = new Progress();
            p.epoch = epoch;
            p.totalEpochs = totalEpochs;
            p.step = step;
            p.stepsPerEpoch = stepsPerEpoch;
            p.trainLoss = trainLoss;
            p.trainAcc = trainAcc;
            p.valLoss = valLoss;
            p.valAcc = valAcc;
            p.finished = finished;
            p.cancelled = cancelled;
            return p;
        }
    }
}
