package com.alex_melan.mnistneural.engine;

/**
 * User-tunable training hyperparameters (exposed in the UI via sliders / fields).
 */
public final class TrainConfig {
    public int epochs = 5;
    public int batchSize = 32;
    public float learningRate = 0.05f;
    public float momentum = 0.9f;
    public float l2 = 0f;
    /** How many samples to actually train on per epoch (subset of the 60k for speed on-device). */
    public int trainSamples = 8000;
    public int valSamples = 2000;
    public long seed = 42L;

    public TrainConfig copy() {
        TrainConfig c = new TrainConfig();
        c.epochs = epochs;
        c.batchSize = batchSize;
        c.learningRate = learningRate;
        c.momentum = momentum;
        c.l2 = l2;
        c.trainSamples = trainSamples;
        c.valSamples = valSamples;
        c.seed = seed;
        return c;
    }
}
