package com.alex_melan.mnistneural.ui

import com.alex_melan.mnistneural.engine.Activation
import com.alex_melan.mnistneural.engine.LayerSpec

/** Layer kinds the user can add in the constructor. */
enum class LayerKind { DENSE, CONV, POOL }

/**
 * Immutable, Compose-friendly description of one hidden layer. The constructor edits these by
 * replacing list entries (so Compose observes the change); they map to the engine's [LayerSpec]
 * at build time.
 */
data class LayerUi(
    val kind: LayerKind,
    val units: Int = 64,
    val activation: Activation = Activation.RELU,
    val filters: Int = 8,
    val kernel: Int = 3,
    val stride: Int = 1,
    val pad: Int = 1,
    val poolKernel: Int = 2,
    val poolStride: Int = 2,
    /** Per-edge connection mask for DENSE layers (length nOut*nIn); null = fully connected. */
    val mask: BooleanArray? = null,
) {
    fun toSpec(): LayerSpec = when (kind) {
        LayerKind.DENSE -> LayerSpec.dense(units, activation).also { it.mask = mask }
        LayerKind.CONV -> LayerSpec.conv(filters, kernel, stride, pad, activation)
        LayerKind.POOL -> LayerSpec.pool(poolKernel, poolStride)
    }

    val title: String
        get() = when (kind) {
            LayerKind.DENSE -> "Dense • $units нейронов"
            LayerKind.CONV -> "Conv2D • $filters фильтров ${kernel}×$kernel"
            LayerKind.POOL -> "MaxPool • ${poolKernel}×$poolKernel"
        }

    // data class with an array field: identity-style equals/hashCode are fine here (we replace whole
    // items on edit), but override to silence the array-in-data-class warning and stay correct.
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is LayerUi) return false
        return kind == other.kind && units == other.units && activation == other.activation &&
                filters == other.filters && kernel == other.kernel && stride == other.stride &&
                pad == other.pad && poolKernel == other.poolKernel && poolStride == other.poolStride &&
                mask === other.mask
    }

    override fun hashCode(): Int {
        var r = kind.hashCode()
        r = 31 * r + units
        r = 31 * r + activation.hashCode()
        r = 31 * r + filters
        r = 31 * r + kernel
        r = 31 * r + stride
        r = 31 * r + pad
        r = 31 * r + poolKernel
        r = 31 * r + poolStride
        r = 31 * r + (mask?.let { System.identityHashCode(it) } ?: 0)
        return r
    }

    companion object {
        fun defaultFor(kind: LayerKind): LayerUi = when (kind) {
            LayerKind.DENSE -> LayerUi(LayerKind.DENSE, units = 64, activation = Activation.RELU)
            LayerKind.CONV -> LayerUi(LayerKind.CONV, filters = 8, kernel = 3, stride = 1, pad = 1,
                activation = Activation.RELU)
            LayerKind.POOL -> LayerUi(LayerKind.POOL, poolKernel = 2, poolStride = 2)
        }
    }
}

sealed interface DatasetStatus {
    data object NotLoaded : DatasetStatus
    data object Loading : DatasetStatus
    data class Loaded(val count: Int) : DatasetStatus
    data class Error(val message: String) : DatasetStatus
}

class Prediction(val digit: Int, val probs: FloatArray)

/** Ready-made architectures the user can load with one tap. */
enum class Preset(val title: String) {
    PERCEPTRON("Перцептрон"),
    MLP("MLP"),
    CNN("CNN"),
}

/** Full-set evaluation: overall + per-class accuracy and a 10×10 confusion matrix. */
class EvalResult(
    val accuracy: Double,
    val perClassCorrect: IntArray,
    val perClassTotal: IntArray,
    val confusion: Array<IntArray>,
    val count: Int,
)
