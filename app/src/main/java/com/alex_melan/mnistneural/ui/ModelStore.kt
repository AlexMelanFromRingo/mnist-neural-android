package com.alex_melan.mnistneural.ui

import android.content.Context
import com.alex_melan.mnistneural.engine.Activation
import com.alex_melan.mnistneural.engine.Loss
import com.alex_melan.mnistneural.engine.Network
import com.alex_melan.mnistneural.engine.NetworkBuilder
import com.alex_melan.mnistneural.engine.Shape
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

/**
 * Persists a trained model to internal storage: the no-code architecture (so it can be restored
 * into the constructor) followed by the engine's weight blob. One slot ({@code model.mnn}).
 */
object ModelStore {

    private const val FILE = "model.mnn"
    private const val MAGIC = 0x4D4E4D31 // 'MNM1'

    fun file(context: Context): File = File(context.filesDir, FILE)

    fun exists(context: Context): Boolean = file(context).exists()

    fun save(context: Context, hidden: List<LayerUi>, loss: Loss, net: Network) {
        DataOutputStream(BufferedOutputStream(FileOutputStream(file(context)))).use { o ->
            o.writeInt(MAGIC)
            o.writeUTF(loss.name)
            o.writeInt(hidden.size)
            for (l in hidden) {
                o.writeUTF(l.kind.name)
                o.writeUTF(l.activation.name)
                o.writeInt(l.units)
                o.writeInt(l.filters)
                o.writeInt(l.kernel)
                o.writeInt(l.stride)
                o.writeInt(l.pad)
                o.writeInt(l.poolKernel)
                o.writeInt(l.poolStride)
                val m = l.mask
                o.writeInt(m?.size ?: 0)
                if (m != null) for (b in m) o.writeBoolean(b)
            }
            net.writeTo(o) // continues in the same stream
        }
    }

    class Loaded(val hidden: List<LayerUi>, val loss: Loss, val net: Network)

    fun load(context: Context, inputShape: Shape, numClasses: Int): Loaded {
        DataInputStream(BufferedInputStream(FileInputStream(file(context)))).use { i ->
            val magic = i.readInt()
            require(magic == MAGIC) { "Неверный формат файла модели" }
            val loss = Loss.valueOf(i.readUTF())
            val n = i.readInt()
            val hidden = ArrayList<LayerUi>(n)
            repeat(n) {
                val kind = LayerKind.valueOf(i.readUTF())
                val act = Activation.valueOf(i.readUTF())
                val units = i.readInt()
                val filters = i.readInt()
                val kernel = i.readInt()
                val stride = i.readInt()
                val pad = i.readInt()
                val poolKernel = i.readInt()
                val poolStride = i.readInt()
                val mlen = i.readInt()
                val mask = if (mlen > 0) BooleanArray(mlen) { i.readBoolean() } else null
                hidden.add(LayerUi(kind, units, act, filters, kernel, stride, pad, poolKernel, poolStride, mask))
            }
            val net = NetworkBuilder.build(inputShape, hidden.map { it.toSpec() }, numClasses, loss, 42L)
            net.readFrom(i) // reads the weight blob written by net.writeTo
            return Loaded(hidden, loss, net)
        }
    }
}
