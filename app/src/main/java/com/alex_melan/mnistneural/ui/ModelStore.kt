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
 * Persists trained models to internal storage. Each model is one file in the `models/` directory:
 * a lightweight header (save time, parameter count, a human-readable summary), then the no-code
 * architecture (so it can be restored into the constructor), then the engine's weight blob.
 *
 * Multiple named models are supported — they can be listed (header read only, no weights loaded),
 * loaded and deleted independently.
 */
object ModelStore {

    private const val MAGIC = 0x4D4E4D32 // 'MNM2'
    private const val DIR = "models"
    private const val EXT = ".mnn"

    private fun dir(context: Context): File = File(context.filesDir, DIR).apply { mkdirs() }

    fun fileFor(context: Context, name: String): File = File(dir(context), sanitize(name) + EXT)

    fun exists(context: Context, name: String): Boolean = fileFor(context, name).exists()

    /**
     * Restricts a user-entered name to a safe, bounded filename stem: keeps letters/digits
     * (incl. Cyrillic), dot, underscore and hyphen; everything else (spaces, slashes, control
     * chars, …) becomes '_'. Idempotent.
     */
    fun sanitize(name: String): String {
        val out = StringBuilder()
        for (c in name.trim()) {
            out.append(if (c.isLetterOrDigit() || c == '.' || c == '_' || c == '-') c else '_')
        }
        return out.toString().take(60).ifBlank { "model" }
    }

    /** Header metadata for the model list — read without touching the weight blob. */
    class Info(
        val name: String,
        val savedAt: Long,
        val paramCount: Int,
        val summary: String,
        val sizeBytes: Long,
    )

    fun save(
        context: Context,
        name: String,
        hidden: List<LayerUi>,
        loss: Loss,
        net: Network,
        savedAt: Long,
        paramCount: Int,
        summary: String,
    ) {
        DataOutputStream(BufferedOutputStream(FileOutputStream(fileFor(context, name)))).use { o ->
            o.writeInt(MAGIC)
            o.writeLong(savedAt)
            o.writeInt(paramCount)
            o.writeUTF(summary)
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

    /** All saved models, newest first. Corrupt or foreign files are skipped. */
    fun list(context: Context): List<Info> {
        val files = dir(context).listFiles { f -> f.isFile && f.name.endsWith(EXT) } ?: return emptyList()
        val out = ArrayList<Info>(files.size)
        for (f in files) {
            try {
                DataInputStream(BufferedInputStream(FileInputStream(f))).use { i ->
                    if (i.readInt() != MAGIC) return@use
                    val savedAt = i.readLong()
                    val params = i.readInt()
                    val summary = i.readUTF()
                    out.add(
                        Info(
                            name = f.name.removeSuffix(EXT),
                            savedAt = savedAt,
                            paramCount = params,
                            summary = summary,
                            sizeBytes = f.length(),
                        )
                    )
                }
            } catch (_: Exception) {
                // skip an unreadable / partially written file
            }
        }
        out.sortByDescending { it.savedAt }
        return out
    }

    fun delete(context: Context, name: String): Boolean = fileFor(context, name).delete()

    class Loaded(val hidden: List<LayerUi>, val loss: Loss, val net: Network)

    fun load(context: Context, name: String, inputShape: Shape, numClasses: Int): Loaded {
        DataInputStream(BufferedInputStream(FileInputStream(fileFor(context, name)))).use { i ->
            require(i.readInt() == MAGIC) { "Неверный формат файла модели" }
            i.readLong() // savedAt
            i.readInt()  // paramCount
            i.readUTF()  // summary
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
