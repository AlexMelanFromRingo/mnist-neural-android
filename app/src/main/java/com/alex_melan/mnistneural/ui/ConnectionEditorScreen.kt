package com.alex_melan.mnistneural.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import kotlin.math.sqrt
import kotlin.random.Random

private const val MAX_INTERACTIVE = 4096

/**
 * Per-neuron connection editor with two modes:
 *  - **Матрица** (only for layers ≤ MAX_INTERACTIVE edges): the full nOut×nIn grid, tap any cell.
 *  - **По нейрону** (works at any scale): pick one output neuron and toggle its incoming edges,
 *    laid out as a square map (28×28 for the 784-input first layer) — meaningful even for big layers.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConnectionEditorScreen(
    nIn: Int,
    nOut: Int,
    initialMask: BooleanArray?,
    onClose: () -> Unit,
    onSave: (BooleanArray?) -> Unit,
) {
    val total = nIn.toLong() * nOut
    val mask = remember(nIn, nOut) {
        BooleanArray(nIn * nOut) { initialMask?.getOrNull(it) ?: true }
    }
    var version by remember { mutableIntStateOf(0) }
    val matrixAvailable = total <= MAX_INTERACTIVE
    var focusMode by remember { mutableStateOf(!matrixAvailable) }
    var neuron by remember { mutableIntStateOf(0) }

    val activeColor = MaterialTheme.colorScheme.primary
    val inactiveColor = MaterialTheme.colorScheme.surfaceVariant

    fun activeCount(): Int {
        var c = 0
        for (b in mask) if (b) c++
        return c
    }

    Surface(Modifier.fillMaxSize()) {
        Column(Modifier.fillMaxSize().padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = onClose) { Icon(Icons.Filled.Close, "закрыть") }
                Text("Связи нейронов", style = MaterialTheme.typography.titleLarge,
                    modifier = Modifier.weight(1f))
            }
            version.let { } // observe so the counters below recompose on toggle
            Text("$nIn входов × $nOut нейронов = $total связей",
                style = MaterialTheme.typography.bodyMedium)
            Text("Активно: ${activeCount()} из ${mask.size}",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)

            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (matrixAvailable) {
                    FilterChip(selected = !focusMode, onClick = { focusMode = false },
                        label = { Text("Матрица") })
                }
                FilterChip(selected = focusMode, onClick = { focusMode = true },
                    label = { Text("По нейрону") })
            }
            Spacer(Modifier.height(8.dp))

            // Scrollable middle so the canvas + per-layer controls never push the action bar off-screen.
            Column(
                Modifier
                    .weight(1f)
                    .verticalScroll(rememberScrollState()),
            ) {
                if (focusMode) {
                    NeuronFocus(nIn, nOut, neuron, mask, version, activeColor, inactiveColor,
                        onPickNeuron = { neuron = it },
                        onToggle = { i -> mask[neuron * nIn + i] = !mask[neuron * nIn + i]; version++ })
                    Row(Modifier.fillMaxWidth().padding(top = 8.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = {
                            for (i in 0 until nIn) mask[neuron * nIn + i] = true; version++
                        }, modifier = Modifier.weight(1f)) { Text("Все (нейрон)") }
                        OutlinedButton(onClick = {
                            for (i in 0 until nIn) mask[neuron * nIn + i] = false; version++
                        }, modifier = Modifier.weight(1f)) { Text("Сброс (нейрон)") }
                    }
                } else {
                    MatrixGrid(nIn, nOut, mask, version, activeColor, inactiveColor) { row, col ->
                        val k = row * nIn + col
                        mask[k] = !mask[k]
                        version++
                    }
                }

                Spacer(Modifier.height(12.dp))
                Text("Ко всему слою:", style = MaterialTheme.typography.titleSmall)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = { mask.fill(true); version++ }, modifier = Modifier.weight(1f)) {
                        Text("Все")
                    }
                    OutlinedButton(onClick = { mask.fill(false); version++ }, modifier = Modifier.weight(1f)) {
                        Text("Сброс")
                    }
                    OutlinedButton(onClick = {
                        val r = Random(42)
                        for (i in mask.indices) mask[i] = r.nextFloat() > 0.5f
                        version++
                    }, modifier = Modifier.weight(1f)) { Text("−50%") }
                }
                Spacer(Modifier.height(12.dp))
            }

            // Pinned action bar — always visible regardless of screen height.
            Row(Modifier.fillMaxWidth().padding(top = 4.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = onClose, modifier = Modifier.weight(1f)) { Text("Отмена") }
                Button(onClick = {
                    onSave(if (mask.all { it }) null else mask.copyOf())
                }, modifier = Modifier.weight(1f)) { Text("Сохранить") }
            }
        }
    }
}

@Composable
private fun MatrixGrid(
    nIn: Int, nOut: Int, mask: BooleanArray, version: Int,
    activeColor: androidx.compose.ui.graphics.Color,
    inactiveColor: androidx.compose.ui.graphics.Color,
    onToggle: (row: Int, col: Int) -> Unit,
) {
    var sizePx by remember { mutableStateOf(IntSize.Zero) }
    Text("Строка — нейрон, столбец — вход. Нажимайте на клетки.",
        style = MaterialTheme.typography.bodySmall)
    Canvas(
        Modifier.fillMaxWidth().height(320.dp).padding(vertical = 8.dp)
            .onSizeChanged { sizePx = it }
            .pointerInput(nIn, nOut) {
                detectTapGestures { off ->
                    if (sizePx.width > 0 && sizePx.height > 0) {
                        val col = (off.x / (sizePx.width.toFloat() / nIn)).toInt().coerceIn(0, nIn - 1)
                        val row = (off.y / (sizePx.height.toFloat() / nOut)).toInt().coerceIn(0, nOut - 1)
                        onToggle(row, col)
                    }
                }
            },
    ) {
        version
        val cw = size.width / nIn
        val ch = size.height / nOut
        val gap = if (cw > 4f && ch > 4f) 1f else 0f
        for (row in 0 until nOut) for (col in 0 until nIn) {
            drawRect(
                color = if (mask[row * nIn + col]) activeColor else inactiveColor,
                topLeft = Offset(col * cw, row * ch),
                size = Size(cw - gap, ch - gap),
            )
        }
    }
}

@Composable
private fun NeuronFocus(
    nIn: Int, nOut: Int, neuron: Int, mask: BooleanArray, version: Int,
    activeColor: androidx.compose.ui.graphics.Color,
    inactiveColor: androidx.compose.ui.graphics.Color,
    onPickNeuron: (Int) -> Unit,
    onToggle: (Int) -> Unit,
) {
    val side = Math.round(sqrt(nIn.toDouble())).toInt()
    val cols = if (side * side == nIn) side else minOf(nIn, 28)
    val rows = (nIn + cols - 1) / cols
    var sizePx by remember { mutableStateOf(IntSize.Zero) }

    IntSliderField("Нейрон", neuron, 0, (nOut - 1).coerceAtLeast(0)) { onPickNeuron(it) }
    Text("Входные связи нейрона $neuron (${cols}×$rows):", style = MaterialTheme.typography.bodySmall)
    Canvas(
        Modifier.fillMaxWidth().height(300.dp).padding(vertical = 8.dp)
            .onSizeChanged { sizePx = it }
            .pointerInput(nIn, nOut, neuron) {
                detectTapGestures { off ->
                    if (sizePx.width > 0 && sizePx.height > 0) {
                        val cell = minOf(sizePx.width.toFloat() / cols, sizePx.height.toFloat() / rows)
                        val col = (off.x / cell).toInt()
                        val row = (off.y / cell).toInt()
                        val i = row * cols + col
                        if (col in 0 until cols && row in 0 until rows && i < nIn) onToggle(i)
                    }
                }
            },
    ) {
        version
        neuron
        val cell = minOf(size.width / cols, size.height / rows)
        val gap = if (cell > 4f) 1f else 0f
        val base = neuron * nIn
        for (i in 0 until nIn) {
            val r = i / cols
            val c = i % cols
            drawRect(
                color = if (mask[base + i]) activeColor else inactiveColor,
                topLeft = Offset(c * cell, r * cell),
                size = Size(cell - gap, cell - gap),
            )
        }
    }
}
