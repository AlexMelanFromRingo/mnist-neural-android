package com.alex_melan.mnistneural.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt

private const val GRID = 28

@Composable
fun DrawScreen(vm: MainViewModel) {
    val grid = remember { FloatArray(GRID * GRID) }
    var version by remember { mutableIntStateOf(0) }
    var canvasSize by remember { mutableStateOf(IntSize.Zero) }
    val prediction by vm.prediction.collectAsState()

    fun paint(off: Offset) {
        if (canvasSize.width == 0 || canvasSize.height == 0) return
        val cw = canvasSize.width.toFloat() / GRID
        val ch = canvasSize.height.toFloat() / GRID
        val gx = off.x / cw
        val gy = off.y / ch
        val r = 1.5f
        val c0 = (gx - r).toInt(); val c1 = (gx + r).toInt()
        val r0 = (gy - r).toInt(); val r1 = (gy + r).toInt()
        for (row in r0..r1) for (col in c0..c1) {
            if (row in 0 until GRID && col in 0 until GRID) {
                val d = hypot(col + 0.5f - gx, row + 0.5f - gy)
                val v = (1f - d / r).coerceIn(0f, 1f)
                val idx = row * GRID + col
                grid[idx] = max(grid[idx], v)
            }
        }
        version++
    }

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Нарисуйте цифру (поле 28×28)", style = MaterialTheme.typography.titleMedium)

        Box(
            Modifier
                .fillMaxWidth()
                .aspectRatio(1f)
                .onSizeChanged { canvasSize = it }
                .pointerInput(Unit) {
                    detectTapGestures { paint(it) }
                }
                .pointerInput(Unit) {
                    detectDragGestures(onDragStart = { paint(it) }) { change, _ ->
                        paint(change.position)
                        change.consume()
                    }
                },
        ) {
            Canvas(Modifier.fillMaxSize()) {
                version // observe for redraw
                val cw = size.width / GRID
                val ch = size.height / GRID
                drawRect(Color.Black, size = size)
                for (row in 0 until GRID) {
                    for (col in 0 until GRID) {
                        val v = grid[row * GRID + col]
                        if (v > 0f) {
                            drawRect(
                                color = Color(v, v, v, 1f),
                                topLeft = Offset(col * cw, row * ch),
                                size = Size(cw, ch),
                            )
                        }
                    }
                }
                // grid lines so the 28×28 squares are visible
                val line = Color.White.copy(alpha = 0.08f)
                for (i in 0..GRID) {
                    drawLine(line, Offset(i * cw, 0f), Offset(i * cw, size.height), 1f)
                    drawLine(line, Offset(0f, i * ch), Offset(size.width, i * ch), 1f)
                }
            }
        }

        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(
                onClick = { grid.fill(0f); version++; vm.clearPrediction() },
                modifier = Modifier.weight(1f),
            ) { Text("Очистить") }
            Button(
                onClick = { vm.predict(centerByMass(grid)) },
                enabled = vm.isTrained,
                modifier = Modifier.weight(1f),
            ) { Text("Распознать") }
        }

        if (!vm.isTrained) {
            Text("Сначала соберите и обучите сеть на вкладке «Обучение».",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                style = MaterialTheme.typography.bodySmall)
        }

        prediction?.let { ResultCard(it) }
    }
}

@Composable
private fun ResultCard(p: Prediction) {
    Card(Modifier.fillMaxWidth()) {
        Row(Modifier.padding(16.dp).fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.width(96.dp)) {
                Text("${p.digit}", fontSize = 64.sp, fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary)
                Text("уверенность ${"%.0f".format(p.probs[p.digit] * 100)}%",
                    style = MaterialTheme.typography.labelSmall)
            }
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                for (d in 0 until 10) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("$d", modifier = Modifier.width(16.dp),
                            style = MaterialTheme.typography.bodySmall)
                        LinearProgressIndicator(
                            progress = { p.probs[d].coerceIn(0f, 1f) },
                            modifier = Modifier.weight(1f).height(8.dp).padding(horizontal = 6.dp),
                        )
                        Text("%.0f%%".format(p.probs[d] * 100),
                            modifier = Modifier.width(40.dp),
                            style = MaterialTheme.typography.labelSmall)
                    }
                }
            }
        }
    }
}

/**
 * Shifts the drawing so its ink center-of-mass sits at the middle of the 28×28 grid, mimicking the
 * centering MNIST applies. This makes recognition robust to where on the canvas the user draws.
 */
private fun centerByMass(src: FloatArray): FloatArray {
    var sum = 0f
    var sx = 0f
    var sy = 0f
    for (r in 0 until GRID) for (c in 0 until GRID) {
        val v = src[r * GRID + c]
        if (v > 0f) {
            sum += v; sx += c * v; sy += r * v
        }
    }
    if (sum <= 0f) return src.copyOf()
    val dx = (GRID / 2f - 0.5f - sx / sum).roundToInt()
    val dy = (GRID / 2f - 0.5f - sy / sum).roundToInt()
    if (dx == 0 && dy == 0) return src.copyOf()
    val out = FloatArray(GRID * GRID)
    for (r in 0 until GRID) for (c in 0 until GRID) {
        val nr = r + dy
        val nc = c + dx
        if (nr in 0 until GRID && nc in 0 until GRID) out[nr * GRID + nc] = src[r * GRID + c]
    }
    return out
}
