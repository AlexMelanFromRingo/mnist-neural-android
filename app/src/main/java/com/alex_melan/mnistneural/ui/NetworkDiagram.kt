package com.alex_melan.mnistneural.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import kotlin.math.min

/**
 * Schematic of the network: input → hidden layers → output, each drawn as a column of nodes with
 * connecting edges. Node counts are capped for legibility (a "+N" note shows the real size).
 */
@Composable
fun NetworkDiagram(hidden: List<LayerUi>, numClasses: Int) {
    val primary = MaterialTheme.colorScheme.primary
    val secondary = MaterialTheme.colorScheme.tertiary
    val edge = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.12f)
    val inputColor = MaterialTheme.colorScheme.secondary

    data class Col(val label: String, val real: Int, val color: Color)

    val cols = buildList {
        add(Col("28×28", 784, inputColor))
        hidden.forEach { l ->
            when (l.kind) {
                LayerKind.DENSE -> add(Col("D${l.units}", l.units, primary))
                LayerKind.CONV -> add(Col("C${l.filters}", l.filters, secondary))
                LayerKind.POOL -> add(Col("P", 4, secondary))
            }
        }
        add(Col("out $numClasses", numClasses, primary))
    }

    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp)) {
            Text("Схема сети", style = MaterialTheme.typography.titleSmall)
            Canvas(
                Modifier
                    .fillMaxWidth()
                    .height(170.dp)
                    .padding(top = 8.dp)
            ) {
                val cap = 8
                val n = cols.size
                val sidePad = 28f
                val topPad = 14f
                val botPad = 14f
                val usableW = size.width - 2 * sidePad
                val usableH = size.height - topPad - botPad
                val radius = 6f

                fun xOf(i: Int) = if (n == 1) size.width / 2 else sidePad + usableW * i / (n - 1)
                fun nodesOf(col: Col) = min(col.real, cap)
                fun yOf(k: Int, shown: Int): Float {
                    if (shown == 1) return topPad + usableH / 2
                    return topPad + usableH * k / (shown - 1)
                }

                // edges
                for (i in 0 until n - 1) {
                    val a = nodesOf(cols[i]); val b = nodesOf(cols[i + 1])
                    val xa = xOf(i); val xb = xOf(i + 1)
                    for (ka in 0 until a) for (kb in 0 until b) {
                        drawLine(
                            color = edge,
                            start = Offset(xa, yOf(ka, a)),
                            end = Offset(xb, yOf(kb, b)),
                            strokeWidth = 1.5f,
                        )
                    }
                }
                // nodes
                for (i in 0 until n) {
                    val shown = nodesOf(cols[i])
                    val x = xOf(i)
                    for (k in 0 until shown) {
                        drawCircle(cols[i].color, radius, Offset(x, yOf(k, shown)))
                    }
                }
            }
            // labels row
            Text(
                cols.joinToString("  →  ") { c ->
                    if (c.real > 8) c.label + "*" else c.label
                },
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
