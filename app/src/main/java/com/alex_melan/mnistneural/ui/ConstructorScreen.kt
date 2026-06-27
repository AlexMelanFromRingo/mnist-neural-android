package com.alex_melan.mnistneural.ui

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowDownward
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.alex_melan.mnistneural.engine.Loss

@Composable
fun ConstructorScreen(vm: MainViewModel) {
    var editingIndex by remember { mutableStateOf<Int?>(null) }

    val idx = editingIndex
    if (idx != null && idx in vm.hidden.indices && vm.hidden[idx].kind == LayerKind.DENSE) {
        val layer = vm.hidden[idx]
        ConnectionEditorScreen(
            nIn = vm.flatInputSizeAt(idx),
            nOut = layer.units,
            initialMask = layer.mask,
            onClose = { editingIndex = null },
            onSave = { mask ->
                vm.updateLayer(idx, layer.copy(mask = mask))
                editingIndex = null
            },
        )
        return
    }

    val plan = vm.plan()

    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item { Spacer(Modifier.height(4.dp)) }

        item { NetworkDiagram(vm.hidden, vm.numClasses) }

        item {
            Column {
                Text("Пресеты", style = MaterialTheme.typography.titleSmall,
                    modifier = Modifier.padding(bottom = 4.dp))
                Row(
                    Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Preset.entries.forEach { p ->
                        FilledTonalButton(
                            onClick = { vm.applyPreset(p) },
                            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                        ) {
                            Text(p.title, maxLines = 1, softWrap = false)
                        }
                    }
                }
            }
        }

        item {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text("Архитектура", style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.height(6.dp))
                    plan.layerLines.forEach {
                        Text(it, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
                    }
                    Spacer(Modifier.height(8.dp))
                    if (plan.valid) {
                        Text("Всего параметров: ${plan.totalParams}",
                            style = MaterialTheme.typography.titleSmall,
                            color = MaterialTheme.colorScheme.primary)
                    } else {
                        Text(plan.error ?: "ошибка", color = MaterialTheme.colorScheme.error)
                    }
                }
            }
        }

        itemsIndexed(vm.hidden, key = { i, _ -> i }) { i, layer ->
            LayerCard(
                index = i,
                count = vm.hidden.size,
                layer = layer,
                inputSize = vm.flatInputSizeAt(i),
                onChange = { vm.updateLayer(i, it) },
                onDelete = { vm.removeLayer(i) },
                onMoveUp = { if (i > 0) vm.moveLayer(i, i - 1) },
                onMoveDown = { if (i < vm.hidden.size - 1) vm.moveLayer(i, i + 1) },
                onEditConnections = { editingIndex = i },
            )
        }

        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                AddLayerButton("Dense", Modifier.weight(1f)) { vm.addLayer(LayerKind.DENSE) }
                AddLayerButton("Conv", Modifier.weight(1f)) { vm.addLayer(LayerKind.CONV) }
                AddLayerButton("Pool", Modifier.weight(1f)) { vm.addLayer(LayerKind.POOL) }
            }
        }

        item {
            Button(
                onClick = { vm.buildNetwork() },
                enabled = plan.valid,
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Собрать сеть") }
        }

        item {
            vm.networkSummary?.let {
                Card(
                    Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
                ) {
                    Column(Modifier.padding(16.dp)) {
                        Text("Сеть собрана ✓", style = MaterialTheme.typography.titleSmall)
                        Spacer(Modifier.height(4.dp))
                        Text(it, style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
            vm.buildError?.let {
                Text(it, color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(top = 8.dp))
            }
        }

        item { Spacer(Modifier.height(24.dp)) }
    }
}

/** Compact "+ Dense/Conv/Pool" button that fits three-up without wrapping its label. */
@Composable
private fun AddLayerButton(label: String, modifier: Modifier = Modifier, onClick: () -> Unit) {
    OutlinedButton(
        onClick = onClick,
        modifier = modifier,
        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 8.dp),
    ) {
        Icon(Icons.Filled.Add, null, modifier = Modifier.size(18.dp))
        Spacer(Modifier.width(4.dp))
        Text(label, maxLines = 1, softWrap = false)
    }
}

@Composable
private fun LayerCard(
    index: Int,
    count: Int,
    layer: LayerUi,
    inputSize: Int,
    onChange: (LayerUi) -> Unit,
    onDelete: () -> Unit,
    onMoveUp: () -> Unit,
    onMoveDown: () -> Unit,
    onEditConnections: () -> Unit,
) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("${index + 1}. ${layer.title}",
                    style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
                IconButton(onClick = onMoveUp, enabled = index > 0) {
                    Icon(Icons.Filled.ArrowUpward, "вверх")
                }
                IconButton(onClick = onMoveDown, enabled = index < count - 1) {
                    Icon(Icons.Filled.ArrowDownward, "вниз")
                }
                IconButton(onClick = onDelete) {
                    Icon(Icons.Filled.Delete, "удалить", tint = MaterialTheme.colorScheme.error)
                }
            }

            when (layer.kind) {
                LayerKind.DENSE -> {
                    IntSliderField("Нейроны", layer.units, 1, 512) { onChange(layer.copy(units = it)) }
                    Spacer(Modifier.height(4.dp))
                    Text("Активация", style = MaterialTheme.typography.bodyMedium)
                    ActivationPicker(layer.activation) { onChange(layer.copy(activation = it)) }
                    Spacer(Modifier.height(8.dp))
                    val edges = inputSize.toLong() * layer.units
                    val maskInfo = layer.mask?.let {
                        val on = it.count { b -> b }
                        " • активно ${on}/${it.size}"
                    } ?: ""
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("Связи: $inputSize×${layer.units} = $edges$maskInfo",
                            style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                        OutlinedButton(onClick = onEditConnections) { Text("Связи") }
                    }
                }

                LayerKind.CONV -> {
                    IntSliderField("Фильтры", layer.filters, 1, 32) { onChange(layer.copy(filters = it)) }
                    IntSliderField("Ядро", layer.kernel, 1, 7) { onChange(layer.copy(kernel = it)) }
                    IntSliderField("Шаг (stride)", layer.stride, 1, 3) { onChange(layer.copy(stride = it)) }
                    IntSliderField("Отступ (pad)", layer.pad, 0, 3) { onChange(layer.copy(pad = it)) }
                    Spacer(Modifier.height(4.dp))
                    Text("Активация", style = MaterialTheme.typography.bodyMedium)
                    ActivationPicker(layer.activation) { onChange(layer.copy(activation = it)) }
                }

                LayerKind.POOL -> {
                    IntSliderField("Ядро пулинга", layer.poolKernel, 2, 4) { onChange(layer.copy(poolKernel = it)) }
                    IntSliderField("Шаг пулинга", layer.poolStride, 1, 4) { onChange(layer.copy(poolStride = it)) }
                }
            }
        }
    }
}
