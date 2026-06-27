package com.alex_melan.mnistneural.ui

import android.content.Intent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.alex_melan.mnistneural.engine.Trainer
import kotlin.math.max

@Composable
fun TrainingScreen(vm: MainViewModel) {
    val context = LocalContext.current
    val datasetStatus by vm.datasetStatus.collectAsState()
    val isTraining by vm.isTraining.collectAsState()
    val progress by vm.progress.collectAsState()
    val ioStatus by vm.ioStatus.collectAsState()
    val evalResult by vm.evalResult.collectAsState()
    val isEvaluating by vm.isEvaluating.collectAsState()
    val savedModels by vm.savedModels.collectAsState()

    var showSaveDialog by remember { mutableStateOf(false) }
    var pendingDelete by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) { vm.refreshModels(context) }

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        DatasetStatusCard(datasetStatus) { vm.loadDataset(context) }

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp)) {
                Text("Параметры обучения", style = MaterialTheme.typography.titleMedium)
                IntSliderField("Эпохи", vm.epochs, 1, 50) { vm.epochs = it }
                IntSliderField("Размер батча", vm.batchSize, 1, 256) { vm.batchSize = it }
                FloatSliderField("Скорость (lr)", vm.learningRate, 0.001f, 0.5f,
                    onChange = { vm.learningRate = it }, format = { String.format("%.3f", it) })
                FloatSliderField("Импульс (momentum)", vm.momentum, 0f, 0.99f,
                    onChange = { vm.momentum = it }, format = { String.format("%.2f", it) })
                FloatSliderField("L2-регуляризация", vm.l2, 0f, 0.01f,
                    onChange = { vm.l2 = it }, format = { String.format("%.4f", it) })
                IntSliderField("Образцов для обучения", vm.trainSamples, 500, 60000) { vm.trainSamples = it }
                IntSliderField("Образцов для валидации", vm.valSamples, 0, 10000) { vm.valSamples = it }
            }
        }

        if (isTraining) {
            OutlinedButton(onClick = { vm.cancelTraining() }, modifier = Modifier.fillMaxWidth()) {
                Text("Остановить")
            }
        } else {
            Button(
                onClick = { vm.startTraining(context) },
                enabled = datasetStatus is DatasetStatus.Loaded,
                modifier = Modifier.fillMaxWidth(),
            ) { Text(if (vm.isTrained) "Дообучить" else "Обучать") }
        }

        vm.buildError?.let {
            Text(it, color = MaterialTheme.colorScheme.error)
        }

        progress?.let { ProgressCard(it, isTraining) }

        OutlinedButton(
            onClick = { showSaveDialog = true },
            enabled = vm.isTrained && !isTraining,
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Сохранить модель") }

        ioStatus?.let {
            Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
        }

        SavedModelsCard(
            models = savedModels,
            enabled = !isTraining,
            onLoad = { vm.loadModel(context, it) },
            onDelete = { pendingDelete = it },
        )

        Button(
            onClick = { vm.evaluateTest() },
            enabled = vm.isTrained && datasetStatus is DatasetStatus.Loaded && !isEvaluating && !isTraining,
            modifier = Modifier.fillMaxWidth(),
        ) { Text(if (isEvaluating) "Оценка…" else "Оценить на тесте") }

        evalResult?.let { r -> EvalCard(r) { shareReport(context, r) } }

        Spacer(Modifier.height(24.dp))
    }

    if (showSaveDialog) {
        var name by remember { mutableStateOf(vm.suggestedModelName()) }
        val willOverwrite = savedModels.any { it.name == ModelStore.sanitize(name) }
        AlertDialog(
            onDismissRequest = { showSaveDialog = false },
            title = { Text("Сохранить модель") },
            text = {
                Column {
                    OutlinedTextField(
                        value = name,
                        onValueChange = { name = it },
                        label = { Text("Имя модели") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth(),
                    )
                    if (willOverwrite) {
                        Spacer(Modifier.height(8.dp))
                        Text("Модель с таким именем будет перезаписана",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error)
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    vm.saveModel(context, name)
                    showSaveDialog = false
                }) { Text(if (willOverwrite) "Перезаписать" else "Сохранить") }
            },
            dismissButton = {
                TextButton(onClick = { showSaveDialog = false }) { Text("Отмена") }
            },
        )
    }

    pendingDelete?.let { name ->
        AlertDialog(
            onDismissRequest = { pendingDelete = null },
            title = { Text("Удалить модель?") },
            text = { Text("«$name» будет удалена без возможности восстановления.") },
            confirmButton = {
                TextButton(onClick = {
                    vm.deleteModel(context, name)
                    pendingDelete = null
                }) { Text("Удалить") }
            },
            dismissButton = {
                TextButton(onClick = { pendingDelete = null }) { Text("Отмена") }
            },
        )
    }
}

@Composable
private fun SavedModelsCard(
    models: List<ModelStore.Info>,
    enabled: Boolean,
    onLoad: (String) -> Unit,
    onDelete: (String) -> Unit,
) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text("Сохранённые модели", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(4.dp))
            if (models.isEmpty()) {
                Text("Пока нет сохранённых моделей. Обучите сеть и нажмите «Сохранить модель».",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            } else {
                models.forEachIndexed { i, m ->
                    if (i > 0) Spacer(Modifier.height(4.dp))
                    SavedModelRow(m, enabled, onLoad, onDelete)
                }
            }
        }
    }
}

@Composable
private fun SavedModelRow(
    m: ModelStore.Info,
    enabled: Boolean,
    onLoad: (String) -> Unit,
    onDelete: (String) -> Unit,
) {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
    ) {
        Column(Modifier.padding(12.dp)) {
            Text(m.name, style = MaterialTheme.typography.titleSmall)
            Text(m.summary, style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text("${formatParams(m.paramCount)} параметров • ${formatSize(m.sizeBytes)} • ${formatDate(m.savedAt)}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(8.dp))
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(
                    onClick = { onLoad(m.name) },
                    enabled = enabled,
                    modifier = Modifier.weight(1f),
                ) { Text("Загрузить") }
                IconButton(onClick = { onDelete(m.name) }, enabled = enabled) {
                    Icon(Icons.Filled.Delete, "удалить", tint = MaterialTheme.colorScheme.error)
                }
            }
        }
    }
}

private fun formatParams(n: Int): String = when {
    n >= 1_000_000 -> "%.1fM".format(n / 1_000_000.0)
    n >= 1_000 -> "%.1fK".format(n / 1_000.0)
    else -> n.toString()
}

private fun formatSize(bytes: Long): String = when {
    bytes >= 1_000_000 -> "%.1f МБ".format(bytes / 1_000_000.0)
    else -> "%d КБ".format((bytes / 1000).coerceAtLeast(1))
}

private fun formatDate(millis: Long): String =
    java.text.SimpleDateFormat("dd.MM.yyyy HH:mm", java.util.Locale.getDefault())
        .format(java.util.Date(millis))

@Composable
private fun EvalCard(r: EvalResult, onExport: () -> Unit) {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.tertiaryContainer),
    ) {
        Column(Modifier.padding(16.dp)) {
            Text("Оценка на тесте (${r.count} образцов)", style = MaterialTheme.typography.titleMedium)
            Text("Точность: %.2f%%".format(r.accuracy * 100),
                style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.height(8.dp))
            Text("По классам:", style = MaterialTheme.typography.bodyMedium)
            for (d in 0 until 10) {
                val tot = r.perClassTotal[d]
                val acc = if (tot > 0) r.perClassCorrect[d].toDouble() / tot else 0.0
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("$d", modifier = Modifier.width(16.dp), style = MaterialTheme.typography.bodySmall)
                    LinearProgressIndicator(
                        progress = { acc.toFloat() },
                        modifier = Modifier.weight(1f).height(8.dp).padding(horizontal = 6.dp),
                    )
                    Text("%.0f%%".format(acc * 100), modifier = Modifier.width(40.dp),
                        style = MaterialTheme.typography.labelSmall)
                }
            }
            Spacer(Modifier.height(12.dp))
            Text("Матрица ошибок (строка — истина, столбец — ответ; диагональ ярче = верно):",
                style = MaterialTheme.typography.bodySmall)
            ConfusionHeatmap(r.confusion)
            Spacer(Modifier.height(8.dp))
            OutlinedButton(onClick = onExport, modifier = Modifier.fillMaxWidth()) {
                Text("Экспортировать отчёт")
            }
        }
    }
}

@Composable
private fun ConfusionHeatmap(confusion: Array<IntArray>) {
    val base = MaterialTheme.colorScheme.primary
    Box(Modifier.fillMaxWidth().height(150.dp).padding(top = 6.dp)) {
        Canvas(Modifier.fillMaxSize()) {
            val n = 10
            val cell = minOf(size.width, size.height) / n
            for (r in 0 until n) {
                var mx = 1
                for (c in 0 until n) if (confusion[r][c] > mx) mx = confusion[r][c]
                for (c in 0 until n) {
                    val v = confusion[r][c].toFloat() / mx
                    drawRect(
                        color = base.copy(alpha = 0.10f + 0.90f * v),
                        topLeft = Offset(c * cell, r * cell),
                        size = Size(cell - 1f, cell - 1f),
                    )
                }
            }
        }
    }
}

private fun shareReport(context: android.content.Context, r: EvalResult) {
    val sb = StringBuilder()
    sb.append("MNIST Neural — отчёт об оценке\n")
    sb.append("Образцов: ${r.count}\n")
    sb.append("Общая точность: %.2f%%\n\n".format(r.accuracy * 100))
    sb.append("class,correct,total,accuracy%\n")
    for (d in 0 until 10) {
        val tot = r.perClassTotal[d]
        val cor = r.perClassCorrect[d]
        val acc = if (tot > 0) cor * 100.0 / tot else 0.0
        sb.append("$d,$cor,$tot,%.2f\n".format(acc))
    }
    sb.append("\nconfusion (row=true, col=pred)\n")
    sb.append("," + (0..9).joinToString(",") + "\n")
    for (t in 0 until 10) sb.append("$t," + r.confusion[t].joinToString(",") + "\n")
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = "text/plain"
        putExtra(Intent.EXTRA_SUBJECT, "MNIST Neural — отчёт об оценке")
        putExtra(Intent.EXTRA_TEXT, sb.toString())
    }
    context.startActivity(Intent.createChooser(intent, "Экспорт отчёта"))
}

@Composable
private fun DatasetStatusCard(status: DatasetStatus, onRetry: () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Row(
            Modifier.padding(16.dp).fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            when (status) {
                is DatasetStatus.NotLoaded -> Text("Датасет не загружен")
                is DatasetStatus.Loading -> {
                    CircularProgressIndicator(Modifier.height(20.dp))
                    Text("Загрузка MNIST…")
                }
                is DatasetStatus.Loaded -> Text("MNIST загружен: ${status.count} образцов ✓")
                is DatasetStatus.Error -> {
                    Text("Ошибка: ${status.message}", color = MaterialTheme.colorScheme.error,
                        modifier = Modifier.weight(1f))
                    OutlinedButton(onClick = onRetry) { Text("Повтор") }
                }
            }
        }
    }
}

@Composable
private fun ProgressCard(p: Trainer.Progress, isTraining: Boolean) {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
    ) {
        Column(Modifier.padding(16.dp)) {
            val title = when {
                p.cancelled -> "Остановлено"
                p.finished && !isTraining -> "Готово ✓"
                else -> "Обучение…"
            }
            Text(title, style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))

            val epochFrac = if (p.stepsPerEpoch > 0) p.step.toFloat() / p.stepsPerEpoch else 0f
            val overall = if (p.totalEpochs > 0)
                ((p.epoch - 1) + epochFrac) / p.totalEpochs else 0f
            LinearProgressIndicator(
                progress = { overall.coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(8.dp))
            Text("Эпоха ${p.epoch}/${p.totalEpochs} • шаг ${p.step}/${max(1, p.stepsPerEpoch)}")
            Text("Train: loss %.4f • acc %.2f%%".format(p.trainLoss, p.trainAcc * 100))
            if (p.valAcc >= 0) {
                Text("Val: loss %.4f • acc %.2f%%".format(p.valLoss, p.valAcc * 100),
                    color = MaterialTheme.colorScheme.primary,
                    style = MaterialTheme.typography.titleSmall)
            }
        }
    }
}
