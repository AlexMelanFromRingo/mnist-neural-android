package com.alex_melan.mnistneural.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.alex_melan.mnistneural.engine.Activation
import kotlin.math.roundToInt

/** A labelled slider paired with an editable numeric field (the user asked for both). */
@Composable
fun IntSliderField(
    label: String,
    value: Int,
    min: Int,
    max: Int,
    modifier: Modifier = Modifier,
    onChange: (Int) -> Unit,
) {
    Column(modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(label, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
            NumberField(value.toString(), onText = {
                it.toIntOrNull()?.let { v -> onChange(v.coerceIn(min, max)) }
            })
        }
        Slider(
            value = value.toFloat(),
            onValueChange = { onChange(it.roundToInt().coerceIn(min, max)) },
            valueRange = min.toFloat()..max.toFloat(),
        )
    }
}

/** A labelled slider paired with an editable field for floating-point hyperparameters. */
@Composable
fun FloatSliderField(
    label: String,
    value: Float,
    min: Float,
    max: Float,
    modifier: Modifier = Modifier,
    format: (Float) -> String = { String.format("%.3f", it) },
    onChange: (Float) -> Unit,
) {
    Column(modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(label, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
            NumberField(format(value), onText = {
                it.replace(',', '.').toFloatOrNull()?.let { v -> onChange(v.coerceIn(min, max)) }
            })
        }
        Slider(
            value = value.coerceIn(min, max),
            onValueChange = { onChange(it.coerceIn(min, max)) },
            valueRange = min..max,
        )
    }
}

@Composable
private fun NumberField(text: String, onText: (String) -> Unit) {
    var local by remember { mutableStateOf(text) }
    LaunchedEffect(text) { if (text != local) local = text }
    OutlinedTextField(
        value = local,
        onValueChange = {
            local = it
            onText(it)
        },
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        modifier = Modifier.width(96.dp),
    )
}

/** Activation picker rendered as a row of filter chips. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ActivationPicker(
    value: Activation,
    options: List<Activation> = listOf(Activation.RELU, Activation.TANH, Activation.SIGMOID),
    onChange: (Activation) -> Unit,
) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        options.forEach { act ->
            FilterChip(
                selected = value == act,
                onClick = { onChange(act) },
                label = { Text(act.name.lowercase()) },
            )
        }
    }
}
