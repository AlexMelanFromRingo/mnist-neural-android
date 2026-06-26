package com.alex_melan.mnistneural.ui

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.alex_melan.mnistneural.data.MnistDataset
import com.alex_melan.mnistneural.engine.Loss
import com.alex_melan.mnistneural.engine.Network
import com.alex_melan.mnistneural.engine.NetworkBuilder
import com.alex_melan.mnistneural.engine.Shape
import com.alex_melan.mnistneural.engine.TrainConfig
import com.alex_melan.mnistneural.engine.Trainer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Single source of truth for the whole app: the editable architecture, training hyperparameters,
 * the built [Network], the [MnistDataset], and async training/prediction state.
 *
 * Architecture + hyperparameters are Compose [mutableStateOf] (edited only on the main thread).
 * Anything written from a background coroutine (dataset load, training progress, prediction) is a
 * [StateFlow] so updates are thread-safe.
 */
class MainViewModel : ViewModel() {

    val inputShape: Shape = Shape(1, 28, 28)
    val numClasses: Int = 10

    // ---- Architecture ----
    val hidden = mutableStateListOf(
        LayerUi(LayerKind.DENSE, units = 64, activation = com.alex_melan.mnistneural.engine.Activation.RELU)
    )
    var loss by mutableStateOf(Loss.SOFTMAX_CROSS_ENTROPY)

    // ---- Hyperparameters ----
    var epochs by mutableStateOf(5)
    var batchSize by mutableStateOf(32)
    var learningRate by mutableStateOf(0.05f)
    var momentum by mutableStateOf(0.9f)
    var l2 by mutableStateOf(0.0f)
    var trainSamples by mutableStateOf(8000)
    var valSamples by mutableStateOf(2000)

    fun plan(): NetworkBuilder.Plan =
        NetworkBuilder.plan(inputShape, hidden.map { it.toSpec() }, numClasses, loss)

    fun flatInputSizeAt(index: Int): Int =
        NetworkBuilder.flatInputSizeAt(inputShape, hidden.map { it.toSpec() }, index)

    fun addLayer(kind: LayerKind) = hidden.add(LayerUi.defaultFor(kind))
    fun removeLayer(index: Int) { if (index in hidden.indices) hidden.removeAt(index) }
    fun updateLayer(index: Int, layer: LayerUi) { if (index in hidden.indices) hidden[index] = layer }
    fun moveLayer(from: Int, to: Int) {
        if (from in hidden.indices && to in hidden.indices) hidden.add(to, hidden.removeAt(from))
    }

    // ---- Dataset ----
    private var dataset: MnistDataset? = null
    private val _datasetStatus = MutableStateFlow<DatasetStatus>(DatasetStatus.NotLoaded)
    val datasetStatus: StateFlow<DatasetStatus> = _datasetStatus.asStateFlow()

    fun loadDataset(context: Context) {
        if (_datasetStatus.value is DatasetStatus.Loaded || _datasetStatus.value is DatasetStatus.Loading) return
        _datasetStatus.value = DatasetStatus.Loading
        val app = context.applicationContext
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val ds = MnistDataset.loadFromAssets(app)
                dataset = ds
                _datasetStatus.value = DatasetStatus.Loaded(ds.totalCount())
            } catch (e: Exception) {
                _datasetStatus.value = DatasetStatus.Error(e.message ?: "ошибка загрузки датасета")
            }
        }
    }

    // ---- Network ----
    var network: Network? = null
        private set
    var networkSummary by mutableStateOf<String?>(null)
        private set
    var buildError by mutableStateOf<String?>(null)
        private set

    /** Rebuilds the engine network from the current architecture. Returns true on success. */
    fun buildNetwork(): Boolean {
        val p = plan()
        if (!p.valid) {
            buildError = p.error
            return false
        }
        return try {
            val net = NetworkBuilder.build(inputShape, hidden.map { it.toSpec() }, numClasses, loss, 42L)
            network = net
            networkSummary = net.layers().joinToString("  →  ") { it.describe() } +
                    "\nПараметров: ${net.paramCount()}"
            buildError = null
            _prediction.value = null
            true
        } catch (e: Exception) {
            buildError = e.message ?: "ошибка сборки сети"
            false
        }
    }

    fun resetNetwork() {
        if (_isTraining.value) return
        network = null
        networkSummary = null
        _progress.value = null
        _prediction.value = null
    }

    // ---- Training ----
    private var trainer: Trainer? = null
    private var trainJob: Job? = null
    private val _progress = MutableStateFlow<Trainer.Progress?>(null)
    val progress: StateFlow<Trainer.Progress?> = _progress.asStateFlow()
    private val _isTraining = MutableStateFlow(false)
    val isTraining: StateFlow<Boolean> = _isTraining.asStateFlow()

    fun startTraining(context: Context) {
        if (_isTraining.value) return
        buildError = null
        if (network == null && !buildNetwork()) return
        val ds = dataset
        if (ds == null) {
            loadDataset(context)
            buildError = "Датасет ещё загружается — нажмите «Обучать» через секунду"
            return
        }
        val cfg = TrainConfig().apply {
            epochs = this@MainViewModel.epochs
            batchSize = this@MainViewModel.batchSize
            learningRate = this@MainViewModel.learningRate
            momentum = this@MainViewModel.momentum
            l2 = this@MainViewModel.l2
            trainSamples = this@MainViewModel.trainSamples
            valSamples = this@MainViewModel.valSamples
        }
        val t = Trainer()
        trainer = t
        _isTraining.value = true
        val net = network!!
        trainJob = viewModelScope.launch(Dispatchers.Default) {
            try {
                t.train(net, ds, cfg) { p -> _progress.value = p }
            } catch (e: Exception) {
                buildError = e.message ?: "ошибка обучения"
            } finally {
                _isTraining.value = false
            }
        }
    }

    fun cancelTraining() {
        trainer?.cancel()
    }

    // ---- Prediction ----
    private val _prediction = MutableStateFlow<Prediction?>(null)
    val prediction: StateFlow<Prediction?> = _prediction.asStateFlow()

    fun predict(grid: FloatArray) {
        val net = network ?: return
        viewModelScope.launch(Dispatchers.Default) {
            val probs = net.predictProbs(grid)
            var best = 0
            for (i in probs.indices) if (probs[i] > probs[best]) best = i
            _prediction.value = Prediction(best, probs)
        }
    }

    fun clearPrediction() {
        _prediction.value = null
    }

    // ---- Presets ----
    fun applyPreset(preset: Preset) {
        if (_isTraining.value) return
        val a = com.alex_melan.mnistneural.engine.Activation.RELU
        val layers = when (preset) {
            Preset.PERCEPTRON -> emptyList()
            Preset.MLP -> listOf(
                LayerUi(LayerKind.DENSE, 128, a),
                LayerUi(LayerKind.DENSE, 64, a),
            )
            Preset.CNN -> listOf(
                LayerUi(LayerKind.CONV, filters = 8, kernel = 3, stride = 1, pad = 1, activation = a),
                LayerUi(LayerKind.POOL, poolKernel = 2, poolStride = 2),
                LayerUi(LayerKind.CONV, filters = 16, kernel = 3, stride = 1, pad = 1, activation = a),
                LayerUi(LayerKind.POOL, poolKernel = 2, poolStride = 2),
                LayerUi(LayerKind.DENSE, 64, a),
            )
        }
        hidden.clear()
        hidden.addAll(layers)
        network = null
        networkSummary = null
        _prediction.value = null
    }

    // ---- Model save / load ----
    private val _ioStatus = MutableStateFlow<String?>(null)
    val ioStatus: StateFlow<String?> = _ioStatus.asStateFlow()

    fun saveModel(context: Context) {
        val net = network
        if (net == null) {
            _ioStatus.value = "Сначала соберите и обучите сеть"
            return
        }
        val app = context.applicationContext
        val snapshot = hidden.toList()
        val l = loss
        viewModelScope.launch(Dispatchers.IO) {
            _ioStatus.value = try {
                ModelStore.save(app, snapshot, l, net)
                "Модель сохранена (${net.paramCount()} параметров)"
            } catch (e: Exception) {
                "Ошибка сохранения: ${e.message}"
            }
        }
    }

    fun loadModel(context: Context) {
        val app = context.applicationContext
        if (!ModelStore.exists(app)) {
            _ioStatus.value = "Нет сохранённой модели"
            return
        }
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val loaded = ModelStore.load(app, inputShape, numClasses)
                withContext(Dispatchers.Main) {
                    hidden.clear()
                    hidden.addAll(loaded.hidden)
                    loss = loaded.loss
                    network = loaded.net
                    networkSummary = loaded.net.layers().joinToString("  →  ") { it.describe() } +
                            "\nПараметров: ${loaded.net.paramCount()}"
                    _prediction.value = null
                }
                _ioStatus.value = "Модель загружена"
            } catch (e: Exception) {
                _ioStatus.value = "Ошибка загрузки: ${e.message}"
            }
        }
    }

    // ---- Test evaluation (per-class accuracy + confusion matrix) ----
    private val _evalResult = MutableStateFlow<EvalResult?>(null)
    val evalResult: StateFlow<EvalResult?> = _evalResult.asStateFlow()
    private val _isEvaluating = MutableStateFlow(false)
    val isEvaluating: StateFlow<Boolean> = _isEvaluating.asStateFlow()

    fun evaluateTest() {
        val net = network ?: run { buildError = "Сначала соберите и обучите сеть"; return }
        val ds = dataset ?: run { buildError = "Датасет не загружен"; return }
        if (_isEvaluating.value) return
        _isEvaluating.value = true
        viewModelScope.launch(Dispatchers.Default) {
            try {
                ds.splitValidation(valSamples)
                val total = ds.valCount()
                val confusion = Array(10) { IntArray(10) }
                val chunk = 256
                var base = 0
                while (base < total) {
                    val count = minOf(chunk, total - base)
                    val xb = Array(count) { FloatArray(MnistDataset.IMG_SIZE) }
                    val yb = IntArray(count)
                    for (k in 0 until count) {
                        val idx = ds.valIndex(base + k)
                        ds.copyImage(idx, xb[k])
                        yb[k] = ds.label(idx)
                    }
                    val out = net.forward(xb, false)
                    for (b in 0 until count) confusion[yb[b]][Network.argMax(out[b])]++
                    base += count
                }
                val perClassTotal = IntArray(10)
                val perClassCorrect = IntArray(10)
                var correct = 0
                for (t in 0 until 10) for (p in 0 until 10) {
                    perClassTotal[t] += confusion[t][p]
                    if (t == p) {
                        perClassCorrect[t] += confusion[t][p]
                        correct += confusion[t][p]
                    }
                }
                val acc = if (total > 0) correct.toDouble() / total else 0.0
                _evalResult.value = EvalResult(acc, perClassCorrect, perClassTotal, confusion, total)
            } catch (e: Exception) {
                buildError = e.message ?: "ошибка оценки"
            } finally {
                _isEvaluating.value = false
            }
        }
    }

    val isTrained: Boolean
        get() = network != null

    override fun onCleared() {
        trainer?.cancel()
        super.onCleared()
    }
}
