package com.alex_melan.mnistneural

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Brush
import androidx.compose.material.icons.outlined.Hub
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.outlined.ModelTraining
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import com.alex_melan.mnistneural.ui.AboutScreen
import com.alex_melan.mnistneural.ui.ConstructorScreen
import com.alex_melan.mnistneural.ui.DrawScreen
import com.alex_melan.mnistneural.ui.MainViewModel
import com.alex_melan.mnistneural.ui.TrainingScreen
import com.alex_melan.mnistneural.ui.theme.MnistNeuralTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MnistNeuralTheme {
                AppRoot()
            }
        }
    }
}

private enum class Tab(val label: String, val icon: ImageVector) {
    CONSTRUCTOR("Сеть", Icons.Outlined.Hub),
    TRAINING("Обучение", Icons.Outlined.ModelTraining),
    DRAW("Рисование", Icons.Outlined.Brush),
    ABOUT("Автор", Icons.Outlined.Info),
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AppRoot(vm: MainViewModel = viewModel()) {
    val context = LocalContext.current
    androidx.compose.runtime.LaunchedEffect(Unit) { vm.loadDataset(context) }

    var tabName by rememberSaveable { mutableStateOf(Tab.CONSTRUCTOR.name) }
    val tab = Tab.valueOf(tabName)

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        topBar = {
            TopAppBar(title = { Text("MNIST Neural — " + tab.label) })
        },
        bottomBar = {
            NavigationBar {
                Tab.entries.forEach { t ->
                    NavigationBarItem(
                        selected = t == tab,
                        onClick = { tabName = t.name },
                        icon = { Icon(t.icon, contentDescription = t.label) },
                        label = { Text(t.label) },
                    )
                }
            }
        },
    ) { inner ->
        Box(Modifier.fillMaxSize().padding(inner)) {
            when (tab) {
                Tab.CONSTRUCTOR -> ConstructorScreen(vm)
                Tab.TRAINING -> TrainingScreen(vm)
                Tab.DRAW -> DrawScreen(vm)
                Tab.ABOUT -> AboutScreen()
            }
        }
    }
}
