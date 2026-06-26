package com.alex_melan.mnistneural.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext

private val LightColors = lightColorScheme(
    primary = md_primary,
    onPrimary = md_onPrimary,
    primaryContainer = md_primaryContainer,
    onPrimaryContainer = md_onPrimaryContainer,
    secondary = md_secondary,
    secondaryContainer = md_secondaryContainer,
    tertiary = md_tertiary,
    tertiaryContainer = md_tertiaryContainer,
    background = md_background,
    surfaceVariant = md_surfaceVariant,
)

private val DarkColors = darkColorScheme(
    primary = md_primary_dark,
    onPrimary = md_onPrimary_dark,
    primaryContainer = md_primaryContainer_dark,
    onPrimaryContainer = md_onPrimaryContainer_dark,
    secondary = md_secondary_dark,
    secondaryContainer = md_secondaryContainer_dark,
    tertiary = md_tertiary_dark,
    tertiaryContainer = md_tertiaryContainer_dark,
    background = md_background_dark,
    surfaceVariant = md_surfaceVariant_dark,
)

/**
 * App theme. Uses Material You dynamic color on Android 12+ (adapts to the user's wallpaper),
 * falling back to a custom indigo scheme on older devices. Light/dark follows the system.
 */
@Composable
fun MnistNeuralTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit,
) {
    val context = LocalContext.current
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S ->
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        darkTheme -> DarkColors
        else -> LightColors
    }
    MaterialTheme(
        colorScheme = colorScheme,
        typography = AppTypography,
        content = content,
    )
}
