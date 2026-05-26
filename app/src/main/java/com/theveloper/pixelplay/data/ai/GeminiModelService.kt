package com.theveloper.pixelplay.data.ai

import com.theveloper.pixelplay.data.repository.MusicRepository
import com.theveloper.pixelplay.data.worker.AiWorkerManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import timber.log.Timber
import javax.inject.Inject
import javax.inject.Singleton

data class GeminiModel(
    val name: String,
    val displayName: String
)

@Singleton
class GeminiModelService @Inject constructor(
    private val orchestrator: AiOrchestrator,
    private val digestGenerator: UserProfileDigestGenerator,
    private val musicRepository: MusicRepository,
    private val workerManager: AiWorkerManager
) {

    /**
     * Fetches available Gemini models using the provided API key.
     * Returns a list of model names that are available for the user.
     */
    suspend fun fetchAvailableModels(apiKey: String): Result<List<GeminiModel>> {
        return withContext(Dispatchers.IO) {
            try {
                if (apiKey.isBlank()) {
                    return@withContext Result.failure(Exception("API Key is required"))
                }

                // Use a lightweight model to test the API key and fetch available models
                // We'll make a request to list models using the Gemini API
                val response = makeModelsListRequest(apiKey)

                Result.success(response)
            } catch (e: Exception) {
                Timber.e(e, "Error fetching Gemini models")
                Result.failure(e)
            }
        }
    }

    private suspend fun makeModelsListRequest(apiKey: String): List<GeminiModel> {
        return withContext(Dispatchers.IO) {
            try {
                // Make HTTP request to Google's Gemini API to list models
                val url = "https://generativelanguage.googleapis.com/v1beta/models?key=$apiKey"
                val connection = java.net.URL(url).openConnection() as java.net.HttpURLConnection

                connection.requestMethod = "GET"
                connection.connectTimeout = 10000
                connection.readTimeout = 10000

                val responseCode = connection.responseCode
                if (responseCode == 200) {
                    val response = connection.inputStream.bufferedReader().use { it.readText() }
                    parseModelsResponse(response)
                } else {
                    val errorMessage = connection.errorStream?.bufferedReader()?.use { it.readText() }
                    Timber.e("Failed to fetch models: $responseCode - $errorMessage")
                    // Return default models if API call fails
                    getDefaultModels()
                }
            } catch (e: Exception) {
                Timber.e(e, "Exception fetching models, returning defaults")
                getDefaultModels()
            }
        }
    }

    private fun parseModelsResponse(jsonResponse: String): List<GeminiModel> {
        try {
            // Parse the JSON response to extract model names
            // Expected format: {"models": [{"name": "models/gemini-...", ...}, ...]}
            val models = mutableListOf<GeminiModel>()

            // Simple JSON parsing - extract model names
            val modelPattern = """"name":\s*"(models/[^"]+)"""".toRegex()
            val matches = modelPattern.findAll(jsonResponse)

            for (match in matches) {
                val fullName = match.groupValues[1]
                val modelName = fullName.removePrefix("models/")

                // Keep the UI focused on the cheapest/fastest Gemini family: Flash only.
                if (isSupportedFlashModel(modelName)) {
                    models.add(GeminiModel(
                        name = modelName,
                        displayName = formatDisplayName(modelName)
                    ))
                }
            }

            return if (models.isNotEmpty()) {
                sortFlashModels(models)
            } else {
                getDefaultModels()
            }
        } catch (e: Exception) {
            Timber.e(e, "Error parsing models response")
            return getDefaultModels()
        }
    }

    /**
     * Estimates the token count for a piece of text.
     * Uses a conservative 4 chars per token rule for non-Gemini providers,
     * but we recommend using the specific countTokens method on AiClient for accuracy.
     */
    fun estimateTokens(text: String): Int {
        return (text.length / 4).coerceAtLeast(1)
    }

    /**
     * High-level method to perform an AI operation.
     * Starts a background worker if [runInBackground] is true, 
     * otherwise executes immediately and returns the result.
     */
    suspend fun performAiTask(
        prompt: String,
        type: AiSystemPromptType,
        runInBackground: Boolean = false,
        temperature: Float = 0.7f
    ): String? {
        if (runInBackground) {
            workerManager.enqueueAiTask(prompt, type, temperature)
            return null
        } else {
            val allSongs = musicRepository.getAllSongsOnce()
            val context = if (type == AiSystemPromptType.PLAYLIST || 
                            type == AiSystemPromptType.TAGGING || 
                            type == AiSystemPromptType.PERSONA) {
                digestGenerator.generateDigest(allSongs)
            } else ""

            return orchestrator.generateContent(
                prompt = prompt,
                type = type,
                temperature = temperature,
                context = context
            )
        }
    }

    private fun isSupportedFlashModel(modelName: String): Boolean {
        return modelName.startsWith("gemini", ignoreCase = true) &&
            modelName.contains("flash", ignoreCase = true) &&
            !modelName.contains("embedding", ignoreCase = true) &&
            !modelName.contains("image", ignoreCase = true)
    }

    private fun sortFlashModels(models: List<GeminiModel>): List<GeminiModel> {
        val preferred = getDefaultModels().map { it.name }
        return models.distinctBy { it.name }.sortedWith(
            compareBy<GeminiModel> { model ->
                preferred.indexOf(model.name).takeIf { it >= 0 } ?: Int.MAX_VALUE
            }.thenBy { it.displayName.lowercase() }
        )
    }

    private fun formatDisplayName(modelName: String): String {
        // Convert "gemini-2.5-flash" to "Gemini 2.5 Flash"
        return modelName
            .split("-")
            .joinToString(" ") { word ->
                word.replaceFirstChar { it.uppercase() }
            }
    }

    private fun getDefaultModels(): List<GeminiModel> {
        return listOf(
            GeminiModel("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite (Cheapest/Fastest Default)"),
            GeminiModel("gemini-2.5-flash", "Gemini 2.5 Flash"),
            GeminiModel("gemini-flash-lite-latest", "Gemini Flash Lite Latest"),
            GeminiModel("gemini-flash-latest", "Gemini Flash Latest"),
            GeminiModel("gemini-2.0-flash-lite", "Gemini 2.0 Flash Lite"),
            GeminiModel("gemini-2.0-flash", "Gemini 2.0 Flash")
        )
    }
}
