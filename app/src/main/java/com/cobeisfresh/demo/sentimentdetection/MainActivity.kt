package com.cobeisfresh.demo.sentimentdetection

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import com.google.android.material.snackbar.Snackbar
import com.google.firebase.ml.modeldownloader.CustomModel
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions
import com.google.firebase.ml.modeldownloader.DownloadType
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.schema.BuiltinOperator
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.io.BufferedReader
import java.io.File
import java.io.InputStream
import java.io.InputStreamReader
import java.lang.IllegalStateException
import java.nio.ByteBuffer
import java.util.*


class MainActivity : AppCompatActivity() {
    private lateinit var interpreter: Interpreter
    private lateinit var dictionary: Map<String, Int>
    private lateinit var labels: List<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initializeSubmitButton()
        downloadModel(
            onSuccess = { model ->
                model.file?.let { file ->
                    initializeModel(file = file)
                    enableUi()
                }
            },
            onError = { t ->
                Snackbar.make(
                    findViewById(android.R.id.content),
                    "Error downloading model. Message: ${t?.message}",
                    Snackbar.LENGTH_SHORT
                ).show()
            }
        )
    }

    private fun classify() {
        val input = tokenizeInputText(input = findViewById<EditText>(R.id.et_input).text.toString())
        val output = Array(size = 1) { FloatArray(labels.size) { 0f } }
        interpreter.run(input, output)
        for (i in labels.indices) {
            Log.d("MainActivity", "${labels[i]}: ${output[0][i]}")
        }
    }

    private fun downloadModel(
        onSuccess: (CustomModel) -> Unit,
        onError: (Throwable?) -> Unit
    ) {
        val conditions = CustomModelDownloadConditions.Builder()
            .requireWifi()
            .build()
        FirebaseModelDownloader.getInstance()
            .getModel("text_classification_v2", DownloadType.LATEST_MODEL, conditions)
            .addOnSuccessListener { customModel ->
                if (customModel != null) {
                    onSuccess(customModel)
                } else {
                    onError(IllegalStateException("Model download successful, but model is null."))
                }
            }
            .addOnFailureListener { e ->
                onError(e)
            }
    }

    private fun initializeModel(file: File) {
        interpreter = Interpreter(file)
        val metadataExtractor = MetadataExtractor(ByteBuffer.wrap(file.readBytes()))
        loadDictionary(metadataExtractor.getAssociatedFile("vocab.txt"))
        loadLabels(metadataExtractor.getAssociatedFile("labels.txt"))
    }

    // refer to https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android
    private fun loadLabels(inputStream: InputStream) {
        val result = mutableListOf<String>()
        val reader = BufferedReader(InputStreamReader(inputStream))
        while (reader.ready()) {
            result.add(reader.readLine())
        }
        labels = result
    }

    // refer to https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android
    private fun loadDictionary(inputStream: InputStream) {
        val result = mutableMapOf<String, Int>()
        val reader = BufferedReader(InputStreamReader(inputStream))
        while (reader.ready()) {
            val (word, index) = reader.readLine().split(" ")
            result[word] = index.toInt()
        }
        dictionary = result
    }

    private fun tokenizeInputText(input: String): Array<IntArray> {
        val temp = IntArray(size = MAX_SENTENCE_LEN) { dictionary[PAD]!! }
        val words = input.split(SIMPLE_SPACE_OR_PUNCTUATION_REGEX)
        var index = 0
        if (dictionary.containsKey(START)) {
            temp[index++] = dictionary[START]!!
        }
        words.forEach { word ->
            if (index >= MAX_SENTENCE_LEN) {
                return@forEach
            }
            temp[index++] = if (dictionary.containsKey(word)) dictionary[word]!! else dictionary[UNKNOWN]!!
        }
        Arrays.fill(temp, index, MAX_SENTENCE_LEN - 1, dictionary[PAD]!!)
        return Array(size = 1) { temp }
    }

    private fun initializeSubmitButton() {
        findViewById<Button>(R.id.btn_submit).setOnClickListener {
            classify()
        }
    }

    private fun enableUi() {
        findViewById<Button>(R.id.btn_submit).isEnabled = true
    }

    companion object {
        private const val MAX_SENTENCE_LEN = 256
        private const val SIMPLE_SPACE_OR_PUNCTUATION_REGEX = " |\\,|\\.|\\!|\\?|\n"
        private const val START = "<START>"
        private const val PAD = "<PAD>"
        private const val UNKNOWN = "<UNKNOWN>"
    }
}