package com.cobeisfresh.demo.sentimentdetection

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import com.google.android.material.snackbar.Snackbar
import com.google.firebase.ml.modeldownloader.CustomModel
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions
import com.google.firebase.ml.modeldownloader.DownloadType
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.io.BufferedReader
import java.io.File
import java.io.InputStream
import java.io.InputStreamReader
import java.lang.IllegalStateException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    private var interpreter: Interpreter? = null

    private var dictionary: Map<String, Int>? = null
    private var labels: List<String>? = null

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
        val input = findViewById<EditText>(R.id.et_input).text.toString()
        val output = ByteBuffer.allocateDirect(1_000).order(ByteOrder.nativeOrder())
        interpreter?.run(input, output)
        findViewById<TextView>(R.id.tv_result).text = String(output.array())
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

    private fun initializeSubmitButton() {
        findViewById<Button>(R.id.btn_submit).setOnClickListener {
            classify()
        }
    }

    private fun enableUi() {
        findViewById<Button>(R.id.btn_submit).isEnabled = true
    }
}