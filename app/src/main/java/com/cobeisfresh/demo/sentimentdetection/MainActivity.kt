package com.cobeisfresh.demo.sentimentdetection

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import com.google.android.material.snackbar.Snackbar
import com.google.firebase.ml.modeldownloader.CustomModel
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions
import com.google.firebase.ml.modeldownloader.DownloadType
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader
import org.tensorflow.lite.Interpreter
import java.io.File
import java.lang.IllegalStateException

class MainActivity : AppCompatActivity() {
    private var interpreter: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initializeSubmitButton()
        downloadModel(
            onSuccess = { model ->
                model.file?.let { file ->
                    initializeInterpreter(modelFile = file)
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

    private fun initializeInterpreter(modelFile: File) {
        interpreter = Interpreter(modelFile)
    }

    private fun initializeSubmitButton() {
        findViewById<Button>(R.id.btn_submit).setOnClickListener {

        }
    }

    private fun enableUi() {
        findViewById<Button>(R.id.btn_submit).isEnabled = true
    }
}