package com.example.cap_final

import android.os.Bundle
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.Executor
import android.content.Intent
import android.view.animation.AnimationUtils
import androidx.biometric.BiometricPrompt
import androidx.core.content.ContextCompat


private lateinit var executor: Executor
private lateinit var biometricPrompt: BiometricPrompt
private lateinit var promptInfo: BiometricPrompt.PromptInfo
open class Start1 : AppCompatActivity() {

    var lock1:ImageView?=null
    var circleanim:ImageView?=null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.start)
        lock1=findViewById(R.id.lock1)
        circleanim=findViewById(R.id.circleanim)
        executor = ContextCompat.getMainExecutor(this@Start1)
        lock1?.setOnClickListener {
            val animation = AnimationUtils.loadAnimation(applicationContext, R.anim.scale)
            circleanim?.startAnimation(animation)
            biometricPrompt = BiometricPrompt(this@Start1, executor,

                object : BiometricPrompt.AuthenticationCallback() {
                    override fun onAuthenticationError(errorCode: Int, errString: CharSequence) {
                        Toast.makeText(getApplicationContext(), "imsi", Toast.LENGTH_LONG).show();

                    }

                    override fun onAuthenticationSucceeded(
                        result: BiometricPrompt.AuthenticationResult
                    ) {
                        super.onAuthenticationSucceeded(result)

                        val intent = Intent(applicationContext, MainActivity::class.java)
                        startActivity(intent)



                    }

                    override fun onAuthenticationFailed() {
                        super.onAuthenticationFailed()


                        Toast.makeText(applicationContext, "다시 시도해 보십시오.", Toast.LENGTH_SHORT).show()
                    }
                })
            promptInfo = BiometricPrompt.PromptInfo.Builder()
                .setTitle("지문 인증")
                .setSubtitle("기기에 등록된 지문을 이용하여 지문을 인증해주세요.")
                .setNegativeButtonText("취소")
                .build()
            biometricPrompt.authenticate(promptInfo)


        }
    }
}