package com.example.cap_final_ver9


import android.annotation.SuppressLint
import android.content.Context
import android.os.Bundle
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.Executor
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.view.View
import android.view.animation.AnimationUtils
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextClock
import androidx.biometric.BiometricPrompt
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.textfield.TextInputEditText


private lateinit var executor: Executor
private lateinit var biometricPrompt: BiometricPrompt
private lateinit var promptInfo: BiometricPrompt.PromptInfo

open class Start1 : AppCompatActivity() {
    lateinit var ip_num:SharedPreferences
    var lock1:ImageView?=null
    var circleanim:ImageView?=null
    var ip_setting: LinearLayout?=null
    var textClock: TextClock?=null
    var ipinput:TextInputEditText?=null
    var save:Button?=null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.start)
        lock1=findViewById(R.id.lock1)
        circleanim=findViewById(R.id.circleanim)
        save=findViewById(R.id.save)
        ip_setting=findViewById(R.id.ip_setting)
        textClock=findViewById(R.id.textClock)
        ipinput=findViewById(R.id.ipinput)
        ip_setting?.visibility=View.GONE
        ip_num=getSharedPreferences("ip", MODE_PRIVATE)

        executor = ContextCompat.getMainExecutor(this@Start1)
        val audioPermission = ContextCompat.checkSelfPermission(this, android.Manifest.permission.RECORD_AUDIO)
        if (audioPermission == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.RECORD_AUDIO), 99)
        }
        save?.setOnClickListener {

            ip_num.edit().putString("ip",ipinput?.text.toString()).apply()
        }
        lock1?.setOnClickListener {
            val animation = AnimationUtils.loadAnimation(applicationContext, R.anim.scale)
            circleanim?.startAnimation(animation)
            biometricPrompt = BiometricPrompt(this@Start1, executor,

                object : BiometricPrompt.AuthenticationCallback() {
                    override fun onAuthenticationError(errorCode: Int, errString: CharSequence) {
                        Toast.makeText(getApplicationContext(), "지문 인증에 오류가 생겨 넘어갑니다.", Toast.LENGTH_LONG).show();
                        ip_num.edit().putString("ip",ipinput?.text.toString()).apply()
                        val intent = Intent(applicationContext, MainActivity::class.java)
                        startActivity(intent)
                    }

                    override fun onAuthenticationSucceeded(
                        result: BiometricPrompt.AuthenticationResult
                    ) {
                        super.onAuthenticationSucceeded(result)
                        ip_num.edit().putString("ip",ipinput?.text.toString()).apply()
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

        textClock?.setOnClickListener {
            if(ip_setting?.visibility==View.GONE){
                ip_setting?.visibility=View.VISIBLE
            }
            else if(ip_setting?.visibility==View.VISIBLE){
                ip_setting?.visibility=View.GONE
            }

        }
    }





    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            99 -> {
                if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                } else {
                    Toast.makeText(this,"권한이 부족하여 음성명령 기능을 이용할 수 없습니다.",Toast.LENGTH_LONG).show()
                }
            }
        }
    }

}