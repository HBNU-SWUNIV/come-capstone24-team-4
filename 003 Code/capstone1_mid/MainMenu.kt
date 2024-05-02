package com.example.capstone_design_app2

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.StrictMode
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Base64
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.biometric.BiometricPrompt
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.content.ContextCompat
import com.google.android.material.textfield.TextInputEditText
import java.io.InputStream
import java.io.OutputStream
import java.net.Socket
import java.util.concurrent.Executor
import kotlin.io.encoding.ExperimentalEncodingApi


private lateinit var executor: Executor
private lateinit var biometricPrompt: BiometricPrompt
private lateinit var promptInfo: BiometricPrompt.PromptInfo

open class MainMenu: AppCompatActivity() {
    var result2=ArrayList<String>()
    var result3=ArrayList<String>()
    var tv1: TextView? = null
    var m1: Button? = null
    var m2: Button? = null
    var m3: Button? = null
    var m4:Button?=null
    var m10: Button? = null
    var mv:Button?=null
    var exit_btn:Button?=null
    var trans:ConstraintLayout?=null
    var m11:Button?=null
    var t:TextView?=null

    var ip_address = "192.168.00.000"
    var port = 000
    var socket:Socket?=null
    var inStream:InputStream?=null
    var outStream:OutputStream?=null
    var c3:ConstraintLayout?=null

    var b1:Boolean=false
    var b2:Boolean=false
    var c2:ConstraintLayout?=null
    var connect_btn:Button?=null
    var cmd1:Int=0
    var connect_btn_chk:Boolean?=false
    var tv51:TextView?=null

    var temptxt:TextView?=null
    var watertxt:TextView?=null
    var send1:Button?=null
    var trans_result:TextView?=null
    var input1:TextInputEditText?=null

    var camlayout:ConstraintLayout?=null
    var camview:ImageView?=null
    var bitmap2:Bitmap?=null
    var emsiemsi=""
    private val handler=Handler()
    @OptIn(ExperimentalEncodingApi::class)
    @SuppressLint("ResourceAsColor", "SuspiciousIndentation")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.mainmenu)
        val policy = StrictMode.ThreadPolicy.Builder().permitAll().build()
        StrictMode.setThreadPolicy(policy)
        executor = ContextCompat.getMainExecutor(this@MainMenu)
        m1 = findViewById(R.id.menu1)
        m2 = findViewById(R.id.menu2)
        m3=findViewById(R.id.menu3)
        m4=findViewById(R.id.menu4)
        m11= findViewById(R.id.menu8)

        c2=findViewById(R.id.c2)
        c3=findViewById(R.id.c3)
        tv1=findViewById(R.id.tv1)
        trans_result=findViewById(R.id.trans_result)
        mv = findViewById(R.id.mv)



        tv51=findViewById(R.id.tv51)


        temptxt=findViewById(R.id.temp_txt)
        watertxt=findViewById(R.id.water_txt)

        exit_btn=findViewById(R.id.exit_btn)
        connect_btn=findViewById(R.id.connectbtn)
        trans=findViewById(R.id.trans)
        send1=findViewById(R.id.send1)
        input1=findViewById(R.id.input1)
        camlayout=findViewById(R.id.cam_layout)

        camview=findViewById(R.id.cam_view)
        camlayout?.visibility=View.GONE
        c2?.visibility= View.VISIBLE
        c3?.visibility=View.GONE

        biometricPrompt = BiometricPrompt(this@MainMenu, executor,

            object : BiometricPrompt.AuthenticationCallback() {
                override fun onAuthenticationError(errorCode: Int, errString: CharSequence) {
                    Toast.makeText(getApplicationContext(), "imsi", Toast.LENGTH_LONG).show();

                }

                override fun onAuthenticationSucceeded(
                    result: BiometricPrompt.AuthenticationResult
                ) {
                    super.onAuthenticationSucceeded(result)

                    Toast.makeText(getApplicationContext(), "환영합니다", Toast.LENGTH_LONG).show();


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
        m11?.setOnClickListener {

            if(connect_btn_chk==true){

                  trans?.visibility=View.VISIBLE

            }
            else {

                    Toast.makeText(this@MainMenu, "서버와 연결안됨", Toast.LENGTH_LONG).show()
            }

        }
        trans?.setOnClickListener {
            trans?.visibility=View.GONE
        }
        send1?.setOnClickListener {

            if(connect_btn_chk==true){
                val data1 = "ke"
                var input_txt=data1+(input1?.text)+data1
                outStream?.write(input_txt.toByteArray())
                Handler().postDelayed({
                    for(i in result3.reversed()) {
                        if(i.contains("ke")){

                            trans_result?.setText(i.replace("ke",""))

                        }

                    }
                },2000)


            }
            else {
                Toast.makeText(this@MainMenu, "서버와 연결안됨", Toast.LENGTH_LONG).show()

            }
        }
        m4?.setOnClickListener {
            camlayout?.visibility=View.VISIBLE
            var data2:String=""
            if(connect_btn_chk==true){
                val data1 = "cam"
                outStream?.write(data1.toByteArray())

            }
            else if(connect_btn_chk==false){
                Toast.makeText(this@MainMenu,"서버와 연결안됨",Toast.LENGTH_LONG).show()
            }

            Handler().postDelayed({
                camview?.setImageBitmap(bitmap2)
            },7000)



        }

        camlayout?.setOnClickListener {
            camlayout?.visibility=View.GONE
        }
        m3?.setOnClickListener {

            if(connect_btn_chk==true){

                if(b2==false){
                    val data1 = "dooropen"
                    outStream?.write(data1.toByteArray())
                    tv51?.setText("문 닫기")
                    m3?.setBackgroundResource(R.drawable.dooropen)
                    b2=true
                }
                else{
                    val data1 = "doorclose"
                    outStream?.write(data1.toByteArray())
                    tv51?.setText("문 열기")
                    m3?.setBackgroundResource(R.drawable.door2)
                    b2=false
                }
            }
            else {
                if(b2==false) {
                    tv51?.setText("문 닫기")
                    Toast.makeText(this@MainMenu, "서버와 연결안됨", Toast.LENGTH_LONG).show()
                    m3?.setBackgroundResource(R.drawable.dooropen)
                    b2=true
                }
                else{
                    tv51?.setText("문 열기")
                    Toast.makeText(this@MainMenu, "서버와 연결안됨", Toast.LENGTH_LONG).show()
                    m3?.setBackgroundResource(R.drawable.door2)
                    b2=false
                }
            }

        }
        connect_btn?.setOnClickListener {

              socket = Socket(ip_address, port)
            outStream = socket?.outputStream
            inStream = socket?.inputStream


            Thread {
                while(true){
                    val inputstream=socket?.getInputStream()
                    var reponse=ByteArray(4096)
                    val data=inputstream?.read(reponse)
                    var resulta=reponse.decodeToString()
                    if(resulta.startsWith("t")) {
                        result2.add(resulta)

                    }
                    if(resulta.startsWith("ke")){
                        result3.add(resulta)
                    }
                    else{
                        Log.v("info2",resulta.toString())
                        emsiemsi=resulta
                        val bytearray= Base64.decode(emsiemsi.toString(),Base64.DEFAULT)
                        bitmap2=BitmapFactory.decodeByteArray(bytearray,0,bytearray.size)

                    }

                    }

            }.start()

            connect_btn_chk=true
            c2?.visibility= View.INVISIBLE

        }


        m2?.setOnClickListener {
            var result=""
            if( connect_btn_chk==true){
                c3?.visibility=View.VISIBLE
                val data2="command11"
                outStream?.write(data2.toByteArray())

                    Handler().postDelayed({
                        for(i in result2.reversed()) {

                                watertxt?.setText(i.substring(1,3))
                                temptxt?.setText(i.substring(4,6))

                        }
                    },2000)


            }
            else{
                Toast.makeText(this@MainMenu,"서버와 연결안됨",Toast.LENGTH_LONG).show()
                c3?.visibility=View.VISIBLE

            }
            c3?.setOnClickListener {
                c3?.visibility=View.GONE
            }


        }


        executor = ContextCompat.getMainExecutor(this@MainMenu)


        m1?.setOnClickListener {

            if(!b1){
                if(connect_btn_chk==true) {
                    val data1 = "lighton"
                    outStream?.write(data1.toByteArray())
                    // outStream?.close()
                    // socket?.close() // 소켓을 닫음
                    m1?.setBackgroundResource(R.drawable.icon_new_final0421final2)
                    cmd1=1
                }
                else{
                    m1?.setBackgroundResource(R.drawable.icon_new_final0421final2)
                    Toast.makeText(this@MainMenu,"서버와 연결안됨",Toast.LENGTH_LONG).show()

                }
                b1=true
            }
            else if(b1){
                if(connect_btn_chk==true) {
                    m1?.setBackgroundResource(R.drawable.light3)
                val data2="lightoff"
                outStream?.write(data2.toByteArray())

                }
                else{
                    m1?.setBackgroundResource(R.drawable.light3)
                    Toast.makeText(this@MainMenu,"서버와 연결안됨",Toast.LENGTH_LONG).show()

                }
                b1=false
            }

        }

        m10?.setOnClickListener {
            inStream?.close()
            outStream?.close()
            socket?.close()
        }

    }

    }