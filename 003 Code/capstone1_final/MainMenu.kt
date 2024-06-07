package com.example.capstone_design_app2

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.wifi.WifiConfiguration
import android.os.Bundle
import android.os.Handler
import android.os.StrictMode
import android.speech.RecognitionListener
import android.speech.SpeechRecognizer
import android.text.Editable
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
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.HttpURLConnection
import java.net.Socket
import java.net.URL
import java.net.URLEncoder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Timer
import java.util.concurrent.Executor
import kotlin.io.encoding.ExperimentalEncodingApi


private lateinit var executor: Executor
private lateinit var biometricPrompt: BiometricPrompt
private lateinit var promptInfo: BiometricPrompt.PromptInfo
private lateinit var timer2:Timer
open class MainMenu: AppCompatActivity()  {
    var result2=ArrayList<String>()
    var result3=ArrayList<String>()
    var tv1: TextView? = null
    var m1: Button? = null
    var m2: Button? = null
    var m3: Button? = null
    var m4:Button?=null
    var m10: Button? = null
    var mv:Button?=null
    var stt_result=""
    var ask_me_btn:Button?=null
    var speechRecognizer: SpeechRecognizer? = null
    var trans:ConstraintLayout?=null
    var m11:Button?=null
    var t:TextView?=null
    var secure_key="=input_key"
        //var ip_address = "ip_address"
    var ip_address = "ip_address"
    var port = "port"
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
    var imsi3=0
    var imsi4=0
    var temptxt:TextView?=null
    var watertxt:TextView?=null
    var send1:Button?=null
    var trans_result:TextView?=null
    var input1:TextInputEditText?=null
    var temphum:TextView?=null
    var camlayout:ConstraintLayout?=null
    var camview:ImageView?=null
    var bitmap2:Bitmap?=null
    var emsiemsi=""
    var lists=ArrayList<String>()
    var swa1=false
    var swa2=false
    var swa3=false
    var swa4=false
    var final_result_dust:String=""
    var final_result_sun1:String=""
    var final_result_sun2:String=""
    var dustinfo:TextView?=null
    var suninfo:TextView?=null
    var currentDate:String=""
    var currentTime:String=""
    var get_year:Int=0
    var get_month:Int=0
    var get_day:Int=0
    var get_hh:Int=0
    var get_mm:Int=0
    private val handler=Handler()
    @OptIn(ExperimentalEncodingApi::class)
    @SuppressLint("ResourceAsColor", "SuspiciousIndentation")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.mainmenu)
        val policy = StrictMode.ThreadPolicy.Builder().permitAll().build()
        StrictMode.setThreadPolicy(policy)
        val sdf = SimpleDateFormat("yyyy-MM-dd")
        val sdf2 = SimpleDateFormat("HH:mm:ss")
        currentDate = sdf.format(Date())
        currentTime = sdf2.format(Date())
        Log.d("currenttimeis",currentTime)
        executor = ContextCompat.getMainExecutor(this@MainMenu)
        m1 = findViewById(R.id.menu1)
        m3=findViewById(R.id.menu3)
        m4=findViewById(R.id.menu4)
        m11= findViewById(R.id.menu8)

        c2=findViewById(R.id.c2)
        c3=findViewById(R.id.c3)
        tv1=findViewById(R.id.tv1)
        trans_result=findViewById(R.id.trans_result)
        mv = findViewById(R.id.mv)
        temphum=findViewById(R.id.temphum)
        suninfo=findViewById(R.id.suninfo)
        dustinfo=findViewById(R.id.dustinfo)
        tv51=findViewById(R.id.tv51)


        temptxt=findViewById(R.id.temp_txt)
        watertxt=findViewById(R.id.water_txt)

        ask_me_btn=findViewById(R.id.ask_me_btn)
        connect_btn=findViewById(R.id.connectbtn)
        trans=findViewById(R.id.trans)
        send1=findViewById(R.id.send1)
        input1=findViewById(R.id.input1)
        camlayout=findViewById(R.id.cam_layout)

        camview=findViewById(R.id.cam_view)
        camlayout?.visibility=View.GONE
        c2?.visibility= View.VISIBLE
        c3?.visibility=View.GONE
        get_year=currentDate.substring(0,3).toInt()
        get_month=currentDate.substring(5,7).toInt()
        get_day=currentDate.substring(8,10).toInt()
        get_hh=currentTime.substring(0,2).toInt()
        get_mm=currentTime.substring(3,5).toInt()


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


        m4?.setOnClickListener {
cam_command()
        }

        camlayout?.setOnClickListener {
            camlayout?.visibility=View.GONE
        }
        ask_me_btn?.setOnClickListener {

            if(connect_btn_chk==true){

                speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this@MainMenu)
                speechRecognizer?.setRecognitionListener(recognitionListener())    // 리스너 설정
                speechRecognizer?.startListening(intent)



            }
            else {
                Toast.makeText(this@MainMenu, "서버와 연결안됨", Toast.LENGTH_LONG).show()

            }

        }
        m3?.setOnClickListener {

      door_command()
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
                    else(resulta.contains(("[A-Za-z0-9!]").toRegex()))
                        Log.v("info2",resulta
                            .toString())
                        emsiemsi=resulta
                    lists.add(emsiemsi)
                    Log.d("listsabc",lists.size.toString())

                    }


            }.start()

            connect_btn_chk=true
            get_dust_data()
            get_sun_data()
            get_temp_hum()
            c2?.visibility= View.INVISIBLE

        }





        executor = ContextCompat.getMainExecutor(this@MainMenu)


        m1?.setOnClickListener {

    light_command()
        }

        m10?.setOnClickListener {
            inStream?.close()
            outStream?.close()
            socket?.close()
        }



    }


    fun light_command(){
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
    fun door_command(){
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
    fun cam_command(){
        camlayout?.visibility=View.VISIBLE
        var data2:String=""
        var timer1=0.5f
        val mTimer= Timer()
        val mHandeler=Handler()
        if(connect_btn_chk==true){
            val data1 = "cam"
            outStream?.write(data1.toByteArray())

        }
        else if(connect_btn_chk==false){
            Toast.makeText(this@MainMenu,"서버와 연결안됨",Toast.LENGTH_LONG).show()
        }
        Log.d("count",lists.size.toString())




        Log.d("result_final125","aaa")

        timer2=kotlin.concurrent.timer(period = 500){
            timer1++
            Log.d("timertimer",timer1.toString())


            if(timer1.toInt()>=7&&timer1.toInt()<=18) {
                get_video_frame(imsi4)
                imsi4=imsi4+1
            }


            runOnUiThread {
                camview?.setImageBitmap(bitmap2)
            }
        }

        //   val mTimerTask= object: TimerTask(){
        //        override fun run(){
        //            mHandeler.postDelayed({
        //                timer1--
        //               Toast.makeText(this@MainMenu,"testing",Toast.LENGTH_LONG).show()
        //               camview?.setImageBitmap(bitmap2)
        //               if(timer1<=0){
        //                  mTimer.cancel()
        //              }
        //          },0)
        //       }

        //    }
        // Handler().postDelayed({
        //     camview?.setImageBitmap(bitmap2)
        // },7000)

        // mTimer.schedule(mTimerTask,0,500)

    }
fun get_video_frame(num:Int){

        var i=lists[num]
        val bytearray= Base64.decode(i.toString(),Base64.DEFAULT)
        bitmap2=BitmapFactory.decodeByteArray(bytearray,0,bytearray.size)
        Log.d("testing214 ",i.toString())

        imsi3=imsi3+1

}


    @SuppressLint("SuspiciousIndentation")
    fun get_dust_data(){
        if(swa1==false) {

            val urlBuilder =
                java.lang.StringBuilder("http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMinuDustFrcstDspth") /*URL*/
            urlBuilder.append(
                "?" + URLEncoder.encode(
                    "serviceKey",
                    "UTF-8"
                ) + "=key"
            ) /*Service Key*/
            urlBuilder.append(
                "&" + URLEncoder.encode(
                    "returnType",
                    "UTF-8"
                ) + "=" + URLEncoder.encode("json", "UTF-8")
            ) /*xml 또는 json*/
            urlBuilder.append(
                "&" + URLEncoder.encode(
                    "numOfRows",
                    "UTF-8"
                ) + "=" + URLEncoder.encode("100", "UTF-8")
            ) /*한 페이지 결과 수(조회 날짜로 검색 시 사용 안함)*/
            urlBuilder.append(
                "&" + URLEncoder.encode(
                    "pageNo",
                    "UTF-8"
                ) + "=" + URLEncoder.encode("1", "UTF-8")
            ) /*페이지번호(조회 날짜로 검색 시 사용 안함)*/
            urlBuilder.append(
                "&" + URLEncoder.encode(
                    "searchDate",
                    "UTF-8"
                ) + "=" + URLEncoder.encode(currentDate, "UTF-8")//2024-05-20
            ) /*통보시간 검색(조회 날짜 입력이 없을 경우 한달동안 예보통보 발령 날짜의 리스트 정보를 확인)*/
            urlBuilder.append(
                "&" + URLEncoder.encode(
                    "InformCode",
                    "UTF-8"
                ) + "=" + URLEncoder.encode("PM10", "UTF-8")
            ) /*통보코드검색(PM10, PM25, O3)*/
            val url = URL(urlBuilder.toString())
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "GET"
            conn.setRequestProperty("Content-type", "application/json")
            val rd: BufferedReader
            rd = if (conn.responseCode >= 200 && conn.responseCode <= 300) {
                BufferedReader(InputStreamReader(conn.inputStream))
            } else {
                BufferedReader(InputStreamReader(conn.errorStream))
            }
            val sb = java.lang.StringBuilder()
            var line: String?
            rd.forEachLine {
                sb.append(it+'\n')
            }
            var sb2=sb.toString()

            rd.close()
            conn.disconnect()
            var s: String?
            var tourists = java.util.ArrayList<Any>()
                Log.d("all_data",sb2)
                var jsonObject: JSONObject? = JSONObject(sb.toString())
                var response: JSONObject? = jsonObject?.getJSONObject("response")
                var body: JSONObject? = response?.getJSONObject("body")
                var items: JSONArray? = body?.get("items") as JSONArray?

                var r1= items?.getJSONObject(5)
                 var r2=r1?.getString("informGrade")

                //var items: JSONObject? = response?.getJSONObject("items")
               // var items: JSONArray? = body?.get("items") as JSONArray?
                Log.d("iteminfo", r2.toString())
                val search="대전"
                val idx=r2.toString().indexOf(search)
                final_result_dust=r2.toString().substring(idx+5,idx+7)
                val final_result="현재 대전의 미세먼지는 "+final_result_dust+"입니다."
            Log.d("final_result_dust",final_result)
            //좋음, 보통, 나쁨
            dustinfo?.setText(final_result)
            swa1=true
        }

    }

    @SuppressLint("SuspiciousIndentation")
    fun get_temp_hum(){
        val data2="command11"
        var temp=""
        var hum=""
        outStream?.write(data2.toByteArray())
        Handler().postDelayed({
            for(i in result2.reversed()) {
                hum=(i.substring(1,3))
                temp=(i.substring(4,6))
            }
            val  final_result="현재 습도는 "+hum+"% 이고 현재온도는 "+temp+"도 입니다."
            temphum?.setText(final_result)
        },2000)


    }

    @SuppressLint("SuspiciousIndentation")
    fun get_sun_data(){
        if(swa2==false) {

            val urlBuilder =
                java.lang.StringBuilder("http://apis.data.go.kr/B090041/openapi/service/RiseSetInfoService/getAreaRiseSetInfo") /*URL*/
            urlBuilder.append(
                "?" + URLEncoder.encode(
                    "serviceKey",
                    "UTF-8"
                ) + "=key"
            ) /*Service Key*/
            urlBuilder.append(
                "&" + URLEncoder.encode(
                    "locdate",
                    "UTF-8"
                ) + "=" + URLEncoder.encode(currentDate.replace("-",""), "UTF-8")
            ) /*xml 또는 json*/
            urlBuilder.append(
                "&" + URLEncoder.encode(
                    "location",
                    "UTF-8"
                ) + "=" + URLEncoder.encode("대전", "UTF-8")
            )
            val url = URL(urlBuilder.toString())
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "GET"
            conn.setRequestProperty("Content-type", "application/json")
            val rd: BufferedReader
            rd = if (conn.responseCode >= 200 && conn.responseCode <= 300) {
                BufferedReader(InputStreamReader(conn.inputStream))
            } else {
                BufferedReader(InputStreamReader(conn.errorStream))
            }
            val sb = java.lang.StringBuilder()
            var line: String?
            rd.forEachLine {
                sb.append(it+'\n')
            }
            var sb2=sb.toString()

            rd.close()
            conn.disconnect()
            var s: String?
            var tourists = java.util.ArrayList<Any>()
            Log.d("all_data",sb2)
            //sb.toString()

            //var items: JSONObject? = response?.getJSONObject("items")
            // var items: JSONArray? = body?.get("items") as JSONArray?
            val search1="<sunrise>"
            val search2="<sunset>"
            val search3="</sunrise>"
            val search4="</sunset>"
            val idx1=sb.toString().indexOf(search1)
            val idx2=sb.toString().indexOf(search2)
            val idx3=sb.toString().indexOf(search3)
            val idx4=sb.toString().indexOf(search4)
            final_result_sun1=sb.toString().substring(idx1+9,idx3-2)
            final_result_sun2=sb.toString().substring(idx2+8,idx4-2)
            Log.d("final_result_sun",final_result_sun1+" "+final_result_sun2)
            var final_result_pre1=final_result_sun1.substring(0,2)+":"+final_result_sun1.substring(2,4)
            var final_result_pre2=final_result_sun2.substring(0,2)+":"+final_result_sun2.substring(2,4)
            //string.toInt()
            //yyyy/MM/dd
            //hh:mm:ss

            var final_result="오늘 일출시간은 "+final_result_pre1+"이고 일몰시간은 "+final_result_pre2+"입니다."
            var calc1=final_result_sun1.substring(0,2).toInt()*60+final_result_sun1.substring(2,4).toInt()
            var calc2=final_result_sun2.substring(0,2).toInt()*60+final_result_sun2.substring(2,4).toInt()
            var calc3=get_hh*60+get_mm.toInt()
            Log.d("calc123 info",calc1.toString()+" "+calc2.toString()+" "+calc3.toString() )
            if (calc1<calc3&& calc3<calc2){

                final_result+=" "+"불을 끕니다."
                suninfo?.setText(final_result)




            }
            else{

                final_result+=" "+"불을 켭니다."
                val data1 = "lighton"
                suninfo?.setText(final_result)
                outStream?.write(data1.toByteArray())
                // outStream?.close()
                // socket?.close() // 소켓을 닫음
                m1?.setBackgroundResource(R.drawable.icon_new_final0421final2)
                cmd1=1



            }
            swa2=true
        }

    }
    fun recognitionListener() = object : RecognitionListener {
        override fun onReadyForSpeech(params: Bundle?) {
            Toast.makeText(applicationContext, "음성 인식을 시작합니다.", Toast.LENGTH_SHORT).show()
        }

        override fun onBeginningOfSpeech() {
            Toast.makeText(applicationContext, "말씀하십시오.", Toast.LENGTH_SHORT).show()
        }

        override fun onRmsChanged(rmsdB: Float) {


        }

        override fun onBufferReceived(buffer: ByteArray?) {


        }

        override fun onEndOfSpeech() {
            speechRecognizer!!.stopListening() //녹음 중지
            Toast.makeText(applicationContext, "음성 기록을 중지합니다.", Toast.LENGTH_SHORT).show()
        }

        override fun onError(error: Int) {
            when (error) {
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> Toast.makeText(
                    applicationContext,
                    "Not enough permission",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }

        @SuppressLint("SuspiciousIndentation")
        override fun onResults(results: Bundle?) {
            val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)!![0]
            stt_result = Editable.Factory.getInstance().newEditable(matches).toString()
            val data1 = "ke"
            var input_txt=data1+(stt_result)+data1
            outStream?.write(input_txt.toByteArray())
            Handler().postDelayed({
                for(i in result3.reversed()) {
                    if(i.contains("ke")){
                    var result= i.replace("ke","")
                        if(result.contains("lighton")||result.contains("lightoff")){
                            light_command()
                            Toast.makeText(this@MainMenu,"전등을 조절하겠습니다.",Toast.LENGTH_LONG).show()
                        }
                        else if(result.contains("dooropen")||result.contains("doorclose")){
                            door_command()
                            Toast.makeText(this@MainMenu,"문 여닫이를 조절하겠습니다.",Toast.LENGTH_LONG).show()
                        }
                        else if(result.contains("cam")){
                            cam_command()
                            Toast.makeText(this@MainMenu,"외부카메라를 표시하겠습니다.",Toast.LENGTH_LONG).show()
                        }
                        else{
                            Toast.makeText(this@MainMenu,"죄송합니다. 명령을 실행하지 못하였습니다.",Toast.LENGTH_LONG).show()
                        }


                    }

                }
            },2000)


        }

        override fun onPartialResults(partialResults: Bundle?) {


        }

        override fun onEvent(eventType: Int, params: Bundle?) {


        }










    }


}


