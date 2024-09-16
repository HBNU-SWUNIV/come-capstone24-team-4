package com.example.cap_final


import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.StrictMode
import android.speech.RecognitionListener
import android.speech.SpeechRecognizer
import android.text.Editable
import android.util.Base64
import android.util.Log
import android.view.View
import android.view.animation.Animation
import android.view.animation.AnimationUtils
import android.view.animation.ScaleAnimation
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.constraintlayout.widget.ConstraintLayout
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


class MainActivity : AppCompatActivity() {

    var secure_key=
    var ip_address =
    var port = 

    var stt_result=""
    var imsi3=""
    var imsi4=0

    var time1=0.0
    var time_l=0
    var time_tv=0
    var e_1=0.0//device
    var e_l=0.0//led
    var e_tv=0.0//
    var sw1=false
    var sw2=false
    var b1:Boolean=false
    var b2:Boolean=false
    var currentDate:String=""
    var currentTime:String=""
    var get_year:Int=0
    var get_month:Int=0
    var get_day:Int=0
    var get_hh:Int=0
    var get_mm:Int=0
    var socket:Socket?=null
    var inStream: InputStream?=null
    var outStream: OutputStream?=null
    var p1=false
    var p2=false
    var p3=false
    var p4=false
    var imsi1=0
    var result2=ArrayList<String>()
    var result3=ArrayList<String>()
    var timer1=0.5f
    var result4=""
    var report:LinearLayout?=null
    var infolayout:ConstraintLayout?=null
    var connected:Boolean=false
    var connected2:Boolean=false
    var report1:TextView?=null
    var dust:ImageView?=null
    var qtext:TextView?=null
    var lightbtn: Button?=null
    var dustinfo: TextView?=null
    var askmebtn:ImageView?=null
    var camlayout:ConstraintLayout?=null
    var bitmap1: Bitmap?=null
    var bitmap2: Bitmap?=null
    var final_result_dust:String=""
    var final_result_sun1:String=""
    var final_result_sun2:String=""
    var lists=ArrayList<String>()
    var lists2=ArrayList<String>()
    var albumbtn:Button?=null
    var opendoor:Button?=null
    var infotxt:TextView?=null
    var info1:TextView?=null
    var speechRecognizer: SpeechRecognizer? = null
    var camview:ImageView?=null
    var askme:ConstraintLayout?=null
    var tmp_text:TextView?=null
    var htext:TextView?=null
    var cambtn:Button?=null
    var loadicon:ImageView?=null
    var connectui:ConstraintLayout?=null
    var menubar:ConstraintLayout?=null
    var cancel_1:Button?=null
    var album_layout:ConstraintLayout?=null
    var c_q:ConstraintLayout?=null
    var timer4=0
    var albumview:ImageView?=null
    private lateinit var timer2:Timer
    private lateinit var timer3:Timer
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
         var r: Animation = AnimationUtils.loadAnimation(applicationContext, R.anim.rotate);


        val policy = StrictMode.ThreadPolicy.Builder().permitAll().build()
        StrictMode.setThreadPolicy(policy)
        val d = SimpleDateFormat("yyyy-MM-dd")
        val t = SimpleDateFormat("HH:mm:ss")
        currentDate = d.format(Date())
        currentTime = t.format(Date())
        report=findViewById(R.id.report)
        qtext=findViewById(R.id.qtext)
        dust=findViewById(R.id.dust)
        albumview=findViewById(R.id.albumview)
        lightbtn=findViewById(R.id.lightbtn)
        dustinfo=findViewById(R.id.dustinfo)
        askmebtn=findViewById(R.id.askmebtn)
        camlayout=findViewById(R.id.cam_layout)
        camview=findViewById(R.id.camview)
        askme=findViewById(R.id.askme)
        tmp_text=findViewById(R.id.tmp_text)
        htext=findViewById(R.id.htext)
        cambtn=findViewById(R.id.cambtn)
        loadicon=findViewById(R.id.loadicon)
        menubar=findViewById(R.id.menubar)
        connectui=findViewById(R.id.connectui)
        cancel_1=findViewById(R.id.cancel_1)
        info1=findViewById(R.id.info1)
        report1=findViewById(R.id.report1)
        infolayout=findViewById(R.id.info_layout)
        infotxt=findViewById(R.id.info_txt)
        opendoor=findViewById(R.id.opendoor)
        albumbtn=findViewById(R.id.albumbtn)
        album_layout=findViewById(R.id.album_layout)
        get_year=currentDate.substring(0,3).toInt()
        get_month=currentDate.substring(5,7).toInt()
        get_day=currentDate.substring(8,10).toInt()
        get_hh=currentTime.substring(0,2).toInt()
        get_mm=currentTime.substring(3,5).toInt()
        infolayout?.visibility=View.GONE
        album_layout?.visibility=View.GONE
        connect()


        askme?.visibility=View.GONE
        camlayout?.visibility=View.GONE
        askmebtn?.setOnClickListener {
            askme?.visibility=View.VISIBLE
            qtext?.setText("말씀하십시오...")
            ask_command()
        }
        lightbtn?.setOnClickListener {
            light_command()
        }
        cambtn?.setOnClickListener {
            sw1=true
            cam_command()
        }
        report?.setOnClickListener {
            val data2 = "report_"
            outStream?.write(data2.toByteArray())

        }
        cancel_1?.setOnClickListener {
            val data2 = "stopcam"
            outStream?.write(data2.toByteArray())
            sw1=false
            camlayout?.visibility= View.INVISIBLE
        }
        askme?.setOnClickListener {
            askme?.visibility=View.GONE
        }
        infolayout?.setOnClickListener {
            infolayout?.visibility=View.GONE
        }
        opendoor?.setOnClickListener {
            val data2 = "dooropen"
            outStream?.write(data2.toByteArray())
        }
        albumbtn?.setOnClickListener {

            album_layout?.visibility=View.VISIBLE
            album_command()
        }
        album_layout?.setOnClickListener {
            album_layout?.visibility=View.GONE
        }
        val animation = AnimationUtils.loadAnimation(applicationContext, R.anim.rotate)
        loadicon?.startAnimation(animation)



        val scaleAnim = ScaleAnimation(1f, 0.9f, 1f, 0.92f,
            Animation.RELATIVE_TO_SELF, 0.5f, Animation.RELATIVE_TO_SELF, 0.5f).apply {
            duration = 5
            isFillEnabled = true
            fillAfter = true

            val animationListener = object : Animation.AnimationListener {
                override fun onAnimationRepeat(animation: Animation?) {}
                override fun onAnimationStart(animation: Animation?) {}
                override fun onAnimationEnd(animation: Animation?) {}
            }

            setAnimationListener(animationListener)
        }

        loadicon?.startAnimation(scaleAnim)

        send_d()
        Thread {
            while(true){
                Thread.sleep(6000)
                calc_e()
            }
        }.start()


    }

    fun send_d(){
        val data1 = "output_"
        outStream?.write(data1.toByteArray())
    }
    fun calc_e(){
        if(connected2){
            val data1 = "device_"
            outStream?.write(data1.toByteArray())
        }
        if(b1 &&connected){
            val data1 = "light_0"
            outStream?.write(data1.toByteArray())
        }
        if(!b1 &&connected){
            val data1 = "light_1"
            outStream?.write(data1.toByteArray())
        }
    }


    fun connect(){
if(p3==false){


        socket = Socket(ip_address, port)
        outStream = socket?.outputStream
        inStream = socket?.inputStream


        Thread {
            while(true){
                val inputstream=socket?.getInputStream()
                var reponse=ByteArray(4096)
                val data=inputstream?.read(reponse)
                var resulta=reponse.decodeToString()
                if(resulta.isNullOrEmpty()){
                    android.os.Process.killProcess(android.os.Process.myPid());
                }

                if(resulta.startsWith("t")) {
                    result2.add(resulta)

                }
                else if(resulta.startsWith("error")){

                    android.os.Process.killProcess(android.os.Process.myPid());
                }
                else if(resulta.startsWith("ke")){
                    result3.add(resulta)
                }
                else if(resulta.startsWith("event")){
                }
                else if(resulta.contains(("[A-Za-z0-9!]").toRegex())&& sw1==true)
                    {
                        imsi3 = resulta.replace("imsi", "")
                        Log.v("info2", imsi3.toString())
                        lists.add(imsi3)
                        Log.d("listsabc", lists.size.toString())

                    }
                else if(resulta.contains(("[A-Za-z0-9!]").toRegex())&& sw2==true)
                {
                    imsi3 = resulta.replace("imsi", "")
                    Log.v("info22", imsi3.toString())
                    lists2.add(imsi3)
                    Log.d("listsabc2", lists2.size.toString())

                }
                else if(resulta.startsWith("predict")){
                    Log.d("report11","예측 전기비:"+resulta+"원")
                    var result_final="예상 전기비: "+resulta.replace(("[^\\d.]").toRegex(), "").toString()+"원"
                    runOnUiThread {
                        report1?.setText(result_final)
                    }


                    }
                runOnUiThread {
                    if(resulta.startsWith("open_door")){
                        infotxt?.visibility=View.VISIBLE
                        infolayout?.visibility=View.VISIBLE

                        infotxt?.setText("문이 열렸습니다.")
                    }
                }



            }


        }.start()

        connected=true
        connected2=true
        get_dust_data()
       get_sun_data()
        get_temp_hum()
        get_event_log()
    connectui?.visibility=View.GONE
    menubar?.visibility=View.VISIBLE
    p3=true

    }}
    fun get_event_log(){

    }
    fun ask_command(){
if (connected){
    speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this@MainActivity)
    speechRecognizer?.setRecognitionListener(recognitionListener())   
    speechRecognizer?.startListening(intent)
}
    }
fun light_command(){
    if(!b1 &&connected){
        val data1 = "lighton"
        outStream?.write(data1.toByteArray())
        b1=true
    }
    else if(b1 &&connected){
        val data2="lightoff"
        outStream?.write(data2.toByteArray())
        b1=false
    }
}

    fun door_command(){
        if(!b2 &&connected){
            val data1 = "dooropen"
            outStream?.write(data1.toByteArray())
            b2=true
        }
        else if(b2 &&connected){
            val data2 = "doorclose"
            outStream?.write(data2.toByteArray())
            b2=false
        }
    }


    fun album_command(){

        if(connected==true){
            album_layout?.visibility= View.VISIBLE
            val data1 = "album"
            outStream?.write(data1.toByteArray())
            Toast.makeText(applicationContext, "이미지 생성중 잠시 기다려 주십시오..", Toast.LENGTH_SHORT).show()
           sw2=true
        }


        timer3=kotlin.concurrent.timer(period = 1000){
            timer4++
            if(timer4.toInt()>=4) {
                var i=lists2[0].toString()
                val bytearray= Base64.decode(i.toString(), Base64.DEFAULT)
                bitmap1= BitmapFactory.decodeByteArray(bytearray,0,bytearray.size)
                Log.d("testing2143 ",i.toString())

            }



            runOnUiThread {
                albumview?.setImageBitmap(bitmap1)
            }
        }


    }


    fun cam_command(){

        if(connected==true){
            camlayout?.visibility= View.VISIBLE
            val data1 = "cam"
            outStream?.write(data1.toByteArray())

        }


        timer2=kotlin.concurrent.timer(period = 1000){
            timer1++
            sw1=true
            Log.d("timertimer",timer1.toString())


            if(timer1.toInt()>=7&&timer1.toInt()<=95) {
                get_video_frame(imsi4)
                imsi4=imsi4+1

            }
            else if(timer1.toInt()>95&&p4){
                val data2 = "stopcam"
                outStream?.write(data2.toByteArray())
                p4=false
                sw1=false

            }


            runOnUiThread {
                camview?.setImageBitmap(bitmap1)
            }
        }


    }
    fun get_video_frame(num:Int){
        Log.d("" +
                "",num.toString())
        Log.d("vf2",lists.size.toString())

if (num<lists.size){
    var i=lists[num]
    val bytearray= Base64.decode(i.toString(), Base64.DEFAULT)
    bitmap1= BitmapFactory.decodeByteArray(bytearray,0,bytearray.size)
    Log.d("testing214 ",i.toString())

    imsi1=imsi1+1
}


    }



    @SuppressLint("SuspiciousIndentation")

    fun get_dust_data(){
        if(p1==false) {

            val urlBuilder =
                java.lang.StringBuilder() /*URL*/
            urlBuilder.append(
                "?" + URLEncoder.encode(
                    "serviceKey",
                    "UTF-8"
                ) + secure_key
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
            )
            urlBuilder.append(
                "&" + URLEncoder.encode(
                    "pageNo",
                    "UTF-8"
                ) + "=" + URLEncoder.encode("1", "UTF-8")
            ) 
            urlBuilder.append(
                "&" + URLEncoder.encode(
                    "searchDate",
                    "UTF-8"
                ) + "=" + URLEncoder.encode(currentDate, "UTF-8")//2024-05-20
            ) 
            urlBuilder.append(
                "&" + URLEncoder.encode(
                    "InformCode",
                    "UTF-8"
                ) + "=" + URLEncoder.encode("PM10", "UTF-8")
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
            var jsonObject: JSONObject? = JSONObject(sb.toString())
            var response: JSONObject? = jsonObject?.getJSONObject("response")
            var body: JSONObject? = response?.getJSONObject("body")
            var items: JSONArray? = body?.get("items") as JSONArray?

            var r1= items?.getJSONObject(5)
            var r2=r1?.getString("informGrade")



            Log.d("iteminfo", r2.toString())
            val search="대전"
            val idx=r2.toString().indexOf(search)
            final_result_dust=r2.toString().substring(idx+5,idx+7)
            val final_result="미세먼지 "+final_result_dust+" 입니다."
            if(final_result_dust.contains("좋음")){
                dust?.setImageResource(R.drawable.dust_good)
            }
            if(final_result_dust.contains("보통")){
                dust?.setImageResource(R.drawable.dust_w)
            }
            if(final_result_dust.contains("나쁨")){
                dust?.setImageResource(R.drawable.dust_c)
            }
            //좋음, 보통, 나쁨
            dustinfo?.setText(final_result)
            p1=true
        }
        fun start_imgview(){



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
                Log.d("error101",i)
                hum=(i.substring(1,3))
                temp=(i.substring(4,6))
            }
            val temp_result=temp+"도"
            tmp_text?.setText(temp_result)
            val h_result=hum+"%"
            htext?.setText(h_result)
        },3000)


    }

            @SuppressLint("SuspiciousIndentation")
            fun get_sun_data(){
                if(p2==false) {

                    val urlBuilder =
                        java.lang.StringBuilder() 
                    urlBuilder.append(
                        "?" + URLEncoder.encode(
                            "serviceKey",
                            "UTF-8"
                        ) + secure_key
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

                    var final_result="오늘 일출시간은 "+final_result_pre1+"이고 일몰시간은 "+final_result_pre2+"입니다."
                    var calc1=final_result_sun1.substring(0,2).toInt()*60+final_result_sun1.substring(2,4).toInt()
                    var calc2=final_result_sun2.substring(0,2).toInt()*60+final_result_sun2.substring(2,4).toInt()
                    var calc3=get_hh*60+get_mm.toInt()
                    Log.d("calc123 info",calc1.toString()+" "+calc2.toString()+" "+calc3.toString() )
                    if (calc1<calc3&& calc3<calc2){
                        info1?.setText("전등을 끕니다.")
                        val data1="lightoff"
                        outStream?.write(data1.toByteArray())
                        b1=false
                    }
                    else{

                        info1?.setText("전등을 켭니다.")
                        val data1 = "lighton"
                        outStream?.write(data1.toByteArray())
                        b1=true

                    }
                    p2=true
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
            qtext?.setText(stt_result)
            val data1 = "ke"
            var input_txt=data1+(stt_result)+data1
            outStream?.write(input_txt.toByteArray())
            Handler().postDelayed({
                for(i in result3.reversed()) {
                    if(i.contains("ke")){
                        var result= i.replace("ke","")
                        if(result.contains("lighton")||result.contains("lightoff")){
                            light_command()
                            qtext?.setText("전등을 조절하겠습니다.")

                        }
                        else if(result.contains("dooropen")||result.contains("doorclose")){
                            door_command()
                            qtext?.setText("문 여닫이를 조절하겠습니다.")

                        }
                        else if(result.contains("cam")){
                            cam_command()
                            qtext?.setText("외부카메라를 표시하겠습니다.")
                        }
                        else{
                            Toast.makeText(this@MainActivity,"죄송합니다. 명령을 실행하지 못하였습니다.",Toast.LENGTH_LONG).show()
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