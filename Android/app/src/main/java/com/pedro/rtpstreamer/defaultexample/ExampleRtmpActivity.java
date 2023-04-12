/*
 * Copyright (C) 2021 pedroSG94.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.pedro.rtpstreamer.defaultexample;

import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.media.ToneGenerator;
import android.os.Build;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatImageButton;

import android.os.Handler;
import android.os.Looper;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.Toast;

import com.amazonaws.AmazonWebServiceClient;
import com.amazonaws.auth.AWSCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.auth.CognitoCachingCredentialsProvider;
import com.amazonaws.services.sqs.AmazonSQS;
import com.amazonaws.services.sqs.model.DeleteMessageRequest;
import com.amazonaws.services.sqs.model.Message;
import com.amazonaws.services.sqs.model.ReceiveMessageRequest;
import com.pedro.encoder.input.video.CameraOpenException;
import com.pedro.rtmp.utils.ConnectCheckerRtmp;
import com.pedro.rtplibrary.rtmp.RtmpCamera1;
import com.pedro.rtpstreamer.R;
import com.pedro.rtpstreamer.utils.PathUtils;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

import com.amazonaws.services.sqs.AmazonSQSClient;



/**
 * More documentation see:
 * {@link com.pedro.rtplibrary.base.Camera1Base}
 * {@link com.pedro.rtplibrary.rtmp.RtmpCamera1}
 */
public class ExampleRtmpActivity extends AppCompatActivity
        implements ConnectCheckerRtmp, View.OnClickListener, SurfaceHolder.Callback {

  private RtmpCamera1 rtmpCamera1;
  private ImageButton button;
  private ImageButton bRecord;
  private EditText etUrl;

  private String currentDateAndTime = "";
  private File folder;

  private Thread sqsThread;

  private SQSClient sqsClient;




  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    setContentView(R.layout.activity_example);
    folder = PathUtils.getRecordPath();
    SurfaceView surfaceView = findViewById(R.id.surfaceView);
    button = findViewById(R.id.b_start_stop);
    button.setOnClickListener(this);
    bRecord = findViewById(R.id.b_record);
    bRecord.setOnClickListener(this);
    ImageButton switchCamera = findViewById(R.id.switch_camera);
    switchCamera.setOnClickListener(this);
    etUrl = findViewById(R.id.et_rtp_url);
    etUrl.setHint(R.string.hint_rtmp);
    rtmpCamera1 = new RtmpCamera1(surfaceView, this);
    rtmpCamera1.setReTries(10);
    surfaceView.getHolder().addCallback(this);
  }

  private void playTone(float beatsPerSecond, float leftVolume, float rightVolume) {
    AudioTrack audioTrack = null;
    try {
      // Set up audio track to play stereo audio
      int sampleRate = 44100;
      int channelConfig = AudioFormat.CHANNEL_OUT_STEREO;
      int audioFormat = AudioFormat.ENCODING_PCM_16BIT;
      int bufferSize = 8192; // or a higher value
      audioTrack = new AudioTrack(
              AudioManager.STREAM_MUSIC,
              sampleRate,
              channelConfig,
              audioFormat,
              bufferSize,
              AudioTrack.MODE_STREAM
      );
      audioTrack.play();

      // Generate audio data for left and right channels
      int samplesPerBeat = (int) (sampleRate / beatsPerSecond);
      Object lock = new Object(); // create a lock
      boolean hasNewBpm = false; // initialize to false
      float newBpmValue = beatsPerSecond; // initialize to current bpm
      while (true) {
        synchronized (lock) { // acquire the lock to prevent changes to beatsPerSecond while generating audio
          for (int i = 0; i < samplesPerBeat; i++) {
            // Generate a square wave at the frequency of the tone
            float value = (i % samplesPerBeat < samplesPerBeat / 2) ? 1.0f : -1.0f;

            // Increase the amplitude of the tone
            value *= 2;

            // Interleave samples for left and right channels
            short leftSample = (short) (value * 32767 * leftVolume);
            short rightSample = (short) (value * 32767 * rightVolume);
            byte[] buffer = new byte[4];
            buffer[0] = (byte) (leftSample & 0xff);
            buffer[1] = (byte) ((leftSample >> 8) & 0xff);
            buffer[2] = (byte) (rightSample & 0xff);
            buffer[3] = (byte) ((rightSample >> 8) & 0xff);
            audioTrack.write(buffer, 0, 4);
          }

          // Check if a new message has arrived
          String messageValue = sqsClient.readFromQueue();
          if (messageValue != null && !messageValue.isEmpty()) {
            System.out.println("messageValue");
            System.out.println(messageValue);
            // Parse message value as float and update the beatsPerSecond value
            synchronized (lock) { // acquire the lock before changing beatsPerSecond
              hasNewBpm = true;
              newBpmValue = Float.parseFloat(messageValue);
            }
          }

          // Check if there is a new BPM value to use
          if (hasNewBpm) {
            // Update the samples per beat value
            samplesPerBeat = (int) (sampleRate / newBpmValue);
            // Reset the flag and value
            hasNewBpm = false;
            newBpmValue = beatsPerSecond;
          }
        }
      }
    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      if (audioTrack != null) {
        audioTrack.stop();
        audioTrack.release();
      }
    }
  }

  private Thread currentToneThread = null;

  private void startSQSListener() {
    // Start a new thread to listen to the SQS queue
    sqsClient = new SQSClient();
    sqsThread = new Thread(new Runnable() {
      @Override
      public void run() {
        while (true) {
          String messageValue = sqsClient.readFromQueue();
          if (messageValue != null && !messageValue.isEmpty()) {
            System.out.println("messageValue");
            System.out.println(messageValue);
            // Parse message value as float and start playing tone in a new thread
            float beatsPerSecond;
            float leftVolume;
            float rightVolume;
            if (messageValue.equals("-1")) {
              beatsPerSecond = 1f;
              leftVolume = 0f;
              rightVolume = 0f;
            } else {
              beatsPerSecond = Float.parseFloat(messageValue);
              leftVolume = 0.5f;
              rightVolume = 0.5f;
            }
            Thread toneThread = new Thread(new Runnable() {
              @Override
              public void run() {
                playTone(beatsPerSecond, leftVolume, rightVolume);
              }
            });

            // Stop the current tone thread if it exists and start the new one
            if (currentToneThread != null) {
              currentToneThread.interrupt();
            }
            currentToneThread = toneThread;
            toneThread.start();
          }
        }
      }
    });
    sqsThread.start();
  }

  @Override
  public void onConnectionStartedRtmp(String rtmpUrl) {
  }

  @Override
  public void onConnectionSuccessRtmp() {
    runOnUiThread(new Runnable() {
      @Override
      public void run() {
        Toast.makeText(ExampleRtmpActivity.this, "Connection success", Toast.LENGTH_SHORT).show();
      }
    });
  }

  @Override
  public void onConnectionFailedRtmp(final String reason) {
    runOnUiThread(new Runnable() {
      @Override
      public void run() {
        if (rtmpCamera1.reTry(5000, reason, null)) {
          Toast.makeText(ExampleRtmpActivity.this, "Retry", Toast.LENGTH_SHORT)
                  .show();
        } else {
          Toast.makeText(ExampleRtmpActivity.this, "Connection failed. " + reason, Toast.LENGTH_SHORT)
                  .show();
          rtmpCamera1.stopStream();
//          button.setText(R.string.start_button);
        }
      }
    });
  }

  @Override
  public void onNewBitrateRtmp(final long bitrate) {

  }

  @Override
  public void onDisconnectRtmp() {
    runOnUiThread(new Runnable() {
      @Override
      public void run() {
        Toast.makeText(ExampleRtmpActivity.this, "Disconnected", Toast.LENGTH_SHORT).show();
        button.setImageResource(R.drawable.play50);

      }
    });
  }

  @Override
  public void onAuthErrorRtmp() {
    runOnUiThread(new Runnable() {
      @Override
      public void run() {
        Toast.makeText(ExampleRtmpActivity.this, "Auth error", Toast.LENGTH_SHORT).show();
        rtmpCamera1.stopStream();
//        button.setText(R.string.start_button);
      }
    });
  }

  @Override
  public void onAuthSuccessRtmp() {
    runOnUiThread(new Runnable() {
      @Override
      public void run() {
        Toast.makeText(ExampleRtmpActivity.this, "Auth success", Toast.LENGTH_SHORT).show();
      }
    });
  }


  @Override
  public void onClick(View view) {
    switch (view.getId()) {
      case R.id.b_start_stop:
        if (!rtmpCamera1.isStreaming()) {
          if (rtmpCamera1.isRecording()
                  || rtmpCamera1.prepareAudio() && rtmpCamera1.prepareVideo()) {
//            button.setText(R.string.stop_button);

            /**
             *  Audio with multithreading here
             */
            startSQSListener();

            rtmpCamera1.startStream("rtmp://18.142.92.149/LiveApp/RenaissanceCapstone");
            button.setImageResource(R.drawable.stop);

          } else {
            Toast.makeText(this, "Error preparing stream, This device cant do it",
                    Toast.LENGTH_SHORT).show();
          }
        } else {
//          button.setText(R.string.start_button);
          rtmpCamera1.stopStream();
        }
        break;
      case R.id.switch_camera:
        try {
          rtmpCamera1.switchCamera();
        } catch (CameraOpenException e) {
          Toast.makeText(this, e.getMessage(), Toast.LENGTH_SHORT).show();
        }
        break;
      case R.id.b_record:
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR2) {
          if (!rtmpCamera1.isRecording()) {
            try {
              if (!folder.exists()) {
                folder.mkdir();
              }
              SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault());
              currentDateAndTime = sdf.format(new Date());
              if (!rtmpCamera1.isStreaming()) {
                if (rtmpCamera1.prepareAudio() && rtmpCamera1.prepareVideo()) {
                  rtmpCamera1.startRecord(
                          folder.getAbsolutePath() + "/" + currentDateAndTime + ".mp4");
//                  bRecord.setText(R.string.stop_record);
                  Toast.makeText(this, "Recording... ", Toast.LENGTH_SHORT).show();
                } else {
                  Toast.makeText(this, "Error preparing stream, This device cant do it",
                          Toast.LENGTH_SHORT).show();
                }
              } else {
                rtmpCamera1.startRecord(
                        folder.getAbsolutePath() + "/" + currentDateAndTime + ".mp4");
//                bRecord.setText(R.string.stop_record);
                Toast.makeText(this, "Recording... ", Toast.LENGTH_SHORT).show();
              }
            } catch (IOException e) {
              rtmpCamera1.stopRecord();
              PathUtils.updateGallery(this, folder.getAbsolutePath() + "/" + currentDateAndTime + ".mp4");
//              bRecord.setText(R.string.start_record);
              Toast.makeText(this, e.getMessage(), Toast.LENGTH_SHORT).show();
            }
          } else {
            rtmpCamera1.stopRecord();
            PathUtils.updateGallery(this, folder.getAbsolutePath() + "/" + currentDateAndTime + ".mp4");
//            bRecord.setText(R.string.start_record);
            Toast.makeText(this,
                    "file " + currentDateAndTime + ".mp4 saved in " + folder.getAbsolutePath(),
                    Toast.LENGTH_SHORT).show();
          }
        } else {
          Toast.makeText(this, "You need min JELLY_BEAN_MR2(API 18) for do it...",
                  Toast.LENGTH_SHORT).show();
        }
        break;
      default:
        break;
    }
  }

  @Override
  public void surfaceCreated(SurfaceHolder surfaceHolder) {

  }

  @Override
  public void surfaceChanged(SurfaceHolder surfaceHolder, int i, int i1, int i2) {
    rtmpCamera1.startPreview();
  }

  @Override
  public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR2 && rtmpCamera1.isRecording()) {
      rtmpCamera1.stopRecord();
      PathUtils.updateGallery(this, folder.getAbsolutePath() + "/" + currentDateAndTime + ".mp4");
//      bRecord.setText(R.string.start_record);
      Toast.makeText(this,
              "file " + currentDateAndTime + ".mp4 saved in " + folder.getAbsolutePath(),
              Toast.LENGTH_SHORT).show();
      currentDateAndTime = "";
    }
    if (rtmpCamera1.isStreaming()) {
      rtmpCamera1.stopStream();
//      button.setText(getResources().getString(R.string.start_button));
    }
    rtmpCamera1.stopPreview();
  }
}