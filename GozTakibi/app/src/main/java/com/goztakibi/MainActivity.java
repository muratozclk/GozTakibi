package com.goztakibi;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.video.Video;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.SeekBar.OnSeekBarChangeListener;

public class MainActivity extends Activity implements CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    public static final int JAVA_DETECTOR = 0;
    private static final int TM_SQDIFF = 0;
    private static final int TM_SQDIFF_NORMED = 1;
    private MenuItem mItemYuz50;
    private MenuItem mItemYuz40;
    private MenuItem mItemYuz30;
    private MenuItem mItemYuz20;
    private MenuItem mItemTıp;
    private Mat mRgba;
    private Mat mGray;
    private Mat sagPen;
    private Mat solPen;
    private File mCascadeFile;
    private CascadeClassifier mJavaDetector;
    private CascadeClassifier mJavaDetectorEye;
    private int mDetectorType = JAVA_DETECTOR;
    private String[] mDetectorName;
    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;
    private CameraBridgeViewBase ekran;
    private  int method = 0;
    double x = -1;
    double y = -1;
    private Mat sablonR;
    private Mat sablonL;
    private int cerceve = 0;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    try {
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);
                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();
                        InputStream iser = getResources().openRawResource(R.raw.haarcascade_lefteye_2splits);
                        File cascadeDirER = getDir("cascadeER", Context.MODE_PRIVATE);
                        File cascadeFileER = new File(cascadeDirER, "haarcascade_eye_right.xml");
                        FileOutputStream oser = new FileOutputStream(cascadeFileER);
                        byte[] buff = new byte[4096];
                        int bytesReadER;
                        while ((bytesReadER = iser.read(buff)) != -1) {
                            oser.write(buff, 0, bytesReadER);
                        }
                        iser.close();
                        oser.close();
                        mJavaDetector = new CascadeClassifier(
                                mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
                        mJavaDetectorEye = new CascadeClassifier(cascadeFileER.getAbsolutePath());
                        if (mJavaDetectorEye.empty()) {Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetectorEye = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
                    ekran.setCameraIndex(1);
                    ekran.enableFpsMeter();
                    ekran.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public MainActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        Log.i(TAG, "Instantiated new " + this.getClass());
    }
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        ekran = (CameraBridgeViewBase) findViewById(R.id.activity_main);
        ekran.setCvCameraViewListener(this);
    }
    @Override
    public void onPause() {
        super.onPause();

        if (ekran!= null)
            ekran.disableView();
    }
    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }
    public void onDestroy() {
        super.onDestroy();
        ekran.disableView();
    }
    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
        sagPen.release();
        solPen.release();
    }
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }
        if (sagPen==null || solPen==null)
            matrisOlusturma();
        MatOfRect yüzler = new MatOfRect();
        if (mJavaDetector != null)
            mJavaDetector.detectMultiScale(mGray, yüzler, 1.1, 2, 2, new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        Rect[] yüzDizisi = yüzler.toArray();
        Rect gozcevresi_sag=new Rect();
        Rect gozcevresi_sol=new Rect();
        for (int i = 0; i < yüzDizisi.length; i++) {
            Core.rectangle(mRgba, yüzDizisi[i].tl(), yüzDizisi[i].br(), FACE_RECT_COLOR, 2);
            x = (yüzDizisi[i].x + yüzDizisi[i].width + yüzDizisi[i].x) / 2;
            y = (yüzDizisi[i].y + yüzDizisi[i].y + yüzDizisi[i].height) / 2;
            Point ortaNokta = new Point(x,y);
            Core.circle(mRgba,ortaNokta, 10, new Scalar(255, 0, 0, 255), 1);
            Rect rect = yüzDizisi[i];
            Rect gozcevresi = new Rect(rect.x + rect.width / 8, (int) (rect.y + (rect.height / 4.5)), rect.width - 2 * rect.width / 8, (int) (rect.height / 3.0));
            gozcevresi_sag = new Rect(rect.x + rect.width / 16, (int) (rect.y + (rect.height / 4.5)), (rect.width - 2 * rect.width / 16) / 2, (int) (rect.height / 3.0));
            gozcevresi_sol = new Rect(rect.x + rect.width / 16 + (rect.width - 2 * rect.width / 16) / 2, (int) (rect.y + (rect.height / 4.5)), (rect.width - 2 * rect.width / 16) / 2, (int) (rect.height / 3.0));
            Core.rectangle(mRgba, gozcevresi_sol.tl(), gozcevresi_sol.br(), new Scalar(0, 0, 255, 255), 2); // iki göz için çevreler
            Core.rectangle(mRgba, gozcevresi_sag.tl(), gozcevresi_sag.br(), new Scalar(0, 0, 255, 255), 2);
            if (cerceve <150000000) {
                sablonR = ayarla(mJavaDetectorEye, gozcevresi_sag, 24);
                sablonL = ayarla(mJavaDetectorEye, gozcevresi_sol, 24);
                cerceve++;
            } else {
                karsilastirma(gozcevresi_sag, sablonR, method);
                karsilastirma(gozcevresi_sol, sablonL, method);
            }
            Imgproc.resize(mRgba.submat(gozcevresi_sol), solPen, solPen.size());
            Imgproc.resize(mRgba.submat(gozcevresi_sag), sagPen, sagPen.size());
        }
        return mRgba;
    }
    public void matrisOlusturma() {
        if (mGray.empty())
            return;
        int rows = mGray.rows();
        int cols = mGray.cols();
        if (solPen == null) {
            sagPen= mRgba.submat(0, rows / 2 - rows / 10, 0,cols / 3 + cols / 10);
            solPen= mRgba.submat(0, rows / 2 - rows / 10, cols / 2 + cols / 10, cols);
        }
    }
    private Mat ayarla(CascadeClassifier cascade, Rect rect, int boyut) {
        Mat matris = new Mat();
        Mat mat = mGray.submat(rect);
        MatOfRect goz = new MatOfRect();
        Point point = new Point();
        Rect  gozb= new Rect();
        cascade.detectMultiScale(mat, goz, 1.15, 2, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());
        Rect[] gozDizisi = goz.toArray();
        for (int i = 0; i < gozDizisi.length;) {
            Rect e = gozDizisi[i];
            e.x = rect.x + e.x;
            e.y = rect.y + e.y;
            Rect rectGoz = new Rect((int) e.tl().x, (int) (e.tl().y + e.height * 0.4), (int) e.width, (int) (e.height * 0.6));
            mat = mGray.submat(rectGoz);
            Mat matr = mRgba.submat(rectGoz);
            Core.MinMaxLocResult sonuc = Core.minMaxLoc(mat);
            Core.circle(matr, sonuc.minLoc, 2, new Scalar(255, 255, 255, 255), 5);
            point.x = sonuc.minLoc.x + rectGoz.x;
            point.y = sonuc.minLoc.y + rectGoz.y;
            gozb = new Rect((int) point.x - boyut / 2, (int) point.y - boyut / 2, boyut, boyut);
            Core.rectangle(mRgba, gozb.tl(),gozb.br(),new Scalar(255, 0, 0, 255), 2);
            matris = (mGray.submat(gozb)).clone();
            return matris;
        }
        return matris;
    }
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemYuz50 = menu.add("Yuz boyutu 50%");
        mItemYuz40 = menu.add("Yuz boyutu 40%");
        mItemYuz30 = menu.add("Yuz boyutu 30%");
        mItemYuz20 = menu.add("Yuz boyutu 20%");
        mItemTıp = menu.add(mDetectorName[mDetectorType]);
        return true;
    }
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item == mItemYuz50)
            setMinFaceSize(0.5f);
        else if (item == mItemYuz40)
            setMinFaceSize(0.4f);
        else if (item == mItemYuz30)
            setMinFaceSize(0.3f);
        else if (item == mItemYuz20)
            setMinFaceSize(0.2f);
        else if (item == mItemTıp) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
        }
        return true;
    }
    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void karsilastirma(Rect rect, Mat matris, int tip) {
        Point point;
        Mat mat = mGray.submat(rect);
        int sutun = mat.cols() - matris.cols() + 1;
        int satir = mat.rows() - matris.rows() + 1;
        if (matris.cols() == 0 || matris.rows() == 0) {
            return ;
        }
        Mat sonuc = new Mat(sutun,satir, CvType.CV_8U);
        Core.MinMaxLocResult sonmat = Core.minMaxLoc(sonuc);
        if (tip == TM_SQDIFF || tip == TM_SQDIFF_NORMED) {
            point = sonmat.minLoc;
        } else {
            point = sonmat.maxLoc;
        }
        Point x = new Point(point.x + rect.x,point.y + rect.y);
        Point y = new Point(point.x + matris.cols() + rect.x, point.y + matris.rows() + rect.y);
        Core.rectangle(mRgba,x, y, new Scalar(255, 255, 0, 255));
        Rect rec = new Rect(x,y);
    }
}

