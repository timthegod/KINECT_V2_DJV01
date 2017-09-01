#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cstdio>
#include <string>
#include <chrono>
#include <typeinfo>
#include <thread>
#include <time.h>
#include <math.h>
#include <windows.h>

// Audio process lib Header
#include <playerUnit.h>
// Frame process lib header
#include <button.h>

// OpenNI Header
#include <OpenNI.h>

// OpenCV Header
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace openni;

const Scalar debugColor0(0, 0, 128);
const Scalar debugColor1(84, 140, 29);
const Scalar debugColor2(255, 255, 255);
// set colors
const Scalar COLOR_BLUE = Scalar(240, 40, 0);
const Scalar COLOR_DARK_GREEN = Scalar(0, 128, 0);
const Scalar COLOR_LIGHT_GREEN = Scalar(0, 255, 0);
const Scalar COLOR_YELLOW = Scalar(0, 128, 200);
const Scalar COLOR_RED = Scalar(0, 0, 255);
const Scalar COLOR_D75B66 = Scalar(102, 91, 215);
const Scalar COLOR_23345C = Scalar(92, 52, 35);     //rgb(35, 52, 92)
const Scalar COLOR_F1BA48 = Scalar(72, 186, 241);   //rgb(241, 186, 72)
const Scalar COLOR_BD8A44 = Scalar(68, 138, 189);   //rgb(189, 138, 68)
const Scalar COLOR_BFAFA0 = Scalar(160, 175, 191);  //rgb(191, 175, 160)

const int cDepthWidth = 512;
const int cDepthHeight = 424;
const int cColorWidth = 1920;
const int cColorHeight = 1080;
const double debugFrameMaxDepth = 4000;

bool          training = true;
const unsigned int nBackgroundTrain = 50;
const unsigned short touchDepthMin = 5;
const unsigned short touchDepthMax = 17;
const unsigned int touchMinArea = 7;
int xMin = 70;
int xMax = 380;
int yMin = 200;
int yMax = 380;
int roi_width = xMax - xMin;
int roi_height = yMax - yMin;
int           iBackgroundTrain = -1;

Mat1s         foreground(cDepthHeight, cDepthWidth);
Mat1b         touchMask(cDepthWidth, cDepthHeight);// touch mask
Mat1s         background(cDepthHeight, cDepthWidth);
vector<Mat1s> buffer(nBackgroundTrain);

void average(vector<Mat1s>& frames, Mat1s& mean) {
    Mat1d acc(mean.size());
    Mat1d frame(mean.size());
    acc = 0.0;
    for (unsigned int i = 0; i<frames.size(); i++) {
        frames[i].convertTo(frame, CV_64FC1);
        acc = acc + frame;
    }
    acc = acc / frames.size();
    acc.convertTo(mean, CV_16SC1);
}

int main()
{
    Mat1s depth_rawFrame(cDepthHeight, cDepthWidth);
    Mat1b depth_8bitFrame(cDepthHeight, cDepthWidth);
    Mat   depth_to_bgr_rawFrame(cDepthHeight, cDepthWidth, CV_8UC3);
    Mat   depth_bgrFrame(cDepthHeight, cDepthWidth, CV_8UC3);
    Mat   colorShow(cDepthHeight, cDepthWidth, CV_8UC3);
    Mat   colorShow_hand(cDepthHeight, cDepthWidth, CV_8UC3);
    
    static int cueArray[10] = { -1, -1 , -1 , -1, -1, -1 , -1 , -1 , -1 , -1 };
    playerUnit player;
    static VinylSample* music1 = new VinylSample("D:/workspace/VS_projects/DJ_V01_KINECTV2_OPENNI/DJ_V01_KINECTV2_OPENNI/audio/Message.wav");

    stereoEffects eqEffect1;
    EFFECTGENERATOR(eq3BandEffect, eq1, eq3BandEffect::MODE::NORMAL_EQ3BAND);
    COMBINEEFFECT(eqEffect1, eq1);
    music1->addEffect(&eqEffect1);

    stereoEffects delayEffect1;
    EFFECTGENERATOR(delayEffect, delay1, delayEffect::MODE::NORMAL_DELAY);
    COMBINEEFFECT(delayEffect1, delay1);
    music1->addEffect(&delayEffect1);

    SETEFFECT_STATICPTR(delay1, setPower(Effect::POWER::OFF));
    SETEFFECT_STATICPTR(eq1, setPower(Effect::POWER::ON));
    SETEFFECT_STATICPTR(delay1, setFeedback(0.8));

    music1->setState(AudioSample::STATE::PAUSING);
    player.addSample(music1);
    //Initial OpenNI
    if (OpenNI::initialize() != STATUS_OK) {
        cerr << "OpenNI Initial Error: " << OpenNI::getExtendedError() << endl;
        return -1;
    }
    Device device;
    VideoStream depthStream;
    VideoStream depth_to_colorStream;
    VideoFrameRef  depthFrame;
    VideoFrameRef  depth_to_colorFrame;
    if (device.open(ANY_DEVICE) != STATUS_OK) {
        cerr << "Can't Open Device: " << OpenNI::getExtendedError() << endl;
        OpenNI::shutdown();
        return -1;
    }

    //Create depth stream  
    if (device.hasSensor(SENSOR_DEPTH)) {
        if (depthStream.create(device, SENSOR_DEPTH) == STATUS_OK) {
            const openni::SensorInfo* sinfo = device.getSensorInfo(openni::SENSOR_DEPTH); // select index=4 640x480, 30 fps, 1mm
            const openni::Array< openni::VideoMode>& modesDepth = sinfo->getSupportedVideoModes();
            if (depthStream.setVideoMode(modesDepth[0]) != STATUS_OK) {
                cout << "Can't apply VideoMode: " << OpenNI::getExtendedError() << endl;
            }
        }
        else {
            cerr << "Can't create depth stream on device: " << OpenNI::getExtendedError() << endl;
            return -1;
        }
    }
    else {
        cerr << "ERROR: This device does not have depth sensor" << endl;
        return -1;
    }
    //Create color stream
    if (device.hasSensor(SENSOR_COLOR)) {
        if (depth_to_colorStream.create(device, SENSOR_COLOR) == STATUS_OK) {
            const openni::SensorInfo* sinfo = device.getSensorInfo(openni::SENSOR_COLOR); // select index=4 640x480, 30 fps, 1mm
            const openni::Array< openni::VideoMode>& modesColor = sinfo->getSupportedVideoModes();
            if (depth_to_colorStream.setVideoMode(modesColor[1]) != STATUS_OK) {
                cout << "Can't apply VideoMode: " << OpenNI::getExtendedError() << endl;
            }
            //image registration 
            //設定自動影像校準技術(深度與彩圖整合)
            //http://www.terasoft.com.tw/support/techpdf/Automating%20Image%20Registration%20with%20MATLAB.pdf
            if (device.isImageRegistrationModeSupported(IMAGE_REGISTRATION_DEPTH_TO_COLOR)) {
                device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
            }
            else {
                cerr << "Can't set ImageRegistration Mode." << endl;
            }
        }
        else {
            cerr << "Can't create depth_to_color stream on device: " << OpenNI::getExtendedError() << endl;
            return -1;
        }
    }
    depthStream.start();
    depth_to_colorStream.start();
    player.startAudio();

     //-------------Add Button-----------------
    vector<Buttons*> vButtons;
    vButtons.push_back(new RectButton("RectButton testing", cv::Point((int)(0.2*roi_width+xMin), (int)(0.3*roi_height+yMin )), 0.07*roi_width, 0.07*roi_width,
        [](Buttons::BUTTON_STATE state, cv::Point point, float height, float width, cv::Mat& target) {
            switch (state)
            {
            case Buttons::PRESSED:
                cv::rectangle(target, Rect(point, Size(width, height)), COLOR_D75B66, 3);
                break;

            case Buttons::UNPRESSED:
            {
                cv::rectangle(target, Rect(point, Size(width, height)), COLOR_23345C, 3);
            }
            break;
            }
        }, 
        [](Buttons::BUTTON_STATE state) {cout << "RectButton1" << endl; }));

    vButtons.push_back(new RectButton("RectButton testing", cv::Point((int)(0.2*roi_width + xMin), (int)(0.3*roi_height + 1 * 0.07*roi_width + yMin)), 0.07*roi_width, 0.07*roi_width,
        [](Buttons::BUTTON_STATE state, cv::Point point, float height, float width, cv::Mat& target) {
        switch (state)
        {
        case Buttons::PRESSED:
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_D75B66, 3);
            break;

        case Buttons::UNPRESSED:
        {
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_23345C, 3);
        }
        break;
        }
    },
        [](Buttons::BUTTON_STATE state) {cout << "RectButton2" << endl; }));

    vButtons.push_back(new RectButton("RectButton testing", cv::Point((int)(0.2*roi_width + xMin), (int)(0.3*roi_height + 2 * 0.07*roi_width + yMin)), 0.07*roi_width, 0.07*roi_width,
        [](Buttons::BUTTON_STATE state, cv::Point point, float height, float width, cv::Mat& target) {
        switch (state)
        {
        case Buttons::PRESSED:
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_D75B66, 3);
            break;

        case Buttons::UNPRESSED:
        {
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_23345C, 3);
        }
        break;
        }
    },
        [](Buttons::BUTTON_STATE state) {cout << "RectButton3" << endl; }));
    vButtons.push_back(new RectButton("RectButton testing", cv::Point((int)(0.2*roi_width + xMin), (int)(0.3*roi_height + 3 * 0.07*roi_width + yMin)), 0.07*roi_width, 0.07*roi_width,
        [](Buttons::BUTTON_STATE state, cv::Point point, float height, float width, cv::Mat& target) {
        switch (state)
        {
        case Buttons::PRESSED:
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_D75B66, 3);
            break;

        case Buttons::UNPRESSED:
        {
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_23345C, 3);
        }
        break;
        }
    },
        [](Buttons::BUTTON_STATE state) {cout << "RectButton4" << endl; }));

    vButtons.push_back(new RectButton("RectButton testing", cv::Point((int)((0.5 - 3.5/20)*roi_width + xMin), (int)(0.9*roi_height + yMin)) , 0.07*roi_width, 0.07*roi_width,
        [](Buttons::BUTTON_STATE state, cv::Point point, float height, float width, cv::Mat& target) {
        switch (state)
        {
        case Buttons::PRESSED:
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_D75B66, 3);
            break;

        case Buttons::UNPRESSED:
        {
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_23345C, 3);
        }
        break;
        }
    },
        [](Buttons::BUTTON_STATE state) {
        music1->setCuetoPosition(cueArray[1]);
    }));
    vButtons.push_back(new RectButton("RectButton testing", cv::Point((int)((0.5 - 2.1 / 20)*roi_width + xMin), (int)(0.9*roi_height + yMin)), 0.07*roi_width, 0.07*roi_width,
        [](Buttons::BUTTON_STATE state, cv::Point point, float height, float width, cv::Mat& target) {
        switch (state)
        {
        case Buttons::PRESSED:
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_D75B66, 3);
            break;

        case Buttons::UNPRESSED:
        {
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_23345C, 3);
        }
        break;
        }
    },
        [](Buttons::BUTTON_STATE state) {
        music1->setCuetoPosition(cueArray[0]);
    }));
    vButtons.push_back(new RectButton("RectButton testing", cv::Point((int)((0.5 - 0.7 / 20)*roi_width + xMin), (int)(0.9*roi_height + yMin)), 0.07*roi_width, 0.07*roi_width,
        [](Buttons::BUTTON_STATE state, cv::Point point, float height, float width, cv::Mat& target) {
        switch (state)
        {
        case Buttons::PRESSED:
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_D75B66, 3);
            break;

        case Buttons::UNPRESSED:
        {
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_23345C, 3);
        }
        break;
        }
    },
        [](Buttons::BUTTON_STATE state) {
        if (cueArray[0] == -1) {
            cueArray[0] = music1->addCue();
        }
        else {
            music1->updateCuePosition(cueArray[0]);
        }
    }));
    vButtons.push_back(new RectButton("RectButton testing", cv::Point((int)((0.5 + 0.7 / 20)*roi_width + xMin), (int)(0.9*roi_height + yMin)), 0.07*roi_width, 0.07*roi_width,
        [](Buttons::BUTTON_STATE state, cv::Point point, float height, float width, cv::Mat& target) {
        switch (state)
        {
        case Buttons::PRESSED:
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_D75B66, 3);
            break;

        case Buttons::UNPRESSED:
        {
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_23345C, 3);
        }
        break;
        }
    },
        [](Buttons::BUTTON_STATE state) {/*cueArray[1] = music1->addCue();*/}));
    vButtons.push_back(new RectButton("RectButton testing", cv::Point((int)((0.5 + 2.1 / 20)*roi_width + xMin), (int)(0.9*roi_height + yMin)), 0.07*roi_width, 0.07*roi_width,
        [](Buttons::BUTTON_STATE state, cv::Point point, float height, float width, cv::Mat& target) {
        switch (state)
        {
        case Buttons::PRESSED:
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_D75B66, 3);
            break;

        case Buttons::UNPRESSED:
        {
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_23345C, 3);
        }
        break;
        }
    },
        [](Buttons::BUTTON_STATE state) {cout << "RectButton9" << endl; }));
    vButtons.push_back(new RectButton("RectButton testing", cv::Point((int)((0.5 + 3.5 / 20)*roi_width + xMin), (int)(0.9*roi_height + yMin)), 0.07*roi_width, 0.07*roi_width,
        [](Buttons::BUTTON_STATE state, cv::Point point, float height, float width, cv::Mat& target) {
        switch (state)
        {
        case Buttons::PRESSED:
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_D75B66, 3);
            if (delay1_1->getPower() == Effect::POWER::OFF) {
                SETEFFECT_STATICPTR(delay1, setPower(Effect::POWER::ON));
            }
            else {
                SETEFFECT_STATICPTR(delay1, setPower(Effect::POWER::OFF));
            }
            break;
        case Buttons::UNPRESSED:
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_23345C, 3);
            break;
        }
    },
        [](Buttons::BUTTON_STATE state) {cout << "RectButton10" << endl; }));

    vButtons.push_back(new ScrollBarButton("ScrollBarButton testing", cv::Point((int)(0.1*roi_width + xMin), (int)(0.5*roi_height + yMin)), 0.7*roi_height,0.05*roi_width, 
        [](Buttons::BUTTON_STATE state, cv::Point point, float height, float width,int offset_y, cv::Mat& target) {
        switch (state)
        {
        case Buttons::PRESSED:
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_BD8A44, 3);  //SCROLL BAR
            //  1/2*height - 1/12*height = 5/12*height 不然小方塊會超過範圍
            cv::rectangle(target, Rect(point.x, point.y + 5*height / 12  + offset_y, width, height / 6), COLOR_BD8A44, 3); //BUTTON  the rate is 6 
            //std::cout << 50 - (offset_y / height) * 100 << "%" << endl;
            break;

        case Buttons::UNPRESSED:
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_BFAFA0, 3);  //SCROLL BAR
            cv::rectangle(target, Rect(point.x,point.y+ 5*height/12 + offset_y, width, height/6), COLOR_BFAFA0, 3);  //BUTTON  the rate is 6 
            break;
        }
    },
        [](Buttons::BUTTON_STATE state, double delta) {
        if (state == Buttons::BUTTON_STATE::PRESSED) {
            //cout << "Original" << delta << endl;
            delta -= 50.0;
            delta /= 100.0;
            music1->setSpeed(1.0 + delta);
            //cout << 1.0 + delta << endl;
        }
    }));
    vButtons.push_back(new ScrollBarButton("ScrollBarButton testing", cv::Point((int)((0.75 + 1/20)*roi_width + xMin), (int)(0.25*roi_height + yMin)), 0.25*roi_height, 0.05*roi_width,
        [](Buttons::BUTTON_STATE state, cv::Point point, float height, float width, int offset_y, cv::Mat& target) {
        switch (state)
        {
        case Buttons::PRESSED:
        {
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_BD8A44, 3);  //SCROLL BAR
            //  1/2*height - 1/12*height = 5/12*height 不然小方塊會歪
            cv::rectangle(target, Rect(point.x, point.y + 5 * height / 12 + offset_y, width, height / 6), COLOR_BD8A44, 3); //BUTTON  the rate is 6 
            //std::cout << 50 - (offset_y / height) * 100 << "%";
        }
        break;

        case Buttons::UNPRESSED:
        {
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_BFAFA0, 3);  //SCROLL BAR
            cv::rectangle(target, Rect(point.x, point.y + 5 * height / 12 + offset_y, width, height / 6), COLOR_BFAFA0, 3);  //BUTTON  the rate is 6 
        }
        break;
        }
    },
        [](Buttons::BUTTON_STATE state, double delta) {
        delta -= 50.0;
        delta /= 50.0;
        SETEFFECT_STATICPTR(eq1, setLowGain(1.0+delta));
        //cout << 1.0 + delta << endl;
    }));
    vButtons.push_back(new ScrollBarButton("ScrollBarButton testing", cv::Point((int)((0.75 + 1.5/20)*roi_width + xMin), (int)(0.25*roi_height + yMin)), 0.25*roi_height, 0.05*roi_width,
        [](Buttons::BUTTON_STATE state, cv::Point point, float height, float width, int offset_y, cv::Mat& target) {
        switch (state)
        {
        case Buttons::PRESSED:
        {
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_BD8A44, 3);  //SCROLL BAR
            //  1/2*height - 1/12*height = 5/12*height 不然小方塊會歪
            cv::rectangle(target, Rect(point.x, point.y + 5 * height / 12 + offset_y, width, height / 6), COLOR_BD8A44, 3); //BUTTON  the rate is 6 
        }
        break;

        case Buttons::UNPRESSED:
        {
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_BFAFA0, 3);  //SCROLL BAR
            cv::rectangle(target, Rect(point.x, point.y + 5 * height / 12 + offset_y, width, height / 6), COLOR_BFAFA0, 3);  //BUTTON  the rate is 6 
        }
        break;
        }
    },
        [](Buttons::BUTTON_STATE state, double delta) {
        delta -= 50.0;
        delta /= 50.0;
        SETEFFECT_STATICPTR(eq1, setMidGain(1.0 + delta));
    }));
    vButtons.push_back(new ScrollBarButton("ScrollBarButton testing", cv::Point((int)((0.75 + 2.95/20)*roi_width + xMin), (int)(0.25*roi_height + yMin)), 0.25*roi_height,0.05*roi_width,
        [](Buttons::BUTTON_STATE state, cv::Point point, float height, float width, int offset_y, cv::Mat& target) {
        switch (state)
        {
        case Buttons::PRESSED:
        {
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_BD8A44, 3);  //SCROLL BAR
            // 1/2*height - 1/12*height = 5/12*height 不然小方塊會歪
            cv::rectangle(target, Rect(point.x, point.y + 5 * height / 12 + offset_y, width, height / 6), COLOR_BD8A44, 3); //BUTTON  the rate is 6 
            //std::cout << 50 - (offset_y / height) * 100 << "%";
        }
        break;

        case Buttons::UNPRESSED:
        {
            cv::rectangle(target, Rect(point, Size(width, height)), COLOR_BFAFA0, 3);  //SCROLL BAR
            cv::rectangle(target, Rect(point.x, point.y + 5 * height / 12 + offset_y, width, height / 6), COLOR_BFAFA0, 3);  //BUTTON  the rate is 6 
        }
        break;
        }
    },
        [](Buttons::BUTTON_STATE state, double delta) {
        delta -= 50.0;
        delta /= 50.0;
        SETEFFECT_STATICPTR(eq1, setHighGain(1.0 + delta));
    }));

    IplImage *src = 0;                                                  //來源影像指標
    IplImage *srctodo = 0;                                              //目標影像指標
    CvSize srctodo_cvsize;                                              //目標影像尺寸

    src = cvLoadImage("D:/workspace/VS_projects/touchBoard_KinectV1/touchBoard_KinectV1/images/vinyl/vinyl_1.jpg", 1);                   //載入影像
    srctodo_cvsize.width = 0.6* roi_height;                                         //目標影像的寬為源影像寬的scale倍
    srctodo_cvsize.height = 0.6 * roi_height;                                        //目標影像的高為源影像高的scale倍
    srctodo = cvCreateImage(srctodo_cvsize, src->depth, src->nChannels);//創立目標影像
    cvResize(src, srctodo, CV_INTER_LINEAR);                            //縮放來源影像到目標影像

    VinylButton vinylButton1("VinylButton testing", cv::Point((int)(0.5*roi_width + xMin), (int)(0.5*roi_height + yMin)),srctodo_cvsize.width / 2, *srctodo,
        [](Buttons::BUTTON_STATE state, cv::Point point, float radius, IplImage& _src, cv::Mat& target) {
        //switch (state)
        //{
        //case Buttons::PRESSED:
        //    //cv::circle(target, point, radius, cv::Scalar(0, 255, 255), CV_FILLED);
        //    break;
        //case Buttons::UNPRESSED:
        //    //cv::circle(target, point, radius, cv::Scalar(255, 255, 0), CV_FILLED);
        //    break;
        //}
        Mat dst(&(_src), 0);
        Mat imgROI = target(Rect(point.x - radius, point.y - radius, radius * 2, radius * 2));  //指定插入的大小和位置
        addWeighted(imgROI, 0, dst, 1, 0, imgROI);
    },
        [](Buttons::BUTTON_STATE state, double sinDelta, double cosDelta) {
        static Buttons::BUTTON_STATE pre_Bstate;
        static AudioSample::STATE pre_Astate;
        if (state == Buttons::BUTTON_STATE::UNPRESSED) {
            if (pre_Bstate != state) {
                //pre_Bstate = state;
                //music1->setState(pre_Astate);



                //music1->exitScratch();
                //music1->clrScratchBuffer();
                //vinylButton1Ptr->setSpinPosition(pos);
            }
            music1->setVinylPressed(false);
        }
        else {
            if (pre_Bstate != state) {
                //pre_Bstate = state;
                //pre_Astate = music1->getState();
                //music1->setScratchBuffer(sinDelta, cosDelta);
                //music1->enterScratch();
                //music1->setState(AudioSample::STATE::SCRATCHING);
                //music1->queuingScratchPosition(-sinDelta, cosDelta);

            }
            music1->setVinylPressed(true);
            //music1->setScratchBuffer(sinDelta, cosDelta);
            //music1->queuingScratchPosition(-sinDelta, cosDelta);
        }
    });

    vButtons.push_back(&vinylButton1);
    vinylButton1.setAudioPositionPtr(music1->getPositionPtr());
    music1->setVinylPositionPtr(vinylButton1.getVinylPositionPtr());
    //music1->setTriFunPtr(vinylButton1.getSinPtr(), vinylButton1.getCosPtr());
    static VinylButton* vinylButton1Ptr = &vinylButton1;

    vButtons.push_back(new CircleButton("CircleButton testing", cv::Point((int)(0.8*roi_width + xMin) , (int)(0.8*roi_height + yMin)), 0.05* roi_height,
        [](Buttons::BUTTON_STATE state, cv::Point point, float radius, cv::Mat& target) {
        switch (state)
        {
        case Buttons::PRESSED:
            cv::circle(target, point, radius, COLOR_F1BA48, CV_FILLED);
            break;

        case Buttons::UNPRESSED:
            cv::circle(target, point, radius, COLOR_BD8A44, CV_FILLED);
            break;
        }
    },
        [](Buttons::BUTTON_STATE state) {
        if (music1->getState() == AudioSample::STATE::PAUSING) {
            music1->setState(AudioSample::STATE::PLAYING);
        }
        else {
            music1->setState(AudioSample::STATE::PAUSING);
            //vinylButton1Ptr->setSpinPosition(-1.0);
        }
        //
    }));

    //-------------END----Add Button-----------------

    //-------Create a Window with controlbar-------
    const char* main_windowName = "GUI interface";
    const char* debug_depth_windowName = "DEBUG depth window";
    cv::namedWindow(main_windowName, WND_PROP_FULLSCREEN);
    cvResizeWindow(main_windowName, 1024, 848);
    cv::namedWindow(debug_depth_windowName, WND_PROP_FULLSCREEN);
    cvResizeWindow(debug_depth_windowName, 1024, 848);
    HWND m_hwnd = (HWND)cvGetWindowHandle(main_windowName);
    int keyboardKeynum = 0;
    while (/*IsWindowVisible(m_hwnd)*/1)
    {
        //get depth frame
        if (depthStream.isValid()) {
            if (depthStream.readFrame(&depthFrame) == STATUS_OK) {
                depth_rawFrame.data = (uchar*)depthFrame.getData();
                cv::flip(depth_rawFrame, depth_rawFrame, 1);
                depth_rawFrame.convertTo(depth_8bitFrame, CV_8U, 255 / debugFrameMaxDepth); // render depth to debug frame
                                                                                            //cv::flip(depth_rawFrame, depth_rawFrame, 1);
            }
        }
        //get color frame
        if (depth_to_colorStream.isValid()) {
            if (depth_to_colorStream.readFrame(&depth_to_colorFrame) == STATUS_OK) {
                //convert data to OpenCV format
               depth_to_bgr_rawFrame.data=(uchar*)depth_to_colorFrame.getData();
                //convert form RGB to BGR
                cv::cvtColor(depth_to_bgr_rawFrame, depth_to_bgr_rawFrame, CV_RGB2BGR);
                //水平翻轉
                cv::flip(depth_to_bgr_rawFrame, depth_to_bgr_rawFrame, 1);
            }
        }
        //-------Train the background-------

        if (training)
        {
            putText(depth_to_bgr_rawFrame, "background training...", Point(66,212 ), 0, 1, Scalar(0, 255, 255), 2);
            if (++iBackgroundTrain < nBackgroundTrain)
            {
                buffer[iBackgroundTrain] = depth_rawFrame;
            }
            else
            {
                average(buffer, background);
                training = false;
            }
        }
        //-------End Trainning-------
        if (!training)
        {
            Mat intermideate;
            Mat skinColor;
            cv::cvtColor(depth_to_bgr_rawFrame, intermideate, CV_BGR2YCrCb);
            cv::inRange(intermideate, Scalar(0, 137, 77), Scalar(256, 177, 127), skinColor);
            cv::erode(skinColor, skinColor, Mat(), Point(-1, -1), 2);
            cv::dilate(skinColor, skinColor, Mat(), Point(-1, -1), 1);

            imshow("test", skinColor);
            //--------計算感興趣區域的contour--------
            foreground = background - depth_rawFrame;
            touchMask = (foreground > touchDepthMin) & (foreground < touchDepthMax);
            Rect roi(xMin, yMin, xMax - xMin, yMax - yMin);
            Mat colorRoi = skinColor(roi);
            Mat touchRoi = touchMask(roi);
            touchRoi = touchRoi & colorRoi;
            cv::erode(touchRoi, touchRoi, Mat(), Point(-1, -1), 1);
            cv::dilate(touchRoi, touchRoi, Mat(), Point(-1, -1), 2);

            vector< vector<Point2i> > contoursANYTHING;
            vector<Point2f> touchPoints;
            cv::findContours(touchRoi, contoursANYTHING, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point2i(xMin, yMin));
            for (unsigned int i = 0; i<contoursANYTHING.size(); i++) {
                Mat contourMat(contoursANYTHING[i]);
                // find touch points by area thresholding
                if (contourArea(contourMat) > touchMinArea && contourArea(contourMat) < 100) {
                    Scalar center = mean(contourMat);
                    Point2i touchPoint(center[0], center[1]);
                    touchPoints.push_back(touchPoint);
                }
            }
            cv::cvtColor(depth_8bitFrame, depth_bgrFrame, CV_GRAY2BGR);
            depth_bgrFrame.setTo(debugColor0, touchMask);// touch mask
            cv::rectangle(depth_to_bgr_rawFrame, roi, debugColor1, 2);
            //cv::rectangle(Depth_BGRFrame, roi, debugColor1, 2);
            for (unsigned int i = 0; i<touchPoints.size(); i++) {// touch points
                circle(depth_bgrFrame, touchPoints[i], 5, debugColor2, CV_FILLED);
                circle(depth_to_bgr_rawFrame, touchPoints[i], 5, debugColor0, CV_FILLED);
            }
        }
        //將每一個button自己感興趣的區域，傳入button確認是否達到按下的狀態
        for (auto itButton = vButtons.begin(); itButton != vButtons.end(); ++itButton)
            (*itButton)->CheckHand(touchMask(Rect(cv::Point((*itButton)->getPoint().x - (int)(*itButton)->getWidth() / 2,
            (*itButton)->getPoint().y - (int)(*itButton)->getHeight() / 2),
                Size((int)(*itButton)->getWidth(), (int)(*itButton)->getHeight()))));

        colorShow_hand = depth_to_bgr_rawFrame.clone();
        //將所有的button畫出來
        for (auto itButton = vButtons.begin(); itButton != vButtons.end(); ++itButton)
            (*itButton)->draw(depth_to_bgr_rawFrame);
        depth_to_bgr_rawFrame = 0.75*depth_to_bgr_rawFrame + 0.25*colorShow_hand;

        keyboardKeynum = waitKey(1);
      //  imshow(main_windowName, depth_to_bgr_rawFrame);
        imshow(debug_depth_windowName, depth_bgrFrame);
        imshow(main_windowName, depth_to_bgr_rawFrame);
    }

    cvDestroyAllWindows();
    depthStream.destroy();
    depth_to_colorStream.destroy();
    device.close();
    OpenNI::shutdown();
    return 0;
}