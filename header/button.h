#pragma once

// STL Header

#include <algorithm>
#include <functional>
#include <string>

// OpenCV Header
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

class Buttons
{
public:
    //描述此按鈕的用途
    enum BUTTON_STATE
    {
        UNPRESSED,
        PRESSED
    };
    std::string application;
    Buttons() {};
    ~Buttons() {};

    Buttons(const std::string& _application, const cv::Point& _position, float _height, float _width, std::function<void(Buttons::BUTTON_STATE, cv::Point, float, float, cv::Mat&)> _drawFunc, std::function<void()> _callbackFunc) {
        state = UNPRESSED;
        height = _height;
        width = _width;
        application = _application;
        position = _position;
    }
    Buttons(const std::string& _application, const cv::Point& _position, float _radius, std::function<void(Buttons::BUTTON_STATE, cv::Point, float, cv::Mat&)> _drawFunc, std::function<void()> _callbackFunc) {
        state = UNPRESSED;
        radius = _radius;
        application = _application;
        position = _position;
    }
    Buttons(const std::string& _application, const cv::Point& _position, float _height, float _width, std::function<void(Buttons::BUTTON_STATE, cv::Point, float, float, int, cv::Mat&)> _drawFunc, std::function<void()> _callbackFunc) {
        state = UNPRESSED;
        height = _height;
        width = _width;
        application = _application;
        position = _position;
    }
    Buttons(const std::string& _application, const cv::Point& _position, float _radius, IplImage& _src, std::function<void(Buttons::BUTTON_STATE, cv::Point, float, IplImage&, cv::Mat&)> _drawFunc, std::function<void()> _callbackFunc) {
        state = UNPRESSED;
        radius = _radius;
        application = _application;
        src = &(_src);
        position = _position;
    }
    float getHeight() { return height; }

    float getWidth() { return width; }

    float getRadius() { return radius; }

    cv::Point getPoint() { return position; }

    virtual void draw(cv::Mat& targetMat) = 0;

    virtual void CheckHand(cv::Mat& buttonRoi) = 0;
    virtual void CheckHand(float x, float y) = 0;


protected:
    //Current button state
    BUTTON_STATE state;
    BUTTON_STATE currentstate;
    IplImage* src;
    //position center coordinate
    cv::Point position;
    //ROI
    float height, width, radius;
    //Draw function
    //std::function<void()> drawFunc;
    //Callback function
    //std::function<void()> callbackFunc;
    //touchMinArea, touchMaxArea
    const unsigned int touchMinArea = 0;
    const unsigned int touchMaxArea = 110;
    //IplImage vector
    vector<IplImage*> imageVector;
    //Check if given position is inside the button
    virtual bool CheckInside(float x, float y) {
        return false;
    };
};

class RectButton : public Buttons
{
protected:
    float halfHeight;
    float halfWidth;
    //Draw function
    std::function<void(Buttons::BUTTON_STATE, cv::Point, float, float, cv::Mat&)> drawFunc;
    //Callback funciotn
    std::function<void(Buttons::BUTTON_STATE)> callbackFunc;
    bool CheckInside(float x, float y) {
        return x <= position.x + halfWidth && position.x - halfWidth < x && y <= position.y + halfHeight && position.y - halfHeight < y;
    }
public:
    RectButton(const std::string& _application, const cv::Point& _position, float _height, float _width, std::function<void(Buttons::BUTTON_STATE, cv::Point, float, float, cv::Mat&)> _drawFunc, std::function<void(Buttons::BUTTON_STATE)> _callbackFunc) {
        state = UNPRESSED;
        height = _height;
        width = _width;
        application = _application;
        position = _position;
        drawFunc = _drawFunc;
        callbackFunc = _callbackFunc;
        halfHeight = (_height / 2);
        halfWidth = (_width / 2);
    }

    void CheckHand(cv::Mat& buttonRoi) {
        vector< vector<cv::Point2i> > contoursANYTHING;
        vector<cv::Point2f> touchPoints;
        int sumContours = 0;
        cv::findContours(buttonRoi, contoursANYTHING, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point2i(position.x, position.y));
        for (unsigned int i = 0; i<contoursANYTHING.size(); i++) {
            cv::Mat contourMat(contoursANYTHING[i]);
            // find touch points by area thresholding
            sumContours += contourArea(contourMat);
        }
        if (sumContours > touchMinArea) {
            if (state == UNPRESSED) {
                state = PRESSED;
                callbackFunc(state);
            }
            else if (state == PRESSED) {

            }
        }
        else {
            state = UNPRESSED;
        }
    }

    void CheckHand(float x, float y) {
        if (CheckInside(x, y)) {
            if (state == UNPRESSED) {
                state = PRESSED;
                callbackFunc(state);
            }
            else if (state == PRESSED) {
            }
        }
        else {
            state = UNPRESSED;
        }
    }

    void draw(cv::Mat& targetMat) {
        drawFunc(state, cv::Point2f(position.x - halfWidth, position.y - halfHeight), height, width, targetMat);
    }
};

class CircleButton : public Buttons
{
protected:
    //Draw function
    std::function<void(Buttons::BUTTON_STATE, cv::Point, float, cv::Mat&)> drawFunc;
    //Callback funciotn
    std::function<void(Buttons::BUTTON_STATE)> callbackFunc;
    bool CheckInside(float x, float y) {
        return (x - position.x)*(x - position.x) + (y - position.y)*(y - position.y) <= radius*radius;
    }
    //The bitwise mask
    cv::Mat mask;
public:
    //note: height, width==2 * _radius
    CircleButton(const std::string& _application, const cv::Point& _position, float _radius, std::function<void(Buttons::BUTTON_STATE, cv::Point, float, cv::Mat&)> _drawFunc, std::function<void(Buttons::BUTTON_STATE)> _callbackFunc) {
        state = UNPRESSED;
        radius = _radius;
        application = _application;
        position = _position;
        drawFunc = _drawFunc;
        callbackFunc = _callbackFunc;

        height = 2 * _radius;
        width = 2 * _radius;
        mask = Mat(Size2f(width, height), CV_8UC1, Scalar::all(0));
        circle(mask, Point(radius, radius), radius, Scalar::all(255), -1);
    }

    void CheckHand(cv::Mat& buttonRoiRaw) {
        Mat buttonRoi = buttonRoiRaw & mask;
        vector< vector<cv::Point2i> > contoursANYTHING;
        vector<cv::Point2f> touchPoints;
        int sumContours = 0;
        cv::findContours(buttonRoi, contoursANYTHING, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point2i(position.x, position.y));
        for (unsigned int i = 0; i<contoursANYTHING.size(); i++) {
            cv::Mat contourMat(contoursANYTHING[i]);
            // find touch points by area thresholding
            sumContours += contourArea(contourMat);
        }
        if (sumContours > touchMinArea) {
            if (state == UNPRESSED) {
                state = PRESSED;
                callbackFunc(state);
            }
            else if (state == PRESSED) {}
        }
        else {
            state = UNPRESSED;
        }
    }

    void CheckHand(float x, float y) {
        if (CheckInside(x, y)) {
            if (state == UNPRESSED) {
                state = PRESSED;
                callbackFunc(state);
            }
            else if (state == PRESSED)
            {
                state = UNPRESSED;
            }
        }
        else {
            state = UNPRESSED;
        }
    }
    void draw(cv::Mat& targetMat) {
        drawFunc(state, cv::Point(position.x, position.y), radius, targetMat);
    }
};

class ScrollBarButton : public Buttons
{
protected:
    float halfHeight;
    float halfWidth;
    int pre_y;      // touched point
    int offset_y;   // offset of y

    //Draw function
    std::function<void(Buttons::BUTTON_STATE, cv::Point, float, float, int, cv::Mat&)> drawFunc;
    //Callback funciotn
    std::function<void(Buttons::BUTTON_STATE, double)> callbackFunc;

public:
    ScrollBarButton(const std::string& _application, const cv::Point& _position, float _height, float _width, std::function<void(Buttons::BUTTON_STATE, cv::Point, float, float, int, cv::Mat&)> _drawFunc, std::function<void(Buttons::BUTTON_STATE, double)> _callbackFunc) {
        state = UNPRESSED;
        height = _height;
        width = _width;
        application = _application;
        position = _position;
        
        callbackFunc = _callbackFunc;
        drawFunc = _drawFunc;
        halfHeight = (_height / 2);
        halfWidth = (_width / 2);
        offset_y = 0;
    }

    void CheckHand(cv::Mat& buttonRoi) {
        vector< vector<cv::Point2i> > contoursANYTHING;
        int sumContours = 0;
        int find_y = 0;
        int maxmoveheight = height / 2;
        int movebound = 12;
        cv::findContours(buttonRoi, contoursANYTHING, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point2i(position.x, position.y));
        //int size = contoursANYTHING.size();
        //if (size)
        //{
        //    for (unsigned int i = 0; i < size; i++) {
        //        cv::Mat contourMat(contoursANYTHING[i]);
        //        if (contourArea(contourMat) > touchMinArea) {
        //            sumContours += contourArea(contourMat);
        //            Scalar center = mean(contourMat);
        //            Point2i touchPoint(center[0], center[1]);
        //            find_y += center[1];
        //        }
        //        
        //    }
        //    find_y /= size;
        //}

        for (unsigned int i = 0; i<contoursANYTHING.size(); i++) {
            cv::Mat contourMat(contoursANYTHING[i]);
            // find touch points by area thresholding
            sumContours += contourArea(contourMat);
            Scalar center = mean(contourMat);
            Point2i touchPoint(center[0], center[1]);
            find_y = center[1];
        }

        if (sumContours > touchMinArea) {
            if (state == UNPRESSED) {
                state = PRESSED;
                pre_y = find_y;
                callbackFunc(state, (double)(50.0 - (offset_y / height) * 100.0));
            }
            else if (state == PRESSED) {
                int tmp_offset;
                tmp_offset = find_y - pre_y;
                if (tmp_offset > movebound)
                {
                    offset_y = offset_y + movebound;

                }
                else if (tmp_offset < -movebound)
                {
                    offset_y = offset_y - movebound;
                }
                else if (tmp_offset < movebound && tmp_offset > -movebound)
                {
                    offset_y = offset_y + tmp_offset;
                }

                if (offset_y < -maxmoveheight)
                    offset_y = -maxmoveheight;
                else if (offset_y > maxmoveheight)
                    offset_y = maxmoveheight;

                pre_y = find_y;

                callbackFunc(state, (double)(50.0 - (offset_y / height) * 100.0));
            }
        }
        else {
            state = UNPRESSED;
        }
    }

    void CheckHand(float x, float y) {
        if (CheckInside(x, y)) {
            if (state == UNPRESSED) {
                state = PRESSED;
                callbackFunc(state, (double)(50.0 - (offset_y / height) * 100.0));
            }
            else if (state == PRESSED) {
            }
        }
        else {
            state = UNPRESSED;
        }
    }

    void draw(cv::Mat& targetMat) {
        drawFunc(state, cv::Point(position.x - halfWidth, position.y - halfHeight), height, width, offset_y, targetMat);
    }
};

class VinylButton : public Buttons
{
protected:
    //Draw function
    std::function<void(Buttons::BUTTON_STATE, cv::Point, float, IplImage& , cv::Mat&)> drawFunc;
    //Callback function
    std::function<void(Buttons::BUTTON_STATE, double, double)> callbackFunc;

    bool CheckInside(float x, float y) {
        return (x - position.x)*(x - position.x) + (y - position.y)*(y - position.y) <= radius*radius;
    }
    //The bitwise mask
    cv::Mat mask;
    IplImage* pImgSrc = NULL;
    IplImage* pImgDst = NULL;
    double delta = 6.0;//2.27272727;//1.666665; //33轉旋轉速度
    double cur_sinTheta = 0, cur_cosTheta = 1;
    double m[6];
    CvMat M;
    double touch_x, touch_y;
    double pre_y, pre_x;

    double *audioPosition;
    //feedBack position = nLoop * 88200 + rPosition
    double vinylPosition;
    int nLoop;
    double rPosition;
    double radian;
public:
    //note: height, width==2 * _radius
    VinylButton(const std::string& _application, const cv::Point& _position, float _radius, IplImage& _src, std::function<void(Buttons::BUTTON_STATE, cv::Point, float, IplImage&, cv::Mat&)> _drawFunc, std::function<void(Buttons::BUTTON_STATE, double, double)> _callbackFunc) {
        state = UNPRESSED;
        radius = _radius;
        application = _application;
        src = &(_src);
        position = _position;

        drawFunc = _drawFunc;
        callbackFunc = _callbackFunc;
        height = 2 * _radius;
        width = 2 * _radius;
        mask = Mat(Size(width, height), CV_8UC1, Scalar::all(0));
        circle(mask, Point(radius, radius), radius, Scalar::all(255), -1);

        pImgSrc = &(_src);
        pImgDst = cvCloneImage(pImgSrc);
    }

    void CheckHand(cv::Mat& buttonRoiRaw) {
        Mat buttonRoi = buttonRoiRaw & mask;
        vector< vector<cv::Point2i> > contoursANYTHING;
        vector<cv::Point2f> touchPoints;
        int sumContours = 0;
        cv::findContours(buttonRoi, contoursANYTHING, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point2i(0, 0));
        for (unsigned int i = 0; i<contoursANYTHING.size(); i++) {
            cv::Mat contourMat(contoursANYTHING[i]);
            double Area = contourArea(contourMat);
            // find touch points by area thresholding
            if (Area > 0.0) {
                sumContours += contourArea(contourMat);
                Scalar center = mean(contourMat);
                Point2i touchPoint(center[0], center[1]);
                touch_y = center[1];
                touch_x = center[0];
            }
        }
        if (sumContours > touchMinArea) {
            if (state == UNPRESSED) {
                state = PRESSED;
                pre_y = touch_y;
                pre_x = touch_x;
                callbackFunc(state, cur_sinTheta, cur_cosTheta);
            } else {
                Mat ImgDst(pImgDst, 0);
                Mat ImgSrc(pImgSrc, 0);
                Point2f src_center(ImgDst.cols / 2.0F, ImgDst.rows / 2.0F);
                double vector1[2] = { pre_x - src_center.x, pre_y - src_center.y };
                double vector2[2] = { touch_x - src_center.x, touch_y - src_center.y };
                double norm_2Vector1 = sqrt(vector1[0] * vector1[0] + vector1[1] * vector1[1]);
                double norm_2Vector2 = sqrt(vector2[0] * vector2[0] + vector2[1] * vector2[1]);
                double dotProduct = vector1[0] * vector2[0] + vector1[1] * vector2[1];
                double crossProduct = vector1[0] * vector2[1] - vector1[1] * vector2[0];
                double cosTheta = dotProduct / (norm_2Vector1*norm_2Vector2);
                double sinTheta = (crossProduct>0)? -sqrt(1 - cosTheta*cosTheta) : sqrt(1 - cosTheta*cosTheta);
                if (!IsNumber(sinTheta)) sinTheta = 0, cosTheta = 1;
                if (!IsNumber(cosTheta)) sinTheta = 0, cosTheta = 1;
                //std::cout << sinTheta << " " << cosTheta << std::endl;
                double new_sinTheta = cur_sinTheta;
                double new_cosTheta = cur_cosTheta;
              //  cout << fabs(atan2(sinTheta, cosTheta)) << endl;
                if (fabs(atan2(sinTheta, cosTheta)) > (1 * CV_PI / 180.0)) {
                    new_sinTheta = cur_sinTheta*cosTheta + cur_cosTheta*sinTheta;
                    new_cosTheta = cur_cosTheta*cosTheta - cur_sinTheta*sinTheta;
                }

                //

                radian = atan2(new_sinTheta, new_cosTheta);
                if (radian < 0) radian += 2 * CV_PI;
                double complementRadian = 2 * CV_PI - radian;
                rPosition = (complementRadian / (2.0 * CV_PI) ) * 88200.0;

                //clockwise
                if (crossProduct > 0) {
                    if (new_cosTheta >= 0 && cur_cosTheta > 0 && new_sinTheta <= 0 && cur_sinTheta > 0) {
                        ++nLoop;
                    }
                }
                else {
                    if (new_cosTheta >= 0 && cur_cosTheta > 0 && new_sinTheta > 0 && cur_sinTheta <= 0) {
                        if (--nLoop < 0) {
                            nLoop = 0;
                            rPosition = 0.0;
                            new_sinTheta = 0.0;
                            new_cosTheta = 1.0;
                        }
                    }
                }

                vinylPosition = nLoop * 88200 + rPosition; 


                callbackFunc(state, new_sinTheta, new_cosTheta);

                //cout << radian * (180 / CV_PI) << endl;
                //cout << rPosition << endl;

                pre_x = touch_x;
                pre_y = touch_y;
                //rotate(cur_cosTheta, cur_sinTheta);
                cur_sinTheta = new_sinTheta;
                cur_cosTheta = new_cosTheta;
            }
        }
        else {
            state = UNPRESSED;
            callbackFunc(state, cur_sinTheta, cur_cosTheta);
            //CheckSpinPosition();
        }
    }
    //指標內容不可變
    double * getSinPtr() {
        return &cur_sinTheta;
    }
    double * getCosPtr() {
        return &cur_cosTheta;
    }

    //void setSpinPosition(double _value) {
    //    /*if (spinPosition>0) spinFlag = false;
    //    else spinFlag = true;*/
    //    spinPosition = _value;
    //}

    double *getVinylPositionPtr() {
        return &vinylPosition;
    }

    void setAudioPositionPtr( double *_audioPosition) {
        audioPosition = _audioPosition;
    }

    //void CheckSpinPosition() {
    //    if (spinPosition>0) {
    //        //maintain_angle();
    //        double cosTheta = cos(-spinPosition*2.0*CV_PI);
    //        double sinTheta = sin(-spinPosition*2.0*CV_PI);
    //        rotate(cosTheta, sinTheta);
    //        cur_sinTheta = sinTheta;
    //        cur_cosTheta = cosTheta;
    //    }
    //}

    void maintain_angle() {
        double cosTheta = cos(-delta * CV_PI / 180.);
        double sinTheta = sin(-delta * CV_PI / 180.);
        double new_sinTheta = cur_sinTheta*cosTheta + cur_cosTheta*sinTheta;
        double new_cosTheta = cur_cosTheta*cosTheta - cur_sinTheta*sinTheta;
        rotate(cur_cosTheta, cur_sinTheta);
        cur_sinTheta = new_sinTheta;
        cur_cosTheta = new_cosTheta;
    }

    void rotate( double cosD, double sinD ) {
        Mat ImgSrc(pImgSrc, 0);
        Mat ImgDst(pImgDst, 0);
        Mat tmpImgDst;
        //cout << sinD << " " << cosD << endl;
        Point2f src_center(ImgDst.cols / 2.0F, ImgDst.rows / 2.0F);

        Mat tempMask = Mat(Size(width, height), ImgSrc.type(), Scalar::all(0));
        circle(tempMask, Point(radius, radius), radius, Scalar::all(255), -1);

        m[0] = cosD;
        m[1] = sinD;
        m[2] = (1 - cosD)*src_center.x-sinD*src_center.y;
        m[3] = -sinD;
        m[4] = cosD;
        m[5] = sinD*src_center.x + (1 - cosD)*src_center.y;
        Mat rot_mat(2, 3, CV_64F, m);
        //cvZero(pImgDst);
        warpAffine(ImgSrc, ImgDst, rot_mat, ImgDst.size(), INTER_NEAREST);
        bitwise_and(ImgDst, tempMask, ImgDst);
        bitwise_not(tempMask, tempMask);
        bitwise_and(ImgSrc, tempMask, tmpImgDst);
        ImgDst = ImgDst + tmpImgDst;
    }

    // This looks like it should always be true, 
    // but it's false if x is a NaN.
    bool IsNumber(double x) {
        return (x == x);
    }

    void CheckHand(float x, float y) {
        if (CheckInside(x, y)) {
            if (state == UNPRESSED) {
                state = PRESSED;
                callbackFunc(state, 0.0, 1.0);
            }
            else if (state == PRESSED)
            {
                state = UNPRESSED;
            }
        }
        else {
            state = UNPRESSED;
        }
    }
    void draw(cv::Mat& targetMat) {
        //參考Audio的position，算出現在Vinyl應變的角度，如現在Vinyl的state是UNPRESSED，那nLoop參考Audio正常算出Vinyl現在轉了第幾圈，
        //若是PRESSED，nLoop則由現在的位置運算現在Vinyl是在第幾圈。
        double pos = *audioPosition;
        double remainder = fmod(pos, 88200.0);
        double rotateScale = remainder / 88200.0;
        double next_sin = -sin(rotateScale * 2 * CV_PI);
        double next_cos = cos(rotateScale * 2 * CV_PI);
        if (state == UNPRESSED) {
            nLoop = pos / 88200;
            rPosition = remainder;
            vinylPosition = pos;
            cur_sinTheta = next_sin;
            cur_cosTheta = next_cos;
        }

        rotate(next_cos, next_sin);
        //cout << "the " << nLoop  << "th" << remainder << endl;
        //cout << *audioPosition << endl;
        drawFunc(state, cv::Point(position.x, position.y), radius, *pImgDst, targetMat);
       
    }
};

