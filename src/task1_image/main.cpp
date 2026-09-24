#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "draw.hpp"

using namespace cv;

const std::string BASE_PATH = "result/task1_images/";

std::string getFullPath(std::string name){
    return BASE_PATH+name;
}

int main(){
    Mat img = imread("resources/test_image.jpg");
    
    if(img.empty()){
        std::cerr << "Failed to read image \n";
        return 1;
    }
    
    // Image reading and color conversion
    Mat gray,hsv,channels[3],hsv_h,hsv_s,hsv_v;
    cvtColor(img,gray,COLOR_BGR2GRAY);
    cvtColor(img,hsv,COLOR_BGR2HSV);
    split(hsv,channels);
    hsv_h = channels[0].clone();
    hsv_s = channels[1].clone();
    hsv_v = channels[2].clone();
    imwrite(getFullPath("gray.png"),gray);
    imwrite(getFullPath("hsv_h.png"),hsv_h);
    imwrite(getFullPath("hsv_s.png"),hsv_s);
    imwrite(getFullPath("hsv_v.png"),hsv_v);

    // Filtering
    Mat meanFiltered,gaussianFiltered,medianFiltered;
    blur(img,meanFiltered,Size(5,5));
    GaussianBlur(img,gaussianFiltered,Size(5,5),1.5);
    medianBlur(img,medianFiltered,3);
    imwrite(getFullPath("mean_filter.png"),meanFiltered);
    imwrite(getFullPath("gaussian_filter.png"),gaussianFiltered);
    imwrite(getFullPath("median_filter.png"),medianFiltered);

    // Extraction
    Mat maskLow, maskHigh, mask;
    inRange(hsv,Scalar(0,120,58),Scalar(15,255,255),maskLow);
    inRange(hsv,Scalar(170,120,58),Scalar(179,255,255),maskHigh);
    bitwise_or(maskLow,maskHigh,mask);
    imwrite(getFullPath("red_mask.png"),mask);

    //Morphology and Contours
    Mat eroded,dilated,opened,closed;
    Mat kernel5 = getStructuringElement(MORPH_RECT,Size(5,5));
    erode(mask,eroded,kernel5);
    dilate(mask,dilated,kernel5);
    morphologyEx(mask,opened,MORPH_OPEN,kernel5);
    morphologyEx(mask,closed,MORPH_CLOSE,kernel5);
    imwrite(getFullPath("erode.png"),eroded);
    imwrite(getFullPath("dilate.png"),dilated);
    imwrite(getFullPath("open.png"),opened);
    imwrite(getFullPath("close.png"),closed);
    
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(closed,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    Mat contoursImg = img.clone();
    for(int i = 0;i < contours.size();i++){
        double area = contourArea(contours[i]);
        if(area < 400.0) continue;

        Rect box = boundingRect(contours[i]);
        drawContours(contoursImg,contours,i,Scalar(0,255,0),2);
        rectangle(contoursImg,box,Scalar(0,0,255),2);

        char text[40];
        memset(text,0,sizeof(text));
        snprintf(text,sizeof(text),"Area:%.2f",area);
        putText(contoursImg,text,Point(box.x,box.y-7),FONT_HERSHEY_PLAIN,1.3,Scalar(255,0,255),2);
    }
    imwrite(getFullPath("contours_boxes.png"),contoursImg);

    //Drawing and transformation
    Mat drawing = img.clone();
    Point center(drawing.size().width/2,drawing.size().height/2);
    circle(drawing,center,200,Scalar(255,255,255),2);
    rectangle(drawing,Point(center.x-300,center.y-250),Point(center.x+300,center.y+250),Scalar(255,255,255),2);
    putTextCenter(drawing,"TEXT",center,FONT_HERSHEY_DUPLEX,1.0,Scalar(255,255,255),2);
    imwrite(getFullPath("drawing.png"),drawing);

    Mat rotated;
    int degree = 35;
    Mat rotationMatrix = getRotationMatrix2D(center,degree,1);
    Size imgSize = img.size();
    int rotatedH = imgSize.width*fabs(sin(degree*M_PI/180)) + imgSize.height*fabs(cos(degree*M_PI/180));
    int rotatedW = imgSize.height*fabs(sin(degree*M_PI/180)) + imgSize.width*fabs(cos(degree*M_PI/180));
    Size rotatedSize(rotatedW,rotatedH);
    rotationMatrix.at<double>(0,2) += (rotatedW-imgSize.width)/2;
    rotationMatrix.at<double>(1,2) += (rotatedH-imgSize.height)/2;
    warpAffine(img,rotated,rotationMatrix,rotatedSize);
    imwrite(getFullPath("rotated_35deg.png"),rotated);

    Mat cropped = img(Range(0,imgSize.height/2),Range(0,imgSize.width/2));
    imwrite(getFullPath("crop_top_left.png"),cropped);

    return 0;
}