#ifndef DRAW_HPP
#define DRAW_HPP

#include <opencv2/opencv.hpp>

using namespace cv;

void putTextCenter( InputOutputArray img, const String& text, Point org,
                         int fontFace, double fontScale, Scalar color,
                         int thickness = 1, int lineType = LINE_8,
                         bool bottomLeftOrigin = false ){
                            int baseline = 0;
                            Size s = getTextSize(text,fontFace,fontScale,thickness,&baseline);
                            Point o = Point(org.x-s.width/2,org.y+(s.height-baseline)/2);
                            putText(img,text,o,fontFace,fontScale,color,thickness,lineType,bottomLeftOrigin);
                        }

Point map2point(double x,double y,Rect drawArea, double minX, double maxX, double minY, double maxY){
    return Point((x-minX)/(maxX-minX)*drawArea.width+drawArea.x,drawArea.y+drawArea.height-((y-minY)/(maxY-minY)*drawArea.height));
}

#endif