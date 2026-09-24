#include "draw.hpp"
#include <ceres/ceres.h>
#include <iostream>

using namespace cv;

struct Residual{
    Residual(double x, double y): _x(x),_y(y){}

    template <typename T> bool operator()(const T* const p,T* residual) const {
        using std::sin;
        residual[0] = T(_y) - (p[1] + p[0]*sin(p[2]*T(_x)+p[3]));
        return true;
    }
    double _x,_y;
};

int main(){
    //Video reader
    VideoCapture vid("resources/task_2.mp4");
    if(!vid.isOpened()){
        std::cerr << "Failed to read video\n";
        return 1;
    }

    //Video writer
    double fps = vid.get(CAP_PROP_FPS);
    double spf = 1/fps;
    int totalFrames = vid.get(CAP_PROP_FRAME_COUNT);
    Size frameSize(vid.get(CAP_PROP_FRAME_WIDTH),vid.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter writer("result/task2_fit/tracking_overlay.mp4",VideoWriter::fourcc('m','p','4','v'),fps,frameSize);
    
    Mat frame;
    int frameCount = 0;
    std::vector<double> omegas; //Starts from the second frame
    double theta[3] = {-100,-100,-100};
    while(true){
        vid >> frame;
        if(frame.empty()){
            break;
        }
        Mat hsv;
        cvtColor(frame,hsv,COLOR_BGR2HSV);
        Mat cyanMask;
        inRange(hsv,Scalar(90,200,200),Scalar(100,255,255),cyanMask);
        std::vector<std::vector<Point>> contours;
        findContours(cyanMask,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
        bool detectedFlag = false;
        for(int i = 0;i < contours.size();i++){
            double circumference = arcLength(contours[i],true);
            if(circumference == 0) continue;
            double area = contourArea(contours[i]);
            double roundness = (4*M_PI*area)/(circumference*circumference);
            if(roundness < 0.4) continue;

            Point2f center(0,0);
            float radius = 0;
            minEnclosingCircle(contours[i],center,radius);
            circle(frame,center,radius,Scalar(0,0,255),2);
            
            double rad = atan2(360-center.y,center.x-480);
            if(rad < 0) rad += 2*M_PI;

            char text[40];
            memset(text,0,sizeof(text));
            snprintf(text,sizeof(text),"Pos: (%d,%d)",(int)center.x,(int)center.y);
            putText(frame,text,Point(20,40),FONT_HERSHEY_DUPLEX,1.0,Scalar(255,0,255),2);
            memset(text,0,sizeof(text));
            snprintf(text,sizeof(text),"Deg: %.2f",rad*180/M_PI);
            putText(frame,text,Point(20,80),FONT_HERSHEY_DUPLEX,1.0,Scalar(255,0,255),2);

            double omega = -1;
            if(theta[0] < 0) theta[0] = rad;
            else if(theta[1] < 0) theta[1] = rad;
            else if(theta[2] < 0) theta[2] = rad;
            else{
                theta[0] = theta[1];
                theta[1] = theta[2];
                theta[2] = rad;
            }
            if(theta[0] >= 0 && theta[1] >= 0 && theta[2] >= 0){
                if(theta[2] - theta[0] < 0){
                    omega = ((theta[2]+(2*M_PI) - theta[0])/2) * fps;
                }else{
                    omega = ((theta[2] - theta[0])/2) * fps;
                }
            }
            if(omega >= 0){
                omegas.push_back(omega);
            }
            // std::cout<<frameCount<<": \n";
            // std::cout<<rad/M_PI*180<<"\n";
            // std::cout<<theta[0]<<" "<<theta[1]<<" "<<theta[2]<<"\n";
            // std::cout<<omega<<"\n";
            // std::cout<<"\n";
            detectedFlag = true;
            break;
        }
        if(!detectedFlag){
            std::cerr << "Detection failed in frame"<<frameCount;
            return 1;
        }
        writer.write(frame);
        // imshow("cyan",frame);
        // waitKey(0);
        frameCount++;
    }
    // std::cout<<omegas.size()<<"/"<<totalFrames<<"\n";
    if(omegas.size() != totalFrames-2){
        std::cerr << "Detection incomplete";
    }

    vid.release();
    writer.release();

    //Optimize
    std::vector<double> tData(omegas.size());
    for(size_t i = 1; i <= omegas.size();i++){
        tData[i-1] = i*spf;
    }

    std::vector<double> xData,yData;
    for(size_t i = 0;i < omegas.size();i+=1){
        xData.push_back(tData[i]);
        yData.push_back(omegas[i]);
    }

    std::cout << "Samples: " << xData.size() << "\n";

    ceres::Problem problem;
    double parameters[4] = {0.5,1.3,1.5,2};
    for(size_t i = 0;i < xData.size();i++){
        auto* cost = new ceres::AutoDiffCostFunction<Residual,1,4>(new Residual(xData[i],yData[i]));
        problem.AddResidualBlock(cost,new ceres::HuberLoss(1.0),parameters);
    }
    problem.SetParameterLowerBound(parameters,3,-M_PI);
    problem.SetParameterUpperBound(parameters,3,M_PI);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "Estimated parameters: A = " << parameters[0] << ", b = " << parameters[1] << ", omega = " << parameters[2] << ", phi = " << parameters[3] << "\n";

    ceres::Problem::EvaluateOptions evalOpts;
    double totalCost = 0;
    std::vector<double> residuals;
    problem.Evaluate(evalOpts,&totalCost,&residuals,nullptr,nullptr);

    double rmse = 0;
    {
        double sum = 0;
        for(int i = 0;i < residuals.size();i++){
            double r = residuals[i];
            sum += r*r;
        }
        rmse = std::sqrt(sum/residuals.size());
    }
    std::cout << "RMSE = " << rmse << "\n";

    std::function<double(double)> w = [parameters](double t) -> double{
        return (parameters[1] + parameters[0]*std::sin(parameters[2]*t+parameters[3]));
    };
    
    Rect drawRect(0,0,1280,720);
    Mat fitComparisonImg(Size(1280,720),CV_8UC3,Scalar(255,255,255));

    std::vector<Point> originalPoints(omegas.size()),fittedPoints(omegas.size());
    for(int i = 0;i < omegas.size();i++){
        originalPoints[i] = map2point(tData[i],omegas[i],drawRect,0,tData.back(),0,2);
        fittedPoints[i] = map2point(tData[i],w(tData[i]),drawRect,0,tData.back(),0,2);
    }
    rectangle(fitComparisonImg,drawRect,Scalar(0,0,0),1);

    polylines(fitComparisonImg,originalPoints,false,Scalar(0,255,0),1);
    polylines(fitComparisonImg,fittedPoints,false,Scalar(0,0,255),1);

    putText(fitComparisonImg,"Original",Point(20,600),FONT_HERSHEY_SIMPLEX,1.0,Scalar(0,255,0),1);
    putText(fitComparisonImg,"Fitted",Point(20,660),FONT_HERSHEY_SIMPLEX,1.0,Scalar(0,0,255),1);

    imwrite("result/task2_fit/fit_comparison.png",fitComparisonImg);

    Mat residualsImg(Size(1280,720),CV_8UC3,Scalar(255,255,255));
    std::vector<Point> residualPoints(omegas.size());

    for(int i = 0;i < omegas.size();i++){
        residualPoints[i] = map2point(tData[i],residuals[i],drawRect,0,tData.back(),-0.2,0.2);
    }

    polylines(residualsImg,residualPoints,false,Scalar(0,0,255),1);

    imwrite("result/task2_fit/residuals.png",residualsImg);
    
    return 0;
}