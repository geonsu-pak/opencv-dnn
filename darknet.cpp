/************************************************************************************************
   ::: prerequisite :::
   1. opencv 4.1.2 download and extract to a folder
      https://sourceforge.net/projects/opencvlibrary/files/4.1.2/opencv-4.1.2-vc14_vc15.exe/download
   2. VS Studio settings ( include folder, library folder & library file = opencv_world412(d).lib )
      Tools menu > Options , then in "Projects and Solutions" branch > "VC++ Directories" & Linker
   3. copy opencv_world412(d).dll to executable folder
*************************************************************************************************/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;

int main()
{
    // download from internet two files below
    std::string model = ".\\yolov3.weights";
    std::string config = ".\\yolov3.cfg";
    // Movie file
    VideoCapture cap(".\\samp.MOV");
    
    // readNetFromDarknet(config, model) is also OKay.
    Net network = readNet(model, config, "Darknet");
    // if your opencv was compiled with cuda, then
    //    network.setPreferableBackend(DNN_BACKEND_CUDA);
    //    network.setPreferableTarget(DNN_TARGET_CUDA);
    network.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_DEFAULT);
    network.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CPU);

    for (;;)
    {
        if (!cap.isOpened()) {
            cout << "Video Capture Fail" << endl;
            break;
        }
        cap >> img;

        static Mat blobFromImg;
        bool swapRB = true;  // opencv's BGR -> RGB
        // (416, 416) is fixed size when the model was trained. please see yolov3.cfg
        blobFromImage(img, blobFromImg, 1, Size(416, 416), Scalar(), swapRB, false);

        // normalization -> because darknet was trained with this condition.
        float scale = 1.0 / 255.0;
        Scalar mean = 0;
        network.setInput(blobFromImg, "", scale, mean);

        Mat outMat;
        network.forward(outMat);
        // rows represent number of detected object (proposed region)
        int rowsNoOfDetection = outMat.rows;

        // The coluns looks like this, The first is region center x, center y, width
        // height, The class 1 - N is the column entries, which gives you a number, 
        // where the biggist one corrsponding to most probable class. 
        // [x ; y ; w; h; class 1 ; class 2 ; class 3 ;  ; ;....]
        //  
        int colsCoordinatesPlusClassScore = outMat.cols;
        // Loop over number of detected object. 
        for (int j = 0; j < rowsNoOfDetection; ++j)
        {
            // for each row, the score is from element 5 up
            // to number of classes index (5 - N columns)
            Mat scores = outMat.row(j).colRange(5, colsCoordinatesPlusClassScore);

            Point PositionOfMax;
            double confidence;

            // This function find indexes of min and max confidence and related index of element. 
            // The actual index is match to the concrete class of the object.
            // First parameter is Mat which is row [5fth - END] scores,
            // Second parameter will gives you min value of the scores. NOT needed 
            // confidence gives you a max value of the scores. This is needed, 
            // Third parameter is index of minimal element in scores
            // the last is position of the maximum value.. This is the class!!
            minMaxLoc(scores, 0, &confidence, 0, &PositionOfMax);

            if (confidence > 0.5)
            {
                // thease four lines are
                // [x ; y ; w; h;
                int centerX = (int)(outMat.at<float>(j, 0) * img.cols);
                int centerY = (int)(outMat.at<float>(j, 1) * img.rows);
                int width = (int)(outMat.at<float>(j, 2) * img.cols + 20);
                int height = (int)(outMat.at<float>(j, 3) * img.rows + 100);

                int left = centerX - width / 2;
                int top = centerY - height / 2;


                stringstream ss;
                ss << PositionOfMax.x;
                string clas = ss.str();
                int color = PositionOfMax.x * 10;
                putText(img, clas, Point(left, top), 1, 2, Scalar(color, 255, 255), 2, false);
                stringstream ss2;
                ss << confidence;
                string conf = ss.str();

                rectangle(img, Rect(left, top, width, height), Scalar(color, 0, 0), 2, 8, 0);
            }
        }

        namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
        imshow("Display window", img);
        waitKey(25);
        break;
    }
    return 0;
}
