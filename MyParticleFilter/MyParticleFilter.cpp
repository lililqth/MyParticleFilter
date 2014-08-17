#include <opencv2/opencv.hpp>
#include <time.h>
#include <iostream>
const int Num = 10; //帧差的间隔
using namespace std;
using namespace cv;
IplImage *curFrame = NULL; // 当前帧
int Wid, Hei;//图像的大小
int WidIn, HeiIn;//  输入的半宽和半高
bool bSelectObject = false;
Point origin;
Rect selection;
bool pause = false; //是否暂停 
bool track = false; //是否跟踪
void mouseHandler(int event, int x, int y, int flags, void *param)
{
	int centerX, centerY;
	if (bSelectObject)
	{
		selection.x = MIN(origin.x, x);
		selection.y = MIN(origin.y, y);
		selection.width = abs(x - origin.x);
		selection.height = abs(y - origin.y);
	}
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		bSelectObject = true;
		track = false; 
		pause = true;
		break;
	case CV_EVENT_LBUTTONUP:
		bSelectObject = false;
		centerX = selection.x + selection.width / 2;
		centerY = selection.y + selection.height / 2;
		WidIn = selection.width / 2;
		HeiIn = selection.height / 2;
		track = true;
		pause = false;
		break;
	default:
		break;
	}
}
int main(int argc, char *argv[])
{
	int FrameNum = 0; //帧号
	CvCapture *capture = 0;
	capture = cvCaptureFromAVI("../13.avi");
	IplImage *Frame[Num];

	for (int i=0; i<Num; i++)
	{
		Frame[i] = NULL;
	}
	bool start = false;

	IplImage *curFrameGray = NULL;
	IplImage *frameGray = NULL;
	while (capture)
	{
		curFrame = cvQueryFrame(capture);

		if(start == false)
		{
			start = true;
		}
		if(track == true)
		{
			cvRectangle(curFrame,cvPoint(selection.x, selection.y),cvPoint(selection.x + selection.width,selection.height + selection.y),cvScalar(255,0,0),2,8,0);
		}
		cvShowImage("vedio", curFrame);
		cvSetMouseCallback("vedio", mouseHandler, 0);
		if(pause)
		{
			cvWaitKey(400);
		}
		else
		{
			cvWaitKey(10);
		}
	}
	cvReleaseImage(&curFrame);
	cvDestroyAllWindows();
		
}