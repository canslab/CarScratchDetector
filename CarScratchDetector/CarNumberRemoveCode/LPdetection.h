#pragma once
#include <vector>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "ConnectedComponent.h"

using namespace cv;

class CQuad;

class LPdetection
{
public:

	LPdetection(void);
	~LPdetection(void);

	std::vector<float> white_point;
	std::vector<float> green_point;

	bool green_plate = false;
	bool white_plate = false;

	void run(Mat src);

private:
	bool inSameLP(CConnectedComponent &a, CConnectedComponent &b, bool isGreen);
	bool GreenLPdetection(IplImage* src);
	bool WhiteLPdetection(IplImage* src);
	//std::vector<Mat> YellowNEWLPdetection(IplImage* src);
	//std::vector<Mat> YellowOLDLPdetection(IplImage* src);

	Mat rectifyImage(Mat src, CQuad quad, int type);
	void lineEstimate(std::vector<CConnectedComponent> a, double &lineTA, double &lineTB, double &lineBA, double &lineBB);
};

class CQuad
{
public:
	Point2f m_LD;
	Point2f m_LU;
	Point2f m_RD;
	Point2f m_RU;

	CQuad(void)
	{
		m_LD.x = 0; m_LD.y = 0;
		m_LU.x = 0; m_LU.y = 0;
		m_RD.x = 0; m_RD.y = 0;
		m_RU.x = 0; m_RU.y = 0;
	}
	CQuad(Point2f LU, Point2f LD, Point2f RU, Point2f RD)
	{
		m_LU = LU;
		m_LD = LD;
		m_RU = RU;
		m_RD = RD;
	}
	~CQuad(void)
	{

	}
};

void RemoveNumberPlate(cv::Mat in_srcImage);