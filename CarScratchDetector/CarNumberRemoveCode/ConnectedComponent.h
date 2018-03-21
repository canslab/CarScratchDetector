#pragma once

#include "IplImageWrapper.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

class CConnectedComponent
{
public:
	CConnectedComponent(void);
	~CConnectedComponent(void);

	// 바이너리 이미지로부터 connected component(CC) 추출
	// CC로 추출되기 위한 최소 픽셀 개수, 최대 픽셀 개수, 가로세로비율 등을 제한
	static std::vector<CConnectedComponent> CCFiltering(Mat binary, const BYTE objectcolor, const BYTE backgroundcolor,
		int minPixel = 50, int maxPixel = 25000, bool longitudeTest = true);

	// vector list로 표현된 CC들 중 index1과 index2에 해당하는 CC 두개를 병합하여 하나의 CC로 만듦
	static void mergeCC(std::vector<CConnectedComponent> &ccList, int index1, int index2);

public:

	Point center;
	// bounding box 정보
	Point lu, ld, ru, rd;
	Point T, B;
	// 높이, 넓이
	int height, width;
	// CC를 구성하는 픽셀들의 좌표벡터(원본 바이너리 이미지 기준)
	std::vector<Point> ptArray;
	int npixels;
};

