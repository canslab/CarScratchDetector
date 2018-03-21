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

	// ���̳ʸ� �̹����κ��� connected component(CC) ����
	// CC�� ����Ǳ� ���� �ּ� �ȼ� ����, �ִ� �ȼ� ����, ���μ��κ��� ���� ����
	static std::vector<CConnectedComponent> CCFiltering(Mat binary, const BYTE objectcolor, const BYTE backgroundcolor,
		int minPixel = 50, int maxPixel = 25000, bool longitudeTest = true);

	// vector list�� ǥ���� CC�� �� index1�� index2�� �ش��ϴ� CC �ΰ��� �����Ͽ� �ϳ��� CC�� ����
	static void mergeCC(std::vector<CConnectedComponent> &ccList, int index1, int index2);

public:

	Point center;
	// bounding box ����
	Point lu, ld, ru, rd;
	Point T, B;
	// ����, ����
	int height, width;
	// CC�� �����ϴ� �ȼ����� ��ǥ����(���� ���̳ʸ� �̹��� ����)
	std::vector<Point> ptArray;
	int npixels;
};

