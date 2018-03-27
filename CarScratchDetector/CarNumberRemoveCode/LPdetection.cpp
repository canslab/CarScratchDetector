#include "LPdetection.h"
#include <fstream>
#include <iostream>

// 함수 설명 : 클래스 생성자
// 리턴 : 없음
LPdetection::LPdetection(void) :toBeRemovedRegionWhite(0)
{
}


// 함수 설명 : 클래스 소멸자
// 리턴 : 없음
LPdetection::~LPdetection(void)
{
}


// 함수 설명 : 두 connected component(CC)가 같은 번호판의 연속한 숫자인지 판단
// 리턴 : 연속한 숫자로 판단되면 true, 아니면 false를 return 
bool LPdetection::inSameLP(
	CConnectedComponent &a,			// 연속한 숫자인지 판단할 두 CC중 하나
	CConnectedComponent &b,			// 연속한 숫자인지 판단할 두 CC중 하나
	bool isGreen					// 연속한 숫자인지 판단할 번호판이 녹색 번호판인지, 흰색번호판인지 가리키는 flag. true면 녹색, false면 흰색.
)
{
	bool Hoverlap = false;
	bool SimilarHeight = false;
	bool nearPosition = false;
	bool SimilarScale = false;

	int maxh = 0, minh = 0;
	if (a.height > b.height)
	{
		maxh = a.height;
		minh = b.height;
	}
	else
	{
		maxh = b.height;
		minh = a.height;
	}


	// 세로 방향으로 일정 범위 안에 두 CC가 있는지 확인
	if ((b.ld.y - a.ru.y > minh*0.5 && a.rd.y - b.lu.y > minh*0.5) || (a.ld.y - b.ru.y > minh*0.5 && b.rd.y - a.lu.y > minh*0.5))
	{
		Hoverlap = true;
	}

	// 비슷한 높이를 가지는지 확인
	if ((double)maxh / (double)minh < 1.1)
	{
		SimilarHeight = true;
	}

	// 인접해 있는지 확인
	if (abs(a.center.x - b.center.x) < 1.1*minh  && abs(a.center.x - b.center.x) > 0.5*minh)
	{
		nearPosition = true;
	}

	// 녹색 번호판, 흰색 번호판인지 여부에 따라 픽셀수가 얼마나 비슷한지 확인
	if (isGreen == true)
	{
		if ((double)a.npixels / b.npixels > 1 / 4 && (double)a.npixels / b.npixels < 4)
		{
			SimilarScale = true;
		}
	}
	else
	{
		if ((double)a.npixels / b.npixels > 1 / 3 && (double)a.npixels / b.npixels < 3)
		{
			SimilarScale = true;
		}
	}

	// 모든 테스트를 통과한 경우 인접한 번호에 해당하는 CC로 판단
	if (Hoverlap == true && SimilarHeight == true && nearPosition == true && SimilarScale == true)
		return true;
	else
		return false;
}


#pragma optimize("gpsy", off)
// 함수 설명 : 주어진 영상에서 녹색 번호판을 detection한 후 detection된 번호판들을 crop하여 그 영상들을 vector array로 출력한다.
// 리턴 : 검출된 번호판들을 각각 crop하고 정면시점으로 warping한 결과영상들을 저장한 vector array
bool LPdetection::GreenLPdetection(
	IplImage* src			// 번호판을 검출할 대상이 되는 입력 영상
)
{
	green_point.clear();
	int height = src->height;
	int width = src->width;
	bool green_alarm = false;

	IplImage* blur1 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	IplImage* blur2 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	IplImage* blur3 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);

	IplImage* binary1 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	IplImage* binary2 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	cvSetZero(binary1);
	cvSetZero(binary2);

	BwImage wblur1(blur1), wblur2(blur2), wblur3(blur3), wbinary1(binary1), wbinary2(binary2), wsrc(src);
	cvSmooth(src, blur1, CV_GAUSSIAN, 5);
	cvSmooth(src, blur2, CV_GAUSSIAN, 9);
	cvSmooth(src, blur3, CV_BLUR, 41, 41);


	// DoG 분석을 통해 edge에 해당하는 픽셀을 binary map으로 추출
	for (int h = 0; h < src->height; h++)
	{
		for (int w = 0; w < src->width; w++)
		{
			if (wblur1[h][w] - wblur2[h][w] < 0)
				wbinary1[h][w] = 255;
		}
	}


	for (int h = 0; h < height; h++)
	{
		for (int w = 0; w < width; w++)
		{
			bool logic1 = (wblur3[h][w] > wsrc[h][w]);
			bool logic2 = abs(wblur3[h][w] - wsrc[h][w]) < 0.01 * 255;
			wbinary2[h][w] = (logic1 || logic2 || wbinary1[h][w] == 255 ? 0 : 255);
		}
	}

	const BYTE object_color = 255, background_color = 0;


	// binary map으로부터 CC 추출
	std::vector<CConnectedComponent> CarNum = CConnectedComponent::CCFiltering(cvarrToMat(binary2), object_color, background_color);


	std::vector<int> temp;
	for (int i = 0; i < CarNum.size(); i++)
	{
		temp.push_back(i);
	}

	for (int i = 0; i < CarNum.size(); i++)
	{
		// 인접한 4개의 번호에 해당하는 CC들을 추출하여 저장(번호판의 4자리 숫자)
		for (int j = i + 1; j < CarNum.size(); j++)
		{
			bool sameLP = false;
			sameLP = inSameLP(CarNum[i], CarNum[j], true);

			if (sameLP == true)
			{
				if (temp[i] == i && temp[j] == j)
				{
					temp[i] = -1;
					temp[j] = i;
				}
				else if (temp[i] == -1 && temp[j] == j)
				{
					temp[j] = i;
				}

				else if (temp[i] == -1 && temp[j] != j)
				{
					temp[i] = temp[j];
					for (int k = 0; k < CarNum.size(); k++)
					{
						if (temp[k] == i)
						{
							temp[k] = temp[j];
						}
					}
				}
				else if (temp[i] != i && temp[j] == j)
				{
					temp[j] = temp[i];
				}
				else if (temp[i] == i && temp[j] != j)
				{
					temp[i] = temp[j];
				}

				else if (temp[i] != i && temp[j] != j && temp[i] != temp[j])
				{
					if (temp[i] < temp[j])
					{
						temp[temp[j]] = temp[i];
						int m = temp[j];
						for (int k = 0; k < CarNum.size(); k++)
						{
							if (temp[k] == m)
								temp[k] = temp[i];
						}
					}
					else
					{
						temp[temp[i]] = temp[j];
						int m = temp[i];
						for (int k = 0; k < CarNum.size(); k++)
						{
							if (temp[k] == m)
								temp[k] = temp[j];
						}
					}
				}
			}

		}
	}


	for (int i = 0; i < CarNum.size(); i++)
	{
		std::vector<CConnectedComponent> a;
		if (temp[i] == -1)
		{
			a.push_back(CarNum[i]);
			for (int j = i + 1; j < CarNum.size(); j++)
			{
				if (temp[j] == i)
				{
					a.push_back(CarNum[j]);
				}
			}
		}


		// 인접한 CC가 4개인 경우 번호판으로 판단, 정보를 저장(후에 인식 모듈로 전달)
		if (a.size() == 4)
		{
			int min_x = src->width, max_x = -1, min_center_x, max_center_y;
			CConnectedComponent bb, cc;

			for (std::vector<CConnectedComponent>::iterator iter = a.begin(); iter != a.end(); ++iter)
			{
				if (min_x > (*iter).ld.x)
				{
					min_x = (*iter).ld.x;
					bb = (*iter);
					min_center_x = (*iter).center.x;
				}
				if (max_x < (*iter).rd.x)
				{
					max_x = (*iter).rd.x;
					cc = (*iter);
					max_center_y = (*iter).center.x;
				}
			}

			double lineTA, lineTB, lineBA, lineBB;
			lineEstimate(a, lineTA, lineTB, lineBA, lineBB);

			//번호판 4개 좌표

			Point2f LD(min_x, lineBA*min_x + lineBB);
			Point2f RD(max_x, lineBA*max_x + lineBB);
			Point2f LU(min_x, lineTA*min_x + lineTB);
			Point2f RU(max_x, lineTA*max_x + lineTB);

			toBeRemovedRegionGreen.push_back(LU);
			toBeRemovedRegionGreen.push_back(LD);
			toBeRemovedRegionGreen.push_back(RD);
			toBeRemovedRegionGreen.push_back(RU);

			green_point.push_back(min_x);
			green_point.push_back(lineTA*min_x + lineTB);
			green_point.push_back(max_x);
			green_point.push_back(lineBA*max_x + lineBB);

			green_alarm = true;
		}

	}

	cvReleaseImage(&blur1);
	cvReleaseImage(&blur2);
	cvReleaseImage(&blur3);
	cvReleaseImage(&binary1);
	cvReleaseImage(&binary2);

	return green_alarm;
}
#pragma optimize("gpsy", on)

// 함수 설명 : 주어진 영상에서 흰색 번호판을 detection한 후 detection된 번호판들을 crop하여 그 영상들을 vector array로 출력한다.
// 리턴 : 검출된 번호판들을 각각 crop하고 정면시점으로 warping한 결과영상들을 저장한 vector array
#pragma optimize("gpsy", off)
bool LPdetection::WhiteLPdetection(
	IplImage* src					// 번호판을 검출할 대상이 되는 입력 영상
)
{
	white_point.clear();
	int height = src->height;
	int width = src->width;
	bool white_alarm = false;

	IplImage* blur1 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	IplImage* blur2 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	IplImage* blur3 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	IplImage* binary = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	cvSetZero(binary);

	BwImage wblur1(blur1), wblur2(blur2), wblur3(blur3), wbinary(binary), wsrc(src);
	cvSmooth(src, blur1, CV_BLUR, 5, 5);
	cvSmooth(src, blur2, CV_BLUR, 11, 11);
	cvSmooth(src, blur3, CV_BLUR, 41, 41);

	for (int h = 0; h < src->height; h++)
	{
		for (int w = 0; w < src->width; w++)
		{
			if (wblur1[h][w] - wblur2[h][w]<0 && abs(wblur2[h][w] - wsrc[h][w]) > 0.01 * 255)
				wbinary[h][w] = 255;
		}
	}

	const BYTE object_color = 255, background_color = 0;

	std::vector<CConnectedComponent> CarNum = CConnectedComponent::CCFiltering(cvarrToMat(binary), object_color, background_color);

	std::vector<int> temp;
	for (int i = 0; i < CarNum.size(); i++) temp.push_back(i);

	for (int i = 0; i < CarNum.size(); i++)
	{
		for (int j = i + 1; j < CarNum.size(); j++)
		{
			bool sameLP = false;
			sameLP = inSameLP(CarNum[i], CarNum[j], false);

			if (sameLP == true)
			{
				if (temp[i] == i && temp[j] == j)
				{
					temp[i] = -1;
					temp[j] = i;
				}
				else if (temp[i] == -1 && temp[j] == j)
				{
					temp[j] = i;
				}

				else if (temp[i] == -1 && temp[j] != j)
				{
					temp[i] = temp[j];
					for (int k = 0; k < CarNum.size(); k++)
					{
						if (temp[k] == i)
						{
							temp[k] = temp[j];
						}
					}
				}
				else if (temp[i] != i && temp[j] == j)
				{
					temp[j] = temp[i];
				}
				else if (temp[i] == i && temp[j] != j)
				{
					temp[i] = temp[j];
				}

				else if (temp[i] != i && temp[j] != j && temp[i] != temp[j])
				{
					if (temp[i] < temp[j])
					{
						temp[temp[j]] = temp[i];
						int m = temp[j];
						for (int k = 0; k < CarNum.size(); k++)
						{
							if (temp[k] == m)
								temp[k] = temp[i];
						}
					}
					else
					{
						temp[temp[i]] = temp[j];
						int m = temp[i];
						for (int k = 0; k < CarNum.size(); k++)
						{
							if (temp[k] == m)
								temp[k] = temp[j];
						}
					}
				}
			}

		}
	}


	for (int i = 0; i < CarNum.size(); i++)
	{
		std::vector<CConnectedComponent> a;
		if (temp[i] == -1)
		{
			a.push_back(CarNum[i]);
			for (int j = i + 1; j < CarNum.size(); j++)
			{
				if (temp[j] == i)
				{
					a.push_back(CarNum[j]);
				}
			}
		}


		if (a.size() == 4)
		{
			int min_x = src->width, max_x = -1;
			CConnectedComponent bb, cc;

			for (std::vector<CConnectedComponent>::iterator iter = a.begin(); iter != a.end(); ++iter)
			{
				if (min_x > (*iter).ld.x)
				{
					min_x = (*iter).ld.x;
					bb = (*iter);

				}
				if (max_x < (*iter).rd.x)
				{
					max_x = (*iter).rd.x;
					cc = (*iter);
				}

			}

			double lineTA, lineTB, lineBA, lineBB;
			lineEstimate(a, lineTA, lineTB, lineBA, lineBB);

			//번호판 4개 좌표
			Point2f LU(min_x, lineTA*min_x + lineTB);
			Point2f LD(min_x, lineBA*min_x + lineBB);
			Point2f RD(max_x, lineBA*max_x + lineBB);
			Point2f RU(max_x, lineTA*max_x + lineTB);

			toBeRemovedRegionWhite.push_back(LU);
			toBeRemovedRegionWhite.push_back(LD);
			toBeRemovedRegionWhite.push_back(RD);
			toBeRemovedRegionWhite.push_back(RU);

			white_point.push_back(min_x);
			white_point.push_back(lineTA*min_x + lineTB);
			white_point.push_back(max_x);
			white_point.push_back(lineBA*max_x + lineBB);

			white_alarm = true;
		}
	}

	cvReleaseImage(&blur1);
	cvReleaseImage(&blur2);
	cvReleaseImage(&blur3);
	cvReleaseImage(&binary);

	return white_alarm;
}
#pragma optimize("gpsy", on)

// 함수 설명 : 주어진 영상에서 번호판을 detection한 후 detection된 번호판들을 crop하여 그 영상들을 vector array로 출력한다.
//			   mask image를 통해 ROI를 설정할 경우 ROI외의 영역을 전부 0으로 만들어 계산량을 줄일 수 있다.
//			   흰색, 녹색 번호판에 대해 각각 검출을 시행하며, 어느 색 검출 알고리즘으로부터 번호판이 검출되었는지도 결과에 함께 저장한다.
// 리턴 : 검출된 번호판들을 각각 crop하고 정면시점으로 warping한 결과영상들을 저장한 vector array
//		  하나의 번호판에 대해 각각 (번호판 영상, 번호판 타입)으로 짝지어진 pair를 vector array 하나의 element로 저장한다.
//		  번호판 타입은 녹색인지 흰색인지를 숫자로 표시한다(1 : 흰색, 3 : 녹색)
void LPdetection::run(
	Mat src					// 번호판을 검출할 대상이 되는 입력 영상
							// Mat()을 넣어주는 경우 ROI를 전체로 설정한 것으로 취급
							// mask Image가 0의 값을 갖는 영역은 입력 영상도 0으로 만들어 검출 영역에서 제외한다.
)
{
	Mat graySrc;
	if (src.channels() == 3)
		cvtColor(src, graySrc, CV_BGR2GRAY);
	else
		graySrc = src;

	IplImage *iplimgSrc = &IplImage(graySrc);

	green_plate = GreenLPdetection(iplimgSrc);
	white_plate = WhiteLPdetection(iplimgSrc);

}


// 함수 설명 : 연속한 숫자들을 나타내는 connected component(CC)로 부터 숫자들의 윗부분을 잇는 선, 아랫부분을 잇는 선의 방정식을 각각 계산한다.
//			   계산된 선의 방정식을 이용해 실제 네 숫자를 둘러싸는 사각형 영역을 계산한 후, 나중에 이 결과를 이용해 번호판 영상을 정면 시점으로 warping한다.
// 리턴 : 없음
void LPdetection::lineEstimate(
	std::vector<CConnectedComponent> a,		// 연속한 숫자들의 CC를 포함하는 vector array
	double &lineTA,							// 윗부분을 잇는 직선의 기울기
	double &lineTB,							// 윗부분을 잇는 직선의 y절편
	double &lineBA,							// 아랫부분을 잇는 직선의 기울기
	double &lineBB							// 아랫부분을 잇는 직선의 y절편
)
{
	Mat sysTX(a.size(), 2, CV_32FC1);
	Mat sysTY(a.size(), 1, CV_32FC1);

	Mat sysBX(a.size(), 2, CV_32FC1);
	Mat sysBY(a.size(), 1, CV_32FC1);

	for (int i = 0; i < a.size(); i++)
	{
		sysTX.at<float>(i, 0) = a[i].T.x;
		sysTX.at<float>(i, 1) = 1;
		sysTY.at<float>(i, 0) = a[i].T.y;

		sysBX.at<float>(i, 0) = a[i].B.x;
		sysBX.at<float>(i, 1) = 1;
		sysBY.at<float>(i, 0) = a[i].B.y;
	}

	Mat lineT;
	solve(sysTX, sysTY, lineT, DECOMP_LU | DECOMP_NORMAL);
	lineTA = lineT.at<float>(0, 0);
	lineTB = lineT.at<float>(1, 0);

	Mat lineB;
	solve(sysBX, sysBY, lineB, DECOMP_LU | DECOMP_NORMAL);
	lineBA = lineB.at<float>(0, 0);
	lineBB = lineB.at<float>(1, 0);
}


// 함수 설명 : 입력 영상에서 검출된 번호판을 둘러싸는 사각형을 입력으로 받아,
//			   이 사각형을 직사각형으로 만드는 warping을 계산한 후 실제로 영상을 warping하여 정면시점으로 변환한 후,
//			   시점변환된 영상에서 실제 번호판에 해당하는 영역을 crop하여 출력한다.
// 리턴 : 정면시점으로 변환되고 번호판 영역에 맞게 crop된 영상
Mat LPdetection::rectifyImage(
	Mat src,				// 번호판이 검출된 입력영상
	CQuad quad,				// 번호판의 네개의 연속된 숫자를 둘러싸는 최소의 사각형 꼭지점 좌표들
	int type				// 녹색 영상에서 검출되었으면 3, 흰색 영상에서 검출되었으면 1
)
{
	Point2f from[4];
	from[0].x = quad.m_LU.x;
	from[0].y = quad.m_LU.y;
	from[1].x = quad.m_LD.x;
	from[1].y = quad.m_LD.y;
	from[2].x = quad.m_RU.x;
	from[2].y = quad.m_RU.y;
	from[3].x = quad.m_RD.x;
	from[3].y = quad.m_RD.y;

	Point2f to[4];
	to[0].x = quad.m_LU.x;
	to[0].y = quad.m_LU.y;
	to[1].x = quad.m_LU.x;
	to[1].y = quad.m_LD.y;
	to[2].x = quad.m_RU.x;
	to[2].y = quad.m_LU.y;
	to[3].x = quad.m_RU.x;
	to[3].y = quad.m_LD.y;
	Point2f warpedCenter;
	warpedCenter.x = (quad.m_LU.x + quad.m_RU.x) / 2.0f;
	warpedCenter.y = (quad.m_LU.y + quad.m_LD.y) / 2.0f;

	Size sizeOfOriginalImage = Mat(src).size();
	Mat warpedImg(sizeOfOriginalImage, CV_8UC3);

	warpPerspective(Mat(src), warpedImg, getPerspectiveTransform(from, to), sizeOfOriginalImage, INTER_CUBIC);

	int increaseBorder = (int)min((quad.m_RU.x - quad.m_LU.x) / 20.0f + 0.5f, (quad.m_LD.y - quad.m_LU.y) / 20.0f + 0.5f);
	Size patchSize(quad.m_RU.x - quad.m_LU.x + increaseBorder, quad.m_LD.y - quad.m_LU.y + increaseBorder);

	if (patchSize.width <= 0 || patchSize.height <= 0)
	{
		return Mat();
	}

	Mat plateImg;
	if (type == 1)
	{
		warpedCenter -= Point2f(0.0f, patchSize.height * 0.35f);
		patchSize.width *= 1.0f;
		patchSize.height *= 1.7f;
		getRectSubPix(warpedImg, patchSize, warpedCenter, plateImg);
	}
	else if (type == 2)
	{
		warpedCenter -= Point2f(patchSize.width * 0.25f, patchSize.height * 0.35f);
		patchSize.width *= 1.5f;
		patchSize.height *= 1.7f;
		getRectSubPix(warpedImg, patchSize, warpedCenter, plateImg);
	}
	else if (type == 0)
	{
		warpedCenter -= Point2f(patchSize.width * 0.6f, patchSize.height * 0.1f);
		patchSize.width *= 2.2f;
		patchSize.height *= 1.2f;
		getRectSubPix(warpedImg, patchSize, warpedCenter, plateImg);
	}


	return plateImg;
}
void RemoveNumberPlate(cv::Mat in_srcImage)
{
	LPdetection LPdetectionInst;
	bool bShouldBeRemoved = false;
	float lx, ly, rx, ry = 0;
	std::vector<std::vector<cv::Point2i>> area;

	LPdetectionInst.run(in_srcImage);

	if (LPdetectionInst.white_plate == true && LPdetectionInst.green_plate == false)
	{
		//lx ly rx ry 순으로 입력
		lx = LPdetectionInst.white_point.at(0);
		ly = LPdetectionInst.white_point.at(1);
		rx = LPdetectionInst.white_point.at(2);
		ry = LPdetectionInst.white_point.at(3);

		area.push_back(LPdetectionInst.toBeRemovedRegionWhite);
		bShouldBeRemoved = true;
	}

	else if (LPdetectionInst.white_plate == true && LPdetectionInst.green_plate == true)
	{
		//lx ly rx ry 순으로 입력
		lx = LPdetectionInst.white_point.at(0);
		ly = LPdetectionInst.white_point.at(1);
		rx = LPdetectionInst.white_point.at(2);
		ry = LPdetectionInst.white_point.at(3);
		area.push_back(LPdetectionInst.toBeRemovedRegionWhite);

		bShouldBeRemoved = true;
	}

	else if (LPdetectionInst.white_plate == false && LPdetectionInst.green_plate == true)
	{
		//lx ly rx ry 순으로 입력
		lx = LPdetectionInst.green_point.at(0);
		ly = LPdetectionInst.green_point.at(1);
		rx = LPdetectionInst.green_point.at(2);
		ry = LPdetectionInst.green_point.at(3);
		area.push_back(LPdetectionInst.toBeRemovedRegionGreen);
		bShouldBeRemoved = true;
	}

	if (bShouldBeRemoved)
	{
		cv::drawContours(in_srcImage, area, 0, cv::Scalar(0, 0, 0), CV_FILLED, 8);
		//Rect roi(int(lx), int(ly), int(rx - lx), int(ry - ly));
		//in_srcImage(roi) = 0;
	}
}