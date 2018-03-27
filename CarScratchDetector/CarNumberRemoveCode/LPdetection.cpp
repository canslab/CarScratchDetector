#include "LPdetection.h"
#include <fstream>
#include <iostream>

// �Լ� ���� : Ŭ���� ������
// ���� : ����
LPdetection::LPdetection(void) :toBeRemovedRegionWhite(0)
{
}


// �Լ� ���� : Ŭ���� �Ҹ���
// ���� : ����
LPdetection::~LPdetection(void)
{
}


// �Լ� ���� : �� connected component(CC)�� ���� ��ȣ���� ������ �������� �Ǵ�
// ���� : ������ ���ڷ� �ǴܵǸ� true, �ƴϸ� false�� return 
bool LPdetection::inSameLP(
	CConnectedComponent &a,			// ������ �������� �Ǵ��� �� CC�� �ϳ�
	CConnectedComponent &b,			// ������ �������� �Ǵ��� �� CC�� �ϳ�
	bool isGreen					// ������ �������� �Ǵ��� ��ȣ���� ��� ��ȣ������, �����ȣ������ ����Ű�� flag. true�� ���, false�� ���.
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


	// ���� �������� ���� ���� �ȿ� �� CC�� �ִ��� Ȯ��
	if ((b.ld.y - a.ru.y > minh*0.5 && a.rd.y - b.lu.y > minh*0.5) || (a.ld.y - b.ru.y > minh*0.5 && b.rd.y - a.lu.y > minh*0.5))
	{
		Hoverlap = true;
	}

	// ����� ���̸� �������� Ȯ��
	if ((double)maxh / (double)minh < 1.1)
	{
		SimilarHeight = true;
	}

	// ������ �ִ��� Ȯ��
	if (abs(a.center.x - b.center.x) < 1.1*minh  && abs(a.center.x - b.center.x) > 0.5*minh)
	{
		nearPosition = true;
	}

	// ��� ��ȣ��, ��� ��ȣ������ ���ο� ���� �ȼ����� �󸶳� ������� Ȯ��
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

	// ��� �׽�Ʈ�� ����� ��� ������ ��ȣ�� �ش��ϴ� CC�� �Ǵ�
	if (Hoverlap == true && SimilarHeight == true && nearPosition == true && SimilarScale == true)
		return true;
	else
		return false;
}


#pragma optimize("gpsy", off)
// �Լ� ���� : �־��� ���󿡼� ��� ��ȣ���� detection�� �� detection�� ��ȣ�ǵ��� crop�Ͽ� �� ������� vector array�� ����Ѵ�.
// ���� : ����� ��ȣ�ǵ��� ���� crop�ϰ� ����������� warping�� ���������� ������ vector array
bool LPdetection::GreenLPdetection(
	IplImage* src			// ��ȣ���� ������ ����� �Ǵ� �Է� ����
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


	// DoG �м��� ���� edge�� �ش��ϴ� �ȼ��� binary map���� ����
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


	// binary map���κ��� CC ����
	std::vector<CConnectedComponent> CarNum = CConnectedComponent::CCFiltering(cvarrToMat(binary2), object_color, background_color);


	std::vector<int> temp;
	for (int i = 0; i < CarNum.size(); i++)
	{
		temp.push_back(i);
	}

	for (int i = 0; i < CarNum.size(); i++)
	{
		// ������ 4���� ��ȣ�� �ش��ϴ� CC���� �����Ͽ� ����(��ȣ���� 4�ڸ� ����)
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


		// ������ CC�� 4���� ��� ��ȣ������ �Ǵ�, ������ ����(�Ŀ� �ν� ���� ����)
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

			//��ȣ�� 4�� ��ǥ

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

// �Լ� ���� : �־��� ���󿡼� ��� ��ȣ���� detection�� �� detection�� ��ȣ�ǵ��� crop�Ͽ� �� ������� vector array�� ����Ѵ�.
// ���� : ����� ��ȣ�ǵ��� ���� crop�ϰ� ����������� warping�� ���������� ������ vector array
#pragma optimize("gpsy", off)
bool LPdetection::WhiteLPdetection(
	IplImage* src					// ��ȣ���� ������ ����� �Ǵ� �Է� ����
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

			//��ȣ�� 4�� ��ǥ
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

// �Լ� ���� : �־��� ���󿡼� ��ȣ���� detection�� �� detection�� ��ȣ�ǵ��� crop�Ͽ� �� ������� vector array�� ����Ѵ�.
//			   mask image�� ���� ROI�� ������ ��� ROI���� ������ ���� 0���� ����� ��귮�� ���� �� �ִ�.
//			   ���, ��� ��ȣ�ǿ� ���� ���� ������ �����ϸ�, ��� �� ���� �˰������κ��� ��ȣ���� ����Ǿ������� ����� �Բ� �����Ѵ�.
// ���� : ����� ��ȣ�ǵ��� ���� crop�ϰ� ����������� warping�� ���������� ������ vector array
//		  �ϳ��� ��ȣ�ǿ� ���� ���� (��ȣ�� ����, ��ȣ�� Ÿ��)���� ¦������ pair�� vector array �ϳ��� element�� �����Ѵ�.
//		  ��ȣ�� Ÿ���� ������� ��������� ���ڷ� ǥ���Ѵ�(1 : ���, 3 : ���)
void LPdetection::run(
	Mat src					// ��ȣ���� ������ ����� �Ǵ� �Է� ����
							// Mat()�� �־��ִ� ��� ROI�� ��ü�� ������ ������ ���
							// mask Image�� 0�� ���� ���� ������ �Է� ���� 0���� ����� ���� �������� �����Ѵ�.
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


// �Լ� ���� : ������ ���ڵ��� ��Ÿ���� connected component(CC)�� ���� ���ڵ��� ���κ��� �մ� ��, �Ʒ��κ��� �մ� ���� �������� ���� ����Ѵ�.
//			   ���� ���� �������� �̿��� ���� �� ���ڸ� �ѷ��δ� �簢�� ������ ����� ��, ���߿� �� ����� �̿��� ��ȣ�� ������ ���� �������� warping�Ѵ�.
// ���� : ����
void LPdetection::lineEstimate(
	std::vector<CConnectedComponent> a,		// ������ ���ڵ��� CC�� �����ϴ� vector array
	double &lineTA,							// ���κ��� �մ� ������ ����
	double &lineTB,							// ���κ��� �մ� ������ y����
	double &lineBA,							// �Ʒ��κ��� �մ� ������ ����
	double &lineBB							// �Ʒ��κ��� �մ� ������ y����
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


// �Լ� ���� : �Է� ���󿡼� ����� ��ȣ���� �ѷ��δ� �簢���� �Է����� �޾�,
//			   �� �簢���� ���簢������ ����� warping�� ����� �� ������ ������ warping�Ͽ� ����������� ��ȯ�� ��,
//			   ������ȯ�� ���󿡼� ���� ��ȣ�ǿ� �ش��ϴ� ������ crop�Ͽ� ����Ѵ�.
// ���� : ����������� ��ȯ�ǰ� ��ȣ�� ������ �°� crop�� ����
Mat LPdetection::rectifyImage(
	Mat src,				// ��ȣ���� ����� �Է¿���
	CQuad quad,				// ��ȣ���� �װ��� ���ӵ� ���ڸ� �ѷ��δ� �ּ��� �簢�� ������ ��ǥ��
	int type				// ��� ���󿡼� ����Ǿ����� 3, ��� ���󿡼� ����Ǿ����� 1
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
		//lx ly rx ry ������ �Է�
		lx = LPdetectionInst.white_point.at(0);
		ly = LPdetectionInst.white_point.at(1);
		rx = LPdetectionInst.white_point.at(2);
		ry = LPdetectionInst.white_point.at(3);

		area.push_back(LPdetectionInst.toBeRemovedRegionWhite);
		bShouldBeRemoved = true;
	}

	else if (LPdetectionInst.white_plate == true && LPdetectionInst.green_plate == true)
	{
		//lx ly rx ry ������ �Է�
		lx = LPdetectionInst.white_point.at(0);
		ly = LPdetectionInst.white_point.at(1);
		rx = LPdetectionInst.white_point.at(2);
		ry = LPdetectionInst.white_point.at(3);
		area.push_back(LPdetectionInst.toBeRemovedRegionWhite);

		bShouldBeRemoved = true;
	}

	else if (LPdetectionInst.white_plate == false && LPdetectionInst.green_plate == true)
	{
		//lx ly rx ry ������ �Է�
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