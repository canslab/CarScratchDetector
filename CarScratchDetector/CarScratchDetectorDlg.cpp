
// CarScratchDetectorDlg.cpp : 구현 파일
//

#include "stdafx.h"
#include "CarScratchDetector.h"
#include "CarScratchDetectorDlg.h"
#include "afxdialogex.h"
#include "AlgorithmCode\AlgorithmCollection.h"
#include "CarNumberRemoveCode\LPdetection.h"
#include "FeatureDialog.h"
#include "RankingDialog.h"
#include "TrainingDialog.h"
#include <fstream>


#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CCarScratchDetectorDlg 대화 상자

CCarScratchDetectorDlg::CCarScratchDetectorDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_CARSCRATCHDETECTOR_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CCarScratchDetectorDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_INPUTIMAGEWIDTHTEXT, m_imageWidthText);
	DDX_Control(pDX, IDC_INPUTIMAGEHEIGHTTEXT, m_imageHeightText);
	DDX_Control(pDX, IDC_SPATIALRADIUSEDIT, m_spatialRadiusEditBox);
	DDX_Control(pDX, IDC_COLORRADIUSEDIT, m_colorRadiusEditBox);
}

BEGIN_MESSAGE_MAP(CCarScratchDetectorDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_OPENIMAGEFILEBTN, &CCarScratchDetectorDlg::OnBnClickedOpenimagefilebtn)
	ON_BN_CLICKED(IDC_CLEARBTN, &CCarScratchDetectorDlg::OnBnClickedClearbtn)
	ON_BN_CLICKED(IDC_BUTTON2, &CCarScratchDetectorDlg::OnCurrentImageTestButton)
	ON_BN_CLICKED(IDC_LOADDBBUTTON, &CCarScratchDetectorDlg::OnBnClickedLoaddbbutton)
	ON_BN_CLICKED(IDC_TRAININGSESSIONBUTTON, &CCarScratchDetectorDlg::OnBnClickedTrainingsessionbutton)
END_MESSAGE_MAP()
// CCarScratchDetectorDlg 메시지 처리기

BOOL CCarScratchDetectorDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

									// TODO: 여기에 추가 초기화 작업을 추가합니다.	
	m_spatialRadiusEditBox.SetWindowTextA("10");
	m_colorRadiusEditBox.SetWindowTextA("16");

	GetDlgItem(IDC_BUTTON2)->EnableWindow(FALSE);
	GetDlgItem(IDC_CLEARBTN)->EnableWindow(FALSE);
	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 응용 프로그램의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CCarScratchDetectorDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CCarScratchDetectorDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CCarScratchDetectorDlg::OnBnClickedOpenimagefilebtn()
{
	// TODO: Add your control notification handler code here
	char szFilter[] = "Image|*.BMP;*.PNG;*.JPG;*.JPEG";
	CFileDialog dlg(TRUE, NULL, NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter, AfxGetMainWnd());

	if (dlg.DoModal() == IDOK)
	{
		CString cstrImgPath = dlg.GetPathName();

		if (m_currentMat.rows > 0)
		{
			m_currentMat.release();
		}

		m_currentMat = cv::imread(cv::String(cstrImgPath).c_str());
		auto originalWidth = m_currentMat.cols;
		auto originalHeight = m_currentMat.rows;

		ResizeImageUsingThreshold(m_currentMat, 100000);

		// If input image exists
		if (m_currentMat.data)
		{
			cv::imshow("Current Input Image", m_currentMat);
			m_imageWidthText.SetWindowTextA(std::to_string(m_currentMat.cols).c_str());
			m_imageHeightText.SetWindowTextA(std::to_string(m_currentMat.rows).c_str());
			GetDlgItem(IDC_OPENIMAGEFILEBTN)->EnableWindow(FALSE);
			GetDlgItem(IDC_CLEARBTN)->EnableWindow(TRUE);

			// Get current file's name
			m_currentFileName = dlg.GetFileTitle();

			// Extract image descriptor from m_currentMat
			ExtractImageDescriptorFromMat(m_currentMat, m_currentImageDescriptor, true);

			// Label UI Update
			std::string averagePointText = "(" + std::to_string(m_currentImageDescriptor.m_largestClusterMeanPosition.x)
				+ ", " + std::to_string(m_currentImageDescriptor.m_largestClusterMeanPosition.y) + ")";
			GetDlgItem(IDC_LARGESTCLUSTERAVERAGEPOINTTEXT)->SetWindowTextA(averagePointText.c_str());
			GetDlgItem(IDC_LARGESTCLUSTERXSTDDEVTEXT)->SetWindowTextA(std::to_string(m_currentImageDescriptor.m_largestClusterStdDev[0]).c_str());
			GetDlgItem(IDC_LARGESTCLUSTERYSTDDEVTEXT)->SetWindowTextA(std::to_string(m_currentImageDescriptor.m_largestClusterStdDev[1]).c_str());
			GetDlgItem(IDC_LARGESTCLUSTERSKEWNESSTEXT)->SetWindowTextA(std::to_string(m_currentImageDescriptor.m_largestClusterSkewness).c_str());
			GetDlgItem(IDC_LARGESTCLUSTERORIENTATIONTEXT)->SetWindowTextA(std::to_string(m_currentImageDescriptor.m_largestClusterOrientation).c_str());
			GetDlgItem(IDC_LARGESTCLUSTERLARGEEIGVALUETEXT)->SetWindowTextA(std::to_string(m_currentImageDescriptor.m_largestClusterLargeEigenValue).c_str());
			GetDlgItem(IDC_LARGESTCLUSTERSMALLEIGVALUETEXT)->SetWindowTextA(std::to_string(m_currentImageDescriptor.m_largestClusterSmallEigenValue).c_str());
			GetDlgItem(IDC_LARGESTCLUSTEREIGVALUERATIOTEXT)->SetWindowTextA(std::to_string(m_currentImageDescriptor.m_largestClusterEigenvalueRatio).c_str());
			GetDlgItem(IDC_LARGESTCLUSTERTOTALPOINTSTEXT)->SetWindowTextA(std::to_string(m_currentImageDescriptor.m_totalNumberOfPointsOfLargestCluster).c_str());

			averagePointText = "(" + std::to_string(m_currentImageDescriptor.m_globalMeanPosition.x)
				+ ", " + std::to_string(m_currentImageDescriptor.m_globalMeanPosition.y) + ")";
			GetDlgItem(IDC_TOTALNUMBEROFPOINTSTEXT)->SetWindowTextA(std::to_string(m_currentImageDescriptor.m_totalNumberOfPointsInROI).c_str());
			GetDlgItem(IDC_AVERAGEPOINTTEXT)->SetWindowTextA(averagePointText.c_str());
			GetDlgItem(IDC_XSTDDEVTEXT)->SetWindowTextA(std::to_string(m_currentImageDescriptor.m_globalStdDev[0]).c_str());
			GetDlgItem(IDC_YSTDDEVTEXT)->SetWindowTextA(std::to_string(m_currentImageDescriptor.m_globalStdDev[1]).c_str());
			GetDlgItem(IDC_SKEWNESSTEXT)->SetWindowTextA(std::to_string(m_currentImageDescriptor.m_globalSkewness).c_str());

			// Button UI Update
			GetDlgItem(IDC_BUTTON2)->EnableWindow(TRUE);
			GetDlgItem(IDC_CLEARBTN)->EnableWindow(TRUE);
		}
	}
}

void CCarScratchDetectorDlg::OnBnClickedClearbtn()
{
	// TODO: Add your control notification handler code here
	if (m_currentMat.data)
	{
		m_currentMat.release();
		m_currentFileName = "";
		m_currentImageDescriptor = ImageDescriptor();
		GetDlgItem(IDC_OPENIMAGEFILEBTN)->EnableWindow(TRUE);
		GetDlgItem(IDC_CLEARBTN)->EnableWindow(FALSE);
		GetDlgItem(IDC_BUTTON2)->EnableWindow(FALSE);

		m_imageHeightText.SetWindowTextA("N/A");
		m_imageWidthText.SetWindowTextA("N/A");

		cv::destroyAllWindows();
	}
}

void CCarScratchDetectorDlg::OnCurrentImageTestButton()
{
	if (m_currentMat.data == nullptr && m_currentImageDescriptor.m_totalNumberOfPointsInROI == 0)
	{
		MessageBox("Open Image Please");
		return;
	}
	else
	{
		cv::destroyAllWindows();
	}

	auto& inputImageDescriptor = m_currentImageDescriptor;

	// TODO: Add your control notification handler code here
	double currentSmallestDistance = 1000000000000000000;
	std::string currentSmallestFileName;

	// 기존에 있던 녀석들과 현재 입력된 테스트 이미지와의 score를 계산한다.
	std::map<std::string, double> distanceRecord;
	for (auto& eachDescriptor : m_loadedImageDescriptorsMap)
	{
		auto curScore = ImageDescriptor::CalculateFeatureDistance(inputImageDescriptor, eachDescriptor.second);
		distanceRecord[eachDescriptor.first] = curScore;
	}

	// Declaring the type of Predicate that accepts 2 pairs and return a bool
	typedef std::function<bool(std::pair<std::string, double>, std::pair<std::string, double>)> Comparator;

	// Defining a lambda function to compare two pairs. It will compare two pairs using second field
	Comparator compFunctor =
		[](std::pair<std::string, double> elem1, std::pair<std::string, double> elem2)
	{
		return elem1.second <= elem2.second;
	};

	// Declaring a set that will store the pairs using above comparision logic
	std::set<std::pair<std::string, double>, Comparator> sortedDistanceSet(
		distanceRecord.begin(), distanceRecord.end(), compFunctor);

	int index = 0;
	std::string result = "";
	RankingDialog dlg;

	for (auto& element : sortedDistanceSet)
	{
		dlg.m_fileNames.push_back(element.first);
		dlg.m_distanceRecord.push_back(element.second);
	}

	dlg.DoModal();
}

// Load DB Button
void CCarScratchDetectorDlg::OnBnClickedLoaddbbutton()
{
	// TODO: Add your control notification handler code here
	char szFilter[] = "Text|*.txt";

	CFileDialog dlg(TRUE, NULL, NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter, AfxGetMainWnd());

	if (dlg.DoModal() == IDOK)
	{
		if (m_loadedImageDescriptorsMap.size() != 0)
		{
			m_loadedImageDescriptorsMap.clear();
			m_currentFileName = "";
			m_currentImageDescriptor = ImageDescriptor();
		}

		int number_of_ImageFiles = -1;
		std::string line;
		std::ifstream myfile(dlg.GetPathName());

		while (std::getline(myfile, line))
		{
			++number_of_ImageFiles;
		}

		myfile.close();
		std::ifstream fileForRead(dlg.GetPathName());
		std::getline(fileForRead, line);

		for (int i = 0; i < number_of_ImageFiles; ++i)
		{
			int findIndex = 0;
			std::string fileName;
			fileForRead >> fileName;

			if (fileName == "")
			{
				continue;
			}

			m_loadedImageDescriptorsMap[fileName] = ImageDescriptor();

			ImageDescriptor& currentImageDescriptor = m_loadedImageDescriptorsMap[fileName];

			fileForRead >> currentImageDescriptor.m_totalNumberOfPointsInROI;
			fileForRead >> currentImageDescriptor.m_globalMeanPosition.x;
			fileForRead >> currentImageDescriptor.m_globalMeanPosition.y;
			fileForRead >> currentImageDescriptor.m_globalStdDev[0];
			fileForRead >> currentImageDescriptor.m_globalStdDev[1];
			fileForRead >> currentImageDescriptor.m_globalSkewness;
			fileForRead >> currentImageDescriptor.m_globalDensityInROI;
			fileForRead >> currentImageDescriptor.m_globalDensityInEffectiveROI;
			fileForRead >> currentImageDescriptor.m_totalNumberOfPointsOfLargestCluster;
			fileForRead >> currentImageDescriptor.m_largestClusterMeanPosition.x;
			fileForRead >> currentImageDescriptor.m_largestClusterMeanPosition.y;
			fileForRead >> currentImageDescriptor.m_largestClusterStdDev[0];
			fileForRead >> currentImageDescriptor.m_largestClusterStdDev[1];
			fileForRead >> currentImageDescriptor.m_largestClusterSkewness;
			fileForRead >> currentImageDescriptor.m_numberOfDenseClusters;
			fileForRead >> currentImageDescriptor.m_largestClusterEigenvalueRatio;
			fileForRead >> currentImageDescriptor.m_largestClusterLargeEigenValue;
			fileForRead >> currentImageDescriptor.m_largestClusterSmallEigenValue;
			fileForRead >> currentImageDescriptor.m_largestClusterOrientation;
		}
	}
}

void CCarScratchDetectorDlg::OnBnClickedTrainingsessionbutton()
{
	// TODO: Add your control notification handler code here
	TrainingDialog dlg;
	dlg.DoModal();
}
