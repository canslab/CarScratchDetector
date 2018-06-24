
// CarScratchDetectorDlg.cpp : ���� ����
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

// CCarScratchDetectorDlg ��ȭ ����

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
// CCarScratchDetectorDlg �޽��� ó����

BOOL CCarScratchDetectorDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// �� ��ȭ ������ �������� �����մϴ�.  ���� ���α׷��� �� â�� ��ȭ ���ڰ� �ƴ� ��쿡��
	//  �����ӿ�ũ�� �� �۾��� �ڵ����� �����մϴ�.
	SetIcon(m_hIcon, TRUE);			// ū �������� �����մϴ�.
	SetIcon(m_hIcon, FALSE);		// ���� �������� �����մϴ�.

									// TODO: ���⿡ �߰� �ʱ�ȭ �۾��� �߰��մϴ�.	
	m_spatialRadiusEditBox.SetWindowTextA("10");
	m_colorRadiusEditBox.SetWindowTextA("16");

	GetDlgItem(IDC_BUTTON2)->EnableWindow(FALSE);
	GetDlgItem(IDC_CLEARBTN)->EnableWindow(FALSE);
	return TRUE;  // ��Ŀ���� ��Ʈ�ѿ� �������� ������ TRUE�� ��ȯ�մϴ�.
}

// ��ȭ ���ڿ� �ּ�ȭ ���߸� �߰��� ��� �������� �׸�����
//  �Ʒ� �ڵ尡 �ʿ��մϴ�.  ����/�� ���� ����ϴ� MFC ���� ���α׷��� ��쿡��
//  �����ӿ�ũ���� �� �۾��� �ڵ����� �����մϴ�.

void CCarScratchDetectorDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // �׸��⸦ ���� ����̽� ���ؽ�Ʈ�Դϴ�.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Ŭ���̾�Ʈ �簢������ �������� ����� ����ϴ�.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// �������� �׸��ϴ�.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// ����ڰ� �ּ�ȭ�� â�� ���� ���ȿ� Ŀ���� ǥ�õǵ��� �ý��ۿ���
//  �� �Լ��� ȣ���մϴ�.
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

	// ������ �ִ� �༮��� ���� �Էµ� �׽�Ʈ �̹������� score�� ����Ѵ�.
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
