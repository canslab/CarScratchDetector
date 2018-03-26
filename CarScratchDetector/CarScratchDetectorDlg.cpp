
// CarScratchDetectorDlg.cpp : 구현 파일
//

#include "stdafx.h"
#include "CarScratchDetector.h"
#include "CarScratchDetectorDlg.h"
#include "afxdialogex.h"
#include "AlgorithmCode\AlgorithmCollection.h"
#include "CarNumberRemoveCode\LPdetection.h"

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
	DDX_Control(pDX, IDC_INPUTIMAGEWIDTH, m_imageWidthText);
	DDX_Control(pDX, IDC_INPUTIMAGEHEIGHT, m_imageHeightText);
	DDX_Control(pDX, IDC_SPATIALRADIUSEDIT, m_spatialRadiusEditBox);
	DDX_Control(pDX, IDC_COLORRADIUSEDIT, m_colorRadiusEditBox);
}

BEGIN_MESSAGE_MAP(CCarScratchDetectorDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_OPENIMAGEFILEBTN, &CCarScratchDetectorDlg::OnBnClickedOpenimagefilebtn)
	ON_BN_CLICKED(IDC_SHOWBODYPART, &CCarScratchDetectorDlg::OnBnClickedShowbodypart)
	ON_BN_CLICKED(IDC_RUNBUTTON, &CCarScratchDetectorDlg::OnBnClickedRunbutton)
	ON_BN_CLICKED(IDC_SAVERESULTBTN, &CCarScratchDetectorDlg::OnBnClickedSaveresultbtn)
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
	m_spatialRadiusEditBox.SetWindowTextA("16");
	m_colorRadiusEditBox.SetWindowTextA("16");

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
		if (m_srcMat.rows > 0)
		{
			m_srcMat.release();
		}

		m_srcMat = cv::imread(cv::String(cstrImgPath).c_str());

		RemoveNumberPlate(m_srcMat);

		// To reduce the image's resolution.
		double aspectRatio = (double)m_srcMat.rows / m_srcMat.cols;
		int resizedWidth = 0, resizedHeight = 0, accValue = 0, totalPixel = 0;

		do
		{
			accValue += 10;
			totalPixel = (int)(accValue * accValue * aspectRatio);

		} while (totalPixel <= 150000);

		resizedWidth = accValue;
		resizedHeight = aspectRatio * accValue;

		if (resizedWidth <= m_srcMat.cols && resizedHeight <= m_srcMat.rows)
		{
			// 이미지 사이즈 조정
			cv::resize(m_srcMat, m_srcMat, cv::Size(resizedWidth, resizedHeight));
		}

		// If input image exists
		if (m_srcMat.rows > 0)
		{
			m_imageWidthText.SetWindowTextA(std::to_string(m_srcMat.cols).c_str());
			m_imageHeightText.SetWindowTextA(std::to_string(m_srcMat.rows).c_str());
		}

		cv::imshow("Input Image", m_srcMat);
	}
}

void CCarScratchDetectorDlg::OnBnClickedShowbodypart()
{
	// TODO: Add your control notification handler code here
	//ExtractCarBody
	AlgorithmParameter param;
	AlgorithmResult result;

	CString spatialRadiusString, colorRadiusString;
	m_spatialRadiusEditBox.GetWindowTextA(spatialRadiusString);
	m_colorRadiusEditBox.GetWindowTextA(colorRadiusString);

	param.SetSpatialBandwidth(atof(spatialRadiusString));
	param.SetColorBandwidth(atof(colorRadiusString));

	if (m_srcMat.data != nullptr)
	{
		ExtractCarBody(m_srcMat, param, result);
	}
}

void CCarScratchDetectorDlg::OnBnClickedRunbutton()
{
	// TODO: Add your control notification handler code here

}

void CCarScratchDetectorDlg::OnBnClickedSaveresultbtn()
{
	// TODO: Add your control notification handler code here
	char szFilter[] = "Image|*.BMP;*.PNG;*.JPG;*.JPEG";

	if (m_resultMat.rows > 0)
	{
		CFileDialog dlg(FALSE, NULL, NULL, OFN_HIDEREADONLY, szFilter, AfxGetMainWnd());
		if (IDOK == dlg.DoModal())
		{
			CString strPathName = dlg.GetPathName();
			cv::imwrite(cv::String(strPathName) + ".jpg", m_resultMat);
			MessageBox("Save Complete");
		}
	}
	else
	{
		MessageBox("Experiment Result doesn't exist");
	}
}