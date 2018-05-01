
// CarScratchDetectorDlg.cpp : ���� ����
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

// CCarScratchDetectorDlg ��ȭ ����

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
	//  DDX_Control(pDX, IDC_EDGEMAPCHECK, m_bCheckEdgeMap);
	DDX_Control(pDX, IDC_LABELMAPCHECK, m_bCheckLabelMap);
	DDX_Control(pDX, IDC_CORNERMAPCHECK, m_bCheckCornerMap);
	//  DDX_Control(pDX, IDC_ANALYZE, m_anaylzeButton);
	//  DDX_Control(pDX, IDC_BODYMAPCHECK, m_bBodyMap);
	//  DDX_Control(pDX, IDC_CONTOURMAPCHECK, m_bCheckBodyMap);
	DDX_Control(pDX, IDC_CONTOURMAPCHECK, m_bCheckContourMap);
	DDX_Control(pDX, IDC_GRADIENTMAPCHECK, m_bGradientMapCheck);
}

BEGIN_MESSAGE_MAP(CCarScratchDetectorDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_OPENIMAGEFILEBTN, &CCarScratchDetectorDlg::OnBnClickedOpenimagefilebtn)
	ON_BN_CLICKED(IDC_RUNBUTTON, &CCarScratchDetectorDlg::OnBnClickedRunbutton)
	ON_BN_CLICKED(IDC_SAVERESULTBTN, &CCarScratchDetectorDlg::OnBnClickedSaveresultbtn)
	ON_BN_CLICKED(IDC_ANALYZEBTN, &CCarScratchDetectorDlg::OnBnClickedAnalyze)
	ON_BN_CLICKED(IDC_CLEARBTN, &CCarScratchDetectorDlg::OnBnClickedClearbtn)
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

	// Anaylze�� �ش��ϴ� üũ �ڽ� ���� ����
	m_bCheckCornerMap.SetCheck(0);
	m_bCheckLabelMap.SetCheck(0);
	m_bCheckContourMap.SetCheck(0);
	m_bGradientMapCheck.SetCheck(0);
	
	GetDlgItem(IDC_ANALYZEBTN)->EnableWindow(FALSE);
	GetDlgItem(IDC_RUNBUTTON)->EnableWindow(FALSE);
	GetDlgItem(IDC_CLEARBTN)->EnableWindow(FALSE);
	GetDlgItem(IDC_SAVERESULTBTN)->EnableWindow(FALSE);
	//GetDlgItem(IDC_ANA)
	//GetDlgItem(IDC_ANALYZE)

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
			// �̹��� ������ ����
			cv::resize(m_srcMat, m_srcMat, cv::Size(resizedWidth, resizedHeight));
		}

		// If input image exists
		if (m_srcMat.data)
		{
			m_imageWidthText.SetWindowTextA(std::to_string(m_srcMat.cols).c_str());
			m_imageHeightText.SetWindowTextA(std::to_string(m_srcMat.rows).c_str());
			GetDlgItem(IDC_OPENIMAGEFILEBTN)->EnableWindow(FALSE);
			GetDlgItem(IDC_ANALYZEBTN)->EnableWindow(TRUE);
			GetDlgItem(IDC_RUNBUTTON)->EnableWindow(TRUE);
			GetDlgItem(IDC_CLEARBTN)->EnableWindow(TRUE);
			GetDlgItem(IDC_SAVERESULTBTN)->EnableWindow(TRUE);
		}
		cv::imshow("Input Image", m_srcMat);
	}
}

void CCarScratchDetectorDlg::OnBnClickedShowbodypart()
{

}

void CCarScratchDetectorDlg::OnBnClickedRunbutton()
{
	// TODO: Add your control notification handler code here
	//ExtractCarBody

	AlgorithmParameter param;
	AlgorithmResult result;

	CString spatialRadiusString, colorRadiusString;
	m_spatialRadiusEditBox.GetWindowTextA(spatialRadiusString);
	m_colorRadiusEditBox.GetWindowTextA(colorRadiusString);

	param.m_spatialBandwidth = atof(spatialRadiusString);
	param.m_colorBandwidth = atof(colorRadiusString);
	
	param.m_bGetGradientMap = (bool)IsDlgButtonChecked(IDC_GRADIENTMAPCHECK);
	param.m_bGetColorLabelMap = (bool)IsDlgButtonChecked(IDC_LABELMAPCHECK);
	param.m_bGetCornerMap = (bool)IsDlgButtonChecked(IDC_CORNERMAPCHECK);
	param.m_bGetContouredMap = (bool)IsDlgButtonChecked(IDC_CONTOURMAPCHECK);

	if (m_srcMat.data != nullptr)
	{
		ExtractCarBody(m_srcMat, param, result);
		GetDlgItem(IDC_ANALYZEBTN)->EnableWindow(TRUE);
	}
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

void CCarScratchDetectorDlg::OnBnClickedAnalyze()
{
	// TODO: Add your control notification handler code here
	
}

void CCarScratchDetectorDlg::OnBnClickedClearbtn()
{
	// TODO: Add your control notification handler code here
	if (m_srcMat.data)
	{
		m_srcMat.release();
		GetDlgItem(IDC_OPENIMAGEFILEBTN)->EnableWindow(TRUE);
		GetDlgItem(IDC_ANALYZEBTN)->EnableWindow(FALSE);
		GetDlgItem(IDC_RUNBUTTON)->EnableWindow(FALSE);
		GetDlgItem(IDC_CLEARBTN)->EnableWindow(FALSE);
		GetDlgItem(IDC_SAVERESULTBTN)->EnableWindow(FALSE);

		m_imageHeightText.SetWindowTextA("N/A");
		m_imageWidthText.SetWindowTextA("N/A");

		cv::destroyAllWindows();
	}
}
