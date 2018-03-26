
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
}

BEGIN_MESSAGE_MAP(CCarScratchDetectorDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_OPENIMAGEFILEBTN, &CCarScratchDetectorDlg::OnBnClickedOpenimagefilebtn)
	ON_BN_CLICKED(IDC_SHOWBODYPART, &CCarScratchDetectorDlg::OnBnClickedShowbodypart)
	ON_BN_CLICKED(IDC_RUNBUTTON, &CCarScratchDetectorDlg::OnBnClickedRunbutton)
	ON_BN_CLICKED(IDC_SAVERESULTBTN, &CCarScratchDetectorDlg::OnBnClickedSaveresultbtn)
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
	m_spatialRadiusEditBox.SetWindowTextA("16");
	m_colorRadiusEditBox.SetWindowTextA("16");

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